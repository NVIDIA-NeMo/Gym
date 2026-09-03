# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.job_bench.vendor import judge


class JobBenchConfig(BaseResourcesServerConfig):
    judge_base_url: str
    judge_api_key: str
    judge_model: str
    max_judge_workers: int = 10
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class JobBenchRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_dir: str
    rubrics_file: str


class JobBenchSeedResponse(BaseSeedSessionResponse):
    sandbox_handle: str


class JobBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_dir: str
    rubrics_file: str


class JobBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str
    score: float
    max_score: float
    passed_count: int
    total_count: int
    judge_model: str
    rubrics: list[dict[str, Any]]


class JobBenchResourcesServer(SimpleResourcesServer):
    config: JobBenchConfig

    def model_post_init(self, context: Any, /) -> None:
        self._sandboxes: dict[str, AsyncSandbox] = {}

    async def seed_session(self, request: Request, body: JobBenchRequest) -> JobBenchSeedResponse:
        task_dir = Path(body.task_dir)
        if not (task_dir / "task_folder" / "TASK_INSTRUCTIONS.txt").is_file():
            raise ValueError(f"Invalid Job-Bench task directory: {task_dir}")

        global_config = get_global_config_dict()
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
        sandbox = AsyncSandbox(provider)
        resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        spec = SandboxSpec(
            image=self.config.sandbox_config["image"],
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir="/workspace",
            env={},
            files={},
            metadata={
                **resolve_provider_metadata(self.config.sandbox_provider, global_config),
                **self.config.sandbox_config.get("metadata", {}),
                "task_id": body.task_id[:63],
            },
            resources=resources,
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        await sandbox.start(spec)

        with tempfile.TemporaryDirectory() as temporary_dir:
            archive = Path(temporary_dir) / "task.tar.gz"
            with tarfile.open(archive, "w:gz", dereference=True) as tar:
                tar.add(task_dir / "task_folder", arcname="task")
            await sandbox.upload(archive, "/tmp/task.tar.gz")
        result = await sandbox.exec(
            "mkdir -p /workspace/output && tar -xzf /tmp/task.tar.gz -C /workspace",
            cwd="/",
        )
        if result.return_code != 0:
            await sandbox.stop()
            raise RuntimeError(f"Failed to seed Job-Bench task: {result.stderr}")

        session_id = request.session[SESSION_ID_KEY]
        self._sandboxes[session_id] = sandbox
        return JobBenchSeedResponse(sandbox_handle=sandbox._handle.sandbox_id)

    def _judge(self, output_dir: Path, rubrics_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rubrics_data = json.loads(rubrics_file.read_text(encoding="utf-8"))
        rubrics = rubrics_data.get("rubrics") or rubrics_data.get("evaluation_rubrics") or []
        if not any(path.is_file() for path in output_dir.rglob("*")):
            results = [
                judge.build_failed_rubric_result(index, rubric, "No output files found in the model output directory.")
                for index, rubric in enumerate(rubrics)
            ]
            return judge.build_scorecard(results), results
        file_contents = judge.extract_all_file_contents(output_dir)
        if not file_contents.strip():
            results = [
                judge.build_failed_rubric_result(
                    index, rubric, "Output files were unreadable or empty after conversion."
                )
                for index, rubric in enumerate(rubrics)
            ]
            return judge.build_scorecard(results), results
        images = judge.collect_image_attachments(output_dir)

        with ThreadPoolExecutor(max_workers=self.config.max_judge_workers) as executor:
            futures = [
                executor.submit(
                    judge.judge_rubric,
                    index,
                    rubric,
                    file_contents,
                    self.config.judge_model,
                    self.config.judge_base_url,
                    self.config.judge_api_key,
                    300,
                    3,
                    images,
                )
                for index, rubric in enumerate(rubrics)
            ]
            results = [future.result()[0] for future in futures]
        return judge.build_scorecard(results), results

    async def verify(self, request: Request, body: JobBenchVerifyRequest) -> JobBenchVerifyResponse:
        sandbox = self._sandboxes.pop(request.session[SESSION_ID_KEY])
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                local_dir = Path(temporary_dir)
                archive = local_dir / "output.tar.gz"
                result = await sandbox.exec("tar -czf /tmp/output.tar.gz -C /workspace/output .")
                if result.return_code != 0:
                    raise RuntimeError(f"Failed to collect Job-Bench output: {result.stderr}")
                await sandbox.download("/tmp/output.tar.gz", archive)
                output_dir = local_dir / "output"
                output_dir.mkdir()
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(output_dir, filter="data")
                scorecard, rubrics = await asyncio.to_thread(self._judge, output_dir, Path(body.rubrics_file))
        finally:
            await sandbox.stop()

        return JobBenchVerifyResponse(
            **body.model_dump(),
            reward=float(scorecard["normalized_score"]),
            score=float(scorecard["total_score"]),
            max_score=float(scorecard["max_score"]),
            passed_count=int(scorecard["passed_count"]),
            total_count=int(scorecard["total_count"]),
            judge_model=self.config.judge_model,
            rubrics=rubrics,
        )


if __name__ == "__main__":
    JobBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = JobBenchResourcesServer.run_webserver()  # noqa: F401
