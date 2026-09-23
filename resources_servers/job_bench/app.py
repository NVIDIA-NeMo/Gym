# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.judge import JudgeError
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.job_bench.task_data import resolve_task_path
from resources_servers.job_bench.vendor import judge


LOG = logging.getLogger(__name__)


class JobBenchConfig(BaseResourcesServerConfig):
    artifact_root: Path | None = None
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
    sandbox_descriptor: dict[str, Any]


class JobBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_dir: str
    rubrics_file: str
    artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


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
        task_dir = await asyncio.to_thread(resolve_task_path, body.task_dir)
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
        try:
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
                raise RuntimeError(f"Failed to seed Job-Bench task: {result.stderr}")
            descriptor = await sandbox.serialize()
        except BaseException:
            try:
                await sandbox.stop()
            except Exception:
                LOG.warning("Job-Bench seed sandbox cleanup failed for task %s", body.task_id, exc_info=True)
            raise

        session_id = request.session[SESSION_ID_KEY]
        self._sandboxes[session_id] = sandbox
        return JobBenchSeedResponse(sandbox_handle=descriptor["sandbox_id"], sandbox_descriptor=descriptor)

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
            judged = [future.result() for future in futures]
        receipt = [
            {
                "result": result,
                "debug": {
                    key: debug[key]
                    for key in (
                        "parse_status",
                        "api_exit_code",
                        "raw_response",
                        "error",
                        "vision_used",
                        "attached_images",
                    )
                    if key in debug
                },
            }
            for result, debug in judged
        ]
        try:
            (output_dir.parent / "judge-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        except (OSError, ValueError, TypeError) as exc:
            LOG.warning("Job-Bench judge receipt retention failed (%s)", type(exc).__name__)
        errors = [debug["error"] for _, debug in judged if debug["api_exit_code"] == 2]
        if errors:
            raise JudgeError("; ".join(errors))
        results = [result for result, _ in judged]
        return judge.build_scorecard(results), results

    async def verify(self, request: Request, body: JobBenchVerifyRequest) -> JobBenchVerifyResponse:
        session_id = str(request.session[SESSION_ID_KEY])
        replay = body.artifact_id is not None
        saved = None
        if self.config.artifact_root is not None:
            body.artifact_id = body.artifact_id or hashlib.sha256(session_id.encode()).hexdigest()
            saved = self.config.artifact_root / body.artifact_id
            if replay:
                body = JobBenchVerifyRequest.model_validate_json((saved / "request.json").read_text())
                if (saved / "result.json").exists():
                    return JobBenchVerifyResponse.model_validate_json((saved / "result.json").read_text())
            else:
                saved.mkdir(parents=True, exist_ok=False)
        elif replay:
            raise HTTPException(status_code=400, detail="Artifact replay requires artifact_root")
        sandbox = None if replay else self._sandboxes.pop(session_id, None)
        if not replay and sandbox is None:
            raise HTTPException(status_code=400, detail="Job-Bench session is not active")
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                local_dir = Path(temporary_dir)
                archive = (saved or local_dir) / "output.tar.gz"
                if sandbox is not None:
                    if saved is not None:
                        rubrics_source = await asyncio.to_thread(resolve_task_path, body.rubrics_file)
                        shutil.copyfile(rubrics_source, saved / "rubrics.json")
                        pending = saved / "request.partial"
                        pending.write_text(body.model_dump_json())
                        pending.replace(saved / "request.json")
                    result = await sandbox.exec("tar -czf /tmp/output.tar.gz -C /workspace/output .")
                    if result.return_code != 0:
                        raise RuntimeError(
                            f"Failed to collect Job-Bench output: {result.stderr}; output={(result.stdout or '')[-4000:]}"
                        )
                    partial = archive.with_suffix(".partial")
                    await sandbox.download("/tmp/output.tar.gz", partial)
                    with partial.open("rb") as stream:
                        os.fsync(stream.fileno())
                    partial.replace(archive)
                output_dir = local_dir / "output"
                output_dir.mkdir()
                with tarfile.open(archive, "r:gz") as tar:
                    # Agent-created environments may contain links outside the output tree.
                    runtime_dirs = {"venv", ".venv", "node_modules", ".git", ".cache", "__pycache__"}
                    members = (
                        member
                        for member in tar
                        if (member.isfile() or member.isdir())
                        and not runtime_dirs.intersection(
                            Path(member.name).parts if member.isdir() else Path(member.name).parts[:-1]
                        )
                    )
                    tar.extractall(output_dir, members=members, filter="data")
                rubrics_file = (
                    saved / "rubrics.json"
                    if saved is not None
                    else await asyncio.to_thread(resolve_task_path, body.rubrics_file)
                )
                try:
                    scorecard, rubrics = await asyncio.to_thread(self._judge, output_dir, rubrics_file)
                finally:
                    receipt = local_dir / "judge-receipt.json"
                    if saved is not None and receipt.is_file():
                        try:
                            with (saved / f"judge-receipt-{uuid4().hex}.json").open("xb") as stream:
                                stream.write(receipt.read_bytes())
                        except OSError as exc:
                            LOG.warning("Job-Bench judge receipt retention failed (%s)", type(exc).__name__)
        finally:
            if sandbox is not None:
                try:
                    await sandbox.stop()
                except Exception:
                    LOG.warning("Job-Bench task sandbox cleanup failed for task %s", body.task_id, exc_info=True)

        response = JobBenchVerifyResponse(
            **body.model_dump(),
            reward=float(scorecard["normalized_score"]),
            score=float(scorecard["total_score"]),
            max_score=float(scorecard["max_score"]),
            passed_count=int(scorecard["passed_count"]),
            total_count=int(scorecard["total_count"]),
            judge_model=self.config.judge_model,
            grading_protocol=judge.INPUT_PROTOCOL,
            rubrics=rubrics,
        )
        if saved is not None:
            pending = saved / "result.partial"
            pending.write_text(response.model_dump_json())
            pending.replace(saved / "result.json")
        return response


if __name__ == "__main__":
    JobBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = JobBenchResourcesServer.run_webserver()  # noqa: F401
