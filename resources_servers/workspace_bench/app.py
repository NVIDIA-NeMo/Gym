# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
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
from resources_servers.workspace_bench.setup_upstream import ensure_upstream


class WorkspaceBenchConfig(BaseResourcesServerConfig):
    judge_base_url: str
    judge_api_key: str
    judge_model: str
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class WorkspaceBenchRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_dir: str


class WorkspaceBenchSeedResponse(BaseSeedSessionResponse):
    sandbox_handle: str


class WorkspaceBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_dir: str


class WorkspaceBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str
    passed_count: int
    total_count: int
    judge_model: str
    rubrics: list[dict[str, Any]]
    dependency_graph: dict[str, Any]


class WorkspaceBenchResourcesServer(SimpleResourcesServer):
    config: WorkspaceBenchConfig

    def model_post_init(self, context: Any, /) -> None:
        self._sandboxes: dict[str, AsyncSandbox] = {}
        self._upstream_dir = ensure_upstream()

    async def seed_session(self, request: Request, body: WorkspaceBenchRequest) -> WorkspaceBenchSeedResponse:
        task_dir = Path(body.task_dir)
        metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
        sandbox_config = self.config.sandbox_config
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        sandbox = AsyncSandbox(provider)
        spec = SandboxSpec(
            image=sandbox_config["image"],
            ttl_s=sandbox_config.get("ttl_s"),
            ready_timeout_s=sandbox_config.get("ready_timeout_s"),
            workdir="/workspace",
            env={},
            files={},
            metadata=resolve_provider_metadata(self.config.sandbox_provider, get_global_config_dict())
            | sandbox_config.get("metadata", {})
            | {"task-id": body.task_id[:63]},
            resources=SandboxResources.from_mapping(sandbox_config.get("resources", {})),
            provider_options=sandbox_config.get("provider_options", {}),
        )
        await sandbox.start(spec)
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                archive = Path(temporary_dir) / "input.tar.gz"
                manifest = metadata.get("data_manifest") or []
                with tarfile.open(archive, "w:gz", dereference=True) as tar:
                    for item in manifest:
                        source = task_dir / item["stored_relpath"]
                        if source.is_file():
                            tar.add(source, arcname=item["filename"])
                await sandbox.upload(archive, "/tmp/input.tar.gz")
            result = await sandbox.exec(
                "mkdir -p /workspace/input /workspace/output /workspace/.opencode "
                "&& ln -s /opt/workspace-bench/office-skills /workspace/.opencode/skills "
                "&& tar -xzf /tmp/input.tar.gz -C /workspace/input",
                cwd="/",
            )
            if result.return_code != 0:
                raise RuntimeError(f"Failed to seed Workspace-Bench input: {result.stderr}")
        except Exception:
            await sandbox.stop()
            raise
        self._sandboxes[str(request.session[SESSION_ID_KEY])] = sandbox
        descriptor = await sandbox.serialize()
        return WorkspaceBenchSeedResponse(sandbox_handle=descriptor["sandbox_id"])

    def _judge(self, case_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        evaluation_dir = self._upstream_dir / "evaluation"
        config_path = case_dir / "judge.yaml"
        config_path.write_text(
            json.dumps(
                {
                    "model_name": "gym-judge",
                    "baseUrl": self.config.judge_base_url,
                    "model": self.config.judge_model,
                    "apiKey": self.config.judge_api_key,
                }
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(evaluation_dir / "src" / "agent_as_a_judge.py"),
                "--task-dir",
                str(case_dir),
                "--eval-yaml",
                str(config_path),
                "--overwrite",
            ],
            cwd=evaluation_dir,
            check=True,
        )
        judged = json.loads((case_dir / "rubrics_judge--gym-judge.json").read_text(encoding="utf-8"))
        graph = json.loads((case_dir / "dependency_graph--gym-judge.json").read_text(encoding="utf-8"))
        return judged["rubrics"], graph

    async def verify(self, request: Request, body: WorkspaceBenchVerifyRequest) -> WorkspaceBenchVerifyResponse:
        sandbox = self._sandboxes.pop(str(request.session[SESSION_ID_KEY]))
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                local_dir = Path(temporary_dir)
                archive = local_dir / "workspace.tar.gz"
                result = await sandbox.exec("tar -czf /tmp/workspace.tar.gz -C /workspace input output")
                if result.return_code != 0:
                    raise RuntimeError(f"Failed to collect Workspace-Bench files: {result.stderr}")
                await sandbox.download("/tmp/workspace.tar.gz", archive)
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(local_dir, filter="data")
                metadata = json.loads((Path(body.task_dir) / "metadata.json").read_text(encoding="utf-8"))
                (local_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                shutil.copytree(local_dir / "input", local_dir / "data")
                rubrics, dependency_graph = await asyncio.to_thread(self._judge, local_dir)
        finally:
            await sandbox.stop()
        passed = sum(item["passed"] for item in rubrics)
        total = len(rubrics)
        return WorkspaceBenchVerifyResponse(
            **body.model_dump(),
            reward=passed / total if total else 0.0,
            passed_count=passed,
            total_count=total,
            judge_model=self.config.judge_model,
            rubrics=rubrics,
            dependency_graph=dependency_graph,
        )


if __name__ == "__main__":
    WorkspaceBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = WorkspaceBenchResourcesServer.run_webserver()  # noqa: F401
