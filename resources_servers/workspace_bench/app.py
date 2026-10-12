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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
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
from resources_servers.workspace_bench.dataset import resolve_task_path


LOG = logging.getLogger(__name__)


class WorkspaceBenchConfig(BaseResourcesServerConfig):
    artifact_root: Path | None = None
    judge_base_url: str
    judge_api_key: str
    judge_model: str
    judge_timeout_s: float = 600
    num_processes: int = 10
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class WorkspaceBenchRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_dir: str


class WorkspaceBenchSeedResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]


class WorkspaceBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_dir: str
    artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


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
        self._reference_dirs: dict[str, tempfile.TemporaryDirectory] = {}
        self._semaphore = asyncio.Semaphore(self.config.num_processes)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                sandboxes = list(self._sandboxes.values())
                self._sandboxes.clear()
                await asyncio.gather(*(sandbox.stop() for sandbox in sandboxes), return_exceptions=True)
                for directory in self._reference_dirs.values():
                    directory.cleanup()
                self._reference_dirs.clear()

        app.router.lifespan_context = lifespan
        return app

    async def seed_session(self, request: Request, body: WorkspaceBenchRequest) -> WorkspaceBenchSeedResponse:
        session_id = str(request.session[SESSION_ID_KEY])
        if session_id in self._sandboxes or session_id in self._reference_dirs:
            raise HTTPException(status_code=409, detail="Workspace-Bench session is already active")
        task_dir = await asyncio.to_thread(resolve_task_path, body.task_dir)
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
            metadata={
                **resolve_provider_metadata(self.config.sandbox_provider, get_global_config_dict()),
                **sandbox_config.get("metadata", {}),
                "task-id": body.task_id[:63],
            },
            resources=SandboxResources.from_mapping(sandbox_config.get("resources", {})),
            provider_options=sandbox_config.get("provider_options", {}),
        )
        reference_dir = tempfile.TemporaryDirectory()
        self._reference_dirs[session_id] = reference_dir
        try:
            await sandbox.start(spec)
            archive = Path(reference_dir.name) / "input.tar.gz"
            manifest = metadata.get("data_manifest") or []

            def build_archive() -> None:
                with tarfile.open(archive, "w:gz", dereference=True) as tar:
                    for item in manifest:
                        source = task_dir / item["stored_relpath"]
                        if source.is_file():
                            tar.add(source, arcname=item["filename"])

            await asyncio.to_thread(build_archive)
            (Path(reference_dir.name) / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            await sandbox.upload(archive, "/tmp/input.tar.gz")
            result = await sandbox.exec(
                "mkdir -p /workspace/input /workspace/output /workspace/.opencode /workspace/.claude "
                "&& ln -s /opt/workspace-bench/office-skills /workspace/.opencode/skills "
                "&& ln -s /opt/workspace-bench/agent-homes/claude/.claude/skills /workspace/.claude/skills "
                "&& tar -xzf /tmp/input.tar.gz -C /workspace/input",
                cwd="/",
            )
            if result.return_code != 0:
                raise RuntimeError(f"Failed to seed Workspace-Bench input: {result.stderr}")
            descriptor = await sandbox.serialize()
        except BaseException:
            self._reference_dirs.pop(session_id)
            reference_dir.cleanup()
            try:
                await sandbox.stop()
            except Exception:
                LOG.warning("Workspace-Bench seed sandbox cleanup failed for task %s", body.task_id, exc_info=True)
            raise
        self._sandboxes[session_id] = sandbox
        return WorkspaceBenchSeedResponse(sandbox_handle=descriptor["sandbox_id"], sandbox_descriptor=descriptor)

    async def _judge(self, case_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        metadata_path = case_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["__metadata_path"] = "/judge/task/metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
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
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        sandbox = AsyncSandbox(provider)
        try:
            await sandbox.start(SandboxSpec(**self.config.sandbox_config))
            with tempfile.TemporaryDirectory() as temporary_dir:
                archive = Path(temporary_dir) / "judge-task.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    tar.add(case_dir, arcname="task")
                await sandbox.upload(archive, "/tmp/judge-task.tar.gz")
            result = await sandbox.exec(
                "mkdir -p /judge && tar -xzf /tmp/judge-task.tar.gz -C /judge && "
                "python3 /workspace/Workspace-Bench/evaluation/src/agent_as_a_judge.py "
                "--task-dir /judge/task --eval-yaml /judge/task/judge.yaml --overwrite",
                cwd="/workspace/Workspace-Bench/evaluation",
                timeout_s=self.config.judge_timeout_s,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Workspace-Bench judge failed: {result.stderr or result.stdout}")
            for name in ("rubrics_judge--gym-judge.json", "dependency_graph--gym-judge.json"):
                await sandbox.download(f"/judge/task/{name}", case_dir / name)
        finally:
            try:
                await sandbox.stop()
            except Exception:
                LOG.warning("Workspace-Bench judge sandbox cleanup failed", exc_info=True)
        judged = json.loads((case_dir / "rubrics_judge--gym-judge.json").read_text(encoding="utf-8"))
        graph = json.loads((case_dir / "dependency_graph--gym-judge.json").read_text(encoding="utf-8"))
        judge_error = (judged.get("judge") or {}).get("error")
        # Unparseable verdicts fail their rubrics, as upstream does. Other judge errors are infrastructure failures.
        if judge_error and judge_error != "Judge output parse failed":
            raise JudgeError(f"Workspace-Bench judge failed: {judge_error}")
        return judged["rubrics"], graph

    async def verify(self, request: Request, body: WorkspaceBenchVerifyRequest) -> WorkspaceBenchVerifyResponse:
        session_id = str(request.session[SESSION_ID_KEY])
        replay = body.artifact_id is not None
        saved = None
        if self.config.artifact_root is not None:
            body.artifact_id = body.artifact_id or hashlib.sha256(session_id.encode()).hexdigest()
            saved = self.config.artifact_root / body.artifact_id
            if replay:
                if not (saved / "input.tar.gz").is_file():
                    raise ValueError("Workspace-Bench artifact lacks frozen original inputs; legacy replay is unsafe")
                body = WorkspaceBenchVerifyRequest.model_validate_json((saved / "request.json").read_text())
                if (saved / "result.json").exists():
                    return WorkspaceBenchVerifyResponse.model_validate_json((saved / "result.json").read_text())
            else:
                saved.mkdir(parents=True, exist_ok=False)
        elif replay:
            raise HTTPException(status_code=400, detail="Artifact replay requires artifact_root")
        if not replay and (session_id not in self._sandboxes or session_id not in self._reference_dirs):
            raise HTTPException(status_code=400, detail="Workspace-Bench session is not active")
        sandbox = None if replay else self._sandboxes.pop(session_id)
        reference_dir = None if replay else self._reference_dirs.pop(session_id)
        references = saved if replay else Path(reference_dir.name)
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                local_dir = Path(temporary_dir) / "judge"
                local_dir.mkdir()
                archive = (saved or Path(temporary_dir)) / "workspace.tar.gz"
                if sandbox is not None:
                    if saved is not None:
                        shutil.copyfile(references / "metadata.json", saved / "metadata.json")
                        shutil.copyfile(references / "input.tar.gz", saved / "input.tar.gz")
                        pending = saved / "request.partial"
                        pending.write_text(body.model_dump_json())
                        pending.replace(saved / "request.json")
                    result = await sandbox.exec("tar -czf /tmp/workspace.tar.gz -C /workspace input output")
                    if result.return_code != 0:
                        raise RuntimeError(
                            f"Failed to collect Workspace-Bench files: {result.stderr}; output={(result.stdout or '')[-4000:]}"
                        )
                    partial = archive.with_suffix(".partial")
                    await sandbox.download("/tmp/workspace.tar.gz", partial)
                    with partial.open("rb") as stream:
                        os.fsync(stream.fileno())
                    partial.replace(archive)

                def prepare_judge_input() -> None:
                    with tarfile.open(archive, "r:gz") as tar:
                        # Grade only real files under output/, never symlinks into the sandbox.
                        members = (
                            member
                            for member in tar
                            if (member.isfile() or member.isdir()) and Path(member.name).parts[:1] == ("output",)
                        )
                        tar.extractall(local_dir, members=members, filter="data")
                    shutil.copyfile(references / "metadata.json", local_dir / "metadata.json")
                    (local_dir / "data").mkdir()
                    with tarfile.open(references / "input.tar.gz", "r:gz") as tar:
                        tar.extractall(local_dir / "data", filter="data")

                await asyncio.to_thread(prepare_judge_input)
                async with self._semaphore:
                    try:
                        rubrics, dependency_graph = await self._judge(local_dir)
                    finally:
                        judge_output = local_dir / "rubrics_judge--gym-judge.json"
                        if saved is not None and judge_output.is_file():
                            try:
                                judged = json.loads(judge_output.read_text(encoding="utf-8"))
                                if not isinstance(judged, dict) or not isinstance(judged.get("judge", {}), dict):
                                    raise ValueError("Invalid judge receipt shape")
                                receipt = {
                                    key: judged[key]
                                    for key in ("taskId", "agentKind", "createdAt", "rubrics", "summary")
                                    if key in judged
                                }
                                receipt["judge"] = {
                                    key: judged["judge"][key]
                                    for key in (
                                        "model",
                                        "modelName",
                                        "usage",
                                        "durationMs",
                                        "tries",
                                        "error",
                                        "rawResponseHead",
                                    )
                                    if key in (judged.get("judge") or {})
                                }
                                # Keep only verdict fields. Endpoint fields can contain credentials.
                                with (saved / f"judge-receipt-{uuid4().hex}.json").open(
                                    "x", encoding="utf-8"
                                ) as stream:
                                    json.dump(receipt, stream)
                            except (OSError, ValueError, TypeError) as exc:
                                LOG.warning("Workspace-Bench judge receipt retention failed (%s)", type(exc).__name__)
        finally:
            if sandbox is not None:
                try:
                    await sandbox.stop()
                except Exception:
                    LOG.warning("Workspace-Bench task sandbox cleanup failed for task %s", body.task_id, exc_info=True)
            if reference_dir is not None:
                reference_dir.cleanup()
        passed = sum(item["passed"] for item in rubrics)
        total = len(rubrics)
        if not total:
            raise ValueError("Workspace-Bench judge returned no rubrics")
        response = WorkspaceBenchVerifyResponse(
            **body.model_dump(),
            reward=passed / total,
            passed_count=passed,
            total_count=total,
            judge_model=self.config.judge_model,
            grading_protocol="workspace-frozen-inputs-v1",
            rubrics=rubrics,
            dependency_graph=dependency_graph,
        )
        if saved is not None:
            pending = saved / "result.partial"
            pending.write_text(response.model_dump_json())
            pending.replace(saved / "result.json")
        return response


if __name__ == "__main__":
    WorkspaceBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = WorkspaceBenchResourcesServer.run_webserver()  # noqa: F401
