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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import BeforeValidator, ConfigDict, Field

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JudgeError
from resources_servers.swebench.sandbox_sessions import SandboxSessionResourcesServer
from resources_servers.workspace_bench.dataset import resolve_task_path
from resources_servers.workspace_bench.task_data import TaskData


LOG = logging.getLogger(__name__)

# The agent works here: the task's files are in input/ and deliverables go to output/.
WORKDIR = "/workspace"


class WorkspaceBenchConfig(BaseResourcesServerConfig):
    artifact_root: Path | None = None
    judge_base_url: str
    judge_api_key: str
    judge_model: str
    judge_timeout_s: float = 600
    num_processes: int = 10
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class WorkspaceBenchRequest(TaskData, BaseSeedSessionRequest):
    """The row an agent's ``/run`` seeds with."""


class WorkspaceBenchSeedResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]


class WorkspaceBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_dir: str
    artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _WorkspaceCapture:
    """What grading needs once the task sandbox is gone: the agent's files and the inputs frozen at seed."""

    workspace: bytes
    metadata: bytes
    inputs: bytes


class WorkspaceBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str
    passed_count: int
    total_count: int
    judge_model: str
    rubrics: list[dict[str, Any]]
    dependency_graph: dict[str, Any]


class WorkspaceBenchResourcesServer(SandboxSessionResourcesServer):
    """Serves Workspace-Bench tasks in a task sandbox the agent works in, and judges its ``/workspace/output``.

    An Environment Server seeds an episode session and lends the agent the sandbox through ``sandbox_access``;
    ``harness_agent``'s ``/run`` seeds with the row and gets the sandbox descriptor. The task's input files are
    frozen on this server at seed, so the judge compares the output against the files the agent started from.
    """

    config: WorkspaceBenchConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._reference_dirs: dict[str, tempfile.TemporaryDirectory] = {}
        self._semaphore = asyncio.Semaphore(self.config.num_processes)

    async def shutdown(self) -> None:
        await super().shutdown()
        # A sandbox whose stop failed keeps its reference directory; the files go with the process.
        for session_id in list(self._reference_dirs):
            self._drop_reference_dir(session_id)

    def _drop_reference_dir(self, session_id: str) -> None:
        reference_dir = self._reference_dirs.pop(session_id, None)
        if reference_dir is not None:
            reference_dir.cleanup()

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._drop_reference_dir(session_id)

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[WorkspaceBenchRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> WorkspaceBenchSeedResponse | ResourcesSeedSessionResponse:
        """Start the task sandbox with the task's files in ``/workspace/input`` and an empty ``/workspace/output``."""
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self.seed_task_sandbox_session(request, body, TaskData)
        session_id = str(request.session[SESSION_ID_KEY])
        async with self._locked_task_session(session_id) as record:
            if record.seed is not None or self._task_session_fences.get(session_id) is not None:
                raise ResourcesSessionConflictError("An agent-run seed cannot replace an episode session")
            if session_id in self._session_id_to_sandbox or session_id in self._reference_dirs:
                raise ResourcesSessionConflictError("Workspace-Bench session is already active")
            try:
                await self._start_task_sandbox(session_id, body)
                descriptor = await self._session_id_to_sandbox[session_id].serialize()
            except BaseException:
                await self._stop_failed_seed_sandbox(session_id)
                raise
        return WorkspaceBenchSeedResponse(sandbox_handle=descriptor["sandbox_id"], sandbox_descriptor=descriptor)

    async def _start_task_sandbox(self, session_id: str, task: TaskData) -> str:
        task_dir = await asyncio.to_thread(resolve_task_path, task.task_dir)
        if not (task_dir / "metadata.json").is_file():
            raise ResourcesSeedRejectedError(f"Invalid Workspace-Bench task directory: {task_dir}")
        try:
            metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
            manifest = metadata.get("data_manifest") or []
            for item in manifest:
                if not isinstance(item.get("stored_relpath"), str) or not isinstance(item.get("filename"), str):
                    raise ValueError(f"data_manifest entry without stored_relpath and filename: {item}")
        except (ValueError, AttributeError) as error:
            # No retry can seed a task whose metadata is broken.
            raise ResourcesSeedRejectedError(f"Invalid Workspace-Bench metadata in {task_dir}: {error}") from error
        sandbox_config = self.config.sandbox_config
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        sandbox = AsyncSandbox(provider)
        spec = SandboxSpec(
            image=sandbox_config["image"],
            ttl_s=sandbox_config.get("ttl_s"),
            ready_timeout_s=sandbox_config.get("ready_timeout_s"),
            workdir=WORKDIR,
            env={},
            files={},
            metadata={
                **resolve_provider_metadata(self.config.sandbox_provider, get_global_config_dict()),
                **sandbox_config.get("metadata", {}),
                "task-id": task.task_id[:63],
            },
            resources=SandboxResources.from_mapping(sandbox_config.get("resources", {})),
            provider_options=sandbox_config.get("provider_options", {}),
        )
        # Own the sandbox before starting it, so a seed that fails or is cancelled from here on still stops it.
        self._session_id_to_sandbox[session_id] = sandbox
        reference_dir = tempfile.TemporaryDirectory()
        self._reference_dirs[session_id] = reference_dir
        try:
            await sandbox.start(spec)
            archive = Path(reference_dir.name) / "input.tar.gz"

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
                f"mkdir -p {WORKDIR}/input {WORKDIR}/output {WORKDIR}/.opencode {WORKDIR}/.claude "
                f"&& ln -s /opt/workspace-bench/office-skills {WORKDIR}/.opencode/skills "
                f"&& ln -s /opt/workspace-bench/agent-homes/claude/.claude/skills {WORKDIR}/.claude/skills "
                f"&& tar -xzf /tmp/input.tar.gz -C {WORKDIR}/input",
                cwd="/",
            )
            if result.return_code != 0 or result.error_type is not None:
                raise RuntimeError(f"Failed to seed Workspace-Bench input ({result.error_type}): {result.stderr}")
        except BaseException:
            # The frozen inputs are of no use without the sandbox, whether or not its stop succeeds.
            self._drop_reference_dir(session_id)
            raise
        return WORKDIR

    async def _collect_workspace(self, session_id: str) -> _WorkspaceCapture:
        """Archive the task sandbox's input and output, then stop the sandbox and drop its frozen inputs.

        An episode session keeps the capture, so a retried verify grades the same files after the sandbox is gone.
        """

        async def collect() -> _WorkspaceCapture:
            sandbox = self._session_id_to_sandbox[session_id]
            references = Path(self._reference_dirs[session_id].name)
            # The agent may have removed input/ or output/; that is graded as no output, not failed as a collection
            # error. Grading takes the inputs from the copy frozen at seed.
            result = await sandbox.exec(
                f"rm -f /tmp/workspace.tar.gz && mkdir -p {WORKDIR}/input {WORKDIR}/output "
                f"&& tar -czf /tmp/workspace.tar.gz -C {WORKDIR} input output"
            )
            # GNU tar exits 1 when a file changed while it was read (a process the agent left running) and still
            # writes the archive; that output is the agent's result. Exit 2 is a fatal tar error.
            if result.return_code == 1 and result.error_type is None:
                LOG.warning(
                    "Workspace-Bench files of session %s changed while they were archived: %s",
                    session_id,
                    result.stderr,
                )
            elif result.return_code != 0 or result.error_type is not None:
                raise RuntimeError(
                    f"Failed to collect Workspace-Bench files ({result.error_type}): {result.stderr}; "
                    f"output={(result.stdout or '')[-4000:]}"
                )
            with tempfile.TemporaryDirectory() as temporary_dir:
                local = Path(temporary_dir) / "workspace.tar.gz"
                await sandbox.download("/tmp/workspace.tar.gz", local)
                workspace = local.read_bytes()
            return _WorkspaceCapture(
                workspace=workspace,
                metadata=(references / "metadata.json").read_bytes(),
                inputs=(references / "input.tar.gz").read_bytes(),
            )

        try:
            capture = await self.capture_task_sandbox_patch(session_id, collect)
        except BaseException:
            # A retry could not grade more than this attempt, and an agent-run session gets no close: stop it now.
            try:
                await self._stop_task_sandbox(session_id)
            except Exception:
                LOG.exception("Failed to stop the Workspace-Bench task sandbox of session %s", session_id)
            raise
        # Stopping the sandbox drops the frozen inputs too; after a failed stop they stay for the retried collection.
        await self.stop_task_sandbox_after_capture(session_id)
        return capture

    async def _stop_agent_run_sandbox(self, session_id: str) -> None:
        """Stop the task sandbox of a session an agent's ``/run`` seeded, whose verify failed before collection.

        Such a session gets no close, so nothing else would stop the sandbox before shutdown. An episode session
        keeps its sandbox for the retried verify or the close.
        """
        record = self._task_sessions.get(session_id)
        if record is None or record.seed is None:
            try:
                await self._stop_task_sandbox(session_id)
            except Exception:
                LOG.exception("Failed to stop the Workspace-Bench task sandbox of session %s", session_id)

    async def _judge(self, case_dir: Path, session_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        # The base stops a registered verifier sandbox even when this request is cancelled, and close and shutdown
        # retry a stop that failed.
        self._verifier_sandboxes[session_id] = sandbox
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
                await self._stop_verifier_sandbox(session_id)
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
        """Judge the task's output; an episode session replays its first verdict to an identical retry.

        A verify for an episode session this server closed, or whose sandbox expired, is masked. Passing a saved
        ``artifact_id`` regrades that trial from its artifacts, without a session.
        """
        if body.artifact_id is not None:
            return await self._grade(request, body)
        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=lambda: self._grade(request, body),
            lost_response=lambda reason, failure_kind: self._lost_session_response(body, reason, failure_kind),
        )

    def _lost_session_response(
        self, body: WorkspaceBenchVerifyRequest, reason: str, failure_kind: str
    ) -> WorkspaceBenchVerifyResponse:
        return WorkspaceBenchVerifyResponse.model_validate(
            body.model_dump()
            | lost_session_verification(reason, failure_kind=failure_kind)
            | {
                "passed_count": 0,
                "total_count": 0,
                "judge_model": self.config.judge_model,
                "rubrics": [],
                "dependency_graph": {},
            }
        )

    async def _grade(self, request: Request, body: WorkspaceBenchVerifyRequest) -> WorkspaceBenchVerifyResponse:
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
                # A retried verify of an episode session reuses the trial directory of its first attempt.
                record = self._task_sessions.get(session_id)
                saved.mkdir(parents=True, exist_ok=record is not None and record.seed is not None)
        elif replay:
            raise HTTPException(status_code=400, detail="Artifact replay requires artifact_root")
        with tempfile.TemporaryDirectory() as temporary_dir:
            local_dir = Path(temporary_dir) / "judge"
            local_dir.mkdir()
            archive = (saved or Path(temporary_dir)) / "workspace.tar.gz"
            references = saved
            if not replay:
                references = Path(temporary_dir) / "references"
                references.mkdir()
                reference_dir = self._reference_dirs.get(session_id)
                if saved is not None:
                    try:
                        # A retried verify saved the frozen inputs on its first attempt, before the sandbox stopped.
                        if reference_dir is not None:
                            shutil.copyfile(Path(reference_dir.name) / "metadata.json", saved / "metadata.json")
                            shutil.copyfile(Path(reference_dir.name) / "input.tar.gz", saved / "input.tar.gz")
                        pending = saved / "request.partial"
                        pending.write_text(body.model_dump_json())
                        pending.replace(saved / "request.json")
                    except BaseException:
                        await self._stop_agent_run_sandbox(session_id)
                        raise
                capture = await self._collect_workspace(session_id)

                def write_capture() -> None:
                    (references / "metadata.json").write_bytes(capture.metadata)
                    (references / "input.tar.gz").write_bytes(capture.inputs)
                    partial = archive.with_suffix(".partial")
                    with partial.open("wb") as stream:
                        stream.write(capture.workspace)
                        stream.flush()
                        os.fsync(stream.fileno())
                    partial.replace(archive)

                await asyncio.to_thread(write_capture)

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
                    rubrics, dependency_graph = await self._judge(local_dir, session_id)
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
                            with (saved / f"judge-receipt-{uuid4().hex}.json").open("x", encoding="utf-8") as stream:
                                json.dump(receipt, stream)
                        except (OSError, ValueError, TypeError) as exc:
                            LOG.warning("Workspace-Bench judge receipt retention failed (%s)", type(exc).__name__)
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
