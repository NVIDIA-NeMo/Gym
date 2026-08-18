# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE v1.1 resources server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shlex import join as shell_join
from time import monotonic
from traceback import format_exc
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, get_first_server_config_dict, is_nemo_gym_fastapi_entrypoint
from resources_servers.deepswe.task_store import EXPECTED_TASK_COUNT, DeepSWETask, DeepSWETaskStore


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]
WORKSPACE_HELPER_LOCAL_PATH = PACKAGE_DIR / "workspace_patch.py"
WORKSPACE_HELPER_REMOTE_PATH = "/tmp/nemo-gym-deepswe-workspace-patch.py"


def _resolve_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (NEMO_GYM_ROOT / expanded).resolve()


class DeepSWEResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE = ReverifyMode.UNSUPPORTED

    tasks_dir: Path
    expected_task_count: int = Field(default=EXPECTED_TASK_COUNT, ge=1)
    is_verifying_golden_patch: bool = False

    sandbox_provider: str
    sandbox_config: dict[str, Any]
    enforce_agent_no_network: bool = True
    sandbox_model_base_url: str | None = None
    sandbox_model_server_name: str | None = None
    max_concurrent_verifications: int = Field(default=128, ge=1)
    workspace_capture_timeout_s: float = Field(default=300, gt=0)
    max_model_patch_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    workspace_excluded_paths: tuple[str, ...] = ()

    logs_dir: Path = Path("resources_servers/deepswe/logs")
    clear_verifier_logs: bool = False
    include_model_patch_in_response: bool = True


class DeepSWEInstanceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    image: str
    verifier_metadata: dict[str, Any] | None = None


class DeepSWESeedSessionRequest(DeepSWEInstanceRequest, BaseSeedSessionRequest):
    pass


class DeepSWESeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]
    initial_tree: str


class DeepSWEVerifyRequest(DeepSWEInstanceRequest, BaseVerifyRequest):
    sandbox_handle: str | None = None
    initial_tree: str | None = None


class DeepSWEVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    evaluation_completed: bool
    apply_failed: bool = False
    verifier_exit_code: int | None = None
    verifier_error: str | None = None

    f2p_total: int = 0
    f2p_passed: int = 0
    p2p_total: int = 0
    p2p_passed: int = 0
    f2p: float = 0.0
    p2p: float = 0.0
    partial: float = 0.0

    model_patch: str | None = None
    model_patch_sha256: str
    model_patch_bytes: int
    changed_paths: int | None = None
    final_tree: str | None = None
    synthetic_tree: str | None = None
    log_dir: str
    workspace_capture_time_s: float
    sandbox_start_time_s: float
    verification_time_s: float


class VerifierResult(BaseModel):
    evaluation_completed: bool
    reward: float
    apply_failed: bool = False
    verifier_exit_code: int | None = None
    verifier_error: str | None = None
    f2p_total: int = 0
    f2p_passed: int = 0
    p2p_total: int = 0
    p2p_passed: int = 0
    f2p: float = 0.0
    p2p: float = 0.0
    partial: float = 0.0


class WorkspacePatchMetadata(BaseModel):
    initial_tree: str
    final_tree: str
    synthetic_tree: str
    changed_paths: int = Field(ge=0)
    patch_bytes: int = Field(ge=0)


@dataclass
class AgentSandboxSession:
    task_id: str
    image: str
    sandbox: AsyncSandbox
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]
    initial_tree: str


def _resolve_task_id(body: DeepSWEInstanceRequest) -> str:
    metadata_task_id = (body.verifier_metadata or {}).get("task_id")
    if body.task_id and metadata_task_id and body.task_id != metadata_task_id:
        raise ValueError(
            f"Conflicting DeepSWE task IDs: task_id={body.task_id!r}, verifier_metadata.task_id={metadata_task_id!r}"
        )
    task_id = body.task_id or metadata_task_id
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("DeepSWE requests must provide verifier_metadata.task_id or task_id")
    return task_id


def _resolve_task(body: DeepSWEInstanceRequest, task_store: DeepSWETaskStore) -> DeepSWETask:
    task_id = _resolve_task_id(body)
    task = task_store.get(task_id)
    if body.image != task.image:
        raise ValueError(f"DeepSWE request image does not match the pinned image for task {task_id!r}")
    return task


class DeepSWEResourcesServer(SimpleResourcesServer):
    config: DeepSWEResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._task_store = DeepSWETaskStore(
            _resolve_repo_path(self.config.tasks_dir),
            expected_task_count=self.config.expected_task_count,
        )
        self._verification_semaphore = asyncio.Semaphore(self.config.max_concurrent_verifications)
        self._agent_sessions: dict[str, AgentSandboxSession] = {}

    def _provider_options(self, *, phase: str) -> dict[str, Any]:
        options = deepcopy(self.config.sandbox_config.get("provider_options", {}))
        if phase != "agent":
            options.pop("network_policy", None)
        model_egress_target = self._model_egress_target() if phase == "agent" else None
        if phase == "agent" and self.config.enforce_agent_no_network:
            options.setdefault("network_policy", {"defaultAction": "deny", "egress": []})
        if phase == "agent" and model_egress_target is not None:
            network_policy = options.setdefault("network_policy", {"defaultAction": "deny", "egress": []})
            if not isinstance(network_policy, dict):
                raise TypeError("DeepSWE sandbox network_policy must be a mapping")
            egress = network_policy.setdefault("egress", [])
            if not isinstance(egress, list):
                raise TypeError("DeepSWE sandbox network_policy.egress must be a list")
            model_rule = {"action": "allow", "target": model_egress_target}
            if model_rule not in egress:
                egress.append(model_rule)

        return options

    def _model_egress_target(self) -> str | None:
        if self.config.sandbox_model_base_url:
            parsed = urlparse(self.config.sandbox_model_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"Invalid DeepSWE sandbox_model_base_url: {self.config.sandbox_model_base_url!r}")
            target = parsed.hostname
        elif self.config.sandbox_model_server_name:
            model_config = get_first_server_config_dict(
                get_global_config_dict(),
                self.config.sandbox_model_server_name,
            )
            target = str(model_config.get("host") or "")
            if not target:
                raise ValueError(f"Model server {self.config.sandbox_model_server_name!r} does not have a host")
        else:
            return None

        if target in {"0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}:
            raise ValueError(
                f"DeepSWE task sandboxes cannot reach loopback model host {target!r}; "
                "set NEMO_GYM_SANDBOX_MODEL_BASE_URL or launch Gym with use_absolute_ip=true"
            )
        return target

    async def _create_sandbox(self, task: DeepSWETask, *, phase: str) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)

        resources = {
            "cpu": task.cpu,
            "memory_mib": task.memory_mib,
            "disk_gib": task.disk_gib,
        }
        resources.update(self.config.sandbox_config.get("resources", {}))
        spec = SandboxSpec(
            image=task.image,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir="/app",
            env=dict(self.config.sandbox_config.get("env", {})),
            files={},
            metadata=provider_metadata
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {
                "benchmark": "deepswe-v1-1",
                "deepswe-task": task.task_id[:63],
                "deepswe-phase": phase,
                "nemo_gym_agent": self.config.name or "deepswe",
            },
            resources=SandboxResources.from_mapping(resources),
            provider_options=self._provider_options(phase=phase),
        )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(spec)
        return sandbox

    async def _stop_sandbox(self, sandbox: AsyncSandbox, *, task_id: str, phase: str) -> None:
        try:
            await sandbox.stop()
        except Exception:
            print(f"Failed to stop DeepSWE {phase} sandbox for {task_id}: {format_exc()}", file=sys.stderr)

    async def _workspace_helper_payload(self, sandbox: AsyncSandbox, arguments: list[str]) -> dict[str, Any]:
        await sandbox.upload(WORKSPACE_HELPER_LOCAL_PATH, WORKSPACE_HELPER_REMOTE_PATH)
        # Isolate imports from agent-created modules next to the helper in /tmp.
        result = await sandbox.exec(
            shell_join(["python3", "-I", WORKSPACE_HELPER_REMOTE_PATH, *arguments]),
            cwd="/app",
            timeout_s=self.config.workspace_capture_timeout_s,
        )
        if result.return_code != 0:
            details = ((result.stderr or "") + (result.stdout or "")).strip()
            raise RuntimeError(f"DeepSWE workspace helper exited with code {result.return_code}: {details[-4000:]}")
        try:
            payload = json.loads((result.stdout or "").strip())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"DeepSWE workspace helper returned invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise RuntimeError("DeepSWE workspace helper must return a JSON object")
        return payload

    async def _capture_initial_tree(self, sandbox: AsyncSandbox) -> str:
        payload = await self._workspace_helper_payload(sandbox, ["snapshot", "--repo", "/app"])
        initial_tree = payload.get("initial_tree")
        if not isinstance(initial_tree, str) or not initial_tree:
            raise RuntimeError("DeepSWE workspace helper did not return an initial tree")
        return initial_tree

    async def _capture_model_patch(
        self,
        sandbox: AsyncSandbox,
        task: DeepSWETask,
        initial_tree: str,
    ) -> tuple[str, WorkspacePatchMetadata]:
        remote_patch_path = f"/tmp/nemo-gym-deepswe-model-{uuid4().hex}.patch"
        arguments = [
            "patch",
            "--repo",
            "/app",
            "--initial-tree",
            initial_tree,
            "--base-commit",
            task.base_commit,
            "--output",
            remote_patch_path,
        ]
        for path in self.config.workspace_excluded_paths:
            arguments.extend(["--exclude-path", path])
        payload = await self._workspace_helper_payload(
            sandbox,
            arguments,
        )
        metadata = WorkspacePatchMetadata.model_validate(payload)
        if metadata.initial_tree != initial_tree:
            raise RuntimeError("DeepSWE workspace helper returned a mismatched initial tree")
        if metadata.patch_bytes > self.config.max_model_patch_bytes:
            raise RuntimeError(
                f"DeepSWE model patch is {metadata.patch_bytes} bytes, exceeding the "
                f"{self.config.max_model_patch_bytes}-byte limit"
            )

        with tempfile.TemporaryDirectory(prefix="nemo-gym-deepswe-workspace-") as temporary_dir:
            local_patch_path = Path(temporary_dir) / "model.patch"
            await sandbox.download(remote_patch_path, local_patch_path)
            patch_bytes = local_patch_path.read_bytes()
        if len(patch_bytes) != metadata.patch_bytes:
            raise RuntimeError(
                f"DeepSWE model patch size changed during download: "
                f"expected {metadata.patch_bytes}, received {len(patch_bytes)}"
            )
        try:
            return patch_bytes.decode("utf-8"), metadata
        except UnicodeDecodeError as error:
            raise RuntimeError("DeepSWE model patch is not valid UTF-8") from error

    async def _stage_verifier(self, sandbox: AsyncSandbox, task: DeepSWETask, model_patch: str) -> None:
        mkdir_result = await sandbox.exec(
            "mkdir -p /tests /logs/artifacts /logs/verifier",
            timeout_s=60,
        )
        if mkdir_result.return_code != 0:
            raise RuntimeError(f"Failed to create DeepSWE verifier directories: {mkdir_result.stderr or ''}")

        uploads = [
            sandbox.upload(local_path, f"/tests/{filename}") for filename, local_path in task.verifier_files.items()
        ]
        with tempfile.TemporaryDirectory(prefix="nemo-gym-deepswe-patch-") as temporary_dir:
            patch_path = Path(temporary_dir) / "model.patch"
            patch_path.write_text(model_patch, encoding="utf-8", errors="surrogateescape")
            uploads.append(sandbox.upload(patch_path, "/logs/artifacts/model.patch"))
            await asyncio.gather(*uploads)

        chmod_result = await sandbox.exec("chmod 0755 /tests/test.sh /tests/grader.py", timeout_s=60)
        if chmod_result.return_code != 0:
            raise RuntimeError(f"Failed to make DeepSWE verifier executable: {chmod_result.stderr or ''}")

    async def _download_if_present(self, sandbox: AsyncSandbox, remote_path: str, local_path: Path) -> bool:
        exists = await sandbox.exec(f"test -f {remote_path}", timeout_s=30)
        if exists.return_code != 0:
            return False
        await sandbox.download(remote_path, local_path)
        return True

    async def _run_verifier(
        self,
        sandbox: AsyncSandbox,
        task: DeepSWETask,
        model_patch: str,
        log_dir: Path,
    ) -> VerifierResult:
        await self._stage_verifier(sandbox, task, model_patch)
        try:
            command_result = await sandbox.exec(
                "bash /tests/test.sh",
                cwd="/app",
                timeout_s=task.verifier_timeout_s,
            )
        except TimeoutError:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_error=f"Verifier timed out after {task.verifier_timeout_s:g} seconds",
            )

        log_dir.mkdir(parents=True, exist_ok=True)
        combined_output = (command_result.stdout or "") + (command_result.stderr or "")
        stdout_path = log_dir / "test-stdout.txt"
        stdout_path.write_text(combined_output, encoding="utf-8", errors="replace")
        await sandbox.upload(stdout_path, "/logs/verifier/test-stdout.txt")

        artifact_paths = {
            "reward.json": log_dir / "reward.json",
            "ctrf.json": log_dir / "ctrf.json",
            "run.log": log_dir / "run.log",
        }
        present = {
            name: await self._download_if_present(sandbox, f"/logs/verifier/{name}", path)
            for name, path in artifact_paths.items()
        }
        if not present["reward.json"]:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error="Verifier did not produce /logs/verifier/reward.json",
            )
        try:
            reward_data = json.loads(artifact_paths["reward.json"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error=f"Invalid verifier reward.json: {error}",
            )

        reward = float(reward_data.get("reward", 0.0))
        if reward not in (0.0, 1.0):
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error=f"Verifier returned non-binary reward: {reward!r}",
            )
        if not present["ctrf.json"]:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error="Verifier did not produce /logs/verifier/ctrf.json",
            )

        return VerifierResult(
            evaluation_completed=True,
            reward=reward,
            apply_failed=bool(reward_data.get("apply_failed", False)),
            verifier_exit_code=command_result.return_code,
            f2p_total=int(reward_data.get("f2p_total", 0)),
            f2p_passed=int(reward_data.get("f2p_passed", 0)),
            p2p_total=int(reward_data.get("p2p_total", 0)),
            p2p_passed=int(reward_data.get("p2p_passed", 0)),
            f2p=float(reward_data.get("f2p", 0.0)),
            p2p=float(reward_data.get("p2p", 0.0)),
            partial=float(reward_data.get("partial", 0.0)),
        )

    async def seed_session(self, request: Request, body: DeepSWESeedSessionRequest) -> DeepSWESeedSessionResponse:
        if self.config.is_verifying_golden_patch:
            raise RuntimeError("DeepSWE seed_session is unavailable in golden-patch mode")

        task = _resolve_task(body, self._task_store)
        session_id = str(request.session[SESSION_ID_KEY])
        previous_session = self._agent_sessions.pop(session_id, None)
        if previous_session is not None:
            await self._stop_sandbox(
                previous_session.sandbox,
                task_id=previous_session.task_id,
                phase="replaced-agent",
            )

        sandbox: AsyncSandbox | None = None
        try:
            sandbox = await self._create_sandbox(task, phase="agent")
            initial_tree = await self._capture_initial_tree(sandbox)
            descriptor = await sandbox.serialize()
            sandbox_handle = descriptor.get("sandbox_id") if isinstance(descriptor, dict) else None
            if not isinstance(sandbox_handle, str) or not sandbox_handle:
                raise RuntimeError("DeepSWE sandbox provider did not return a sandbox_id")
            sandbox_descriptor = dict(descriptor)
            self._agent_sessions[session_id] = AgentSandboxSession(
                task_id=task.task_id,
                image=task.image,
                sandbox=sandbox,
                sandbox_handle=sandbox_handle,
                sandbox_descriptor=sandbox_descriptor,
                initial_tree=initial_tree,
            )
            return DeepSWESeedSessionResponse(
                sandbox_handle=sandbox_handle,
                sandbox_descriptor=sandbox_descriptor,
                initial_tree=initial_tree,
            )
        except Exception:
            if sandbox is not None:
                await self._stop_sandbox(sandbox, task_id=task.task_id, phase="failed-agent-seed")
            raise

    async def verify(self, request: Request, body: DeepSWEVerifyRequest) -> DeepSWEVerifyResponse:
        task = _resolve_task(body, self._task_store)
        session_id = str(request.session.get(SESSION_ID_KEY, "golden"))
        initial_tree = body.initial_tree
        sandbox_handle = body.sandbox_handle
        workspace_metadata: WorkspacePatchMetadata | None = None
        workspace_capture_time_s = 0.0
        patch_error: str | None = None

        if self.config.is_verifying_golden_patch:
            model_patch = task.solution_patch_path.read_text(encoding="utf-8", errors="replace")
        else:
            model_patch = ""
            agent_session = self._agent_sessions.pop(session_id, None)
            if agent_session is None:
                patch_error = f"No DeepSWE agent sandbox exists for session {session_id!r}"
            else:
                initial_tree = agent_session.initial_tree
                sandbox_handle = agent_session.sandbox_handle
                started = monotonic()
                try:
                    if agent_session.task_id != task.task_id:
                        raise RuntimeError(
                            f"DeepSWE session task {agent_session.task_id!r} does not match verify task {task.task_id!r}"
                        )
                    if agent_session.image != task.image:
                        raise RuntimeError(
                            f"DeepSWE session image {agent_session.image!r} does not match verify image {task.image!r}"
                        )
                    model_patch, workspace_metadata = await self._capture_model_patch(
                        agent_session.sandbox,
                        task,
                        agent_session.initial_tree,
                    )
                except Exception as error:
                    print(f"Failed to capture DeepSWE workspace for {task.task_id}: {format_exc()}", file=sys.stderr)
                    patch_error = f"{type(error).__name__}: {error}"
                finally:
                    workspace_capture_time_s = monotonic() - started
                    await self._stop_sandbox(agent_session.sandbox, task_id=task.task_id, phase="agent")

        patch_bytes = model_patch.encode("utf-8", errors="surrogateescape")
        patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
        log_dir = _resolve_repo_path(self.config.logs_dir) / task.task_id / session_id

        sandbox: AsyncSandbox | None = None
        sandbox_start_time_s = 0.0
        verification_time_s = 0.0
        result = VerifierResult(evaluation_completed=False, reward=0.0, verifier_error=patch_error)
        if patch_error is None:
            async with self._verification_semaphore:
                try:
                    started = monotonic()
                    phase = "golden-verifier" if self.config.is_verifying_golden_patch else "verifier"
                    sandbox = await self._create_sandbox(task, phase=phase)
                    sandbox_start_time_s = monotonic() - started
                    started = monotonic()
                    result = await self._run_verifier(sandbox, task, model_patch, log_dir)
                    verification_time_s = monotonic() - started
                except Exception as error:
                    print(f"DeepSWE verifier failed for {task.task_id}: {format_exc()}", file=sys.stderr)
                    result = VerifierResult(
                        evaluation_completed=False,
                        reward=0.0,
                        verifier_error=f"{type(error).__name__}: {error}",
                    )
                finally:
                    if sandbox is not None:
                        await self._stop_sandbox(sandbox, task_id=task.task_id, phase="verifier")

        if self.config.clear_verifier_logs and result.evaluation_completed:
            for path in log_dir.glob("*"):
                path.unlink(missing_ok=True)
            log_dir.rmdir()

        return DeepSWEVerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": result.reward,
                "task_id": task.task_id,
                "sandbox_handle": sandbox_handle,
                "initial_tree": initial_tree,
                "evaluation_completed": result.evaluation_completed,
                "apply_failed": result.apply_failed,
                "verifier_exit_code": result.verifier_exit_code,
                "verifier_error": result.verifier_error,
                "f2p_total": result.f2p_total,
                "f2p_passed": result.f2p_passed,
                "p2p_total": result.p2p_total,
                "p2p_passed": result.p2p_passed,
                "f2p": result.f2p,
                "p2p": result.p2p,
                "partial": result.partial,
                "model_patch": model_patch if self.config.include_model_patch_in_response else None,
                "model_patch_sha256": patch_sha256,
                "model_patch_bytes": len(patch_bytes),
                "changed_paths": workspace_metadata.changed_paths if workspace_metadata is not None else None,
                "final_tree": workspace_metadata.final_tree if workspace_metadata is not None else None,
                "synthetic_tree": workspace_metadata.synthetic_tree if workspace_metadata is not None else None,
                "log_dir": str(log_dir),
                "workspace_capture_time_s": workspace_capture_time_s,
                "sandbox_start_time_s": sandbox_start_time_s,
                "verification_time_s": verification_time_s,
            }
        )


if __name__ == "__main__":
    DeepSWEResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DeepSWEResourcesServer.run_webserver()  # noqa: F401
