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
from math import ceil
from pathlib import Path
from shutil import rmtree
from time import monotonic
from traceback import format_exc
from typing import Any

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
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, get_first_server_config_dict, is_nemo_gym_fastapi_entrypoint
from resources_servers.deepswe.task_store import (
    EXPECTED_TASK_COUNT,
    DeepSWETaskStore,
    Task,
    task_collect_hook,
    task_id,
    task_image,
    task_sandbox_resources,
    task_solution_patch_path,
    task_verifier_files,
)


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]


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
    task_cpu_multiplier: float = Field(default=2.0, gt=0)
    task_memory_multiplier: float = Field(default=2.0, gt=0)

    sandbox_provider: str
    sandbox_config: dict[str, Any]
    enforce_agent_no_network: bool = True
    sandbox_model_server: ModelServerRef | None = None

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


class DeepSWEVerifyRequest(DeepSWEInstanceRequest, BaseVerifyRequest):
    sandbox_handle: str | None = None


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
    log_dir: str
    patch_collection_time_s: float
    sandbox_start_time_s: float
    verification_time_s: float


class VerifierResult(BaseModel):
    evaluation_completed: bool
    reward: float
    apply_failed: bool = False
    verifier_exit_code: int | None = None
    verifier_error: str | None = None
    test_output: str | None = None
    f2p_total: int = 0
    f2p_passed: int = 0
    p2p_total: int = 0
    p2p_passed: int = 0
    f2p: float = 0.0
    p2p: float = 0.0
    partial: float = 0.0


@dataclass
class AgentSandboxSession:
    task_id: str
    image: str
    sandbox: AsyncSandbox
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]


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


def _resolve_task(body: DeepSWEInstanceRequest, task_store: DeepSWETaskStore) -> Task:
    requested_task_id = _resolve_task_id(body)
    task = task_store.get(requested_task_id)
    if body.image != task_image(task):
        raise ValueError(f"DeepSWE request image does not match the pinned image for task {requested_task_id!r}")
    return task


class DeepSWEResourcesServer(SimpleResourcesServer):
    config: DeepSWEResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._task_store = DeepSWETaskStore(
            _resolve_repo_path(self.config.tasks_dir),
            expected_task_count=self.config.expected_task_count,
        )
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
        if self.config.sandbox_model_server:
            model_config = get_first_server_config_dict(
                get_global_config_dict(),
                self.config.sandbox_model_server.name,
            )
            target = str(model_config.get("host") or "")
            if not target:
                raise ValueError(f"Model server {self.config.sandbox_model_server.name!r} does not have a host")
        else:
            return None

        if target in {"0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}:
            raise ValueError(
                f"DeepSWE task sandboxes cannot reach loopback model host {target!r}; "
                "set NEMO_GYM_SANDBOX_MODEL_BASE_URL or launch Gym with use_absolute_ip=true"
            )
        return target

    async def _create_sandbox(self, task: Task, *, phase: str) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)

        current_task_id = task_id(task)
        resources = task_sandbox_resources(task, phase=phase)
        resources["cpu"] *= self.config.task_cpu_multiplier
        resources["memory_mib"] = ceil(resources["memory_mib"] * self.config.task_memory_multiplier)
        resources.update(self.config.sandbox_config.get("resources", {}))
        spec = SandboxSpec(
            image=task_image(task),
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir="/app",
            env=dict(self.config.sandbox_config.get("env", {})),
            files={},
            metadata=provider_metadata
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {
                "benchmark": "deepswe-v1-1",
                "deepswe-task": current_task_id[:63],
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

    async def _collect_model_patch(self, sandbox: AsyncSandbox, task: Task) -> bytes:
        collect = task_collect_hook(task)
        result = await sandbox.exec(collect.command, timeout_s=collect.timeout_sec)
        if result.return_code != 0:
            details = ((result.stderr or "") + (result.stdout or "")).strip()
            raise RuntimeError(f"DeepSWE collect hook exited with code {result.return_code}: {details[-4000:]}")

        with tempfile.TemporaryDirectory(prefix="nemo-gym-deepswe-collect-") as temporary_dir:
            local_patch_path = Path(temporary_dir) / "model.patch"
            await sandbox.download("/logs/artifacts/model.patch", local_patch_path)
            return local_patch_path.read_bytes()

    async def _stage_verifier(self, sandbox: AsyncSandbox, task: Task, model_patch: bytes) -> None:
        mkdir_result = await sandbox.exec(
            "mkdir -p /tests /logs/artifacts /logs/verifier",
            timeout_s=60,
        )
        if mkdir_result.return_code != 0:
            raise RuntimeError(f"Failed to create DeepSWE verifier directories: {mkdir_result.stderr or ''}")

        uploads = [
            sandbox.upload(local_path, f"/tests/{filename}")
            for filename, local_path in task_verifier_files(task).items()
        ]
        with tempfile.TemporaryDirectory(prefix="nemo-gym-deepswe-patch-") as temporary_dir:
            patch_path = Path(temporary_dir) / "model.patch"
            patch_path.write_bytes(model_patch)
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
        task: Task,
        model_patch: bytes,
        log_dir: Path,
    ) -> VerifierResult:
        await self._stage_verifier(sandbox, task, model_patch)
        try:
            command_result = await sandbox.exec(
                "bash /tests/test.sh",
                cwd="/app",
                timeout_s=task.config.verifier.timeout_sec,
            )
        except TimeoutError:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_error=f"Verifier timed out after {task.config.verifier.timeout_sec:g} seconds",
            )

        log_dir.mkdir(parents=True, exist_ok=True)
        combined_output = (command_result.stdout or "") + (command_result.stderr or "")

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
                test_output=combined_output,
            )
        try:
            reward_data = json.loads(artifact_paths["reward.json"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error=f"Invalid verifier reward.json: {error}",
                test_output=combined_output,
            )

        reward = float(reward_data.get("reward", 0.0))
        if reward not in (0.0, 1.0):
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error=f"Verifier returned non-binary reward: {reward!r}",
                test_output=combined_output,
            )
        if not present["ctrf.json"]:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error="Verifier did not produce /logs/verifier/ctrf.json",
                test_output=combined_output,
            )

        return VerifierResult(
            evaluation_completed=True,
            reward=reward,
            apply_failed=bool(reward_data.get("apply_failed", False)),
            verifier_exit_code=command_result.return_code,
            test_output=combined_output,
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
            current_task_id = task_id(task)
            current_task_image = task_image(task)
            descriptor = await sandbox.serialize()
            sandbox_handle = descriptor.get("sandbox_id") if isinstance(descriptor, dict) else None
            if not isinstance(sandbox_handle, str) or not sandbox_handle:
                raise RuntimeError("DeepSWE sandbox provider did not return a sandbox_id")
            sandbox_descriptor = dict(descriptor)
            self._agent_sessions[session_id] = AgentSandboxSession(
                task_id=current_task_id,
                image=current_task_image,
                sandbox=sandbox,
                sandbox_handle=sandbox_handle,
                sandbox_descriptor=sandbox_descriptor,
            )
            return DeepSWESeedSessionResponse(
                sandbox_handle=sandbox_handle,
                sandbox_descriptor=sandbox_descriptor,
            )
        except Exception:
            if sandbox is not None:
                await self._stop_sandbox(sandbox, task_id=task_id(task), phase="failed-agent-seed")
            raise

    async def verify(self, request: Request, body: DeepSWEVerifyRequest) -> DeepSWEVerifyResponse:
        task = _resolve_task(body, self._task_store)
        current_task_id = task_id(task)
        current_task_image = task_image(task)
        session_id = str(request.session.get(SESSION_ID_KEY, "golden"))
        sandbox_handle = body.sandbox_handle
        patch_collection_time_s = 0.0
        patch_error: str | None = None

        if self.config.is_verifying_golden_patch:
            model_patch = task_solution_patch_path(task).read_bytes()
        else:
            model_patch = b""
            agent_session = self._agent_sessions.pop(session_id, None)
            if agent_session is None:
                patch_error = f"No DeepSWE agent sandbox exists for session {session_id!r}"
            else:
                sandbox_handle = agent_session.sandbox_handle
                started = monotonic()
                try:
                    if agent_session.task_id != current_task_id:
                        raise RuntimeError(
                            f"DeepSWE session task {agent_session.task_id!r} does not match verify task "
                            f"{current_task_id!r}"
                        )
                    if agent_session.image != current_task_image:
                        raise RuntimeError(
                            f"DeepSWE session image {agent_session.image!r} does not match verify image "
                            f"{current_task_image!r}"
                        )
                    model_patch = await self._collect_model_patch(agent_session.sandbox, task)
                except Exception as error:
                    print(
                        f"Failed to collect DeepSWE model patch for {current_task_id}: {format_exc()}",
                        file=sys.stderr,
                    )
                    patch_error = f"{type(error).__name__}: {error}"
                finally:
                    patch_collection_time_s = monotonic() - started
                    await self._stop_sandbox(agent_session.sandbox, task_id=current_task_id, phase="agent")

        patch_sha256 = hashlib.sha256(model_patch).hexdigest()
        log_dir = _resolve_repo_path(self.config.logs_dir) / current_task_id / session_id

        sandbox: AsyncSandbox | None = None
        sandbox_start_time_s = 0.0
        verification_time_s = 0.0
        result = VerifierResult(evaluation_completed=False, reward=0.0, verifier_error=patch_error)
        if patch_error is None:
            try:
                started = monotonic()
                phase = "golden-verifier" if self.config.is_verifying_golden_patch else "verifier"
                sandbox = await self._create_sandbox(task, phase=phase)
                sandbox_start_time_s = monotonic() - started
                started = monotonic()
                result = await self._run_verifier(sandbox, task, model_patch, log_dir)
                verification_time_s = monotonic() - started
            except Exception as error:
                print(f"DeepSWE verifier failed for {current_task_id}: {format_exc()}", file=sys.stderr)
                result = VerifierResult(
                    evaluation_completed=False,
                    reward=0.0,
                    verifier_error=f"{type(error).__name__}: {error}",
                )
            finally:
                if sandbox is not None:
                    await self._stop_sandbox(sandbox, task_id=current_task_id, phase="verifier")

        if self.config.clear_verifier_logs:
            rmtree(str(log_dir), ignore_errors=True)
            log_dir = ""

        return DeepSWEVerifyResponse.model_validate(
            body.model_dump()
            | result.model_dump()
            | {
                "task_id": current_task_id,
                "sandbox_handle": sandbox_handle,
                "model_patch": model_patch.decode("utf-8", errors="replace")
                if self.config.include_model_patch_in_response
                else None,
                "model_patch_sha256": patch_sha256,
                "model_patch_bytes": len(model_patch),
                "log_dir": str(log_dir),
                "patch_collection_time_s": patch_collection_time_s,
                "sandbox_start_time_s": sandbox_start_time_s,
                "verification_time_s": verification_time_s,
            }
        )


if __name__ == "__main__":
    DeepSWEResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DeepSWEResourcesServer.run_webserver()  # noqa: F401
