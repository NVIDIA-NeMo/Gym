# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NL2RepoBench resources server."""

from __future__ import annotations

import hashlib
import re
import shlex
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from time import monotonic
from traceback import format_exc
from typing import Any, ClassVar

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
from resources_servers.nl2repobench.task_store import (
    EXPECTED_TASK_COUNT,
    NL2RepoBenchTaskStore,
    Task,
    task_id,
    task_image,
)


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]

# Timeout for the agent-side workspace collection tar command.
WORKSPACE_COLLECT_TIMEOUT_S = 300.0
# Per-command timeout for install/test commands run in the fresh verifier sandbox.
TEST_COMMAND_TIMEOUT_S = 1800.0
# Combined stdout+stderr for all test commands is capped to keep JSONL rows reasonably sized.
TEST_OUTPUT_MAX_CHARS = 50_000

_WORKSPACE_EXCLUDES = "--exclude=.git --exclude=__pycache__ --exclude=.venv --exclude=node_modules"
# GNU tar treats --exclude as positional once it appears after other
# positional args (-C DIR .) — it must precede them or it's silently ignored
# (older tar) or rejected outright (newer tar, exit code 2).
_COLLECT_COMMAND = f"tar czf /tmp/workspace.tar.gz {_WORKSPACE_EXCLUDES} -C /workspace ."

_PASSED_RE = re.compile(r"(\d+)\s+passed")
_FAILED_RE = re.compile(r"(\d+)\s+failed")
_ERROR_RE = re.compile(r"(\d+)\s+error")


def _resolve_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (NEMO_GYM_ROOT / expanded).resolve()


class NL2RepoBenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    tasks_dir: Path
    expected_task_count: int = Field(default=EXPECTED_TASK_COUNT, ge=1)

    sandbox_provider: str
    enforce_agent_no_network: bool = True
    sandbox_model_server: ModelServerRef | None = None
    sandbox_config: dict[str, Any]

    logs_dir: Path = Path("resources_servers/nl2repobench/logs")
    clear_verifier_logs: bool = True
    include_workspace_in_response: bool = False
    workspace_tar_max_bytes: int = 209_715_200


class NL2RepoBenchInstanceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    image: str
    verifier_metadata: dict[str, Any] | None = None


class NL2RepoBenchSeedSessionRequest(NL2RepoBenchInstanceRequest, BaseSeedSessionRequest):
    pass


class NL2RepoBenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]


class NL2RepoBenchVerifyRequest(NL2RepoBenchInstanceRequest, BaseVerifyRequest):
    sandbox_handle: str | None = None


class NL2RepoBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    evaluation_completed: bool
    verifier_exit_code: int | None = None
    verifier_error: str | None = None

    tests_passed: int = 0
    tests_failed: int = 0
    tests_error: int = 0
    test_case_count: int = 0
    success_rate: float = 0.0

    workspace_bytes: int | None = None
    workspace_sha256: str | None = None
    workspace_tar: str | None = None
    test_output: str | None = None

    log_dir: str
    workspace_collection_time_s: float
    sandbox_start_time_s: float
    verification_time_s: float


class VerifierResult(BaseModel):
    evaluation_completed: bool
    reward: float
    verifier_exit_code: int | None = None
    verifier_error: str | None = None
    test_output: str | None = None
    tests_passed: int = 0
    tests_failed: int = 0
    tests_error: int = 0
    test_case_count: int = 0
    success_rate: float = 0.0


@dataclass
class AgentSandboxSession:
    task_id: str
    image: str
    sandbox: AsyncSandbox
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]


def _resolve_task_id(body: NL2RepoBenchInstanceRequest) -> str:
    metadata_task_id = (body.verifier_metadata or {}).get("task_id")
    if body.task_id and metadata_task_id and body.task_id != metadata_task_id:
        raise ValueError(
            f"Conflicting NL2RepoBench task IDs: task_id={body.task_id!r}, verifier_metadata.task_id={metadata_task_id!r}"
        )
    current_task_id = body.task_id or metadata_task_id
    if not isinstance(current_task_id, str) or not current_task_id:
        raise ValueError("NL2RepoBench requests must provide verifier_metadata.task_id or task_id")
    return current_task_id


def _resolve_task(body: NL2RepoBenchInstanceRequest, task_store: NL2RepoBenchTaskStore) -> Task:
    requested_task_id = _resolve_task_id(body)
    task = task_store.get(requested_task_id)
    if body.image != task_image(task):
        raise ValueError(f"NL2RepoBench request image does not match the pinned image for task {requested_task_id!r}")
    return task


def _parse_pytest_summary(combined_output: str) -> tuple[int, int, int]:
    """Return (passed, failed, error) from the LAST matching pytest summary occurrences."""

    passed_matches = _PASSED_RE.findall(combined_output)
    failed_matches = _FAILED_RE.findall(combined_output)
    error_matches = _ERROR_RE.findall(combined_output)
    tests_passed = int(passed_matches[-1]) if passed_matches else 0
    tests_failed = int(failed_matches[-1]) if failed_matches else 0
    tests_error = int(error_matches[-1]) if error_matches else 0
    return tests_passed, tests_failed, tests_error


def _compute_reward(tests_passed: int, test_case_count: int) -> float:
    if test_case_count > 0:
        reward = min(tests_passed / test_case_count, 1.0)
    else:
        reward = 0.0
    return max(reward, 0.0)


class NL2RepoBenchResourcesServer(SimpleResourcesServer):
    config: NL2RepoBenchResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._task_store = NL2RepoBenchTaskStore(
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
                raise TypeError("NL2RepoBench sandbox network_policy must be a mapping")
            egress = network_policy.setdefault("egress", [])
            if not isinstance(egress, list):
                raise TypeError("NL2RepoBench sandbox network_policy.egress must be a list")
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
                f"NL2RepoBench task sandboxes cannot reach loopback model host {target!r}; "
                "set NEMO_GYM_SANDBOX_MODEL_BASE_URL or launch Gym with use_absolute_ip=true"
            )
        return target

    async def _create_sandbox(self, task: Task, *, phase: str) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)

        current_task_id = task_id(task)
        resources = dict(self.config.sandbox_config.get("resources", {}))
        spec = SandboxSpec(
            image=task_image(task),
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir="/workspace",
            env=dict(self.config.sandbox_config.get("env", {})),
            # Upstream NL2RepoBench copies start.md into the task workspace at launch
            # (shutil.copy2 onto a host bind-mount); this is our equivalent for the
            # sandbox-provider path. Agent-only: the verifier sandbox just runs tests
            # against whatever the agent produced and has no use for the spec file.
            files={"/workspace/start.md": task.start_md} if phase == "agent" else {},
            metadata=provider_metadata
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {
                "benchmark": "nl2repobench-v1",
                "nl2repobench-task": current_task_id[:63],
                "nl2repobench-phase": phase,
                "nemo_gym_agent": self.config.name or "nl2repobench",
            },
            resources=SandboxResources.from_mapping(resources),
            provider_options=self._provider_options(phase=phase),
        )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(spec)
        if phase == "agent":
            # The pinned image is shared with the verifier sandbox and ships the real,
            # graded test suite baked in (task_image() returns the same reference for
            # both phases). Strip it from the agent's own copy so "hidden pytest suite"
            # is actually hidden from the agent, not just undocumented in start.md — the
            # verifier rebuilds its sandbox from this same image independently, so this
            # has no effect on grading.
            test_paths = " ".join(shlex.quote(f"/workspace/{p}") for p in task.test_files.files)
            if test_paths:
                await sandbox.exec(command=f"rm -rf {test_paths}", timeout_s=60)
        return sandbox

    async def _stop_sandbox(self, sandbox: AsyncSandbox, *, task_id: str, phase: str) -> None:
        try:
            await sandbox.stop()
        except Exception:
            print(f"Failed to stop NL2RepoBench {phase} sandbox for {task_id}: {format_exc()}", file=sys.stderr)

    async def seed_session(
        self, request: Request, body: NL2RepoBenchSeedSessionRequest
    ) -> NL2RepoBenchSeedSessionResponse:
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
                raise RuntimeError("NL2RepoBench sandbox provider did not return a sandbox_id")
            sandbox_descriptor = dict(descriptor)
            self._agent_sessions[session_id] = AgentSandboxSession(
                task_id=current_task_id,
                image=current_task_image,
                sandbox=sandbox,
                sandbox_handle=sandbox_handle,
                sandbox_descriptor=sandbox_descriptor,
            )
            return NL2RepoBenchSeedSessionResponse(
                sandbox_handle=sandbox_handle,
                sandbox_descriptor=sandbox_descriptor,
            )
        except Exception:
            if sandbox is not None:
                await self._stop_sandbox(sandbox, task_id=task_id(task), phase="failed-agent-seed")
            raise

    async def _collect_workspace(self, sandbox: AsyncSandbox) -> tuple[bytes | None, str | None]:
        """Tar up /workspace in the agent sandbox and download it. Returns (tar_bytes, error)."""

        result = await sandbox.exec(_COLLECT_COMMAND, timeout_s=WORKSPACE_COLLECT_TIMEOUT_S)
        if result.return_code != 0:
            details = ((result.stderr or "") + (result.stdout or "")).strip()
            return None, f"Workspace collection exited with code {result.return_code}: {details[-4000:]}"

        with tempfile.TemporaryDirectory(prefix="nemo-gym-nl2repobench-collect-") as temporary_dir:
            local_tar_path = Path(temporary_dir) / "workspace.tar.gz"
            await sandbox.download("/tmp/workspace.tar.gz", local_tar_path)
            tar_bytes = local_tar_path.read_bytes()
        if len(tar_bytes) > self.config.workspace_tar_max_bytes:
            return None, (
                f"Collected workspace tar ({len(tar_bytes)} bytes) exceeds workspace_tar_max_bytes "
                f"({self.config.workspace_tar_max_bytes} bytes)"
            )
        return tar_bytes, None

    async def _run_verifier(self, sandbox: AsyncSandbox, task: Task, workspace_tar: bytes) -> VerifierResult:
        mkdir_result = await sandbox.exec("mkdir -p /workspace", timeout_s=60)
        if mkdir_result.return_code != 0:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_error=f"Failed to create /workspace: {mkdir_result.stderr or ''}",
                test_case_count=task.test_case_count,
            )

        with tempfile.TemporaryDirectory(prefix="nemo-gym-nl2repobench-restore-") as temporary_dir:
            local_tar_path = Path(temporary_dir) / "workspace.tar.gz"
            local_tar_path.write_bytes(workspace_tar)
            await sandbox.upload(local_tar_path, "/tmp/workspace.tar.gz")

        extract_result = await sandbox.exec("tar xzf /tmp/workspace.tar.gz -C /workspace", timeout_s=300)
        if extract_result.return_code != 0:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=extract_result.return_code,
                verifier_error=f"Failed to extract workspace tar: {extract_result.stderr or ''}",
                test_case_count=task.test_case_count,
            )

        output_parts: list[str] = []
        verifier_error: str | None = None
        last_exit_code: int | None = None
        commands = task.test_commands.commands
        for index, command in enumerate(commands):
            is_last = index == len(commands) - 1
            command_result = await sandbox.exec(command, cwd="/workspace", timeout_s=TEST_COMMAND_TIMEOUT_S)
            last_exit_code = command_result.return_code
            output_parts.append(
                f"$ {command}\n{command_result.stdout or ''}{command_result.stderr or ''}"
                f"\n[exit code: {command_result.return_code}]\n"
            )
            if command_result.return_code != 0 and not is_last:
                verifier_error = f"install command failed: {command}"
                break

        combined_output = ("\n" + "-" * 40 + "\n").join(output_parts)
        if len(combined_output) > TEST_OUTPUT_MAX_CHARS:
            combined_output = combined_output[-TEST_OUTPUT_MAX_CHARS:]

        tests_passed, tests_failed, tests_error = _parse_pytest_summary(combined_output)
        if verifier_error is not None:
            tests_passed = 0
        reward = _compute_reward(tests_passed, task.test_case_count)

        return VerifierResult(
            evaluation_completed=True,
            reward=reward,
            verifier_exit_code=last_exit_code,
            verifier_error=verifier_error,
            test_output=combined_output,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_error=tests_error,
            test_case_count=task.test_case_count,
            success_rate=reward,
        )

    async def verify(self, request: Request, body: NL2RepoBenchVerifyRequest) -> NL2RepoBenchVerifyResponse:
        task = _resolve_task(body, self._task_store)
        current_task_id = task_id(task)
        current_task_image = task_image(task)
        session_id = str(request.session.get(SESSION_ID_KEY, "unknown"))
        sandbox_handle = body.sandbox_handle

        workspace_error: str | None = None
        workspace_tar = b""
        workspace_collection_time_s = 0.0

        agent_session = self._agent_sessions.pop(session_id, None)
        if agent_session is None:
            workspace_error = f"no agent session found for session {session_id!r}"
        else:
            sandbox_handle = agent_session.sandbox_handle
            started = monotonic()
            try:
                if agent_session.task_id != current_task_id:
                    raise RuntimeError(
                        f"NL2RepoBench session task {agent_session.task_id!r} does not match verify task "
                        f"{current_task_id!r}"
                    )
                if agent_session.image != current_task_image:
                    raise RuntimeError(
                        f"NL2RepoBench session image {agent_session.image!r} does not match verify image "
                        f"{current_task_image!r}"
                    )
                collected, collect_error = await self._collect_workspace(agent_session.sandbox)
                if collect_error is not None:
                    workspace_error = collect_error
                else:
                    workspace_tar = collected or b""
            except Exception as error:
                print(
                    f"Failed to collect NL2RepoBench workspace for {current_task_id}: {format_exc()}",
                    file=sys.stderr,
                )
                workspace_error = f"{type(error).__name__}: {error}"
            finally:
                workspace_collection_time_s = monotonic() - started
                await self._stop_sandbox(agent_session.sandbox, task_id=current_task_id, phase="agent")

        workspace_sha256 = hashlib.sha256(workspace_tar).hexdigest()
        workspace_bytes = len(workspace_tar)
        log_dir = _resolve_repo_path(self.config.logs_dir) / current_task_id / session_id

        sandbox: AsyncSandbox | None = None
        sandbox_start_time_s = 0.0
        verification_time_s = 0.0
        result = VerifierResult(
            evaluation_completed=False,
            reward=0.0,
            verifier_error=workspace_error,
            test_case_count=task.test_case_count,
        )
        if workspace_error is None:
            try:
                started = monotonic()
                sandbox = await self._create_sandbox(task, phase="verifier")
                sandbox_start_time_s = monotonic() - started
                started = monotonic()
                result = await self._run_verifier(sandbox, task, workspace_tar)
                verification_time_s = monotonic() - started
            except Exception as error:
                print(f"NL2RepoBench verifier failed for {current_task_id}: {format_exc()}", file=sys.stderr)
                result = VerifierResult(
                    evaluation_completed=False,
                    reward=0.0,
                    verifier_error=f"{type(error).__name__}: {error}",
                    test_case_count=task.test_case_count,
                )
            finally:
                if sandbox is not None:
                    await self._stop_sandbox(sandbox, task_id=current_task_id, phase="verifier")

        # No per-task artifacts are persisted to disk (test_output is captured in-memory and
        # returned directly), so log_dir is a nominal per-session path; clearing is a no-op unless
        # something else wrote into it.
        if self.config.clear_verifier_logs:
            rmtree(str(log_dir), ignore_errors=True)
            log_dir_str = ""
        else:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_dir_str = str(log_dir)

        return NL2RepoBenchVerifyResponse.model_validate(
            body.model_dump()
            | result.model_dump()
            | {
                "task_id": current_task_id,
                "sandbox_handle": sandbox_handle,
                "workspace_bytes": workspace_bytes,
                "workspace_sha256": workspace_sha256,
                "workspace_tar": workspace_tar.hex() if self.config.include_workspace_in_response else None,
                "log_dir": log_dir_str,
                "workspace_collection_time_s": workspace_collection_time_s,
                "sandbox_start_time_s": sandbox_start_time_s,
                "verification_time_s": verification_time_s,
            }
        )


if __name__ == "__main__":
    NL2RepoBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NL2RepoBenchResourcesServer.run_webserver()  # noqa: F401
