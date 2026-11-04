# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE v1.1 resources server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
import uuid
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

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef, get_first_server_config_dict, get_global_config_dict
from nemo_gym.core import (
    ENVIRONMENT_SERVER_FAILED,
    SESSION_LOST,
    VERIFIER_ERROR,
    FailureBoundary,
    NgFailureEvent,
    current_verifier_invocation_id,
    observe_failure,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import bounded_text, is_nemo_gym_fastapi_entrypoint
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


# The longest failure_reason a failure event accepts.
_MAX_FAILURE_REASON_CHARS = 2000
# What the verifier reports when it ran but left no verdict file: a fresh sandbox may still produce one.
_NO_VERDICT_FILE = "Verifier did not produce /logs/verifier/reward.json"


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
    # A verifier attempt that produced no verdict (no reward.json, or a sandbox that could not be created or run) is
    # run again on a fresh sandbox with the same collected patch, at most this many more times. A malformed verdict,
    # a timeout and a conclusive result are the verifier's answer and are not run again; a patch that breaks the
    # test script leaves no verdict every time, so after the bound the outcome stays masked as before.
    inconclusive_verification_retries: int = Field(default=2, ge=0)


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
    # A re-verification of a saved result whose verifier produced no verdict: the workspace patch that verifier was
    # given, with its saved sha256. It is verified on a fresh sandbox only when no agent session exists.
    saved_model_patch: str | None = None
    saved_model_patch_sha256: str | None = None


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
    # How many verifier attempts ran (0 when the patch was never collected), and why each one before the last
    # produced no verdict (``inconclusive_verification_retries``).
    verification_attempts: int = 0
    inconclusive_attempts: list[str] = Field(default_factory=list)
    # Whether the verified patch is a saved one, re-verified without its agent session.
    replayed_saved_patch: bool = False


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


def _inconclusive_reason(result: VerifierResult, error: Exception | None) -> str | None:
    """Why a verifier attempt produced no verdict that a fresh sandbox could still produce, or None.

    A sandbox that could not be created or run, and a verifier that left no reward.json, are run again. A malformed
    verdict and a timeout are the verifier's own answer about this patch, and a completed evaluation is a verdict.
    """
    if error is not None:
        return f"{type(error).__name__}: {error}"
    if result.evaluation_completed or result.verifier_error != _NO_VERDICT_FILE:
        return None
    return result.verifier_error


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

        async def _run_setup(started: AsyncSandbox) -> None:
            if phase != "agent":
                return
            # The task images ship no git user identity, and git's identity
            # auto-detection is hostname-dependent: an FQDN hostname passes
            # with a warning, while a domainless one (e.g. a K8s pod name ->
            # 'root@<pod>.(none)') makes every `git commit` fail with "Author
            # identity unknown". v1.1 grading collects only committed work,
            # so seed an explicit identity instead of relying on the sandbox
            # provider's hostname format.
            await started.exec(
                'git config --global user.email "agent@nemo-gym.local"'
                ' && git config --global user.name "NeMo Gym Agent"'
                " && git config --global --add safe.directory /app || true",
                timeout_s=60,
            )

        await sandbox.start_with_setup(spec, _run_setup)
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

    def _verifier_workdir(self, task: Task) -> str:
        """Directory test.sh runs from. DeepSWE tasks live at /app; subclasses may read it from the row."""
        return "/app"

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
                cwd=self._verifier_workdir(task),
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
                verifier_error=_NO_VERDICT_FILE,
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

        if not isinstance(reward_data, dict) or "reward" not in reward_data:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error="Verifier reward.json is missing its reward field",
                test_output=combined_output,
            )
        reward = reward_data["reward"]
        # Only a real 0 or 1 is a verdict: a bool is an int, and a numeric string is not a number, so
        # neither may become a pass (or a fail) through float().
        if type(reward) not in (int, float) or reward not in (0.0, 1.0):
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error=f"Verifier returned non-binary reward: {reward!r}",
                test_output=combined_output,
            )
        reward = float(reward)
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
        failure_kind = VERIFIER_ERROR
        failure_cause: Exception | None = None
        replayed = False

        if self.config.is_verifying_golden_patch:
            model_patch = task_solution_patch_path(task).read_bytes()
        else:
            model_patch = b""
            agent_session = self._agent_sessions.pop(session_id, None)
            if agent_session is None and body.saved_model_patch is not None:
                # A re-verification: the saved patch is verified as itself, never collected again.
                saved = body.saved_model_patch.encode("utf-8")
                if hashlib.sha256(saved).hexdigest() != body.saved_model_patch_sha256:
                    patch_error = "The saved DeepSWE model patch does not match its saved sha256"
                    failure_kind = SESSION_LOST
                else:
                    model_patch = saved
                    replayed = True
                    session_id = f"replay-{body.saved_model_patch_sha256[:16]}-{uuid.uuid4().hex[:8]}"
            elif agent_session is None:
                patch_error = f"No DeepSWE agent sandbox exists for session {session_id!r}"
                failure_kind = SESSION_LOST
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
                    failure_kind = ENVIRONMENT_SERVER_FAILED
                    failure_cause = error
                finally:
                    patch_collection_time_s = monotonic() - started
                    await self._stop_sandbox(agent_session.sandbox, task_id=current_task_id, phase="agent")

        patch_sha256 = hashlib.sha256(model_patch).hexdigest()
        log_dir = _resolve_repo_path(self.config.logs_dir) / current_task_id / session_id

        sandbox_start_time_s = 0.0
        verification_time_s = 0.0
        verification_attempts = 0
        inconclusive_attempts: list[str] = []
        result = VerifierResult(evaluation_completed=False, reward=0.0, verifier_error=patch_error)
        # One verification operation, however many attempts it takes; outside a verifier invocation it gets its own id.
        operation_id = current_verifier_invocation_id() or uuid.uuid4().hex
        if patch_error is None:
            # The same collected patch is verified on a fresh sandbox each time; an attempt without a verdict is run
            # again within the bound (``_inconclusive_reason``), and the last attempt's result stands.
            attempts = 1 + self.config.inconclusive_verification_retries
            phase = "golden-verifier" if self.config.is_verifying_golden_patch else "verifier"
            for attempt in range(1, attempts + 1):
                verification_attempts = attempt
                sandbox: AsyncSandbox | None = None
                attempt_error: Exception | None = None
                try:
                    started = monotonic()
                    sandbox = await self._create_sandbox(task, phase=phase)
                    sandbox_start_time_s = monotonic() - started
                    started = monotonic()
                    result = await self._run_verifier(sandbox, task, model_patch, log_dir)
                    verification_time_s = monotonic() - started
                except Exception as error:
                    print(f"DeepSWE verifier failed for {current_task_id}: {format_exc()}", file=sys.stderr)
                    attempt_error = error
                    result = VerifierResult(
                        evaluation_completed=False,
                        reward=0.0,
                        verifier_error=f"{type(error).__name__}: {error}",
                    )
                finally:
                    if sandbox is not None:
                        await self._stop_sandbox(sandbox, task_id=current_task_id, phase="verifier")
                reason = _inconclusive_reason(result, attempt_error)
                if reason is None or attempt == attempts:
                    failure_cause = attempt_error
                    break
                inconclusive_attempts.append(bounded_text(reason, _MAX_FAILURE_REASON_CHARS))
                # An attempt that is run again is observed as its own failure, as swebench_pro observes its
                # verification attempts, so every infrastructure retry is visible in the run's facts. Only an
                # exception before a sandbox was assigned names allocation as the failing operation.
                allocation_failed = attempt_error is not None and sandbox is None
                observed = NgFailureEvent(
                    boundary=FailureBoundary.RESOURCES_SERVER,
                    failure_kind=VERIFIER_ERROR,
                    failure_reason=(
                        "Verifier sandbox allocation failed"
                        if allocation_failed
                        else "Verification produced no conclusive verdict"
                    ),
                    operation="verify.sandbox.allocate" if allocation_failed else "verify",
                    operation_id=operation_id,
                    attempt_id=f"{operation_id}:{attempt}",
                    stage="verification",
                    exception_type=type(attempt_error).__name__ if attempt_error is not None else None,
                )
                observed.__cause__ = attempt_error
                observe_failure(observed)
                print(
                    f"Retrying DeepSWE verification for {current_task_id} on a fresh sandbox "
                    f"(attempt {attempt}/{attempts} was inconclusive: {reason})",
                    file=sys.stderr,
                )

        if self.config.clear_verifier_logs:
            rmtree(str(log_dir), ignore_errors=True)
            log_dir = ""

        if not result.evaluation_completed:
            # A lost session or failed patch collection happens before verification. Keep that
            # boundary and its cause distinct from an inconclusive verifier result.
            failure = NgFailureEvent(
                boundary=FailureBoundary.RESOURCES_SERVER,
                failure_kind=failure_kind,
                # The event field is bounded; a long provider diagnostic must not cost the result and patch.
                failure_reason=bounded_text(
                    result.verifier_error or "DeepSWE verification did not complete", _MAX_FAILURE_REASON_CHARS
                ),
                operation="collect_model_patch" if patch_error is not None else "verify",
                operation_id=operation_id,
                # The last of the attempts that ran; the earlier ones were observed as they were run again.
                attempt_id=f"{operation_id}:{verification_attempts}" if verification_attempts else None,
                stage="verification",
            )
            failure.__cause__ = failure_cause
            observe_failure(failure)

        return DeepSWEVerifyResponse.model_validate(
            body.model_dump(exclude={"saved_model_patch", "saved_model_patch_sha256"})
            | result.model_dump()
            | {
                "task_id": current_task_id,
                "sandbox_handle": sandbox_handle,
                "model_patch": model_patch.decode("utf-8", errors="replace")
                if self.config.include_model_patch_in_response and patch_error is None
                else None,
                "model_patch_sha256": patch_sha256,
                "model_patch_bytes": len(model_patch),
                "log_dir": str(log_dir),
                "patch_collection_time_s": patch_collection_time_s,
                "sandbox_start_time_s": sandbox_start_time_s,
                "verification_time_s": verification_time_s,
                "verification_attempts": verification_attempts,
                "inconclusive_attempts": inconclusive_attempts,
                "replayed_saved_patch": replayed,
                "mask_sample": not result.evaluation_completed,
                "failure_kind": None if result.evaluation_completed else failure_kind,
                "failure_reason": None if result.evaluation_completed else result.verifier_error,
            }
            | (lost_session_verification(patch_error) if failure_kind == SESSION_LOST else {})
        )


if __name__ == "__main__":
    DeepSWEResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DeepSWEResourcesServer.run_webserver()  # noqa: F401
