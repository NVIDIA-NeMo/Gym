# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE v1.1 resources server.

The server serves two kinds of seed. An Environment Server such as ``single_agent_turn_legacy`` seeds an episode
session with a ``ResourcesSeedSessionRequest`` and hands the agent sandbox to the agent through ``sandbox_access``;
the session lifecycle (verify once, close, expiry, shutdown) is ``SandboxSessionResourcesServer``'s. An agent that
runs the episode itself through ``/run`` seeds with the flat task row and gets the sandbox handle back.
"""

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
from typing import Annotated, Any

from fastapi import Request
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef, get_first_server_config_dict, get_global_config_dict
from nemo_gym.core import (
    ENVIRONMENT_SERVER_FAILED,
    PROVIDER_OOM_KILLED,
    SESSION_LOST,
    VERIFIER_ERROR,
    FailureBoundary,
    FailureReport,
    FailureReportRow,
    NgFailureEvent,
    OutcomeAction,
    RequestFailureLog,
    SandboxFailureDetails,
    StageOutcomePolicyConfig,
    choose_stage_outcome,
    current_verifier_invocation_id,
    failure_kinds,
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
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.trajectory import NG_AGENT_SANDBOX_OBSERVATION_KEY, SandboxObservation
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
from resources_servers.swebench.sandbox_sessions import SandboxSessionResourcesServer


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]
# The task repository in every DeepSWE image; the agent works and commits here.
AGENT_WORKDIR = "/app"


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
    outcome_policy: StageOutcomePolicyConfig = Field(default_factory=StageOutcomePolicyConfig)

    sandbox_provider: str
    sandbox_config: dict[str, Any]
    enforce_agent_no_network: bool = True
    sandbox_model_server: ModelServerRef | None = None

    logs_dir: Path = Path("resources_servers/deepswe/logs")
    clear_verifier_logs: bool = False
    include_model_patch_in_response: bool = True
    # A verifier attempt that produced no verdict (no reward.json, or a sandbox that could not be created or run) is
    # run again on a fresh sandbox with the same collected patch, at most this many more times. A malformed verdict,
    # a reported command deadline and a conclusive result are not run again; a patch that breaks the
    # test script leaves no verdict every time, so after the bound the outcome stays masked as before.
    inconclusive_verification_retries: int = Field(default=2, ge=0)

    @field_validator("outcome_policy")
    @classmethod
    def require_supported_stage_actions(cls, policy: StageOutcomePolicyConfig) -> StageOutcomePolicyConfig:
        """Validate actions against the operations this resources server can perform."""
        supported = {
            "agent": {OutcomeAction.FAIL, OutcomeAction.COUNT_AS_ZERO},
            "verification": {OutcomeAction.FAIL, OutcomeAction.RETRY},
            None: {OutcomeAction.FAIL},
        }
        for stage, allowed in supported.items():
            rules = policy.by_stage.get(stage, policy.default_policy)
            actions = {
                rules.default_action,
                *rules.by_failure_kind.values(),
                *rules.by_status_code.values(),
                *rules.by_failure_class.values(),
            }
            if actions - allowed:
                raise ValueError(
                    f"DeepSWE {stage or 'unknown-stage'} policy supports only {', '.join(sorted(allowed))}"
                )
        return policy


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
    failure_report: FailureReport | None = None


class DeepSWEVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    failure_report: FailureReport | None = Field(default=None, exclude_if=lambda value: value is None)

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
    verdict and a confirmed command deadline are not run again. An unclassified TimeoutError is still an unknown
    infrastructure outcome and can be retried; it does not prove that the verifier exhausted its command budget.
    """
    if isinstance(error, NgFailureEvent) and error.failure_kind == failure_kinds.PROVIDER_COMMAND_TIMEOUT:
        return None
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


class DeepSWEResourcesServer(SandboxSessionResourcesServer):
    """DeepSWE grading of the patch an agent committed in its task sandbox.

    An agent-run seed keeps its sandbox in ``_agent_sessions``. An episode session's sandbox lives in the shared
    base's ``_session_id_to_sandbox``, with what the seed recorded about it in ``_episode_agent_sandboxes``.
    """

    config: DeepSWEResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._task_store = DeepSWETaskStore(
            _resolve_repo_path(self.config.tasks_dir),
            expected_task_count=self.config.expected_task_count,
        )
        self._agent_sessions: dict[str, AgentSandboxSession] = {}
        self._episode_agent_sandboxes: dict[str, AgentSandboxSession] = {}

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

    async def _create_sandbox(self, task: Task, *, phase: str, session_id: str | None = None) -> AsyncSandbox:
        """Start a sandbox for ``phase``; with ``session_id``, register it as that episode session's task sandbox.

        Registration happens before the start awaits, so a cancelled start stays reachable by the session's cleanup.
        """
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
            workdir=AGENT_WORKDIR,
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
        if session_id is not None:
            self._session_id_to_sandbox[session_id] = sandbox

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
            setup_result = await started.exec(
                'git config --global user.email "agent@nemo-gym.local"'
                ' && git config --global user.name "NeMo Gym Agent"'
                f" && git config --global --add safe.directory {AGENT_WORKDIR}",
                timeout_s=60,
            )
            if setup_result.return_code != 0 or setup_result.error_type is not None:
                raise RuntimeError(
                    f"DeepSWE agent git setup failed with code {setup_result.return_code}, "
                    f"error_type={setup_result.error_type!r}: {setup_result.stderr or ''}"
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
        if result.return_code != 0 or result.error_type is not None:
            details = ((result.stderr or "") + (result.stdout or "")).strip()
            status = f"exited with code {result.return_code}"
            if result.error_type is not None:
                status = f"failed with code {result.return_code}, error_type={result.error_type!r}"
            raise RuntimeError(f"DeepSWE collect hook {status}: {details[-4000:]}")

        with tempfile.TemporaryDirectory(prefix="nemo-gym-deepswe-collect-") as temporary_dir:
            local_patch_path = Path(temporary_dir) / "model.patch"
            await sandbox.download("/logs/artifacts/model.patch", local_patch_path)
            return local_patch_path.read_bytes()

    async def _stage_verifier(self, sandbox: AsyncSandbox, task: Task, model_patch: bytes) -> None:
        mkdir_result = await sandbox.exec(
            "mkdir -p /tests /logs/artifacts /logs/verifier",
            timeout_s=60,
        )
        if mkdir_result.return_code != 0 or mkdir_result.error_type is not None:
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
        if chmod_result.return_code != 0 or chmod_result.error_type is not None:
            raise RuntimeError(f"Failed to make DeepSWE verifier executable: {chmod_result.stderr or ''}")

    async def _download_if_present(self, sandbox: AsyncSandbox, remote_path: str, local_path: Path) -> bool:
        exists = await sandbox.exec(f"test -f {remote_path}", timeout_s=30)
        if exists.error_type is not None or exists.return_code not in (0, 1):
            raise RuntimeError(
                f"Could not check DeepSWE verifier artifact {remote_path}: "
                f"code={exists.return_code}, error_type={exists.error_type!r}"
            )
        if exists.return_code == 1:
            return False
        await sandbox.download(remote_path, local_path)
        return True

    def _verifier_workdir(self, task: Task) -> str:
        """Directory test.sh runs from. DeepSWE tasks live at /app; subclasses may read it from the row."""
        return AGENT_WORKDIR

    async def _run_verifier(
        self,
        sandbox: AsyncSandbox,
        task: Task,
        model_patch: bytes,
        log_dir: Path,
    ) -> VerifierResult:
        await self._stage_verifier(sandbox, task, model_patch)
        command_result = await sandbox.exec(
            "bash /tests/test.sh",
            cwd=self._verifier_workdir(task),
            timeout_s=task.config.verifier.timeout_sec,
        )

        log_dir.mkdir(parents=True, exist_ok=True)
        combined_output = (command_result.stdout or "") + (command_result.stderr or "")

        if command_result.error_type == "timeout":
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error=(
                    f"Verifier command failed with code {command_result.return_code}, error_type='timeout'"
                ),
                test_output=combined_output,
            )

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
        # All 113 tasks at the pinned DeepSWE revision run failing test suites under `set +e`;
        # their shared grader writes reward 0 plus CTRF and exits 0. A nonzero verifier exit
        # therefore is not the benchmark's completed failing verdict.
        if command_result.return_code != 0 or command_result.error_type is not None:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error=(
                    f"Verifier command failed with code {command_result.return_code}, "
                    f"error_type={command_result.error_type!r}"
                ),
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
        try:
            ctrf_data = json.loads(artifact_paths["ctrf.json"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error=f"Invalid verifier ctrf.json: {error}",
                test_output=combined_output,
            )
        if not isinstance(ctrf_data, dict):
            return VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=command_result.return_code,
                verifier_error="Invalid verifier ctrf.json: expected a JSON object",
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

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[DeepSWESeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> DeepSWESeedSessionResponse | ResourcesSeedSessionResponse:
        """Start the agent sandbox for a task.

        An Environment Server seeds an episode session and gets the sandbox back as ``sandbox_access``. An agent's
        ``/run`` seeds with the task row and gets the sandbox handle; repeating that seed replaces the sandbox.
        """
        if self.config.is_verifying_golden_patch:
            if isinstance(body, ResourcesSeedSessionRequest):
                raise ResourcesSeedRejectedError("DeepSWE seed_session is unavailable in golden-patch mode")
            raise RuntimeError("DeepSWE seed_session is unavailable in golden-patch mode")
        if isinstance(body, ResourcesSeedSessionRequest):
            try:
                task = _resolve_task(DeepSWEInstanceRequest.model_validate(body.task_data), self._task_store)
            except (KeyError, ValueError) as error:
                raise ResourcesSeedRejectedError(f"Invalid DeepSWE task_data: {error}") from error
            if body.task_id.task_id != task_id(task):
                raise ResourcesSeedRejectedError(
                    f"TaskId {body.task_id.task_id!r} does not match DeepSWE task {task_id(task)!r}"
                )
            return await self.seed_task_sandbox_session(request, body, DeepSWEInstanceRequest)

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

    async def _start_task_sandbox(self, session_id: str, task: BaseModel) -> str:
        """Start an episode session's agent sandbox and return the repository the agent works in."""
        resolved = _resolve_task(DeepSWEInstanceRequest.model_validate(task), self._task_store)
        self._forget_task_sandbox_state(session_id)
        sandbox = await self._create_sandbox(resolved, phase="agent", session_id=session_id)
        descriptor = await sandbox.serialize()
        sandbox_handle = descriptor.get("sandbox_id") if isinstance(descriptor, dict) else None
        if not isinstance(sandbox_handle, str) or not sandbox_handle:
            raise RuntimeError("DeepSWE sandbox provider did not return a sandbox_id")
        self._episode_agent_sandboxes[session_id] = AgentSandboxSession(
            task_id=task_id(resolved),
            image=task_image(resolved),
            sandbox=sandbox,
            sandbox_handle=sandbox_handle,
            sandbox_descriptor=dict(descriptor),
        )
        return AGENT_WORKDIR

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._episode_agent_sandboxes.pop(session_id, None)

    def _is_episode_session(self, session_id: str) -> bool:
        """Whether an Environment Server seeded ``session_id``, including after its close or expiry."""
        record = self._task_sessions.get(session_id)
        return (record is not None and record.seed is not None) or self._task_session_fences.get(
            session_id
        ) is not None

    async def verify(self, request: Request, body: DeepSWEVerifyRequest) -> DeepSWEVerifyResponse:
        """Grade the committed patch of the session's agent sandbox, or the task's golden patch.

        An episode session is verified once: an identical retry replays the verdict, and a verify after close or
        expiry is masked as a lost session.
        """
        session_id = request.session.get(SESSION_ID_KEY)
        if self.config.is_verifying_golden_patch or session_id is None or not self._is_episode_session(session_id):
            return await self._verify(request, body)
        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=lambda: self._verify(request, body),
            lost_response=lambda reason, failure_kind: self._lost_session_response(body, reason, failure_kind),
        )

    def _lost_session_response(
        self, body: DeepSWEVerifyRequest, reason: str, failure_kind: str
    ) -> DeepSWEVerifyResponse:
        return DeepSWEVerifyResponse.model_validate(
            body.model_dump(exclude={"saved_model_patch", "saved_model_patch_sha256"})
            | {
                "task_id": _resolve_task_id(body),
                "evaluation_completed": False,
                "verifier_error": reason,
                "model_patch_sha256": hashlib.sha256(b"").hexdigest(),
                "model_patch_bytes": 0,
                "log_dir": "",
                "patch_collection_time_s": 0.0,
                "sandbox_start_time_s": 0.0,
                "verification_time_s": 0.0,
            }
            | lost_session_verification(reason, failure_kind=failure_kind)
        )

    async def _verify(self, request: Request, body: DeepSWEVerifyRequest) -> DeepSWEVerifyResponse:
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
        agent_failure_action: OutcomeAction | None = None
        failure_outcome_attribution = None
        failure_log = RequestFailureLog.current()
        if body.failure_report is not None and failure_log is not None:
            failure_log.add(body.failure_report)
        try:
            raw_agent_observation = (body.response.metadata or {}).get(NG_AGENT_SANDBOX_OBSERVATION_KEY)
            agent_oom_observation = None
            agent_oom_event = None
            if raw_agent_observation is not None:
                agent_oom_observation = SandboxObservation.model_validate_json(raw_agent_observation)
                if (
                    agent_oom_observation.role != "agent"
                    or agent_oom_observation.outcome != "oom"
                    or agent_oom_observation.stage != "agent"
                ):
                    raise ValueError("Invalid agent sandbox OOM observation for DeepSWE scoring")
                report = body.failure_report
                if report is None or report.omitted or report.failure_id is not None:
                    raise ValueError("DeepSWE agent sandbox OOM requires a complete diagnostic failure report")
                agent_oom_event = report.events.get(agent_oom_observation.failure_id)
                if (
                    agent_oom_event is None
                    or agent_oom_event.boundary != FailureBoundary.SANDBOX
                    or agent_oom_event.failure_kind != PROVIDER_OOM_KILLED
                    or not isinstance(agent_oom_event.details, SandboxFailureDetails)
                    or not agent_oom_event.details.provider
                    or not agent_oom_event.details.sandbox_id
                    or agent_oom_event.stage not in (None, "agent")
                ):
                    raise ValueError("DeepSWE agent sandbox OOM requires its canonical sandbox failure event")
                details = agent_oom_event.details
                provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
                if details.provider not in provider_config:
                    raise ValueError("Agent sandbox OOM provider does not match the configured DeepSWE provider")
                if (
                    agent_oom_observation.provider is not None and agent_oom_observation.provider != details.provider
                ) or (
                    agent_oom_observation.sandbox_id is not None
                    and agent_oom_observation.sandbox_id != details.sandbox_id
                ):
                    raise ValueError("Agent sandbox OOM observation conflicts with its canonical failure details")

            if agent_oom_observation is not None and (self.config.is_verifying_golden_patch or body.response.output):
                raise ValueError("An agent sandbox OOM zero requires a non-golden run with an empty agent response")
        except Exception:
            # These are sessions created by seed_session, never a request's borrowed sandbox handle.
            rejected_session = self._agent_sessions.pop(session_id, None)
            if rejected_session is not None:
                # _stop_sandbox logs cleanup failures so the evidence rejection keeps its original cause.
                await self._stop_sandbox(
                    rejected_session.sandbox, task_id=rejected_session.task_id, phase="rejected-agent-evidence"
                )
            raise

        if self.config.is_verifying_golden_patch:
            model_patch = task_solution_patch_path(task).read_bytes()
        else:
            model_patch = b""
            episode = self._is_episode_session(session_id)
            captured = False
            if episode:
                # The shared base already matched this verify task against the seed. A retry after verify stopped
                # the sandbox finds no sandbox here and grades the capture the session saved.
                agent_session = self._episode_agent_sandboxes.get(session_id)
                record = self._task_sessions.get(session_id)
                captured = record is not None and record.capture_ready
            else:
                agent_session = self._agent_sessions.pop(session_id, None)
            if agent_session is None and agent_oom_observation is not None:
                raise RuntimeError("Cannot score an agent sandbox OOM without its seeded DeepSWE session")
            if agent_session is None and not episode and body.saved_model_patch is not None:
                # A re-verification: the saved patch is verified as itself, never collected again.
                saved = body.saved_model_patch.encode("utf-8")
                if hashlib.sha256(saved).hexdigest() != body.saved_model_patch_sha256:
                    patch_error = "The saved DeepSWE model patch does not match its saved sha256"
                    failure_kind = SESSION_LOST
                else:
                    model_patch = saved
                    replayed = True
                    session_id = f"replay-{body.saved_model_patch_sha256[:16]}-{uuid.uuid4().hex[:8]}"
            elif agent_session is None and not captured:
                patch_error = f"No DeepSWE agent sandbox exists for session {session_id!r}"
                failure_kind = SESSION_LOST
            else:
                if agent_session is not None:
                    sandbox_handle = agent_session.sandbox_handle
                started = monotonic()
                try:
                    if agent_session is not None and agent_session.task_id != current_task_id:
                        raise RuntimeError(
                            f"DeepSWE session task {agent_session.task_id!r} does not match verify task "
                            f"{current_task_id!r}"
                        )
                    if agent_session is not None and agent_session.image != current_task_image:
                        raise RuntimeError(
                            f"DeepSWE session image {agent_session.image!r} does not match verify image "
                            f"{current_task_image!r}"
                        )
                    if agent_oom_observation is not None:
                        if agent_oom_event.details.sandbox_id != agent_session.sandbox_handle:
                            raise ValueError("Agent sandbox OOM observation does not match the seeded DeepSWE sandbox")
                        decision = choose_stage_outcome(
                            failure_kind=agent_oom_event.failure_kind,
                            http_status=agent_oom_event.http_status,
                            stage=agent_oom_observation.stage,
                            config=self.config.outcome_policy,
                        )
                        agent_failure_action = decision.action
                        failure_outcome_attribution = decision.attribution
                        patch_error = (
                            "Agent sandbox was OOM-killed during agent execution; no patch could be collected. "
                            f"{agent_oom_event.failure_reason}"
                        )
                    elif episode:

                        async def capture() -> tuple[str, bytes]:
                            return agent_session.sandbox_handle, await self._collect_model_patch(
                                agent_session.sandbox, task
                            )

                        # Saved with the session, so a retry after the sandbox stopped grades the same patch.
                        sandbox_handle, model_patch = await self.capture_task_sandbox_patch(session_id, capture)
                    else:
                        model_patch = await self._collect_model_patch(agent_session.sandbox, task)
                except Exception as error:
                    if agent_oom_observation is not None:
                        raise
                    print(
                        f"Failed to collect DeepSWE model patch for {current_task_id}: {format_exc()}",
                        file=sys.stderr,
                    )
                    patch_error = f"{type(error).__name__}: {error}"
                    failure_kind = ENVIRONMENT_SERVER_FAILED
                    failure_cause = error
                finally:
                    patch_collection_time_s = monotonic() - started
                    if episode:
                        # A failed stop fails this verify and keeps the sandbox registered, so a retry stops it.
                        await self.stop_task_sandbox_after_capture(session_id)
                    else:
                        await self._stop_sandbox(agent_session.sandbox, task_id=current_task_id, phase="agent")

        patch_sha256 = hashlib.sha256(model_patch).hexdigest()
        log_dir = _resolve_repo_path(self.config.logs_dir) / current_task_id / session_id

        sandbox_start_time_s = 0.0
        verification_time_s = 0.0
        verification_attempts = 0
        verification_failure_action = None
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
                decision = None
                if reason is not None:
                    # Policy decides whether an admitted inconclusive attempt can be retried; the existing
                    # operation boundary still owns safety and the finite fresh-sandbox attempt budget.
                    decision = choose_stage_outcome(
                        failure_kind=attempt_error.failure_kind
                        if isinstance(attempt_error, NgFailureEvent)
                        else VERIFIER_ERROR,
                        http_status=attempt_error.http_status if isinstance(attempt_error, NgFailureEvent) else None,
                        # This loop establishes verifier ownership only. Preserve a different known
                        # origin on the event and use the conservative policy for that unowned context.
                        stage=None
                        if isinstance(attempt_error, NgFailureEvent)
                        and attempt_error.stage not in (None, "verification")
                        else "verification",
                        config=self.config.outcome_policy,
                    )
                if reason is None or attempt == attempts or decision.action != OutcomeAction.RETRY:
                    failure_cause = attempt_error
                    if decision is not None:
                        # A non-retry action or an exhausted retry budget ends this attempt sequence ungraded.
                        verification_failure_action = OutcomeAction.FAIL
                        failure_outcome_attribution = decision.attribution
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

        if not result.evaluation_completed and agent_oom_observation is None:
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

        agent_failure_fields = {}
        if agent_failure_action == OutcomeAction.FAIL:
            # This is a handled, terminal failure, not a failed HTTP request for the caller to retry.
            # Select the original admitted event without rebuilding its identity or searching its causes.
            selected_report = body.failure_report.model_copy(update={"failure_id": agent_oom_event.event_id})
            agent_failure_fields = {
                **FailureReportRow.fields(selected_report),
                "failure_report": selected_report,
                "_ng_failure_class": PROVIDER_OOM_KILLED,
                "_ng_failure_terminal": True,
                "_ng_failure_counts_as_zero": False,
            }

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
                "mask_sample": not result.evaluation_completed and agent_failure_action != OutcomeAction.COUNT_AS_ZERO,
                "failure_outcome_action": agent_failure_action or verification_failure_action,
                "failure_outcome_attribution": failure_outcome_attribution,
                "failure_kind": PROVIDER_OOM_KILLED
                if agent_oom_observation is not None
                else (None if result.evaluation_completed else failure_kind),
                "failure_reason": None if result.evaluation_completed else result.verifier_error,
            }
            | agent_failure_fields
            | (lost_session_verification(patch_error) if failure_kind == SESSION_LOST else {})
        )


if __name__ == "__main__":
    DeepSWEResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DeepSWEResourcesServer.run_webserver()  # noqa: F401
