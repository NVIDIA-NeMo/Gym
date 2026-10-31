# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TB4 sandbox provisioning, verification, and resource cleanup."""

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import ClassVar, Literal
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError

from nemo_gym.agent_utils import lifecycle_observation
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import SESSION_EXPIRED, SESSION_LOST, VERIFIER_ERROR, EpisodeId, NeMoGymResponse, rollout_context
from nemo_gym.rollout_collection import NG_TERMINAL_KEY
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
    observe_verifier_sandbox,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.environment import EnvironmentConfig, ExecLimitConfigurationError
from resources_servers.terminal_bench_4.lifecycle import NATIVE_VERSION, Session
from resources_servers.terminal_bench_4.models import (
    AgentTermination,
    ResourcesSessionVerifyRequest,
    SandboxedVerifyRequest,
    SandboxedVerifyResponse,
    SeedSessionResponse,
    TaskIdentity,
    TerminalBench4RunRequest,
    VerifierTestCounts,
)
from resources_servers.terminal_bench_4.task import PackageLoader


logger = logging.getLogger(__name__)


REPOSITORY = Path(__file__).resolve().parents[2]
BENCHMARK = REPOSITORY / "benchmarks" / "terminal_bench_4"


class TerminalBench4Config(BaseResourcesServerConfig):
    num_workers: Literal[1] = 1
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    manifest_path: Path = BENCHMARK / "manifest.json"
    artifacts_dir: Path = Path("results/terminal_bench_4/resources")
    environment: EnvironmentConfig
    max_concurrent_sessions: int = Field(default=8, gt=0)
    shutdown_timeout_sec: float = Field(default=30, ge=0)
    seeded_session_timeout_sec: float = Field(default=10 * 60 * 60, gt=0)
    task_download_dir: Path | None = None
    # Top-level sandbox provider block equivalent to environment.sandbox_provider. Agents that
    # borrow the task sandbox through SandboxAccess (Environment Server sessions) reconnect with it.
    sandbox_provider_ref: str | None = None
    session_close_retry_window_seconds: float = Field(
        default=300,
        gt=0,
        allow_inf_nan=False,
        description="Retain completed close fences for this many seconds; cover the caller's retry horizon.",
    )


# Users a SandboxAccess borrower runs as: the access carries no user, so harnesses execute as
# the sandbox's default user, which is root for every image this server provisions.
_DEFAULT_AGENT_USERS = (None, "root", 0, "0")
_RESOURCES_SESSION_KEY = "tb4_resources_session_id"


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


# Verifier outcomes that Terminal-Bench never retries and counts as a 0: Harbor excludes them from retries and counts a
# trial without a reward as 0. They are scored zeros here too, so the task stays in the pass-rate denominator.
_ZERO_SCORED_VERIFIER_FAILURES = frozenset(
    {"VerifierTimeoutError", "RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError"}
)


def _scored_verifier_failure(diagnostics) -> str | None:
    """The outcome a verifier that ran ended in, when Terminal-Bench scores it as 0. Read from the verifier's own
    diagnostic: an agent that exited on a timeout or a nonzero code is recorded first, as the trial's exception."""
    for diagnostic in reversed(diagnostics or []):
        if (
            isinstance(diagnostic, dict)
            and diagnostic.get("subphase") == "verifier_execution"
            and diagnostic.get("exception_type") in _ZERO_SCORED_VERIFIER_FAILURES
        ):
            return diagnostic["exception_type"]
    return None


class TerminalBench4ResourcesServer(SimpleResourcesServer):
    config: TerminalBench4Config

    def model_post_init(self, context):
        super().model_post_init(context)
        manifest_path = self.config.manifest_path
        if not manifest_path.is_absolute():
            # Servers run from their own directory; relative manifests name repository files.
            manifest_path = REPOSITORY / manifest_path
        self._manifest = json.loads(manifest_path.read_text())
        # Task packages are published under the dataset's org, e.g. terminal-bench-science/<task>.
        org = self._manifest["dataset"].split("/")[0]
        self._tasks = {f"{org}/{task['name']}": task for task in self._manifest["tasks"]}
        # A per-task setting keyed by a misspelled task name would silently apply to nothing.
        names = {task["name"] for task in self._manifest["tasks"]}
        environment = self.config.environment
        for setting, keys in (
            ("verifier_timeout_overrides", environment.verifier_timeout_overrides),
            ("task_images.by_task", environment.task_images.by_task),
            ("compose_service_overrides", environment.compose_service_overrides),
        ):
            if unknown := sorted(set(keys) - names):
                raise ValueError(f"environment.{setting} names tasks that {manifest_path.name} lacks: {unknown}")
        # Overrides are known before any task loads; package budgets are checked when each task loads.
        for name, timeout_sec in environment.verifier_timeout_overrides.items():
            environment.require_exec_limit_covers(name, timeout_sec)
        self._sessions: dict[str, Session] = {}
        self._by_identity: dict[str, str] = {}
        # Environment Server sessions: resources_session_id -> TB4 session ID, closed IDs (which
        # fence late seeds) with their episode, per-ID locks for seed/close ordering, and when each
        # completed close is forgotten. Closes whose cleanup failed or is pending never expire.
        self._resources_sessions: dict[str, str] = {}
        self._closed_resources: dict[str, EpisodeId] = {}
        self._resources_locks: dict[str, asyncio.Lock] = {}
        self._close_expiry: OrderedDict[str, float] = OrderedDict()
        self._slots = asyncio.Semaphore(self.config.max_concurrent_sessions)
        self._loader = PackageLoader(self.config.task_download_dir)
        self._closing = False
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/seed_session")(self.seed_session)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                self._closing = True
                await lifecycle.shutdown(list(self._sessions.values()), self.config.shutdown_timeout_sec)

        app.router.lifespan_context = lifespan
        return app

    def _owner(self, request):
        return hashlib.sha256(
            request.session.get("tb4_client_session_id", request.session[SESSION_ID_KEY]).encode()
        ).hexdigest()

    def _state_path(self, identity):
        return self.config.artifacts_dir / f"{identity}.json"

    def _persist(self, session):
        resources = []
        for env in (session.environment, session.verifier_environment, session.shared_logs):
            if env is not None:
                resources.append(
                    {
                        "session_id": env.session_id,
                        "closed": env.closed,
                        "resources": env.resources or env.resource_identities(),
                        "cleanup_errors": env.cleanup_errors,
                    }
                )
        atomic_json(
            self._state_path(session.identity),
            {
                "record_version": 3,
                "runtime": "gym-tb4-native",
                "runtime_version": NATIVE_VERSION,
                "session_id": session.session_id,
                "owner": session.owner,
                "request": session.request.model_dump() | {"_ng_rollout_id": session.request.capture_rollout_id},
                "phase": session.phase,
                "subphase": session.subphase,
                "result": session.result,
                "identity": session.identity,
                "deadlines": session.deadlines,
                "termination": session.termination.model_dump() if session.termination else None,
                "verify_body": session.verify_body.model_dump(mode="json") if session.verify_body else None,
                "verified_response": session.verified_response.model_dump(mode="json")
                if session.verified_response
                else None,
                "resources": resources or session.recorded_resources,
                "diagnostics": session.diagnostics,
                "resources_session_id": session.resources_request.resources_session_id
                if session.resources_request is not None
                else None,
            },
        )
        lookup = self.config.artifacts_dir / f"{session.session_id}.state"
        temporary = lookup.with_suffix(".tmp")
        temporary.write_text(session.identity)
        temporary.replace(lookup)
        if session.directory.exists():
            atomic_json(session.directory / "result.json", session.result)

    def _new_session(self, identity, owner, body, session_id, **kwargs):
        kwargs.setdefault("result", {"runtime": "gym-tb4-native", "runtime_version": NATIVE_VERSION})
        session = Session(identity, owner, body, session_id, self.config.artifacts_dir / session_id, **kwargs)
        session.slots = self._slots
        session.config = self.config
        session.persist = lambda: self._persist(session)
        return session

    def _check_task_identity(self, body: TaskIdentity) -> None:
        task = self._tasks.get(body.task_name)
        if task is None or body.task_ref != task["ref"] or body.dataset_ref != self._manifest["ref"]:
            raise HTTPException(422, "Task identity does not match the configured dataset pin")

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | TerminalBench4RunRequest
    ) -> ResourcesSeedSessionResponse | SeedSessionResponse:
        if self._closing:
            raise HTTPException(503, "Resources server is shutting down")
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self._seed_resources_session(request, body)
        self._check_task_identity(body)
        if body.client_session_id:
            request.session["tb4_client_session_id"] = body.client_session_id
        owner = self._owner(request)
        identity = hashlib.sha256(f"{owner}:{body.rollout_id}".encode()).hexdigest()
        session_id = self._by_identity.get(identity)
        if session_id is None:
            if self._state_path(identity).exists():
                state = json.loads(self._state_path(identity).read_text())
                session = self._session(request, state["session_id"])
            else:
                session_id = "tb4-" + uuid4().hex
                session = self._new_session(identity, owner, body.model_copy(deep=True), session_id)
                self._sessions[session_id] = session
                self._by_identity[identity] = session_id
                session.persist()
                session.execution = asyncio.create_task(self._prepare_session(session))
        else:
            session = self._sessions[session_id]
        if session.request != body:
            raise HTTPException(409, "Rollout identity is already bound to another request")
        if session.execution is not None:
            # Retried seed requests share provisioning, including before the first cookie response.
            await asyncio.shield(session.execution)
        self._check_expiry(session)
        if session.verified_response is not None:
            return SeedSessionResponse(session_id=session.session_id, verified_response=session.verified_response)
        if session.seed_response is None:
            raise HTTPException(409, "Episode has no recorded seed response")
        return session.seed_response

    async def _prepare_session(self, session: Session) -> None:
        session.started.set()
        with rollout_context(session.request.capture_rollout_id):
            try:
                await lifecycle.prepare_session(session, self._loader)
                session.seed_response = SeedSessionResponse(
                    session_id=session.session_id,
                    task_id=session.request.task_name,
                    sandbox_descriptor=await session.environment.main.serialize(),
                    sandbox_provider=session.environment.provider_config,
                    instruction=session.task.instruction,
                    user=session.task.config.agent.user,
                    agent_timeout_sec=session.task.config.agent.timeout_sec,
                    mcp_servers=[s.model_dump() for s in session.task.config.environment.mcp_servers],
                    skills_dir=session.task.config.environment.skills_dir,
                )
                session.phase = "ready"
                session.agent_deadline = asyncio.get_running_loop().time() + self.config.seeded_session_timeout_sec
                session.deadlines["agent_expires_at"] = (
                    datetime.now(timezone.utc) + timedelta(seconds=self.config.seeded_session_timeout_sec)
                ).isoformat()
                session.persist()
                session.expiry_task = asyncio.create_task(self._watch_agent_deadline(session))
            except (Exception, asyncio.CancelledError) as exc:
                session.termination = AgentTermination(
                    reason="cancelled" if isinstance(exc, asyncio.CancelledError) else "infrastructure_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                lifecycle.exception(session, exc)
                await lifecycle.cleanup(session)
                session.seed_response = SeedSessionResponse(
                    session_id=session.session_id, termination=session.termination
                )

    async def _seed_resources_session(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> ResourcesSeedSessionResponse:
        """Provision an Environment Server episode and lend its task sandbox through SandboxAccess.

        The episode reuses TB4's own seed and verify lifecycle: Resources owns the task and verifier sandboxes, the
        borrowing agent runs in the task sandbox, ``/verify`` grades in the separate offline verifier,
        and ``/close_session`` releases whatever remains. Seeds are idempotent per
        ``resources_session_id``; a changed request, or any seed after close, is rejected.
        """
        if self.config.sandbox_provider_ref is None:
            raise HTTPException(422, "Set sandbox_provider_ref to lend task sandboxes through SandboxAccess")
        if self.config.environment.sandbox_split_endpoints:
            raise HTTPException(422, "Split sandbox endpoints cannot be lent through one provider reference")
        try:
            identity = TaskIdentity.model_validate(body.task_data)
        except ValidationError as error:
            raise HTTPException(422, f"Task data does not name a pinned package: {error}") from error
        self._check_task_identity(identity)
        key = body.resources_session_id
        self._expire_close_records()
        async with self._resources_lock(key):
            if key in self._closed_resources:
                raise HTTPException(409, "Resources session is already closed")
            session_id = self._resources_sessions.get(key)
            if session_id is None:
                rollout_id = body.episode_id.capture_key
                run_request = TerminalBench4RunRequest(
                    # Bookkeeping only: the agent receives its prompt from the Environment Server.
                    responses_create_params={"input": []},
                    rollout_id=rollout_id,
                    _ng_rollout_id=rollout_id,
                    **identity.model_dump(),
                )
                owner = hashlib.sha256(key.encode()).hexdigest()
                episode = hashlib.sha256(f"resources-session:{key}".encode()).hexdigest()
                session_id = "tb4-" + uuid4().hex
                session = self._new_session(
                    episode, owner, run_request, session_id, resources_request=body.model_copy(deep=True)
                )
                self._sessions[session_id] = session
                self._by_identity[episode] = session_id
                self._resources_sessions[key] = session_id
                session.persist()
                session.execution = asyncio.create_task(self._prepare_resources_session(session))
            else:
                session = self._sessions[session_id]
                if session.resources_request != body:
                    raise HTTPException(409, "resources_session_id is already bound to a different request")
        # Retried seeds share provisioning; close fences it, then waits for it before releasing.
        await asyncio.shield(session.execution)
        if session.resources_error is not None:
            raise HTTPException(*session.resources_error)
        if key in self._closed_resources:
            raise HTTPException(409, "Resources session is already closed")
        self._check_expiry(session)
        if session.verify_body is not None:
            raise HTTPException(409, "Resources session is no longer available for seeding")
        request.session[_RESOURCES_SESSION_KEY] = key
        return session.resources_response

    async def _prepare_resources_session(self, session: Session) -> None:
        key = session.resources_request.resources_session_id
        await self._prepare_session(session)
        if session.seed_response.termination is not None:
            # Provisioning failed and was cleaned up; the episode may be retried.
            session.resources_error = (503, f"Task sandbox setup failed: {session.termination.detail}")
            self._remember_close(key, session)
            return
        with rollout_context(session.request.capture_rollout_id):
            try:
                # A close that arrived during setup fenced it: release instead of lending.
                self._raise_if_closed_during_setup(session)
                user = session.task.config.agent.user
                if user not in _DEFAULT_AGENT_USERS:
                    session.resources_error = (
                        422,
                        f"{session.request.task_name} runs its agent as {user!r}; SandboxAccess borrowers "
                        "run as the sandbox's default user",
                    )
                    raise ValueError(session.resources_error[1])
                workdir = await session.environment.agent_workdir()
                if not PurePosixPath(workdir).is_absolute():
                    raise RuntimeError(f"Task working directory is not absolute: {workdir!r}")
                self._raise_if_closed_during_setup(session)
                session.resources_response = ResourcesSeedSessionResponse(
                    resources_session_id=session.resources_request.resources_session_id,
                    sandbox_access=SandboxAccess(
                        connection=DirectSandboxConnection(
                            provider_config_ref=self.config.sandbox_provider_ref,
                            descriptor=session.seed_response.sandbox_descriptor,
                        ),
                        workdir=workdir,
                    ),
                )
                session.persist()
            except (Exception, asyncio.CancelledError) as exc:
                closed = isinstance(exc, asyncio.CancelledError) or key in self._closed_resources
                session.termination = AgentTermination(
                    reason="cancelled" if closed else "infrastructure_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                session.resources_error = session.resources_error or (503, f"Task sandbox setup failed: {exc}")
                lifecycle.exception(session, exc)
                await lifecycle.cleanup(session)
                session.seed_response = SeedSessionResponse(
                    session_id=session.session_id, termination=session.termination
                )
                # A late seed that performs a fenced close's teardown starts its retention window.
                self._remember_close(key, session)

    def _raise_if_closed_during_setup(self, session: Session) -> None:
        if session.resources_request.resources_session_id in self._closed_resources:
            session.resources_error = (409, "Resources session closed during setup")
            raise RuntimeError(session.resources_error[1])

    @asynccontextmanager
    async def _resources_lock(self, key: str):
        """Serialize seed and close for one resources session."""
        while True:
            lock = self._resources_locks.setdefault(key, asyncio.Lock())
            async with lock:
                # Expiry can retire a lock while a caller is waiting on it; join the replacement.
                if self._resources_locks.get(key) is not lock:
                    continue
                yield
                return

    def _remember_close(self, key: str, session: Session | None) -> None:
        """Start a completed close's retention window once; close retries do not extend it.

        Only fenced sessions whose sandboxes all stopped expire: failed or pending cleanup keeps
        its fence, so a late seed can never lend a sandbox for a closed episode.
        """
        if key not in self._closed_resources or key in self._close_expiry:
            return
        if session is not None and not lifecycle.released(session):
            return
        self._close_expiry[key] = monotonic() + self.config.session_close_retry_window_seconds

    def _expire_close_records(self) -> None:
        """Forget completed closes past the retry window, so the fences do not grow without bound."""
        now = monotonic()
        while self._close_expiry:
            key, expires_at = next(iter(self._close_expiry.items()))
            if expires_at > now:
                break
            del self._close_expiry[key]
            self._closed_resources.pop(key, None)
            self._resources_locks.pop(key, None)
            session = self._sessions.pop(self._resources_sessions.pop(key, ""), None)
            if session is not None:
                self._by_identity.pop(session.identity, None)

    def _expire_seed(self, session: Session) -> None:
        # No await between claiming expiry and installing the cleanup task: /verify
        # either owns finalization already or must reject this expired session.
        if session.phase != "ready" or session.finalization is not None:
            return
        session.phase = "expiring"
        session.deadlines["agent_expired_at"] = lifecycle.now()
        session.termination = AgentTermination(reason="timeout", detail="Seeded session deadline expired")
        lifecycle.exception(session, session.termination.detail, "SeededSessionExpired")
        session.persist()
        session.finalization = asyncio.create_task(lifecycle.finalize_session(session, grade=False))

    def _expired(self, session: Session) -> bool:
        if session.agent_deadline is not None and asyncio.get_running_loop().time() >= session.agent_deadline:
            self._expire_seed(session)
        return "agent_expired_at" in session.deadlines

    def _check_expiry(self, session: Session) -> None:
        if self._expired(session):
            raise HTTPException(410, "Seeded session deadline expired")

    async def _watch_agent_deadline(self, session: Session) -> None:
        await asyncio.sleep(max(0, session.agent_deadline - asyncio.get_running_loop().time()))
        self._expire_seed(session)

    async def _finalize_session(self, session: Session) -> None:
        body = session.verify_body
        with rollout_context(session.request.capture_rollout_id):
            session.termination = session.termination or body.termination
            session.result.update(body.agent_timings)
            if body.termination.reason == "infrastructure_error":
                lifecycle.exception(session, body.termination.detail, "AgentInfrastructureError")
            elif not body.agent_started and body.termination.reason == "timeout":
                lifecycle.exception(session, body.termination.detail, "AgentSetupTimeoutError")
            try:
                session.directory.mkdir(parents=True, exist_ok=True)
                (session.directory / "gym-agent.json").write_text(body.model_dump_json(indent=2))
            except OSError as exc:
                lifecycle.exception(session, exc, "AgentRecordError")
            session.phase = "verifying" if body.agent_started else "cleaning"
            await lifecycle.finalize_session(
                session, grade=body.agent_started and session.seed_response.termination is None
            )
            # The resource owner retains the actual handles after cleanup. Bind these
            # facts to the verify invocation, independently of the agent's bundle.
            for environment, role in ((session.environment, "agent"), (session.verifier_environment, "verifier")):
                if environment is None:
                    continue
                try:
                    sandboxes = (
                        tuple(environment.compose.services.values())
                        if environment.uses_compose
                        else (environment.main,)
                    )
                except Exception:
                    logger.warning("Sandbox owner handles unavailable for verification observation")
                    continue
                for sandbox in sandboxes:
                    try:
                        observation = lifecycle_observation(sandbox, role=role)
                        if observation is not None:
                            observe_verifier_sandbox(observation)
                    except Exception:
                        # Observability must not replace a calculated grade or its original failure.
                        logger.warning("Sandbox lifecycle observation unavailable")
            session.verified_response = self._verified_response(session, body.harness_metadata)
            session.persist()
        if session.resources_request is not None:
            self._remember_close(session.resources_request.resources_session_id, session)

    def _verified_response(self, session, extra):
        result = session.result or {}
        verifier_result = result.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}
        completed = "reward" in rewards
        counts = VerifierTestCounts.model_validate(verifier_result)
        # Like the reward, 0 when the evaluation did not complete, so the two means share a denominator.
        partial_credit = 0.0
        if completed:
            # Never below the official grade: a passing run with skipped tests keeps full
            # credit, and without usable test counts the official reward is the finest signal.
            fraction = 0.0
            if counts.num_tests_total and counts.num_tests_passed is not None:
                fraction = counts.num_tests_passed / counts.num_tests_total
            partial_credit = max(float(rewards["reward"]), fraction)
        failure = None
        failure_type = None
        verifier_failure = None
        if not completed:
            # An agent infrastructure failure stays retryable whatever its verifier did next: Harbor decides a retry
            # from the trial's first exception, which is the agent's.
            if session.termination.reason != "infrastructure_error":
                verifier_failure = _scored_verifier_failure(session.diagnostics)
            if verifier_failure is None:
                failure = (result.get("exception_info") or {}).get("exception_type", "MissingOfficialReward")
                failure_type = failure
        elif session.termination.reason == "infrastructure_error" and not result.get("sandbox_resource_limit"):
            failure = session.termination.detail or "Agent infrastructure failure"
        return SandboxedVerifyResponse(
            **(
                session.verify_body.model_dump(
                    exclude={"termination", "agent_started", "agent_timings", "harness_metadata"}
                )
                | extra
                | {"task_id": session.request.task_name}
                | counts.model_dump()
                | {"sandbox_resource_limit": result.get("sandbox_resource_limit")}
            ),
            reward=float(rewards.get("reward", 0)),
            partial_credit=partial_credit,
            evaluation_completed=completed,
            termination=session.termination,
            infrastructure_error=failure,
            failure_reason=failure or verifier_failure,
            failure_kind=VERIFIER_ERROR if verifier_failure else None,
            artifacts={"trial": str(session.directory)},
            timings={
                key: result.get(key) for key in ("environment_setup", "agent_setup", "agent_execution", "verifier")
            },
            **({"_ng_failure_class": "infrastructure_error"} if failure else {}),
            **({"_ng_failure_type": failure_type} if failure_type else {}),
            # A retry would fail the same way: only a configuration change can fix it.
            **({NG_TERMINAL_KEY: True} if failure_type == ExecLimitConfigurationError.__name__ else {}),
        )

    def _session(self, request, session_id):
        session = self._sessions.get(session_id)
        if session is None:
            if not re.fullmatch(r"tb4-[a-f0-9]{32}", session_id):
                raise HTTPException(404, "Unknown session")
            lookup = self.config.artifacts_dir / f"{session_id}.state"
            if lookup.exists():
                identity = lookup.read_text()
                if not re.fullmatch(r"[a-f0-9]{64}", identity):
                    raise HTTPException(409, "Invalid recorded episode identity")
                state = json.loads(self._state_path(identity).read_text())
                if state["owner"] != self._owner(request):
                    raise HTTPException(404, "Unknown session")
                if state["phase"] != "closed":
                    raise HTTPException(
                        409, "Resources process restarted; episode cannot resume; provider TTL applies"
                    )
                if state.get("record_version", 0) != 3:
                    raise HTTPException(409, "Unsupported recorded episode version")
                session = self._new_session(
                    state["identity"],
                    state["owner"],
                    TerminalBench4RunRequest.model_validate(state["request"]),
                    session_id,
                    phase="closed",
                    result=state.get("result") or {},
                    deadlines=state.get("deadlines") or {},
                    diagnostics=state.get("diagnostics") or [],
                    recorded_resources=state.get("resources") or [],
                )
                if state.get("termination"):
                    session.termination = AgentTermination.model_validate(state["termination"])
                if state.get("verify_body"):
                    session.verify_body = SandboxedVerifyRequest.model_validate(state["verify_body"])
                if state.get("verified_response"):
                    session.verified_response = SandboxedVerifyResponse.model_validate(state["verified_response"])
                self._sessions[session_id] = session
        if session is None or session.owner != self._owner(request):
            raise HTTPException(404, "Unknown session")
        return session

    async def verify(
        self, request: Request, body: SandboxedVerifyRequest | ResourcesSessionVerifyRequest
    ) -> SandboxedVerifyResponse:
        if isinstance(body, ResourcesSessionVerifyRequest):
            return await self._verify_resources_session(request, body)
        session = self._session(request, body.session_id)
        if session.execution is not None:
            await asyncio.shield(session.execution)
        self._check_expiry(session)
        if session.verify_body is not None and session.verify_body != body:
            raise HTTPException(409, "Session is already bound to another verification request")
        if session.verified_response is not None:
            return session.verified_response
        if session.seed_response is None:
            raise HTTPException(409, "Episode has no recorded seed response")
        if session.finalization is None:
            if session.expiry_task is not None:
                session.expiry_task.cancel()
            session.agent_deadline = None
            session.deadlines["verification_started_at"] = lifecycle.now()
            session.verify_body = body.model_copy(deep=True)
            session.finalization = asyncio.create_task(self._finalize_session(session))
        await asyncio.shield(session.finalization)
        return session.verified_response

    async def _verify_resources_session(
        self, request: Request, body: ResourcesSessionVerifyRequest
    ) -> SandboxedVerifyResponse | JSONResponse:
        """Grade an Environment Server episode once; identical retries replay the verdict.

        The Environment Server closes the agent first, so the task sandbox is quiet. Grading uses
        the same finalizer as TB4's own verify path: collect artifacts, stop the task sandbox, then run
        the official tests in the separate offline verifier. A closed, unknown or expired session has
        nothing left to grade and answers with a masked sample.
        """
        key = request.session.get(_RESOURCES_SESSION_KEY)
        session = self._sessions.get(self._resources_sessions.get(key, ""))
        if key in self._closed_resources:
            expired = session is not None and "agent_expired_at" in session.deadlines
            return _lost_session(
                body, "Resources session is already closed", SESSION_EXPIRED if expired else SESSION_LOST
            )
        if session is None:
            return _lost_session(body, "No resources session is bound to this cookie", SESSION_LOST)
        if session.execution is not None:
            await asyncio.shield(session.execution)
        if self._expired(session):
            return _lost_session(body, "Seeded session deadline expired", SESSION_EXPIRED)
        if session.resources_verify_body is not None and session.resources_verify_body != body:
            raise HTTPException(409, "Session is already bound to another verification request")
        if session.verified_response is not None:
            return session.verified_response
        if session.resources_response is None:
            raise HTTPException(409, "Resources session has no task sandbox to verify")
        if session.finalization is None:
            seeded = TaskIdentity.model_validate(session.request.model_dump())
            if TaskIdentity.model_validate(body.model_dump()) != seeded:
                raise HTTPException(409, "Verification task does not match the seeded task")
            if session.expiry_task is not None:
                session.expiry_task.cancel()
            session.agent_deadline = None
            session.deadlines["verification_started_at"] = lifecycle.now()
            session.resources_verify_body = body.model_copy(deep=True)
            session.verify_body = SandboxedVerifyRequest(
                session_id=session.session_id,
                responses_create_params=body.responses_create_params,
                response=body.response,
                termination=_termination_from_response(body.response),
                agent_started=True,
                # As observed by Resources: from the ready sandbox (agent setup included) to grading.
                agent_timings={
                    "agent_execution": {
                        "started_at": session.result["environment_setup"]["finished_at"],
                        "finished_at": session.deadlines["verification_started_at"],
                    }
                },
            )
            session.finalization = asyncio.create_task(self._finalize_session(session))
        await asyncio.shield(session.finalization)
        return session.verified_response

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Release an Environment Server episode, verified or not; repeated closes succeed.

        Close fences the session first, so a seed still provisioning releases its sandboxes instead
        of lending them, even if this close's caller stops waiting. It then waits for in-flight
        provisioning or grading and stops any sandbox still running without grading it. A close
        before (or instead of) the seed fences a late seed. If a sandbox cannot be stopped, close
        raises and keeps the fence; a repeated close retries, and the provider TTL bounds the orphan.
        A completed close is remembered for ``session_close_retry_window_seconds``.
        """
        key = body.resources_session_id
        self._expire_close_records()
        session = self._sessions.get(self._resources_sessions.get(key, ""))
        expected = self._closed_resources.get(key) or (
            session.resources_request.episode_id if session is not None else None
        )
        if expected is not None and expected != body.episode_id:
            raise HTTPException(409, "episode_id does not match the resources session")
        # Fence before waiting: a timed-out close must not leave a late allocation running.
        self._closed_resources[key] = body.episode_id
        async with self._resources_lock(key):
            session = self._sessions.get(self._resources_sessions.get(key, ""))
            if session is not None:
                for task in (session.execution, session.finalization):
                    if task is not None:
                        await asyncio.shield(task)
                if session.phase != "closed" or not lifecycle.released(session):
                    # The session owns its teardown: a caller that stops waiting cannot abandon a sandbox,
                    # the admission slot, or the retention record.
                    session.finalization = asyncio.create_task(self._release_closed_session(key, session))
                    await asyncio.shield(session.finalization)
                if not lifecycle.released(session):
                    raise HTTPException(503, "Task sandboxes could not be stopped; the provider TTL applies")
            self._remember_close(key, session)
            return ResourcesCloseSessionResponse(resources_session_id=key)

    async def _release_closed_session(self, key: str, session: Session) -> None:
        with rollout_context(session.request.capture_rollout_id):
            if session.termination is None:
                session.termination = AgentTermination(
                    reason="cancelled", detail="Resources session closed before verification"
                )
            await lifecycle.cleanup(session)
        self._remember_close(key, session)


def _lost_session(body: ResourcesSessionVerifyRequest, reason: str, failure_kind: str) -> JSONResponse:
    # The task row and the agent's response, as a graded result carries them, without TB4's grading fields.
    return JSONResponse(body.model_dump(mode="json") | lost_session_verification(reason, failure_kind=failure_kind))


def _termination_from_response(response: NeMoGymResponse) -> AgentTermination:
    """Map a closed agent's final response onto TB4's termination record (grading still runs)."""
    if response.status in (None, "completed"):
        return AgentTermination(reason="completed")
    if response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else None
        if reason == "max_output_tokens":
            return AgentTermination(reason="nonzero_exit", detail="The final model call hit max_output_tokens")
        return AgentTermination(reason="timeout", detail="The agent stopped at its time limit")
    return AgentTermination(reason="nonzero_exit", detail=f"Agent response status: {response.status}")


if __name__ == "__main__":
    TerminalBench4ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TerminalBench4ResourcesServer.run_webserver()
