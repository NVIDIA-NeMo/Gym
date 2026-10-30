# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import contextvars
import ipaddress
import logging
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from functools import wraps
from time import monotonic
from typing import Any, ClassVar, Optional
from urllib.parse import urlsplit
from warnings import warn

from fastapi import Body, FastAPI, HTTPException, Request, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

from nemo_gym._checkpoint.agent import (
    ActivationOutOfOrderError,
    AgentSessionParticipant,
    RestartOnlyAgentParticipant,
    RestartOnlyTrackingMiddleware,
    RestoredAgentSession,
)
from nemo_gym._checkpoint.control import CheckpointMode
from nemo_gym._checkpoint.errors import StaleAttemptError
from nemo_gym._checkpoint.participant_workers import install_server_participant
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym._checkpoint.steps import CHECKPOINT_RESTART_HEADER
from nemo_gym.agent_activation_types import (
    AgentContinuationCapabilities,
    AgentContinuationRequirements,
)
from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.agent_utils.ordered_operations import ActivationConflictError, OrderedOperationLedger
from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.config.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT, ModelServerRef
from nemo_gym.config.global_config import (
    OBSERVABILITY_ENABLED_KEY_NAME,
    TOKEN_ID_CAPTURE_BLOCK,
    get_first_server_config_dict,
)
from nemo_gym.core.episode import EpisodeId, TaskId, maybe_rollout_id_from_run_body, rollout_context
from nemo_gym.core.responses_api_types import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
    apply_rollout_prefix,
    rollout_path_prefix,
)
from nemo_gym.session_routing import SessionClaims, SessionKind
from nemo_gym.telemetry.endpoints import traced_endpoint, traced_rollout_endpoint
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.tool_access import ToolAccess


LOG = logging.getLogger(__name__)


# Session cookie key that binds later activation and close requests to their agent session.
AGENT_SESSION_COOKIE_KEY = "agent_session_id"
# An agent session is named by agent_session_id, in its seed and close bodies and in the cookie after its seed
# (see nemo_gym.session_routing).
AGENT_SESSIONS = SessionKind(
    cookie_key=AGENT_SESSION_COOKIE_KEY,
    id_field="agent_session_id",
    seed_path="/v1/agent_sessions",
    close_path="/v1/agent_sessions/close",
    cookie_session=False,
)
# Request header with an activation's index: its 1-based count within its agent session.
# With checkpointing,
# an agent that supports it answers an activation it already completed with the reply it gave,
# without running it again.
# An environment server that can invoke an activation again after a restore must send it:
# its checkpoint may precede the agent's reply while the agent's checkpoint follows it.
# Multi-turn continuation sessions use this same header without checkpointing.
# Other adapters may treat requests without the header as new activations.
AGENT_ACTIVATION_HEADER = "x-ng-agent-activation"
LOG = logging.getLogger(__name__)


class AgentSeedSessionRequest(BaseModel):
    """Idempotently initialize agent-server state under a caller-assigned identifier.

    Repeating the same identifier and episode must return the existing session.
    Closing an unknown identifier must prevent a racing seed within the retry window.
    The shared implementation retains close responses for session_close_retry_window_seconds;
    callers must use unique IDs and finish retries within that window. External resources
    should use provider TTLs when available; there is no active-session expiry timer.
    """

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str = Field(min_length=1)
    episode_id: EpisodeId
    task_id: TaskId
    tool_accesses: list[ToolAccess] = Field(default_factory=list)
    sandbox_access: SandboxAccess | None = None
    continuation: AgentContinuationRequirements | None = Field(
        default=None,
        description=(
            "Opt into continuation by resuming the harness's saved session and require these observations at seed time. "
            "Omitted for ordinary sessions."
        ),
    )
    # The task's own time budget for the harness's work, in seconds (Harbor's ``[agent].timeout_sec``).
    # The environment server enforces it; a harness that reads it can stop cleanly and return a partial turn.
    agent_timeout_sec: float | None = Field(default=None, gt=0)

    @field_validator("tool_accesses")
    @classmethod
    def require_unique_tool_names(cls, tool_accesses: list[ToolAccess]) -> list[ToolAccess]:
        names = [access.name for access in tool_accesses]
        if len(names) != len(set(names)):
            raise ValueError("tool access names must be unique within an agent session")
        return tool_accesses

    # Keep the model schema; a serializer return annotation would replace it.
    @model_serializer(mode="wrap")
    def _serialize_optional_metadata(self, handler: SerializerFunctionWrapHandler):
        result = handler(self)
        if self.continuation is None:
            result.pop("continuation", None)
        return result


class AgentSeedSessionResponse(BaseModel):
    """Confirm the caller-assigned agent session identifier and declare what the harness can use."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str
    # Whether the harness drives the task's own tool declarations (Harbor's ``[[environment.mcp_servers]]``
    # and ``skills_dir``). An environment server refuses such a task to a harness that does not, instead of
    # running it without its tools.
    supports_task_tool_context: bool = False
    capabilities: AgentContinuationCapabilities | None = Field(
        default=None,
        description=(
            "Continuation mode and observations supported by this adapter. "
            "Returned only when continuation was requested."
        ),
    )

    # Keep the model schema; a serializer return annotation would replace it.
    @model_serializer(mode="wrap")
    def _serialize_optional_metadata(self, handler: SerializerFunctionWrapHandler):
        result = handler(self)
        if self.capabilities is None:
            result.pop("capabilities", None)
        return result


class AgentSessionCapabilitiesResponse(BaseModel):
    """What this adapter's sessions can do, asked before any session is seeded.

    An environment server reads it to refuse a run before provisioning anything: a multi-step task whose turns
    must continue one harness session needs ``continuation``, as Harbor's multi-step trial refuses
    ``resume_trajectory`` for an agent without ``SUPPORTS_RESUME`` before any environment spend.
    """

    model_config = ConfigDict(extra="forbid")

    supports_task_tool_context: bool = False
    continuation: AgentContinuationCapabilities | None = None


class AgentCloseSessionRequest(BaseModel):
    """Close agent-server state."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str
    episode_id: EpisodeId


class AgentCloseSessionResponse(BaseModel):
    """Confirm closure and return captured observations."""

    model_config = ConfigDict(extra="forbid")

    agent_session_id: str
    agent_observations: AgentObservationBundle | None = None
    cleanup_confirmed: bool | None = Field(
        default=None,
        description=(
            "True means no harness was launched, termination was confirmed, or an owned sandbox was stopped. "
            "False means a final negative termination receipt. None is omitted when the adapter does not report "
            "confirmation. Connection release can succeed with false; the activation remains ungradable."
        ),
    )
    resources_cookies: dict[str, str] | None = Field(
        default=None,
        description=(
            "The Resources Server cookie jar after the agent's tool calls. "
            "None or an empty jar leaves the Environment Server's jar from the Resources seed unchanged."
        ),
    )

    trajectory: TrajectoryRecord | None = Field(
        default=None,
        description=(
            "The harness's own account of the run's decisions: the turns it took and the model calls behind "
            "each. Without it rollout collection rebuilds turns from every captured model call, so helper calls "
            "(context recovery, summarization) count as decisions and the step counts are lost. Cumulative "
            "across a continuation session's activations. Omitted when not reported."
        ),
    )

    # Keep the model schema; a serializer return annotation would replace it.
    @model_serializer(mode="wrap")
    def _serialize_optional_metadata(self, handler: SerializerFunctionWrapHandler):
        result = handler(self)
        for key in ("cleanup_confirmed", "trajectory"):
            if getattr(self, key) is None:
                result.pop(key, None)
        return result


@dataclass(frozen=True)
class ModelEndpoint:
    """The OpenAI-compatible endpoint a harness sends its model calls to.

    ``base_url`` ends in the API version (for example ``/v1``).
    ``model`` is set only when the endpoint dictates the model name; otherwise the harness keeps its own.
    """

    base_url: str
    model: str | None = None


class BaseResponsesAPIAgentConfig(BaseRunServerInstanceConfig):
    session_close_retry_window_seconds: float = Field(
        default=300,
        gt=0,
        allow_inf_nan=False,
        description="Retain close responses for this many seconds after cleanup; cover the caller's retry horizon.",
    )
    skip_verification: bool = False
    skip_verification_reward: float = 0.0
    # Whether this agent's rollouts participate in training token capture.
    # Native agents already receive token ids inline and normally leave this disabled.
    # Opaque external harnesses enable it because their returned output has no token ids.
    # The run-level ``token_id_capture.enabled`` setting gates the capture infrastructure.
    # The run-level ``token_id_capture.all_agents`` setting overrides this agent-level choice.
    token_id_capture: bool = False
    tool_accesses: list[ToolAccess] = Field(default_factory=list)

    @field_validator("tool_accesses")
    @classmethod
    def require_unique_tool_names(cls, tool_accesses: list[ToolAccess]) -> list[ToolAccess]:
        names = [access.name for access in tool_accesses]
        if len(names) != len(set(names)):
            raise ValueError("configured tool access names must be unique")
        return tool_accesses


class BaseResponsesAPIAgent(BaseServer):
    """An agent server: the harness Gym runs against the model for each task.

    ``calls_model_from_sandbox``: declare ``True`` on the class the entrypoint runs when the harness calls the
    model server from inside the task sandbox, where the loopback address is the container itself. A Harbor run
    (``gym eval run <target> --agent <name>``) reads that literal from the entrypoint source, as Gym reads
    ``ray_enabled``, and advertises a reachable host (``use_absolute_ip``) by default for such a harness. It is
    not read through inheritance: each such class declares it itself.
    """

    config: BaseResponsesAPIAgentConfig
    calls_model_from_sandbox: ClassVar[bool] = False


@dataclass
class AgentSessionState:
    """Harness-owned session state with the immutable caller-assigned seed binding."""

    request: AgentSeedSessionRequest


class AgentSeedRejectedError(HTTPException, ValueError):
    """Refuse an agent session seed this agent cannot serve, such as a grant or sandbox it does not support.

    FastAPI answers it with HTTP 422, which an Environment Server treats as terminal, so a configuration error
    fails the episode once with this message instead of being retried as if it were transient. It is also a
    ``ValueError``, so code that validates a seed outside a request (a checkpoint restore, a direct call in a test)
    keeps catching it as before.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(422, detail)


def _log_seed_refusal(agent_session_id: str, error: AgentSeedRejectedError) -> None:
    # FastAPI answers an HTTPException inside the router, so no exception middleware logs the refusal.
    LOG.warning("Refused agent session seed %s: %s", agent_session_id, error.detail)


class AgentSessionSetupError(Exception):
    """Retain incomplete setup for close while propagating the original setup error."""

    def __init__(self, state: AgentSessionState, *, error: BaseException) -> None:
        super().__init__(str(error))
        self.state: AgentSessionState = state
        self.error: BaseException = error


@dataclass
class _AgentSessionRecord:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state: AgentSessionState | None = None
    closing: bool = False
    episode_id: EpisodeId | None = None
    close_response: AgentCloseSessionResponse | None = None
    expires_at: float = float("inf")
    activations: OrderedOperationLedger[NeMoGymResponse] = field(default_factory=OrderedOperationLedger)
    close_task: asyncio.Task[AgentCloseSessionResponse] | None = None


# Sandbox providers whose processes share the host network namespace, so a loopback model URL still resolves
# to the host. Docker only does so when the daemon runs on the host and the container uses host networking.
_HOST_NETWORK_SANDBOX_PROVIDERS = frozenset({"local"})
_LOCAL_DOCKER_SANDBOX_PROVIDERS = frozenset({"docker"})
# An agent either exports its sessions or restarts its rollouts; it has no stateless mode.
_AGENT_CHECKPOINT_MODES = ("restart_only", "exported")


def is_loopback_host(url: str) -> bool:
    """Whether ``url`` points at the caller's own loopback interface (127.0.0.0/8, ::1, localhost)."""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host.lower() == "localhost" or host.lower().endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def assert_model_url_reachable_from_sandbox(url: str, *, provider_name: str | None) -> None:
    """Refuse a model URL that a sandboxed agent could not reach.

    Agents that run inside a sandbox (Hermes, mini-SWE) hand the Model Server URL to code in the
    container. Inside a container on a remote provider (OpenSandbox, E2B, Daytona, ...), a loopback
    host is the container itself, so every model call fails and the episode silently scores 0.

    Raises:
        ValueError: when ``url`` is loopback and ``provider_name`` is not a host-network or local
            Docker provider. Local Docker only logs a warning: loopback works there only when the
            daemon shares the host network.
    """
    if not is_loopback_host(url):
        return
    if provider_name in _HOST_NETWORK_SANDBOX_PROVIDERS:
        return
    if provider_name in _LOCAL_DOCKER_SANDBOX_PROVIDERS:
        LOG.warning(
            "Model Server URL %s is a loopback address; a %s sandbox reaches it only when the container shares "
            "the host network. Set ++use_absolute_ip=true or configure a reachable model URL otherwise.",
            url,
            provider_name,
        )
        return
    raise ValueError(
        f"Model Server URL {url} is a loopback address, which the {provider_name or 'remote'} sandbox cannot "
        "reach: inside the container, 127.0.0.1 is the container itself. Set ++use_absolute_ip=true so the "
        "agent advertises the host's network address, or configure a reachable model URL."
    )


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, AggregateMetricsMixin, SimpleServer):
    config: BaseResponsesAPIAgentConfig
    routes_sessions_to_owner = True
    _session_records: dict[str, _AgentSessionRecord] = PrivateAttr(default_factory=dict)
    _closed_session_records: OrderedDict[str, _AgentSessionRecord] = PrivateAttr(default_factory=OrderedDict)
    # Seeds whose setup failed, kept for the caller's close until session_close_retry_window_seconds passes.
    _failed_seed_records: OrderedDict[str, _AgentSessionRecord] = PrivateAttr(default_factory=OrderedDict)
    _failed_seed_releases: set[asyncio.Task] = PrivateAttr(default_factory=set)
    # With several workers, which worker claimed each agent_session_id (see nemo_gym.session_routing).
    _agent_session_claims: Optional[SessionClaims] = PrivateAttr(default=None)

    def claimed_session_routing(self) -> dict[str, Any]:
        return {"kind": AGENT_SESSIONS, "agent_sessions": self}

    def holds_agent_session(self, agent_session_id: str) -> bool:
        """Whether this worker keeps any record of the session: open, failed and awaiting close, or closed."""
        return agent_session_id in self._session_records

    def use_agent_session_claims(self, claims: SessionClaims) -> None:
        """Release a session's claim when this worker drops its last record (see nemo_gym.session_routing)."""
        self._agent_session_claims = claims

    def _release_agent_session_claim(self, agent_session_id: str) -> None:
        if self._agent_session_claims is not None:
            self._agent_session_claims.release(agent_session_id)

    def expire_agent_session_records(self) -> None:
        """Drop the close receipts whose window has passed, and release the failed seeds nobody closed."""
        now = monotonic()
        while self._closed_session_records:
            key, record = next(iter(self._closed_session_records.items()))
            if record.expires_at > now:
                break
            self._closed_session_records.pop(key)
            # A retire may have freed the session, and a new one may have taken its ID since.
            if self._session_records.get(key) is record:
                del self._session_records[key]
                self._release_agent_session_claim(key)
        while self._failed_seed_records:
            key, record = next(iter(self._failed_seed_records.items()))
            if record.expires_at > now:
                break
            self._failed_seed_records.pop(key)
            if self._session_records.get(key) is record and record.close_response is None:
                # Nobody closed the failed seed within the retry window, so release what its setup left.
                # A fresh context: the task must not run under the rollout of the request that found it expired.
                task = asyncio.create_task(self._release_failed_seed(key, record), context=contextvars.Context())
                self._failed_seed_releases.add(task)
                task.add_done_callback(self._failed_seed_releases.discard)

    @asynccontextmanager
    async def _locked_agent_session(self, session_id: str) -> AsyncIterator[_AgentSessionRecord]:
        self.expire_agent_session_records()
        while True:
            record = self._session_records.setdefault(session_id, _AgentSessionRecord())
            async with record.lock:
                # A waiter can outlive a failed seed or expired receipt. Never use its old lock
                # to access a replacement record for the same ID.
                if self._session_records.get(session_id) is not record:
                    continue
                try:
                    yield record
                finally:
                    if (
                        record.state is None
                        and record.close_response is None
                        and self._session_records.get(session_id) is record
                    ):
                        del self._session_records[session_id]
                        self._release_agent_session_claim(session_id)
                return

    def _keep_failed_seed(self, agent_session_id: str, record: _AgentSessionRecord) -> None:
        """Keep a failed seed's state for the caller's close, for as long as a close receipt would be kept."""
        record.closing = True
        record.expires_at = monotonic() + self.config.session_close_retry_window_seconds
        self._failed_seed_records.pop(agent_session_id, None)
        self._failed_seed_records[agent_session_id] = record

    async def _undo_seed(self, agent_session_id: str, record: _AgentSessionRecord) -> None:
        """Release a seeded session's state with the close hook, or keep it as a failed seed if the hook fails."""
        try:
            await self._close_agent_session_state(record.state)
        except Exception:
            LOG.exception("Could not release agent session %r after its seed failed", agent_session_id)
            self._keep_failed_seed(agent_session_id, record)
            return
        record.state = None

    async def _release_failed_seed(self, agent_session_id: str, record: _AgentSessionRecord) -> None:
        """Run the close hook on a failed seed's state that no close reached, then drop the record.

        The hook runs under the session's own rollout, as it would for the caller's close.
        If it fails, the record keeps its state for a later close, and the release is tried again a window later.
        """
        async with self._locked_agent_session(agent_session_id) as current:
            if current is not record or record.close_response is not None or record.state is None:
                return
            episode_id = record.episode_id
            try:
                with rollout_context(episode_id.capture_key if episode_id is not None else None):
                    await self._close_agent_session_state(record.state)
            except Exception:
                LOG.exception(
                    "Could not release agent session %r, whose seed failed and was never closed; retrying in %s s",
                    agent_session_id,
                    self.config.session_close_retry_window_seconds,
                )
                record.expires_at = monotonic() + self.config.session_close_retry_window_seconds
                self._failed_seed_records[agent_session_id] = record
                return
            # The lock's exit drops the record, now that it holds no state.
            record.state = None

    @staticmethod
    def _agent_session_id_from_request(request: Request | None) -> str | None:
        if request is None:
            return None
        try:
            session = request.session
        except (AssertionError, AttributeError):
            return None
        if not isinstance(session, Mapping) or AGENT_SESSION_COOKIE_KEY not in session:
            return None
        marker = session[AGENT_SESSION_COOKIE_KEY]
        if not isinstance(marker, str) or not marker:
            raise HTTPException(409, "Invalid agent session marker")
        return marker

    def _require_agent_session(self, agent_session_id: str) -> AgentSessionState:
        record = self._session_records.get(agent_session_id)
        if record is None or record.state is None or record.closing:
            raise HTTPException(409, "Unknown or closing agent_session_id")
        return record.state

    # How this agent takes part in partial-rollout checkpoints.
    # Agents that implement the AgentSessionHooks methods declare "exported".
    # The default fails closed: unfinished rollouts are restarts, which start over from their input after a crash.
    checkpoint_mode: ClassVar[CheckpointMode] = "restart_only"
    _checkpoint_participant: Optional[AgentSessionParticipant] = PrivateAttr(default=None)
    _restart_only: Optional[RestartOnlyAgentParticipant] = PrivateAttr(default=None)

    @property
    def checkpoint_participant(self) -> Optional[AgentSessionParticipant]:
        """The checkpoint participant, or ``None`` when checkpointing is off for this agent."""
        return self._checkpoint_participant

    def setup_checkpoint_participant(self, app: FastAPI) -> None:
        # An agent whose setup_webserver builds its own app still takes part, as restart-only if it has no hooks.
        self.setup_agent_checkpoint(app)

    def setup_agent_checkpoint(self, app: FastAPI) -> None:
        """Install this agent's checkpoint participant on ``app``, once."""
        if getattr(app.state, "nemo_gym_agent_checkpoint_installed", False):
            return
        app.state.nemo_gym_agent_checkpoint_installed = True
        # The declaration is a class attribute nothing else checks: a typo would act as a mode of its own.
        if self.checkpoint_mode not in _AGENT_CHECKPOINT_MODES:
            raise ValueError(
                f"{type(self).__name__} declares checkpoint_mode={self.checkpoint_mode!r}; "
                f"expected one of {_AGENT_CHECKPOINT_MODES}"
            )
        settings = checkpoint_settings(getattr(self.server_client, "global_config_dict", None))
        if settings is None:
            return
        if self.checkpoint_mode == "restart_only":
            # Fail closed: in-flight work is reported as restarts,
            # which never hold up a checkpoint and start over after a crash.
            self._restart_only = RestartOnlyAgentParticipant(self.retire_agent_session)
            participant = self._restart_only
            app.add_middleware(RestartOnlyTrackingMiddleware, participant=self._restart_only)
        else:
            self._checkpoint_participant = AgentSessionParticipant(self)
            participant = self._checkpoint_participant
        install_server_participant(
            app,
            participant,
            num_workers=self.config.num_workers or 1,
            auth_token=settings.control_auth_token,
            lease_grace_seconds=settings.lease_grace_seconds,
            instance_name=self.config.name,
        )

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        """Return the state of every session in ``session_keys`` for a checkpoint."""
        raise NotImplementedError

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        """Validate every session, then install all of them; never install a partial set."""
        raise NotImplementedError

    async def retire_agent_session(self, session_key: str) -> None:
        """Discard a session whose attempt was retired.

        The default releases the session with the agent's close hook, without a receipt,
        so a restart-only agent's retire stops the sandbox a seed started.
        """
        async with self._locked_agent_session(session_key) as record:
            if record.state is None or record.close_response is not None:
                return
            if type(self)._close_agent_session_state is not SimpleResponsesAPIAgent._close_agent_session_state:
                if record.state.request.continuation is not None:
                    record.closing = True
                    await self._close_continuation_session(record)
                else:
                    await self._close_agent_session_state(record.state)
            record.state = None

    def effective_tool_accesses(self, request: AgentSeedSessionRequest) -> list[ToolAccess]:
        """Overlay episode-scoped tool access onto configured declarations by name."""
        accesses = {access.name: access for access in self.config.tool_accesses}
        accesses.update((access.name, access) for access in request.tool_accesses)
        return list(accesses.values())

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)
        self.setup_agent_checkpoint(app)

        agent_attributes = {"nemo.gym.server.name": self.config.name}
        traced_responses = traced_endpoint(GymSpanGroup.AGENT, "gym.agent.responses", self.responses, agent_attributes)
        app.post("/v1/responses")(traced_responses)
        # A self-call made with ``url_path_for_run`` lands on a prefixed twin.
        # ``responses`` recovers the rollout id from the path.
        # The same handler serves prefixed and unprefixed calls.
        app.post(f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/v1/responses")(traced_responses)
        app.post(f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/{TOKEN_CAPTURE_PATH_SEGMENT}/v1/responses")(traced_responses)

        # Traced *inside* rollout_context, not outside it. The span reads
        # `current_rollout_id()` when it starts, so wrapping the other way round would
        # start the span before the ContextVar is set and every rollout span would be
        # missing its `nemo.gym.rollout.id` — which is exactly what a first run on real
        # hardware showed.
        run = traced_rollout_endpoint(self.run, agent_attributes)

        @wraps(run)
        async def run_with_rollout_context(*args: Any, **kwargs: Any) -> BaseVerifyResponse:
            body = kwargs.get("body")
            if body is None:
                body = next((arg for arg in args if isinstance(arg, BaseRunRequest)), None)
            with rollout_context(self.rollout_id_from_run(body)):
                return await run(*args, **kwargs)

        app.post("/run")(run_with_rollout_context)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.post("/v1/agent_sessions")(self.seed_agent_session)
        app.post("/v1/agent_sessions/close")(self.close_agent_session)
        app.get("/v1/agent_sessions/capabilities")(self.agent_session_capabilities)

        return app

    async def agent_session_capabilities(self) -> AgentSessionCapabilitiesResponse:
        """Declare the session capabilities a seed would confirm, without seeding: the task-tool context and the
        multi-turn continuation this adapter supports (``None`` for one-activation adapters and for agents that
        implement no sessions at all)."""
        return AgentSessionCapabilitiesResponse(
            supports_task_tool_context=self.supports_task_tool_context(),
            continuation=self._agent_continuation_capabilities(),
        )

    async def seed_agent_session(
        self,
        request: Request,
        body: AgentSeedSessionRequest,
        response: Response = None,
    ) -> AgentSeedSessionResponse:
        """Seed once per caller ID; identical retries reuse the same harness state."""
        if type(self)._seed_agent_session_state is SimpleResponsesAPIAgent._seed_agent_session_state:
            error = AgentSeedRejectedError("This agent does not implement episode sessions")
            _log_seed_refusal(body.agent_session_id, error)
            raise error
        if self._restart_only is not None and response is not None:
            # This agent cannot capture the session, so the episode that seeds it starts over after a crash.
            response.headers[CHECKPOINT_RESTART_HEADER] = "1"
        capabilities = self._agent_continuation_capabilities() if body.continuation is not None else None
        if body.continuation is not None:
            if capabilities is None or capabilities.mode != body.continuation.mode:
                raise HTTPException(422, "Agent does not support the required multi-turn continuation mode")
            missing = set(body.continuation.observations) - set(capabilities.observations)
            if missing:
                raise HTTPException(422, f"Agent lacks required continuation observations: {sorted(missing)}")
        current = self._agent_session_id_from_request(request)
        if current is not None and current != body.agent_session_id:
            raise HTTPException(409, "agent_session_id does not match the session cookie")
        participant = self._checkpoint_participant
        # With checkpointing, a seed waits out an open checkpoint and is refused for a retired attempt.
        # One in progress holds up prepare, and a retire waits for it.
        seeding = participant or self._restart_only
        async with (
            seeding.seeding(body.agent_session_id, body.episode_id) if seeding else nullcontext(),
            self._locked_agent_session(body.agent_session_id) as record,
        ):
            if record.close_response is not None:
                raise HTTPException(409, "Agent session is already closed")
            if record.closing:
                raise HTTPException(409, "Agent session is closing")
            if record.state is None:
                if current is not None:
                    raise HTTPException(409, "Agent session cookie has expired")
                record.episode_id = body.episode_id
                try:
                    record.state = await self._seed_agent_session_state(body.model_copy(deep=True))
                except AgentSessionSetupError as error:
                    record.state = error.state
                    self._keep_failed_seed(body.agent_session_id, record)
                    raise error.error from None
                except AgentSeedRejectedError as error:
                    _log_seed_refusal(body.agent_session_id, error)
                    raise
                try:
                    if participant is not None:
                        participant.open_session(body.agent_session_id, body.episode_id, seed=True)
                    elif self._restart_only is not None:
                        self._restart_only.open_session(body.agent_session_id, body.episode_id)
                except Exception:
                    # Nothing tracks the session, so undo the seed: release its state now,
                    # so the lock's exit drops the record and, with several workers, its claim.
                    await self._undo_seed(body.agent_session_id, record)
                    raise
            elif record.state.request != body:
                raise HTTPException(409, "agent_session_id is already bound to another seed request")
            request.session[AGENT_SESSION_COOKIE_KEY] = body.agent_session_id
            return AgentSeedSessionResponse(
                agent_session_id=body.agent_session_id,
                capabilities=capabilities,
                supports_task_tool_context=self.supports_task_tool_context(),
            )

    async def close_agent_session(
        self,
        request: Request,
        body: AgentCloseSessionRequest,
    ) -> AgentCloseSessionResponse:
        """Retain successful close responses for a bounded retry window, including observations.

        Failed release keeps state for retry. Closing an unknown ID prevents a delayed seed
        within the same window. No timer cancels active sessions: the episode owner and sandbox
        provider retain responsibility for normal and crash cleanup.
        """
        if type(self)._close_agent_session_state is SimpleResponsesAPIAgent._close_agent_session_state:
            raise NotImplementedError("This agent does not implement episode sessions")
        current = self._agent_session_id_from_request(request)
        if current is not None and current != body.agent_session_id:
            raise HTTPException(409, "agent_session_id does not match the session cookie")
        async with self._locked_agent_session(body.agent_session_id) as record:
            if record.episode_id is not None and record.episode_id != body.episode_id:
                raise HTTPException(409, "episode_id does not match the seeded agent session")
            if record.close_response is not None:
                return record.close_response.model_copy(deep=True)
            if record.state is None:
                if current is not None:
                    raise HTTPException(409, "Agent close receipt has expired")
                result = AgentCloseSessionResponse(agent_session_id=body.agent_session_id)
            else:
                result = await self._close_session_state(body.agent_session_id, record)
            record.state = None
            record.episode_id = body.episode_id
            record.close_response = result.model_copy(deep=True)
            record.expires_at = monotonic() + self.config.session_close_retry_window_seconds
            self._failed_seed_records.pop(body.agent_session_id, None)
            self._closed_session_records[body.agent_session_id] = record
            # Keep the marker so a stale /responses request cannot fall back to the non-session path.
            request.session[AGENT_SESSION_COOKIE_KEY] = body.agent_session_id
            return result

    async def _close_session_state(
        self, agent_session_id: str, record: _AgentSessionRecord
    ) -> AgentCloseSessionResponse:
        """Stop the session's activation, then release its state with the close hook.

        A close cancelled while it stops the activation leaves the hook to its retry, so the hook runs once.
        A retire can free the session while the close awaits.
        The close then stops there and leaves the session to the retire.
        """
        participant = self._checkpoint_participant
        if participant is not None and not record.closing:
            # The close waits out an open checkpoint, as a seed does, and only then marks the session closing.
            # The commit then exports the session as it was before the close,
            # which is where the environment server's checkpoint continues it.
            # A close that waited after marking the session would hold up prepare instead.
            await participant.wait_open()
            self._require_current_session(agent_session_id, record)
        record.closing = True
        if participant is not None and not await participant.close_session(agent_session_id):
            raise StaleAttemptError(f"agent session {agent_session_id!r} was retired while it closed")
        self._require_current_session(agent_session_id, record)
        if record.state.request.continuation is None:
            result = await self._close_agent_session_state(record.state)
        else:
            result = await self._close_continuation_session(record)
        self._require_current_session(agent_session_id, record)
        if self._restart_only is not None and record.episode_id is not None:
            self._restart_only.close_session(agent_session_id, record.episode_id)
        return result

    async def _close_continuation_session(self, record: _AgentSessionRecord) -> AgentCloseSessionResponse:
        if record.close_task is None or (
            record.close_task.done() and (record.close_task.cancelled() or record.close_task.exception() is not None)
        ):
            record.close_task = asyncio.create_task(self._close_agent_session_record(record))
            record.close_task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(record.close_task)

    def _require_current_session(self, agent_session_id: str, record: _AgentSessionRecord) -> None:
        if self._session_records.get(agent_session_id) is not record:
            raise StaleAttemptError(f"agent session {agent_session_id!r} was retired while it closed")

    def _activation_index(self, request: Request) -> Optional[int]:
        """Read the existing 1-based header for checkpointed or multi-turn continuation sessions."""
        session_id = self._agent_session_id_from_request(request)
        state = self._session_state(session_id) if session_id is not None else None
        continuation = state is not None and state.request.continuation is not None
        if self._checkpoint_participant is None and not continuation:
            return None
        raw = request.headers.get(AGENT_ACTIVATION_HEADER)
        if raw is None:
            return None
        try:
            index = int(raw)
        except ValueError:
            index = 0
        if index < 1:
            raise ActivationOutOfOrderError(f"{AGENT_ACTIVATION_HEADER} must be a positive integer, got {raw!r}")
        return index

    def _session_state(self, agent_session_id: str) -> Optional[AgentSessionState]:
        """The open state of a session, or ``None``; checkpoint hooks export it."""
        record = self._session_records.get(agent_session_id)
        return record.state if record is not None and not record.closing else None

    def _install_restored_session(self, agent_session_id: str, state: AgentSessionState) -> None:
        """Install a session restored from a checkpoint, as its replacement attempt's seed would have."""
        self._session_records[agent_session_id] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)

    def _free_session(self, agent_session_id: str) -> None:
        """Free a retired session that has no close receipt: its episode's close then finds nothing to free."""
        record = self._session_records.get(agent_session_id)
        if record is not None and record.close_response is None:
            self._session_records.pop(agent_session_id, None)
            self._release_agent_session_claim(agent_session_id)

    def _agent_continuation_capabilities(self) -> AgentContinuationCapabilities | None:
        """Opt into multi-turn continuation; existing one-activation adapters stay unchanged."""
        return None

    async def activate_agent_session(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """Run or replay one header-indexed delta in a seeded harness session."""
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "Activation requires the seeded agent session cookie")
        async with self._locked_agent_session(session_id) as record:
            state = self._require_agent_session(session_id)
            if state.request.continuation is None:
                raise HTTPException(422, "Session was not seeded for multi-turn continuation")
            rollout_id = request.path_params.get("rollout_id")
            if rollout_id is not None and rollout_id != state.request.episode_id.capture_key:
                raise HTTPException(409, "Activation rollout does not match the seeded session")
            try:
                index = self._activation_index(request)
            except ActivationOutOfOrderError as error:
                raise HTTPException(409, str(error)) from error
            if index is None:
                raise HTTPException(422, f"Multi-turn continuation requires {AGENT_ACTIVATION_HEADER}")
            params = body.model_copy(deep=True)
            # Reject malformed inputs before binding an activation index, including on retries.
            self._validate_agent_activation(state, params)

        try:
            return await record.activations.execute(
                index=index,
                request=params,
                operation=lambda: self._activate_agent_session_state(state, params, activation_id=index),
            )
        except ActivationConflictError as error:
            raise HTTPException(409, str(error)) from error

    def _validate_agent_activation(
        self, state: AgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> None:
        """Validate a delta without mutating the session or consuming its index."""
        raise NotImplementedError("This agent does not implement multi-turn continuation")

    async def _activate_agent_session_state(
        self, state: AgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming, *, activation_id: int
    ) -> NeMoGymResponse:
        """Run one input delta with the retained harness session and sandbox."""
        raise NotImplementedError("This agent does not implement multi-turn continuation")

    def _continuation_close_timeout(self) -> float:
        """Return the adapter's configured bound for stopping an active continuation."""
        raise NotImplementedError("This agent does not implement multi-turn continuation")

    async def _close_agent_session_record(self, record: _AgentSessionRecord) -> AgentCloseSessionResponse:
        await record.activations.close(timeout=self._continuation_close_timeout())
        assert record.state is not None
        # The adapter may release a borrowed connection after a final negative
        # termination receipt. Preserve that receipt; release errors still raise.
        return await self._close_agent_session_state(record.state)

    def supports_task_tool_context(self) -> bool:
        """Whether this harness uses the task's own tool declarations.

        Harbor tasks declare ``[[environment.mcp_servers]]`` and ``skills_dir``. A harness that drives them
        from inside the sandbox answers True, and the seed response carries the answer to the environment
        server, which refuses such a task to a harness that would run it without its tools.
        """
        return False

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        """Validate grants and initialize harness state.

        Raise AgentSeedRejectedError for a seed this agent cannot serve, such as a sandbox or a required grant it
        does not support: the seed answers a final 422 and binds nothing. An HTTPException keeps its explicit status;
        any other, unexpected exception answers 500, which an Environment Server retries as transient.
        If setup fails and cleanup cannot finish, raise AgentSessionSetupError with the
        partial state and original error. The base retains it for close, never activation.
        """
        raise NotImplementedError("This agent does not implement episode sessions")

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        """Release harness state, or raise without losing the handle needed for another close."""
        raise NotImplementedError("This agent does not implement episode sessions")

    def _capture_correlation_enabled(self) -> bool:
        """Return whether this agent needs rollout correlation.

        Evaluation uses ``/ng-rollout/<id>/...`` for every agent.
        Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
        Training capture requires ``token_id_capture.enabled``.
        It also requires the static agent flag or run-level ``all_agents``.
        Partial-rollout checkpointing also needs it for this agent.
        Missing global configuration disables correlation.
        """
        return (
            self._model_call_capture_enabled()
            or self._token_id_capture_enabled()
            # Checkpointing keys a legacy /run by its rollout, so its self-dispatch must carry the rollout.
            or self._checkpoint_participant is not None
            or self._restart_only is not None
        )

    def _model_call_capture_enabled(self) -> bool:
        """Whether evaluation model-call observability is enabled."""
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return False
        return bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False))

    def _token_id_capture_enabled(self) -> bool:
        """Whether this agent explicitly opted into training-token capture."""
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return False
        block = global_config.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        if not isinstance(block, Mapping) or not block.get("enabled", False):
            return False
        return bool(block.get("all_agents", False)) or bool(
            getattr(getattr(self, "config", None), "token_id_capture", False)
        )

    def rollout_id_from_run(self, body: Any) -> Optional[str]:
        """Return the capture id for a run request.

        Return ``None`` when capture is disabled.
        Return ``None`` when the body has no usable identity.
        """
        if not self._capture_correlation_enabled():
            return None
        return maybe_rollout_id_from_run_body(body)

    def url_path_for_run(self, url_path: str, body: Any) -> str:
        """Apply this run's capture path to a downstream URL path.

        Evaluation uses ``/ng-rollout/<id>/...``.
        Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
        Calls without a rollout id remain unchanged.
        """
        return (
            f"{rollout_path_prefix(self.rollout_id_from_run(body), token_capture=self._token_id_capture_enabled())}"
            f"{url_path}"
        )

    def base_url_for_run(self, base_url: str, body: Any) -> str:
        """Apply this run's capture path to a model-server root URL.

        Append the API-version suffix after this method returns. While traces export, the URL also
        names this run's current span, so the model server parents calls that arrive without a
        ``traceparent`` header (a harness in a sandbox) to it.
        """
        return apply_rollout_prefix(
            base_url,
            self.rollout_id_from_run(body),
            token_capture=self._token_id_capture_enabled(),
        )

    def url_path_for_request(self, url_path: str, request: Optional[Request]) -> str:
        """Carry an inbound capture path onto a downstream URL path.

        Prefixed self-calls expose the rollout id as a path parameter.
        Training-capture requests preserve their dedicated path segment.
        Unprefixed requests remain unchanged.
        """
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        request_path = getattr(getattr(request, "url", None), "path", "")
        token_capture = f"/{TOKEN_CAPTURE_PATH_SEGMENT}/" in request_path
        return f"{rollout_path_prefix(rollout_id, token_capture=token_capture)}{url_path}"

    def resolve_model_base_url(
        self, model_server_name: str, rollout_id: Optional[str] = None, *, trace_context: bool = True
    ) -> str:
        """Resolve a model-server URL with an optional rollout prefix (and trace context, as ``base_url_for_run``).

        Pass ``trace_context=False`` for a URL that a client cached across rollouts will keep using.
        """
        server_config = get_first_server_config_dict(self.server_client.global_config_dict, model_server_name)
        base_url = self.server_client._build_server_base_url(server_config)
        prefixed = apply_rollout_prefix(
            base_url, rollout_id, token_capture=self._token_id_capture_enabled(), trace_context=trace_context
        )
        return f"{prefixed}/v1"

    def model_endpoint(
        self,
        *,
        model_server: ModelServerRef | None,
        rollout_id: Optional[str] = None,
        base_url: Optional[str] = None,
        session_endpoint: ModelEndpoint | None = None,
    ) -> ModelEndpoint:
        """Choose the endpoint for a harness's model calls.

        In order: ``session_endpoint``, an endpoint the episode's session supplies (for example a component
        that records model calls from inside the task sandbox); the Gym model server, with this rollout's
        capture prefix; the agent's configured ``base_url``, used verbatim because it has no prefix routing.
        """
        if session_endpoint is not None:
            return session_endpoint
        if model_server is not None:
            return ModelEndpoint(base_url=self.resolve_model_base_url(model_server.name, rollout_id))
        if base_url:
            return ModelEndpoint(base_url=base_url)
        raise ValueError("No model endpoint: configure model_server or a model base URL")

    # TODO: right now there is no validation on the TypedDict NeMoGymResponseCreateParamsNonStreaming
    # We should explicitly add validation at this server level or we should explicitly not validate so that there is flexibility in this API.
    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    @abstractmethod
    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        pass

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Default: same RewardProfiler aggregation as resources server. Override to proxy."""
        if self.config.skip_verification:
            warn(
                "Skipping aggregate metrics because skip_verification=True; "
                "use disable_aggregation=True to avoid writing aggregate metric files.",
                RuntimeWarning,
                stacklevel=2,
            )
            return AggregateMetrics()

        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
            compute_repeat_metrics_fn=self.compute_repeat_metrics,
        )

    def classify_outcome(self, native_result: Mapping[str, Any]) -> AgentOutcome:
        """Override to classify native failures; the environment decides grading and retry eligibility."""
        return AgentOutcome(failure_kind="unknown")

    def report_outcome(self, response: NeMoGymResponse, native_result: Mapping[str, Any]) -> NeMoGymResponse:
        """Attach validated evidence through the existing string-valued response metadata."""
        if response.status != "failed":
            return response
        outcome = AgentOutcome.model_validate(self.classify_outcome(native_result))
        return response.model_copy(
            update={
                "metadata": {
                    **(response.metadata or {}),
                    AGENT_OUTCOME_METADATA_KEY: outcome.model_dump_json(exclude_none=True),
                }
            }
        )
