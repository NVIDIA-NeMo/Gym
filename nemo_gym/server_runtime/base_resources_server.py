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
import logging
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from inspect import iscoroutinefunction
from time import monotonic
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Optional, TypedDict, TypeVar, get_args

from fastapi import FastAPI, HTTPException, Request
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    StrictBool,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from starlette.datastructures import Headers
from starlette.middleware import Middleware


if TYPE_CHECKING:
    from itsdangerous import URLSafeSerializer

    # Type-only: importing MCPTool at runtime would be circular (mcp_auto_exposure imports this
    # module) and would pull the mcp SDK into agent/model processes that never need it.
    from nemo_gym.server_runtime.mcp_auto_exposure import MCPTool

from nemo_gym.client.server_client import SESSION_ID_KEY
from nemo_gym.config.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.config.global_config import MODEL_SAMPLING_KEY_NAME
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.core.failure_kinds import SESSION_LOST, SESSION_RELEASE_FAILED, FailureClass, validate_failure_kind
from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.failure_transport import observe_failure
from nemo_gym.core.model_call_sampling import ModelCallSampling, model_call_sampling_config
from nemo_gym.core.outcome_policy import OutcomeAction
from nemo_gym.core.responses_api_types import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.metrics.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_runtime._checkpoint.control import CheckpointMode
from nemo_gym.server_runtime._checkpoint.participant_workers import install_server_participant
from nemo_gym.server_runtime._checkpoint.resources import (
    MCPTokenClaims,
    ResourcesCheckpointMiddleware,
    ResourcesParticipant,
)
from nemo_gym.server_runtime._checkpoint.settings import checkpoint_settings
from nemo_gym.server_runtime._checkpoint.steps import CHECKPOINT_CLOSE_FOLLOWS_HEADER, StepType
from nemo_gym.server_runtime.base_server import BaseRunServerInstanceConfig, BaseServer, SimpleServer
from nemo_gym.server_runtime.request_handling.model_sampling import CallerSamplingMiddleware
from nemo_gym.server_runtime.request_handling.rollout_context import RolloutContextMiddleware
from nemo_gym.server_runtime.request_handling.session_routing import SessionKind, mcp_session_token
from nemo_gym.server_runtime.verifier_observability import observe_verifier
from nemo_gym.telemetry.endpoints import traced_verify_endpoint
from nemo_gym.trajectory.trajectory_record import VerifierInvocation
from nemo_gym.verification.judge import judge_failsafe


logger = logging.getLogger(__name__)

NEMO_GYM_MCP_SESSION_TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
NEMO_GYM_MCP_METADATA_KEY = "mcp"
# The MCP session token claim that names the attempt whose seed minted the token, by its capture key.
# Only a server with checkpointing on adds it.
MCP_TOKEN_EPISODE_CLAIM = "episode"
# Salt namespacing the signed MCP session token, so it can't be confused with another signer
# that happens to share the same session-middleware secret.
_MCP_TOKEN_SALT = "nemo-gym-mcp-session-token"

# The session hooks an ``exported`` resources server must implement.
_SESSION_HOOKS = ("export_session_states", "restore_session_states", "retire_session_state")


def mcp_session_token_serializer(secret: str) -> "URLSafeSerializer":
    """The signer of Gym MCP session tokens; build it once per server."""
    from itsdangerous import URLSafeSerializer

    return URLSafeSerializer(secret, salt=_MCP_TOKEN_SALT)


def mcp_session_token_payload(serializer: "URLSafeSerializer", token: str) -> Any:
    """The payload of a Gym MCP session token whose signature verifies, or ``None`` if it does not."""
    from itsdangerous import BadSignature

    try:
        return serializer.loads(token)
    except BadSignature:
        return None


def normalize_tool_name(name: str, server_name: Optional[str] = None) -> str:
    """Map a trajectory tool-call name to the server's bare tool name.

    HTTP-driven agents record bare tool names ("email_reply_email"); MCP-native agents (e.g.
    Claude Code) record them namespaced per server ("mcp__workplace_assistant__email_reply_email").
    Verifiers compare trajectory names against dataset/ground-truth vocabulary, so names are
    normalized before verify sees them and rollouts score identically on both transports.
    Non-namespaced names pass through unchanged. When ``server_name`` is given, only that server's
    prefix is stripped (robust to tool names that themselves contain double underscores).
    This runs only for servers exposed over MCP and mirrors how MCP clients namespace tool names,
    so a real tool that is itself named ``mcp__<server>__x`` being stripped is accepted.
    """
    if not name.startswith("mcp__"):
        return name
    if server_name is not None:
        prefix = f"mcp__{server_name}__"
        return name[len(prefix) :] if name.startswith(prefix) else name
    _, sep, tool = name[len("mcp__") :].partition("__")
    return tool if sep else name


# Tool names that would collide with the resources server's own endpoints if advertised over MCP.
RESERVED_MCP_TOOL_NAMES = frozenset({"verify", "seed_session", "close_session", "aggregate_metrics", "mcp"})


class ReverifyMode(str, Enum):
    STATELESS = "stateless"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class BaseResourcesServerConfig(BaseRunServerInstanceConfig):
    # Opt in to serve this server's tool routes over MCP; default off.
    expose_tools_over_mcp: bool = False
    mcp_server_name: Optional[str] = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="The name the MCP endpoint lists itself under in /seed_session's MCP metadata, which clients "
        "name its tools by (default: this server's name). For an endpoint that code a model can read calls, "
        "such as a harness in a sandbox. Needs expose_tools_over_mcp",
    )
    mcp_bearer_token: bool = Field(
        default=False,
        description="Hand out MCP session tokens as a standard 'Authorization: Bearer' header instead of "
        f"{NEMO_GYM_MCP_SESSION_TOKEN_HEADER}, and require one for tools/list too, which then lists only the "
        "session's tools; a tool outside the session's grant is answered as an unknown tool, and errors name no "
        "framework. For clients that pass the seed's MCP headers as given, as the environment server's MCP grants "
        "do. Needs expose_tools_over_mcp",
    )
    session_close_retry_window_seconds: float = Field(
        default=300,
        gt=0,
        allow_inf_nan=False,
        description=(
            "With several workers, how long a session stays with the worker that created it "
            "after its last request, once it is closed, verified or failed to seed; "
            "cover the caller's retry horizon and the server's own close fences. "
            "Servers that fence closed sessions with ResourcesSessionFences keep each fence this long, "
            "with any number of workers."
        ),
    )
    # The mode of reverification (for gym eval reverify) of this server.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNKNOWN

    @model_validator(mode="after")
    def _mcp_options_need_the_mcp_endpoint(self) -> "BaseResourcesServerConfig":
        """Refuse the MCP endpoint's options on a server that serves no MCP endpoint, where they would do nothing."""
        options = [name for name in ("mcp_server_name", "mcp_bearer_token") if getattr(self, name)]
        if options and not self.expose_tools_over_mcp:
            raise ValueError(
                f"{' and '.join(options)} configure this server's MCP endpoint, which it serves only with "
                "expose_tools_over_mcp: true"
            )
        return self


class BaseResourcesServer(BaseServer):
    config: BaseResourcesServerConfig


class BaseRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    capture_rollout_id: Optional[str] = Field(
        default=None,
        alias="_ng_rollout_id",
        exclude=True,
    )
    # The training framework's sampling values for this rollout's agent, which its model URLs carry.
    model_sampling: Optional[ModelCallSampling] = Field(
        default=None,
        alias=MODEL_SAMPLING_KEY_NAME,
        exclude=True,
    )
    # True turns token capture off for this rollout's model calls; null counts as unset. See
    # nemo_gym.core.episode_context.
    skip_token_capture: Optional[StrictBool] = Field(
        default=False,
        alias="_ng_skip_token_capture",
        exclude=True,
    )


class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float
    failure_outcome_action: OutcomeAction | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Failure outcome decision already applied by the owning boundary; "
            "independent of completion and attribution."
        ),
    )
    failure_outcome_attribution: FailureClass | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Shared failure classification applied by the owning boundary alongside failure_outcome_action.",
    )
    ng_verifier_invocations: list[VerifierInvocation] = Field(default_factory=list, exclude_if=lambda value: not value)

    mask_sample: bool = Field(
        default=False,
        description=(
            "Whether this completed sample should be excluded from evaluation scores and "
            "other downstream quality calculations because its reward is not a valid "
            "measurement of the evaluated system. Set it when the environment or its "
            "infrastructure failed rather than the evaluated system: a lost session, an "
            "unavailable judge, an OOM-killed container, a reset that timed out. The "
            "default means the reward is a valid measurement. It is independent of the "
            "diagnostic fields: a sample that degraded but was still measured validly "
            "keeps mask_sample=False while naming a failure_kind/failure_reason."
        ),
    )

    failure_kind: Optional[str] = Field(
        default=None,
        description=(
            "Which kind of failure this was, from the shared vocabulary in "
            "nemo_gym.core.failure_kinds. Stable and low cardinality, so it is safe to group by "
            "in logs, metrics and traces — unlike failure_reason. An environment that needs "
            "something the shared set should not grow can use '<server>:<kind>'. Orthogonal "
            "to mask_sample: naming the kind does not decide whether the sample is usable."
        ),
    )

    # Human-readable diagnosis of why `reward` may not reflect policy quality. Occurrence
    # detail, so never a metric label; `failure_kind` is the groupable half.
    failure_reason: Optional[str] = None

    @field_validator("failure_kind")
    @classmethod
    def _warn_on_unregistered_failure_kind(cls, value: Optional[str]) -> Optional[str]:
        """Producers learn about a name outside the vocabulary without losing the failure.

        Validation warns rather than rejects: an unregistered kind is a migration signal,
        and dropping the response over it would replace a visible wrong label with an
        invisible lost failure.
        """
        return validate_failure_kind(value)


class BaseMultiRewardVerifyResponse(BaseVerifyResponse):
    """Base verify response for environments with multiple reward objectives.

    Subclass this response instead of declaring ``reward_components`` on an
    environment-specific ``BaseVerifyResponse`` subclass. The mapping is required, and
    its objective keys should remain consistent across every task in the environment.

    Set the inherited ``reward`` to the scalar aggregate expected by single-reward
    consumers. To include individual objectives in aggregate metrics, also expose them
    as top-level numeric fields because metrics do not descend into this mapping. See
    ``resources_servers/example_tool_call_multireward`` for a complete example.
    """

    reward_components: dict[str, float]


class BaseSeedSessionRequest(BaseModel):
    pass


class BaseSeedSessionResponse(BaseModel):
    pass


class MCPServerMetadata(BaseModel):
    """Metadata returned from /seed_session for per-rollout Gym MCP access."""

    server_name: str
    url_path: str = "/mcp"
    transport: str = "http"
    headers: dict[str, str]


class ResourcesSeedSessionRequest(BaseModel):
    """Idempotently initialize resources state under a caller-assigned identifier.

    Repeating the same identifier and episode must return the existing session.
    Closing an unknown identifier must prevent a racing seed from creating it
    for session_close_retry_window_seconds (see ResourcesSessionFences);
    callers must use unique IDs and finish seed retries within that window.
    External resources should use provider TTLs when available. Current process-local
    session records may remain until the server shuts down.
    """

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str = Field(min_length=1)
    episode_id: EpisodeId
    task_id: TaskId
    task_data: dict[str, JsonValue]


def reject_partial_typed_seed(value: object) -> object:
    """Keep a malformed typed seed from falling through to a permissive cookie-seed request model.

    Use as a Pydantic ``BeforeValidator`` on the cookie-seed arm of a seed request union. A body that names a
    ``resources_session_id``, which is how the session router recognises a typed seed, or carries both typed
    ``episode_id`` and ``task_id`` objects, must validate as ``ResourcesSeedSessionRequest`` instead of creating a
    cookie session. A dataset row's own scalar ``task_id`` does not make a body typed.
    """
    if isinstance(value, dict) and (
        "resources_session_id" in value
        or (isinstance(value.get("episode_id"), dict) and isinstance(value.get("task_id"), dict))
    ):
        raise ValueError("A typed seed must supply a valid resources session, episode, task and task_data")
    return value


class ResourcesSeedSessionResponse(BaseModel):
    """Confirm resources state and return optional agent access and an initial observation."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    resources_tools: MCPServerMetadata | None = None
    sandbox_access: SandboxAccess | None = None
    # The single_agent_turn Environment Server appends this to the agent's input as one user message.
    observation: str | None = None

    # A return annotation would replace the model's serialization schema with that container schema.
    @model_serializer(mode="wrap")
    def _serialize_observation(self, handler: SerializerFunctionWrapHandler):
        """Preserve the seed wire shape when the server supplies no observation field."""
        result = handler(self)
        if "observation" not in self.model_fields_set:
            result.pop("observation", None)
        return result


VerificationInputT = TypeVar("VerificationInputT")


class ResourcesVerifyRequest(BaseModel, Generic[VerificationInputT]):
    """Carry typed environment output to a resources server."""

    model_config = ConfigDict(extra="forbid")

    episode_id: EpisodeId
    task_id: TaskId
    verification_input: VerificationInputT


class ResourcesCloseSessionRequest(BaseModel):
    """Close resources-server state."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    episode_id: EpisodeId


class ResourcesCloseSessionResponse(BaseModel):
    """Confirm resources-server state was closed."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str


class ResourcesSeedRejectedError(HTTPException, ValueError):
    """Refuse a seed whose input no retry can fix, such as ``task_data`` that fails validation.

    FastAPI answers it with HTTP 422, which an Environment Server treats as terminal, so the episode fails once
    with this message instead of being dispatched again with the same input. It is also a ``ValueError``, so code
    that seeds outside a request (a direct call in a test) keeps catching it as before.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(422, detail)


class ResourcesSessionConflictError(HTTPException, ValueError):
    """Refuse a seed or close that conflicts with what this server recorded for the session.

    Examples: a ``resources_session_id`` already bound to another episode or task, a close that names another
    episode, or a seed for a session that already closed. FastAPI answers it with HTTP 409, which an Environment
    Server treats as terminal. It is also a ``ValueError`` for direct callers.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(409, detail)


class ResourcesSessionLostError(HTTPException):
    """Answer a tool call for a session this server no longer holds with HTTP 404.

    Raise this instead of returning an ordinary tool observation that says the session is gone. What a harness does
    with the 404 varies: simple_agent returns its body to the model as the tool output and keeps going, Stirrup
    fails the run so the episode is dispatched again, and an MCP client reports a tool error. In every case the
    trajectory records a failed tool call whose detail names the ``failure_kind``, which a log can group by. The
    verify for the session answers with ``lost_session_verification``, whose masked sample keeps the episode out
    of the score.
    """

    def __init__(self, message: str, *, failure_kind: str = SESSION_LOST) -> None:
        super().__init__(404, {"failure_kind": validate_failure_kind(failure_kind), "message": message})


class LostSessionVerification(TypedDict):
    """The verify response fields of a masked sample for a session the server no longer holds."""

    reward: float
    mask_sample: bool
    failure_kind: str
    failure_reason: str


def lost_session_verification(reason: str, *, failure_kind: str = SESSION_LOST) -> LostSessionVerification:
    """Return the verify response fields for an episode whose session this server no longer holds.

    The sample is masked: the environment lost the state it would measure, so a reward of 0 would read as a
    policy that solved nothing. Merge the result over the request's fields so these win, for example
    ``BaseVerifyResponse(**(body.model_dump() | lost_session_verification("sandbox gone")))``; spreading both as
    keyword arguments raises ``TypeError`` when a request model that allows extra fields already carries a
    ``reward``. Use ``failure_kind=SESSION_EXPIRED`` when the server closed the session itself because its lifetime
    ran out.

    Each call logs a warning, because a masked sample is otherwise visible only in coverage metrics: many of them
    point at a wiring fault such as a session cookie that never reaches ``/verify``.
    """
    logger.warning("Masking the verify of a session this server no longer holds (%s): %s", failure_kind, reason)
    return {"reward": 0.0, "mask_sample": True, "failure_kind": failure_kind, "failure_reason": reason}


# A resources session is the cookie session: an episode seed or close names it by resources_session_id,
# which the cookie then carries as its SESSION_ID_KEY (see nemo_gym.server_runtime.request_handling.session_routing).
RESOURCES_SESSIONS = SessionKind(
    cookie_key=SESSION_ID_KEY,
    id_field="resources_session_id",
    seed_path="/seed_session",
    close_path="/close_session",
    cookie_session=True,
)


FenceRecordT = TypeVar("FenceRecordT")


class ResourcesSessionFences(Generic[FenceRecordT]):
    """The fences of the resources sessions this worker closed, each lasting the close retry window.

    A fence is the record a close leaves so that a seed arriving after the close (a retry, or a reply that raced
    the close) is refused instead of creating the session again. Each fence holds a record of the server's choice,
    such as the episode that first closed the session, which a retried close can check.
    A fence expires ``close_retry_window_seconds`` after the first close; a retried close does not extend it.
    Construct it with the server's ``session_close_retry_window_seconds``.

    With several workers, the worker's ownership of the ID expires that long after the last request for the session,
    and the fence is set before the close's reply is sent.
    So ownership never expires before the fence
    (see ``nemo_gym.server_runtime.request_handling.session_routing.ResourcesSessionLifetimes``).
    While ownership lasts, a late seed reaches the worker that holds the fence and is refused.
    After it expires, every worker treats the ID as new, so the answer does not depend on which worker gets it.
    """

    def __init__(self, close_retry_window_seconds: float) -> None:
        if not close_retry_window_seconds > 0:
            raise ValueError(f"close_retry_window_seconds must be positive, got {close_retry_window_seconds!r}")
        self._close_retry_window_seconds = close_retry_window_seconds
        # Earliest expiry first: every fence lasts the same window, and a fence is never moved.
        self._fences: OrderedDict[str, tuple[float, FenceRecordT]] = OrderedDict()

    def fence(self, resources_session_id: str, record: FenceRecordT) -> None:
        """Fence a closed session; call it only after its state was released.

        A retried close keeps the first close's record and expiry.
        """
        self._expire()
        if resources_session_id not in self._fences:
            self._fences[resources_session_id] = (monotonic() + self._close_retry_window_seconds, record)

    def get(self, resources_session_id: str) -> Optional[FenceRecordT]:
        """Return the record of the session's fence, or None if it is not fenced or its fence expired.

        A fence may hold a None record, so test ``resources_session_id in fences`` to ask whether a session is fenced.
        """
        self._expire()
        entry = self._fences.get(resources_session_id)
        return None if entry is None else entry[1]

    def raise_if_fenced(self, resources_session_id: str) -> None:
        """Refuse a seed for a fenced session with 409, which the environment server treats as terminal."""
        if resources_session_id in self:
            raise ResourcesSessionConflictError(f"Resources session is already closed: {resources_session_id}")

    def clear(self) -> None:
        """Delete every fence, for a server that drops all of its sessions at shutdown."""
        self._fences.clear()

    def __contains__(self, resources_session_id: object) -> bool:
        self._expire()
        return resources_session_id in self._fences

    def __len__(self) -> int:
        self._expire()
        return len(self._fences)

    def _expire(self) -> None:
        now = monotonic()
        while self._fences and next(iter(self._fences.values()))[0] <= now:
            self._fences.popitem(last=False)


class ResourcesSessionLocks:
    """Lock each resources session so that its seed and close never interleave.

    A session's lock exists only while a request holds or waits for it.
    """

    def __init__(self) -> None:
        self._locks: dict[str, _SessionLock] = {}

    @asynccontextmanager
    async def locked(self, resources_session_id: str) -> AsyncIterator[None]:
        """Hold the lock for ``resources_session_id``."""
        entry = self._locks.setdefault(resources_session_id, _SessionLock())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._locks[resources_session_id]


@dataclass
class _SessionLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@dataclass
class ResourcesVerification:
    """The live state or retained evidence admitted by ``SimpleResourcesServer.verifying``.

    A verifier masks ``is_lost`` with ``lost_session_verification``. Otherwise it scores live state when
    ``is_live`` or uses ``retained`` on a retry. Call ``retain`` before an awaited judge or before releasing
    anything needed by a retry. Payloads belong to the verifier; snapshot mutable state rather than retaining
    a reference to a live table. Call ``retain(None)`` explicitly for a verifier that scores only the request.
    Omitting ``retain`` makes a subsequent retry lost, rather than silently grading without its evidence.
    """

    is_live: bool
    is_lost: bool
    retained: object | None = None
    _retained: bool = field(default=False, repr=False)

    def retain(self, payload: object | None) -> None:
        """Save compact evidence for retries after the context releases the live resources."""
        if self.is_lost:
            raise RuntimeError("Cannot retain verification evidence for a lost resources session")
        self.retained = payload
        self._retained = True


@dataclass
class _VerificationReceipt:
    identity: tuple[EpisodeId, TaskId] | None
    payload: object | None
    retained: bool


class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    """Resources server with an optional shared stateful lifecycle.

    Implement the three state hooks to inherit seed, close and checkpoint retirement, and use ``verifying``
    around scoring to release after verification. Servers without state hooks keep their permissive seed and
    no-op close. Existing custom protocol overrides remain registered unchanged.

    Session state, locks, fences and verification receipts live in the worker process. With several workers,
    session routing keeps a session's requests on the worker that owns it, so a hook server needs no
    ``num_workers: 1``.
    """

    config: BaseResourcesServerConfig
    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = BaseSeedSessionRequest
    _session_episodes: dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)
    _initializing_sessions: dict[str, tuple[EpisodeId, TaskId] | None] = PrivateAttr(default_factory=dict)
    _cookie_sessions: set[str] = PrivateAttr(default_factory=set)
    _lifecycle_locks: ResourcesSessionLocks = PrivateAttr(default_factory=ResourcesSessionLocks)
    _lifecycle_closed_sessions: ResourcesSessionFences[EpisodeId] | None = PrivateAttr(default=None)
    _verification_receipts: OrderedDict[str, tuple[float, _VerificationReceipt]] = PrivateAttr(
        default_factory=OrderedDict
    )
    _pending_verifications: dict[str, _VerificationReceipt] = PrivateAttr(default_factory=dict)
    _cookie_seed_warning_emitted: bool = PrivateAttr(default=False)
    # How this server's sessions take part in partial-rollout checkpoints. "exported" servers implement the
    # ResourcesSessionHooks methods.
    # The default fails closed: its sessions' rollouts are restarts, which never hold up a checkpoint,
    # are left out of it, and start over after a crash.
    checkpoint_mode: ClassVar[CheckpointMode] = "restart_only"
    # "replay" declares that /verify is safe to run again after a crash
    # and does not change checkpointed session state, so a checkpoint need not wait for it.
    # Callers learn it from the /seed_session reply.
    checkpoint_verify: ClassVar[StepType] = "wait"
    # The routes that start a session for the request's rollout, such as a Gymnasium server's /reset.
    # Each waits out a checkpoint before its handler runs, needs a rollout prefix,
    # and binds the session to that rollout when it succeeds.
    checkpoint_session_starts: ClassVar[tuple[str, ...]] = ("/seed_session",)
    # True for a server whose /verify may ask a model through a client that notes no request (an external SDK such
    # as openai or anthropic). Its verifier then saves no grading record of its own, so its trials are observed only
    # through a recorded grading operation or a saved grading artifact.
    grading_may_use_uninstrumented_model: ClassVar[bool] = False
    _checkpoint: Optional[ResourcesParticipant] = PrivateAttr(default=None)
    _mcp_token_serializer: Any = PrivateAttr(default=None)
    routes_sessions_to_owner = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        hooks = ("has_session_state", "initialize_session_state", "release_session_state")
        implemented = [name for name in hooks if getattr(cls, name) is not getattr(SimpleResourcesServer, name)]
        if not implemented:
            return
        missing = sorted(set(hooks) - set(implemented))
        if missing:
            raise TypeError(f"{cls.__name__} must implement all session state hooks; missing {missing}")
        for name in ("initialize_session_state", "release_session_state"):
            if not iscoroutinefunction(getattr(cls, name)):
                raise TypeError(f"{cls.__name__}.{name} must be async")
        if iscoroutinefunction(cls.has_session_state):
            raise TypeError(f"{cls.__name__}.has_session_state must be synchronous")

    def _manages_session_state(self) -> bool:
        return type(self).initialize_session_state is not SimpleResourcesServer.initialize_session_state

    def has_session_state(self, session_id: str) -> bool:
        """Whether the verifier can use live state for this ID; keep this synchronous and cheap.

        Implement all three hooks to opt in. This also protects unbound state created outside ``seed_session``
        from a typed seed taking it over. Failed initialization is tracked separately by the base.
        """
        return False

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """Create state from validated task data, under the session lock.

        The base releases the preceding cookie run first and does not initialize again on a successful typed
        retry. On any exception, including cancellation, it calls ``release_session_state`` before re-raising.
        Make partial allocations reachable by release before another await. A cleanup that fails or is itself
        cancelled stays owned and is retried by the next seed or close of the session, before another initialization;
        a seed that returns an error is not tracked by checkpoint retirement. No separate pending-resource table
        is required.
        """

    async def release_session_state(self, session_id: str) -> None:
        """Idempotently release live and partially initialized resources under the session lock.

        Called before cookie replacement, after failed initialization, on leaving ``verifying`` (also when
        scoring raises), and by close and retirement. Unknown and already released IDs must succeed. Raise on
        failure and retain resource references needed to retry; the base retains ownership until success.

        The base holds ``session_lock`` when it calls this hook, and the lock is not reentrant: do not take it here.
        Tool handlers are not locked by the base: they take ``session_lock`` themselves, or this hook awaits their
        in-flight work.
        Sessions abandoned before verify/close have no automatic expiry; providers should enforce a TTL.
        """

    @asynccontextmanager
    async def session_lock(self, session_id: str) -> AsyncIterator[None]:
        """Serialize a session's lifecycle and cooperating tools; this lock is not reentrant."""
        async with self._lifecycle_locks.locked(session_id):
            yield

    def _get_lifecycle_fences(self) -> ResourcesSessionFences[EpisodeId]:
        # Lazy creation also works for subclasses whose model_post_init does not call super().
        if self._lifecycle_closed_sessions is None:
            self._lifecycle_closed_sessions = ResourcesSessionFences(self.config.session_close_retry_window_seconds)
        return self._lifecycle_closed_sessions

    def session_is_closed(self, session_id: str) -> bool:
        """Whether a successful close still fences this ID within the close retry window."""
        return session_id in self._get_lifecycle_fences()

    def _verification_receipt(self, session_id: str) -> _VerificationReceipt | None:
        now = monotonic()
        while self._verification_receipts and next(iter(self._verification_receipts.values()))[0] <= now:
            self._verification_receipts.popitem(last=False)
        pending = self._pending_verifications.get(session_id)
        if pending is not None:
            return pending
        record = self._verification_receipts.get(session_id)
        return record[1] if record is not None else None

    def session_is_verified(self, session_id: str) -> bool:
        """Whether a verification attempt is retained, including one whose cleanup must retry.

        The attempt may lack retry evidence; ``verifying`` then reports it as lost rather than scoring again.
        """
        return self._verification_receipt(session_id) is not None

    def session_release_pending(self, session_id: str) -> bool:
        """Whether verification left resources that close or retirement must still release.

        A successful score does not end checkpoint or worker ownership while its cleanup is pending.
        """
        return session_id in self._pending_verifications

    def session_identity(self, session_id: str) -> tuple[EpisodeId, TaskId] | None:
        """Return active, partially initialized, or retained verification ownership for this ID."""
        receipt = self._verification_receipt(session_id)
        return (
            self._session_episodes.get(session_id)
            or self._initializing_sessions.get(session_id)
            or (receipt.identity if receipt is not None else None)
        )

    def restore_session_identity(self, session_id: str, *, episode_id: EpisodeId, task_id: TaskId) -> None:
        """Restore a checkpoint's binding before replaying the continuation's typed seed.

        Restore the continuation's episode (``next_attempt``), not the checkpoint's preceding attempt. Call
        while restoring state synchronously or holding ``session_lock``. A closed or verified ID cannot restore.
        """
        self.restore_session_identities({session_id: (episode_id, task_id)})

    def restore_session_identities(self, identities: Mapping[str, tuple[EpisodeId, TaskId] | None]) -> None:
        """Validate all checkpoint bindings, then install them atomically without an await.

        ``None`` represents an unbound checkpoint. Validate and decode the complete state batch before calling
        this method, then install its state without awaiting, so a failure cannot leave half a restored batch.
        """
        for session_id, identity in identities.items():
            self._get_lifecycle_fences().raise_if_fenced(session_id)
            if self.session_is_verified(session_id) or session_id in self._initializing_sessions:
                raise ResourcesSessionConflictError(
                    "Cannot restore a verified or partially initialized resources session"
                )
            if session_id in self._cookie_sessions:
                raise ResourcesSessionConflictError("Cannot restore over a cookie-seeded resources session")
            previous = self.session_identity(session_id)
            if previous is not None and previous != identity:
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
        for session_id, identity in identities.items():
            if identity is not None:
                self._session_episodes[session_id] = identity

    @asynccontextmanager
    async def verifying(self, session_id: str, *, allow_missing: bool = False) -> AsyncIterator[ResourcesVerification]:
        """Own verification's lock, release and bounded retry evidence.

        Mask ``verification.is_lost`` before scoring. Otherwise use live state or ``retained`` and call
        ``retain`` with compact evidence. On exit, including a scoring exception, release resources and move
        ownership into a retry record. Failed release is observed without replacing the score or scoring error,
        and keeps ownership and evidence for a later verify/close/retire. Cancellation still propagates: a request
        cancelled by its caller's disconnect does not finish its release, so the pending record keeps ownership and
        the next verify retry, close or retirement releases the session (the provider's TTL does if none comes).
        A close that arrives meanwhile waits for the lock, and is cancelled without side effects if its caller gives up.
        Successful retries do not extend the record's window; a record whose cleanup is pending has no deadline until
        the cleanup succeeds. Close always invalidates it. A retry with no
        explicitly retained evidence is lost; pending cleanup is still attempted when it returns that mask.

        ``allow_missing`` is for request-scored verifiers that intentionally accept unseeded calls. It never
        permits verification of a closed session. The caller still masks any environment-specific state loss.
        """
        if not self._manages_session_state():
            raise TypeError("verifying requires all session state hooks")
        async with self.session_lock(session_id):
            receipt = self._verification_receipt(session_id)
            previous = self._verification_receipts.get(session_id)
            retry_deadline = previous[0] if previous is not None else None
            live = receipt is None and self.has_session_state(session_id)
            lost = (
                self.session_is_closed(session_id)
                or session_id in self._initializing_sessions
                or (receipt is not None and not receipt.retained)
                or (not live and receipt is None and not allow_missing)
            )
            verification = ResourcesVerification(
                live, lost, receipt.payload if receipt is not None else None, receipt.retained if receipt else False
            )
            if lost and not self.session_release_pending(session_id):
                yield verification
                return
            identity = self.session_identity(session_id)
            try:
                yield verification
            finally:
                pending = _VerificationReceipt(identity, verification.retained, verification._retained)
                self._pending_verifications[session_id] = pending
                try:
                    await self.release_session_state(session_id)
                except Exception as error:
                    failure = NgFailureEvent(
                        boundary=FailureBoundary.RESOURCES_SERVER,
                        failure_kind=SESSION_RELEASE_FAILED,
                        failure_reason="Resources session cleanup after verification failed",
                        operation="verify.cleanup",
                        stage="cleanup",
                    )
                    failure.__cause__ = error
                    observe_failure(failure)
                    logger.warning("Failed to release verified resources session %s", session_id, exc_info=True)
                else:
                    self._session_episodes.pop(session_id, None)
                    self._initializing_sessions.pop(session_id, None)
                    self._cookie_sessions.discard(session_id)
                    self._pending_verifications.pop(session_id, None)
                    # Another session may have expired this record while the judge awaited. A successful retry
                    # must not resurrect it with a fresh window when it finishes.
                    now = monotonic()
                    deadline = (
                        retry_deadline
                        if retry_deadline is not None
                        else now + self.config.session_close_retry_window_seconds
                    )
                    if deadline > now:
                        self._verification_receipts[session_id] = (deadline, pending)
                    else:
                        self._verification_receipts.pop(session_id, None)

    async def _initialize_session(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        try:
            await self.initialize_session_state(session_id, body)
        except BaseException:
            # Includes cancellation: a published partial handle must remain reachable until release succeeds.
            try:
                await self.release_session_state(session_id)
            except Exception:
                logger.warning(
                    "Failed to release a partially initialized resources session %s", session_id, exc_info=True
                )
            else:
                self._initializing_sessions.pop(session_id, None)
                self._cookie_sessions.discard(session_id)
            raise

    def _forget_session(self, session_id: str) -> None:
        self._session_episodes.pop(session_id, None)
        self._initializing_sessions.pop(session_id, None)
        self._cookie_sessions.discard(session_id)
        self._pending_verifications.pop(session_id, None)
        self._verification_receipts.pop(session_id, None)

    def owned_session_routing(self) -> dict[str, Any]:
        return {
            "kind": RESOURCES_SESSIONS,
            "close_retry_window_seconds": self.config.session_close_retry_window_seconds,
            "session_starts": tuple(self.checkpoint_session_starts),
            "session_ends": ("/verify",),
            "close_follows_header": CHECKPOINT_CLOSE_FOLLOWS_HEADER,
            "session_release_pending": self.session_release_pending,
        }

    def setup_webserver(self) -> FastAPI:
        app = self.create_resources_app()
        self.register_protocol_routes(app)
        return app

    def create_resources_app(self) -> FastAPI:
        """Build the app with what every resources server shares, whatever its episode protocol.

        That is the session and rollout middleware, the checkpoint participant,
        ``/aggregate_metrics`` and ``/reverify_mode``.
        A server with another protocol, such as Gymnasium's ``/reset`` and ``/step``,
        overrides ``register_protocol_routes`` instead of building its own app.
        """
        app = FastAPI()

        self.setup_session_middleware(app)
        app.add_middleware(RolloutContextMiddleware)
        if model_call_sampling_config(getattr(self.server_client, "global_config_dict", None)) is not None:
            app.add_middleware(CallerSamplingMiddleware)
        self.setup_resources_checkpoint(app)

        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.get("/reverify_mode")(self.get_reverify_mode)
        return app

    def register_protocol_routes(self, app: FastAPI) -> None:
        """Register the episode routes ``/seed_session``, ``/close_session`` and ``/verify``."""
        if type(self).seed_session is SimpleResourcesServer.seed_session:
            # Inject Request at the HTTP boundary while keeping seed_session(body) valid for direct callers.
            cookie_body = (
                Annotated[self.seed_request_model, BeforeValidator(reject_partial_typed_seed)]
                if self._manages_session_state()
                else BaseSeedSessionRequest
            )
            # A real annotation object, not a string: this module must not use `from __future__ import annotations`,
            # or FastAPI cannot resolve the local name and reads `body` as a query parameter.
            seed_body = ResourcesSeedSessionRequest | cookie_body

            async def seed_endpoint(
                request: Request, body: seed_body
            ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
                return await self.seed_session(body, request=request)

            app.post("/seed_session", name="seed_session")(seed_endpoint)
        else:
            app.post("/seed_session")(self.seed_session)
        app.post("/close_session")(self.close_resources_session)
        config = getattr(self.server_client, "global_config_dict", {})
        # Wrapped outside judge_failsafe so the span covers the failsafe's own handling too.
        app.post("/verify")(
            traced_verify_endpoint(
                observe_verifier(
                    judge_failsafe(self.verify),
                    enabled=TypeAdapter(bool).validate_python(config.get("observability_enabled", False)),
                    verifier_name=self.config.name,
                    uninstrumented_models=self.grading_may_use_uninstrumented_model,
                ),
                static_attributes={"nemo.gym.server.name": self.config.name},
            )
        )

    def checkpoint_mode_in_effect(self) -> CheckpointMode:
        """How this instance takes part in checkpoints; the class's ``checkpoint_mode`` unless an override knows better.

        Status, discovery and seed replies report it.
        A server whose capture depends on its configuration, such as on a sandbox provider that can snapshot,
        overrides it.
        """
        return self.checkpoint_mode

    def setup_resources_checkpoint(self, app: FastAPI) -> None:
        # The declarations are class attributes nothing else checks: a typo would act as a mode of its own.
        for name, allowed in (("checkpoint_mode", CheckpointMode), ("checkpoint_verify", StepType)):
            if getattr(self, name) not in get_args(allowed):
                raise ValueError(
                    f"{type(self).__name__} declares {name}={getattr(self, name)!r}; expected one of {get_args(allowed)}"
                )
        if not all(isinstance(path, str) and path.startswith("/") for path in self.checkpoint_session_starts):
            raise ValueError(
                f"{type(self).__name__} declares checkpoint_session_starts={self.checkpoint_session_starts!r}; "
                "expected a tuple of route paths"
            )
        settings = checkpoint_settings(getattr(self.server_client, "global_config_dict", None))
        if settings is None:
            return
        mode = self.checkpoint_mode_in_effect()
        if mode not in get_args(CheckpointMode):
            raise ValueError(f"{type(self).__name__}.checkpoint_mode_in_effect() returned {mode!r}")
        if mode == "exported":
            # Hooks from an intermediate base class or a mixin count; only the stubs here do not.
            missing = [
                name
                for name in _SESSION_HOOKS
                if getattr(type(self), name) is getattr(SimpleResourcesServer, name)
                and not (name == "retire_session_state" and self._manages_session_state())
            ]
            if missing:
                raise ValueError(
                    f"{type(self).__name__} declares checkpoint_mode='exported' but does not implement {missing}"
                )
        participant = ResourcesParticipant(
            self,
            mode,
            verify_step_type=self.checkpoint_verify,
            session_release_pending=self.session_release_pending,
        )
        participant.keeps_external_state = (
            mode == "exported" and type(self).delete_external_state is not SimpleResourcesServer.delete_external_state
        )
        self._checkpoint = participant
        install_server_participant(
            app,
            participant,
            num_workers=self.config.num_workers or 1,
            auth_token=settings.control_auth_token,
            lease_grace_seconds=settings.lease_grace_seconds,
            instance_name=self.config.name,
        )
        # Innermost, so the session middleware has already resolved the cookie session ID.
        mcp_token_claims = self._mcp_token_claims if self.config.expose_tools_over_mcp else None
        app.user_middleware.append(
            Middleware(
                ResourcesCheckpointMiddleware,
                participant=participant,
                mcp_token_claims=mcp_token_claims,
                session_starts=frozenset(self.checkpoint_session_starts),
            )
        )
        app.state.nemo_gym_resources_checkpoint_installed = True

    def setup_checkpoint_participant(self, app: FastAPI) -> None:
        # Fail closed rather than install the participant here:
        # an app this class did not build may lack the middleware it relies on, and declares no session starts,
        # so the participant would track nothing, and commits would go ahead without the server's live sessions.
        if getattr(app.state, "nemo_gym_resources_checkpoint_installed", False):
            return
        if checkpoint_settings(getattr(self.server_client, "global_config_dict", None)) is not None:
            raise ValueError(
                f"{type(self).__name__}.setup_webserver did not install the checkpoint participant; "
                "build the app with SimpleResourcesServer.setup_webserver, or call setup_resources_checkpoint on it"
            )

    @property
    def checkpoint_participant(self) -> Optional[ResourcesParticipant]:
        """The participant that takes part in partial-rollout checkpoints, or ``None`` with checkpointing off."""
        return self._checkpoint

    def _mcp_token_claims(self, scope: dict[str, Any]) -> Optional[MCPTokenClaims]:
        """The session and attempt an MCP request's signed token names.

        ``None`` without a token or for one that does not verify; the MCP endpoint rejects a bad token.
        An attempt claim that does not decode counts as missing.
        """
        header = "Authorization" if self.config.mcp_bearer_token else NEMO_GYM_MCP_SESSION_TOKEN_HEADER
        text = mcp_session_token(Headers(scope=scope), header)
        if text is None:
            return None
        if self._mcp_token_serializer is None:
            self._mcp_token_serializer = mcp_session_token_serializer(self.get_session_secret())
        payload = mcp_session_token_payload(self._mcp_token_serializer, text)
        if not isinstance(payload, dict) or not isinstance(payload.get("sid"), str):
            return None
        episode_id = None
        capture_key = payload.get(MCP_TOKEN_EPISODE_CLAIM)
        if isinstance(capture_key, str):
            try:
                episode_id = EpisodeId.from_capture_key(capture_key)
            except ValueError:
                pass
        return MCPTokenClaims(session_id=payload["sid"], episode_id=episode_id)

    def checkpoint_session_ended(self, request: Request) -> None:
        """Report that the request's session ended; standard ``/close_session`` and ``/verify`` do this.

        Call it from a protocol route that ends a session under another name, such as a terminal step.
        """
        session_id = request.session.get(SESSION_ID_KEY)
        if self._checkpoint is not None and session_id is not None:
            self._checkpoint.ended(session_id)
        # With several workers, this worker's ownership of the session ends a window after its last request from now on.
        # A request without an app, such as a test's stand-in, has no routing and so no ownership to end.
        app = getattr(request, "app", None)
        lifetimes = getattr(getattr(app, "state", None), "nemo_gym_resources_session_lifetimes", None)
        if lifetimes is not None and session_id is not None:
            lifetimes.ended(session_id)

    async def export_session_states(self, session_ids: list[str], *, checkpoint_id: str) -> dict[str, JsonValue]:
        """Return each session's state for checkpoint ``checkpoint_id``, leaving out sessions this server no longer
        holds."""
        raise NotImplementedError

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        """Validate every state, then install all of them; never install a partial set."""
        raise NotImplementedError

    async def retire_session_state(self, session_id: str) -> None:
        """Release a retired attempt's state without fencing a future checkpoint restoration.

        The checkpoint participant refuses the retired attempt's requests. It also calls this method to undo a
        restoration or drop an unclaimed session, so a later restore of the same ID must remain possible. A
        failed release preserves active, pending or verified ownership for an owning close to retry.
        """
        if not self._manages_session_state():
            return
        async with self.session_lock(session_id):
            await self.release_session_state(session_id)
            self._forget_session(session_id)

    async def delete_external_state(self, kept: frozenset[str]) -> None:
        """Delete what exports of checkpoints outside ``kept`` kept outside Gym, such as sandbox snapshots.

        Called after the controller published a checkpoint and pruned old ones, only in the ``exported`` mode;
        an empty ``kept`` means nothing will be restored again. The default keeps nothing outside Gym.

        - It holds the control lock: no prepare, commit, or restore runs meanwhile, and no checkpoint is open.
        - It is cancelled at the release request's deadline, and the controller releases again later,
          so it must tolerate state an earlier call already deleted and resume a deletion one left half done.
        - Admission stays open: episodes keep running, so it must not touch what live sessions use.
        - With several workers every worker runs it, so deleting what another worker already deleted must not fail.
        - Keep by reference, not by age: keep whatever a checkpoint in ``kept`` refers to, and delete the rest.
        """

    def normalize_tool_name(self, name: str) -> str:
        """Strip this server's MCP namespace from a trajectory tool-call name (see module function)."""
        # MCP clients namespace tools by the name the endpoint lists itself under
        return normalize_tool_name(name, self.config.mcp_server_name or self.config.name or self.__class__.__name__)

    def mcp_tools(self, harvested: list["MCPTool"], catchall: Optional[Any]) -> Optional[list["MCPTool"]]:
        """Return the MCP tools to expose (default: the auto-harvested typed POST routes).

        Override to exclude (filter harvested), add catch-all-backed tools (harvested + [catchall.tool(...)]),
        or disable (return None). 'catchall' is None unless the server has one parameterized catch-all route.
        """
        return harvested

    def mcp_allowed_tools_for_session(self, seed_body: dict[str, Any]) -> Optional[list[str]]:
        """Per-session tool restriction: return the tool names allowed for this rollout's MCP token,
        or ``None`` (the default) for unrestricted. ``seed_body`` is the JSON body POSTed to
        ``/seed_session``, or its ``task_data`` when an Environment Server seeds the session.
        """
        return None

    async def seed_session(
        self,
        body: ResourcesSeedSessionRequest | BaseSeedSessionRequest,
        *,
        request: Request | None = None,
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Seed state through the hooks, or acknowledge a permissive stateless request.

        Stateful calls require ``request`` to bind the cookie after successful initialization. A typed retry
        reuses the same episode/task; another owner, a completed verify or a close conflicts with 409. Invalid
        task data returns 422. Construction or cleanup failures remain retryable server errors.

        A served override declares ``request: Request`` before ``body`` and delegates to this method with
        ``super().seed_session(body, request=request)``. To accept a cookie seed that is more permissive than
        ``seed_request_model``, it declares ``body: ResourcesSeedSessionRequest | Annotated[CookieModel,
        BeforeValidator(reject_partial_typed_seed)]``; without the validator a malformed typed seed is accepted as
        a cookie seed. The conformance checker exercises close ownership to catch an override that bypasses the
        binding. Stateless ``seed_session(body)`` remains supported.

        A server that overrides this method instead of implementing the hooks refuses a seed that no retry can make
        succeed with ``ResourcesSeedRejectedError`` (422, input that fails validation) or
        ``ResourcesSessionConflictError`` (409, an ID bound to another episode or task, or a closed session), never a
        plain exception (HTTP 500), which the Environment Server retries by dispatching the whole episode again. A
        failure a retry may fix, such as an unavailable sandbox provider, stays a 5xx.
        """
        if not self._manages_session_state():
            if isinstance(body, ResourcesSeedSessionRequest):
                return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)
            return BaseSeedSessionResponse()
        if request is None:
            raise TypeError("Stateful seed_session requires request= to bind the session cookie")
        if not isinstance(body, ResourcesSeedSessionRequest):
            session_id = request.session[SESSION_ID_KEY]
            async with self.session_lock(session_id):
                self._get_lifecycle_fences().raise_if_fenced(session_id)
                if self.session_identity(session_id) is not None:
                    raise ResourcesSessionConflictError("A cookie seed cannot replace a typed session")
                if not self._cookie_seed_warning_emitted:
                    logger.warning(
                        "Cookie-based /seed_session is deprecated. Migrate to an Environment Server that sends "
                        "ResourcesSeedSessionRequest with resources_session_id, episode_id, task_id, and task_data."
                    )
                    self._cookie_seed_warning_emitted = True
                # A release failure must prevent replacement and leave the previous run reachable.
                if (
                    self.has_session_state(session_id)
                    or session_id in self._cookie_sessions
                    or session_id in self._pending_verifications
                ):
                    await self.release_session_state(session_id)
                    # Nothing is left to release, so the finished run's receipt expires with its window
                    # instead of pinning ownership as pending cleanup if the replacement fails to initialize.
                    if (pending := self._pending_verifications.pop(session_id, None)) is not None:
                        self._verification_receipts.setdefault(
                            session_id, (monotonic() + self.config.session_close_retry_window_seconds, pending)
                        )
                self._cookie_sessions.add(session_id)
                self._initializing_sessions[session_id] = None
                await self._initialize_session(session_id, body)
                self._initializing_sessions.pop(session_id, None)
                # Clear only after success: failed initialization preserves the preceding finished run's receipt.
                self._verification_receipts.pop(session_id, None)
                self._pending_verifications.pop(session_id, None)
            return BaseSeedSessionResponse()

        session_id = body.resources_session_id
        identity = (body.episode_id, body.task_id)
        async with self.session_lock(session_id):
            self._get_lifecycle_fences().raise_if_fenced(session_id)
            if self.session_is_verified(session_id):
                raise ResourcesSessionConflictError("resources session has already been verified")
            previous = self.session_identity(session_id)
            if previous is not None and previous != identity:
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
            if session_id in self._cookie_sessions or (previous is None and self.has_session_state(session_id)):
                raise ResourcesSessionConflictError("A typed seed cannot replace a cookie-seeded session")
            if session_id not in self._session_episodes:
                try:
                    task = self.seed_request_model.model_validate(body.task_data)
                except ValidationError as error:
                    raise ResourcesSeedRejectedError(
                        f"task_data is not a valid {self.seed_request_model.__name__}: {error}"
                    ) from error
                if session_id in self._initializing_sessions:
                    # The preceding initialization failed and its cleanup also failed. Retry cleanup before
                    # allocating again, while keeping the original episode's ownership if cleanup still fails.
                    await self.release_session_state(session_id)
                self._initializing_sessions[session_id] = identity
                await self._initialize_session(session_id, task)
                self._initializing_sessions.pop(session_id, None)
                self._session_episodes[session_id] = identity
            request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Release by ID, then fence the session for the configured close retry window.

        Unknown IDs and an owning retried close succeed. Another episode cannot close active, partially
        initialized, verified or already closed state. Release failure retains ownership and resource references
        so the owner can retry. A close also drops verification payloads; only its episode fence remains.
        Stateless servers acknowledge the request without retaining state.

        A server that overrides this method instead of implementing the hooks must succeed when the session is
        already closed or was never seeded; keep a later seed from creating it again for
        ``session_close_retry_window_seconds`` (``ResourcesSessionFences``); release by ``resources_session_id`` even
        when the request carries no session cookie; and raise when releasing fails, so the caller can retry.
        ``check_resources_session_contract`` checks these rules from a server's tests.
        """
        if not self._manages_session_state():
            return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)
        session_id = body.resources_session_id
        async with self.session_lock(session_id):
            closed = self._get_lifecycle_fences().get(session_id)
            if closed is not None and closed != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            identity = self.session_identity(session_id)
            if identity is not None and identity[0] != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            await self.release_session_state(session_id)
            self._forget_session(session_id)
            self._get_lifecycle_fences().fence(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Score the episode.

        A server whose score depends on per-rollout state answers a verify for a session it no longer holds
        (never seeded, closed, or lost) with a masked sample built by ``lost_session_verification``, never a
        plain reward of 0 and never an exception that dispatches the episode again. A tool call for such a
        session raises ``ResourcesSessionLostError`` (404).
        """

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Compute aggregate metrics from verify responses.

        RewardProfiler provides baseline stats. Override compute_metrics(),
        compute_repeat_metrics(), and/or get_key_metrics() for benchmark-specific customization.
        """
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
            compute_repeat_metrics_fn=self.compute_repeat_metrics,
        )

    async def get_reverify_mode(self) -> ReverifyMode:
        return self.config.REVERIFY_MODE
