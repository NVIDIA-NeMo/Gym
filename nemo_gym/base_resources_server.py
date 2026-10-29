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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Optional, TypedDict, TypeVar, get_args

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, TypeAdapter, field_validator
from starlette.middleware import Middleware


if TYPE_CHECKING:
    from itsdangerous import URLSafeSerializer

    # Type-only: importing MCPTool at runtime would be circular (mcp_auto_exposure imports this
    # module) and would pull the mcp SDK into agent/model processes that never need it.
    from nemo_gym.mcp_auto_exposure import MCPTool

from nemo_gym._checkpoint.control import CheckpointMode
from nemo_gym._checkpoint.participant_workers import install_server_participant
from nemo_gym._checkpoint.resources import (
    MCPTokenClaims,
    ResourcesCheckpointMiddleware,
    ResourcesParticipant,
)
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym._checkpoint.steps import CHECKPOINT_CLOSE_FOLLOWS_HEADER, StepType
from nemo_gym.client.server_client import SESSION_ID_KEY
from nemo_gym.config.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core.episode import EpisodeId, TaskId
from nemo_gym.core.failure_kinds import SESSION_LOST, validate_failure_kind
from nemo_gym.core.responses_api_types import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.judge import judge_failsafe
from nemo_gym.observability.verifiers import observe_verifier
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from nemo_gym.rollout_observability import VerifierInvocation
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer
from nemo_gym.session_routing import SessionKind
from nemo_gym.telemetry.endpoints import traced_verify_endpoint


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
    session_close_retry_window_seconds: float = Field(
        default=300,
        gt=0,
        allow_inf_nan=False,
        description=(
            "With several workers, how long a session stays with the worker that created it "
            "after its last request, once it is closed, verified or failed to seed; "
            "cover the caller's retry horizon and the server's own close fences. "
            "Servers that remember closed sessions with ClosedResourcesSessionRecords keep them this long, "
            "with any number of workers."
        ),
    )
    # The mode of reverification (for gym eval reverify) of this server.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNKNOWN


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


class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float
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
    for session_close_retry_window_seconds (see ClosedResourcesSessionRecords and ResourcesSessionFence);
    callers must use unique IDs and finish seed retries within that window.
    External resources should use provider TTLs when available. Current process-local
    session records may remain until the server shuts down.
    """

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str = Field(min_length=1)
    episode_id: EpisodeId
    task_id: TaskId
    task_data: dict[str, JsonValue]


class ResourcesSeedSessionResponse(BaseModel):
    """Confirm resources state and return optional agent access."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    resources_tools: MCPServerMetadata | None = None
    sandbox_access: SandboxAccess | None = None


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


# A resources session is the cookie session: a typed seed or close names it by resources_session_id,
# which the cookie then carries as its SESSION_ID_KEY (see nemo_gym.session_routing).
RESOURCES_SESSIONS = SessionKind(
    cookie_key=SESSION_ID_KEY,
    id_field="resources_session_id",
    seed_path="/seed_session",
    close_path="/close_session",
    cookie_session=True,
)


ClosedRecordT = TypeVar("ClosedRecordT")


class ClosedResourcesSessionRecords(Generic[ClosedRecordT]):
    """The resources sessions this worker closed, each kept for a fixed window after its first close.

    A server uses it to refuse a late seed of a closed ``resources_session_id``, or to check a retried close.
    Construct it with the server's ``session_close_retry_window_seconds``.
    With several workers, the routing claim on the ID ends that long after the last request for the session,
    and the close is recorded before its reply is sent.
    So the claim never ends before the record (see ``nemo_gym.session_routing.ResourcesSessionLifetimes``).
    While the claim lasts, a late seed reaches the worker that remembers the close and is refused.
    After it ends, every worker treats the ID as new, so the answer does not depend on which worker gets it.
    """

    def __init__(self, window_seconds: float) -> None:
        if not window_seconds > 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds!r}")
        self._window_seconds = window_seconds
        # Earliest expiry first: every record lasts the same window, and a record is never moved.
        self._records: OrderedDict[str, tuple[float, ClosedRecordT]] = OrderedDict()

    def remember(self, resources_session_id: str, record: ClosedRecordT) -> None:
        """Record a close; a retried close keeps the first record and its expiry."""
        self._expire()
        if resources_session_id not in self._records:
            self._records[resources_session_id] = (monotonic() + self._window_seconds, record)

    def get(self, resources_session_id: str) -> Optional[ClosedRecordT]:
        """Return the record of the session's close, or None if it is not closed or its window has passed."""
        self._expire()
        entry = self._records.get(resources_session_id)
        return None if entry is None else entry[1]

    def __contains__(self, resources_session_id: object) -> bool:
        self._expire()
        return resources_session_id in self._records

    def __len__(self) -> int:
        self._expire()
        return len(self._records)

    def _expire(self) -> None:
        now = monotonic()
        while self._records and next(iter(self._records.values()))[0] <= now:
            self._records.popitem(last=False)


class ResourcesSessionFence:
    """Serialize seed and close per resources session, and refuse a seed for a session closed within the window.

    A seed that arrives after its session closed (a retry, or a reply that raced the close) must not create the
    session again, so the fence remembers each closed ``resources_session_id`` and the episode that first closed it,
    in ``ClosedResourcesSessionRecords``. Construct it with the server's ``session_close_retry_window_seconds``, so a
    record lasts exactly as long as the session's routing claim with several workers. A per-session lock exists only
    while a request holds or waits for it.

    TODO: also expire sessions that are never closed, as when the Environment Server dies before its close. Planned
    rule: a fixed lifetime from the seed (the sandbox ``ttl_s`` when set, else 24 hours), checked on the next request
    and at shutdown; an expired session is released, fenced, and verifies as ``SESSION_EXPIRED``.
    """

    def __init__(self, window_seconds: float) -> None:
        self._window_seconds = window_seconds
        self._locks: dict[str, _FenceLock] = {}
        self._closed: ClosedResourcesSessionRecords[EpisodeId] = ClosedResourcesSessionRecords(window_seconds)

    @asynccontextmanager
    async def locked(self, resources_session_id: str) -> AsyncIterator[None]:
        """Hold the lock for ``resources_session_id`` so its seed and close never interleave."""
        entry = self._locks.setdefault(resources_session_id, _FenceLock())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._locks[resources_session_id]

    def closed_episode(self, resources_session_id: str) -> EpisodeId | None:
        """Return the episode that first closed ``resources_session_id`` within the window, else ``None``."""
        return self._closed.get(resources_session_id)

    def raise_if_closed(self, resources_session_id: str) -> None:
        """Refuse a seed for a closed session with 409, which the Environment Server does not retry."""
        if self.closed_episode(resources_session_id) is not None:
            raise ResourcesSessionConflictError(f"Resources session is already closed: {resources_session_id}")

    def mark_closed(self, resources_session_id: str, episode_id: EpisodeId) -> None:
        """Record ``resources_session_id`` as closed; call it only after its state was released.

        A retried close keeps the first close's episode and expiry.
        """
        self._closed.remember(resources_session_id, episode_id)

    def clear(self) -> None:
        """Forget every record, for a server that drops all of its sessions at shutdown."""
        self._closed = ClosedResourcesSessionRecords(self._window_seconds)


@dataclass
class _FenceLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    config: BaseResourcesServerConfig
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
    _checkpoint: Optional[ResourcesParticipant] = PrivateAttr(default=None)
    _mcp_token_serializer: Any = PrivateAttr(default=None)
    routes_sessions_to_owner = True

    def claimed_session_routing(self) -> dict[str, Any]:
        return {
            "kind": RESOURCES_SESSIONS,
            "close_window_seconds": self.config.session_close_retry_window_seconds,
            "session_starts": tuple(self.checkpoint_session_starts),
            "session_ends": ("/verify",),
            "close_follows_header": CHECKPOINT_CLOSE_FOLLOWS_HEADER,
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
        self.setup_resources_checkpoint(app)

        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.get("/reverify_mode")(self.get_reverify_mode)
        return app

    def register_protocol_routes(self, app: FastAPI) -> None:
        """Register the episode routes ``/seed_session``, ``/close_session`` and ``/verify``."""
        app.post("/seed_session")(self.seed_session)
        app.post("/close_session")(self.close_resources_session)
        config = getattr(self.server_client, "global_config_dict", {})
        # Wrapped outside judge_failsafe so the span covers the failsafe's own handling too.
        app.post("/verify")(
            traced_verify_endpoint(
                observe_verifier(
                    judge_failsafe(self.verify),
                    enabled=TypeAdapter(bool).validate_python(config.get("observability_enabled", False)),
                ),
                static_attributes={"nemo.gym.server.name": self.config.name},
            )
        )

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
        if self.checkpoint_mode == "exported":
            # Hooks from an intermediate base class or a mixin count; only the stubs here do not.
            missing = [
                name for name in _SESSION_HOOKS if getattr(type(self), name) is getattr(SimpleResourcesServer, name)
            ]
            if missing:
                raise ValueError(
                    f"{type(self).__name__} declares checkpoint_mode='exported' but does not implement {missing}"
                )
        participant = ResourcesParticipant(self, self.checkpoint_mode, verify_step_type=self.checkpoint_verify)
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
        header = NEMO_GYM_MCP_SESSION_TOKEN_HEADER.lower().encode()
        token = next((value for key, value in scope.get("headers", ()) if key.lower() == header), None)
        if token is None:
            return None
        try:
            text = token.decode()
        except UnicodeDecodeError:
            return None
        if self._mcp_token_serializer is None:
            self._mcp_token_serializer = mcp_session_token_serializer(self.get_session_middleware_key())
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
        # With several workers, the session's routing claim ends a window after its last request from now on.
        # A request without an app, such as a test's stand-in, has no routing and so no claim to end.
        app = getattr(request, "app", None)
        lifetimes = getattr(getattr(app, "state", None), "nemo_gym_resources_session_lifetimes", None)
        if lifetimes is not None and session_id is not None:
            lifetimes.ended(session_id)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        """Return each session's state for a checkpoint, leaving out sessions this server no longer holds."""
        raise NotImplementedError

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        """Validate every state, then install all of them; never install a partial set."""
        raise NotImplementedError

    async def retire_session_state(self, session_id: str) -> None:
        """Discard a session whose attempt was retired; called in every checkpoint mode.

        The default does nothing.
        A server that keeps per-session state frees it here, since a retired session's late requests,
        a terminal step among them, are refused.
        """

    def normalize_tool_name(self, name: str) -> str:
        """Strip this server's MCP namespace from a trajectory tool-call name (see module function)."""
        return normalize_tool_name(name, self.config.name or self.__class__.__name__)

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
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Seed per-rollout state; the default keeps none.

        An Environment Server seeds with a ``ResourcesSeedSessionRequest`` and gets its caller-assigned
        ``resources_session_id`` back. An Agent's ``/run`` seeds with its legacy body and gets an empty
        response. Servers that keep per-rollout state override this method.

        An override refuses a seed that no retry can make succeed with a final 4xx, never a plain exception
        (HTTP 500), which the Environment Server retries by dispatching the whole episode again:

        - ``ResourcesSeedRejectedError`` (422) for input that fails validation, such as ``task_data`` or a
          ``TaskId`` that does not match the task row;
        - ``ResourcesSessionConflictError`` (409) for an ID bound to another episode or task, or a closed session.

        A failure that a retry may fix, such as an unavailable sandbox provider, stays a 5xx.
        """
        if isinstance(body, ResourcesSeedSessionRequest):
            return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)
        return BaseSeedSessionResponse()

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Close state an Environment Server seeded; the default keeps none.

        Served at ``/close_session``. Servers that keep per-rollout state override this method, and may accept
        other close bodies by overriding it with a different signature. The name differs from the
        ``close_session`` helpers some servers already define for their own state.

        An override must:

        - succeed when the session is already closed or was never seeded;
        - keep a seed that arrives after the close from creating the session again, for a bounded time
          (``ResourcesSessionFence`` does this for ``session_close_retry_window_seconds``);
        - release the session's state by ``resources_session_id`` even when the request carries no session cookie;
        - raise when releasing fails rather than recording the session as closed, so the caller can retry.

        A served resources server routes sessions (see ``nemo_gym.session_routing``): a typed seed's session cookie
        carries ``resources_session_id`` as its ``SESSION_ID_KEY``, and a typed close that carries no cookie, as when
        the seed's reply was lost, is given that ID too, so a server may key state by either; they name the same
        session. An app built without session routing, such as one under ``TestClient``, gives a new session a random
        ID instead, so release by ``body.resources_session_id`` to be correct either way.

        ``nemo_gym.testing.session_conformance.check_resources_session_contract`` checks these rules from a server's tests.
        """
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

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
