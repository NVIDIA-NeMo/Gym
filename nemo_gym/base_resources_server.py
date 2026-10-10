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
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Optional, TypeVar

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, field_validator
from starlette.middleware import Middleware


if TYPE_CHECKING:
    from itsdangerous import URLSafeSerializer

    # Type-only: importing MCPTool at runtime would be circular (mcp_auto_exposure imports this
    # module) and would pull the mcp SDK into agent/model processes that never need it.
    from nemo_gym.mcp_auto_exposure import MCPTool

from nemo_gym._checkpoint.control import install_participant
from nemo_gym._checkpoint.resources import (
    ResourcesCheckpointMiddleware,
    ResourcesCheckpointMode,
    ResourcesParticipant,
)
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym._checkpoint.steps import StepMode
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.failure_kinds import validate_failure_kind
from nemo_gym.judge import judge_failsafe
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import RolloutContextMiddleware, current_episode_id
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_utils import SESSION_ID_KEY, BaseRunServerInstanceConfig, BaseServer, SimpleServer
from nemo_gym.telemetry.endpoints import traced_verify_endpoint


NEMO_GYM_MCP_SESSION_TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
NEMO_GYM_MCP_METADATA_KEY = "mcp"
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
            "nemo_gym.failure_kinds. Stable and low cardinality, so it is safe to group by "
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
    Closing an unknown identifier must prevent a racing seed from creating it later.
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


class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    config: BaseResourcesServerConfig
    # How this server's sessions take part in partial-rollout checkpoints. "exported" servers implement the
    # ResourcesSessionHooks methods.
    # The default fails closed: its sessions' rollouts are restarts, which never hold up a checkpoint,
    # are left out of it, and start over after a crash.
    checkpoint_mode: ClassVar[ResourcesCheckpointMode] = "restart_only"
    # "replay" declares that /verify is safe to run again after a crash
    # and does not change checkpointed session state, so a checkpoint need not wait for it.
    # Callers learn it from the /seed_session reply.
    checkpoint_verify: ClassVar[StepMode] = "wait"
    _checkpoint: Optional[ResourcesParticipant] = PrivateAttr(default=None)
    _mcp_token_serializer: Any = PrivateAttr(default=None)

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
        # Wrapped outside judge_failsafe so the span covers the failsafe's own handling too.
        app.post("/verify")(
            traced_verify_endpoint(
                judge_failsafe(self.verify),
                static_attributes={"nemo.gym.server.name": self.config.name},
            )
        )

    def setup_resources_checkpoint(self, app: FastAPI) -> None:
        settings = checkpoint_settings(getattr(self.server_client, "global_config_dict", None))
        if settings is None:
            return
        if (self.config.num_workers or 1) != 1:
            raise ValueError("resources checkpointing requires num_workers=1: sessions live in one process")
        if self.checkpoint_mode == "exported":
            # Hooks from an intermediate base class or a mixin count; only the stubs here do not.
            missing = [
                name for name in _SESSION_HOOKS if getattr(type(self), name) is getattr(SimpleResourcesServer, name)
            ]
            if missing:
                raise ValueError(
                    f"{type(self).__name__} declares checkpoint_mode='exported' but does not implement {missing}"
                )
        participant = ResourcesParticipant(self, self.checkpoint_mode, verify_mode=self.checkpoint_verify)
        self._checkpoint = participant
        install_participant(
            app,
            participant,
            auth_token=settings.control_auth_token,
            lease_grace_seconds=settings.lease_grace_seconds,
            instance_name=self.config.name,
        )
        # Innermost, so the session middleware has already resolved the cookie session ID.
        mcp_session_id = self._mcp_session_id if self.config.expose_tools_over_mcp else None
        app.user_middleware.append(
            Middleware(ResourcesCheckpointMiddleware, participant=participant, mcp_session_id=mcp_session_id)
        )

    def _mcp_session_id(self, scope: dict[str, Any]) -> Optional[str]:
        """The session an MCP request's signed token names, or ``None``; the MCP endpoint rejects a bad token."""
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
        return payload.get("sid") if isinstance(payload, dict) else None

    def checkpoint_session_started(self, request: Request) -> None:
        """Report that the request's session began an episode; standard ``/seed_session`` does this itself.

        Call it from a protocol route that starts a session under another name, such as ``/reset``.
        """
        if self._checkpoint is None:
            return
        episode_id = current_episode_id()
        session_id = request.session.get(SESSION_ID_KEY)
        if episode_id is not None and session_id is not None:
            self._checkpoint.seeded(session_id, episode_id)

    def checkpoint_session_ended(self, request: Request) -> None:
        """Report that the request's session ended; standard ``/close_session`` and ``/verify`` do this.

        Call it from a protocol route that ends a session under another name, such as a terminal step.
        """
        session_id = request.session.get(SESSION_ID_KEY)
        if self._checkpoint is not None and session_id is not None:
            self._checkpoint.ended(session_id)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        """Return each session's state for a checkpoint, leaving out sessions this server no longer holds."""
        raise NotImplementedError

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        """Validate every state, then install all of them; never install a partial set."""
        raise NotImplementedError

    async def retire_session_state(self, session_id: str) -> None:
        """Discard a session whose attempt was retired."""
        raise NotImplementedError

    def setup_resources_checkpoint(self, app: FastAPI) -> None:
        settings = checkpoint_settings(getattr(self.server_client, "global_config_dict", None))
        if settings is None:
            return
        if (self.config.num_workers or 1) != 1:
            raise ValueError("resources checkpointing requires num_workers=1: sessions live in one process")
        if self.checkpoint_mode == "exported":
            # Hooks from an intermediate base class or a mixin count; only the stubs here do not.
            missing = [
                name for name in _SESSION_HOOKS if getattr(type(self), name) is getattr(SimpleResourcesServer, name)
            ]
            if missing:
                raise ValueError(
                    f"{type(self).__name__} declares checkpoint_mode='exported' but does not implement {missing}"
                )
        participant = ResourcesParticipant(self, self.checkpoint_mode, verify_mode=self.checkpoint_verify)
        self._checkpoint = participant
        install_participant(
            app,
            participant,
            auth_token=settings.control_auth_token,
            lease_grace_seconds=settings.lease_grace_seconds,
            instance_name=self.config.name,
        )
        # Innermost, so the session middleware has already resolved the cookie session ID.
        mcp_session_id = self._mcp_session_id if self.config.expose_tools_over_mcp else None
        app.user_middleware.append(
            Middleware(ResourcesCheckpointMiddleware, participant=participant, mcp_session_id=mcp_session_id)
        )

    def _mcp_session_id(self, scope: dict[str, Any]) -> Optional[str]:
        """The session an MCP request's signed token names, or ``None``; the MCP endpoint rejects a bad token."""
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
        return payload.get("sid") if isinstance(payload, dict) else None

    def checkpoint_session_started(self, request: Request) -> None:
        """Report that the request's session began an episode; standard ``/seed_session`` does this itself.

        Call it from a protocol route that starts a session under another name, such as ``/reset``.
        """
        if self._checkpoint is None:
            return
        episode_id = current_episode_id()
        session_id = request.session.get(SESSION_ID_KEY)
        if episode_id is not None and session_id is not None:
            self._checkpoint.seeded(session_id, episode_id)

    def checkpoint_session_ended(self, request: Request) -> None:
        """Report that the request's session ended; standard ``/close_session`` and ``/verify`` do this.

        Call it from a protocol route that ends a session under another name, such as a terminal step.
        """
        session_id = request.session.get(SESSION_ID_KEY)
        if self._checkpoint is not None and session_id is not None:
            self._checkpoint.ended(session_id)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        """Return each session's state for a checkpoint, leaving out sessions this server no longer holds."""
        raise NotImplementedError

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        """Validate every state, then install all of them; never install a partial set."""
        raise NotImplementedError

    async def retire_session_state(self, session_id: str) -> None:
        """Discard a session whose attempt was retired."""
        raise NotImplementedError

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
        - keep a seed that arrives after the close from creating the session again;
        - raise when releasing fails rather than recording the session as closed, so the caller can retry.

        The default seed does not tie the session cookie to ``resources_session_id``. A server that keys state
        by the cookie's ``SESSION_ID_KEY`` releases it by that key, from ``request.session``, or sets
        ``request.session[SESSION_ID_KEY] = body.resources_session_id`` in its own seed.

        ``nemo_gym.testing.session_conformance.check_resources_session_contract`` checks these rules from a server's tests.
        """
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass

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
