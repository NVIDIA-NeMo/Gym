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
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


if TYPE_CHECKING:
    # Type-only: importing MCPTool at runtime would be circular (mcp_auto_exposure imports this
    # module) and would pull the mcp SDK into agent/model processes that never need it.
    from nemo_gym.mcp_auto_exposure import MCPTool

from nemo_gym._checkpoint.control import ControlCapabilities, checkpoint_control_auth_token
from nemo_gym._checkpoint.resources import (
    ResourcesCheckpointParticipant,
    ResourceSnapshot,
    ResourcesRouteKind,
    install_resources_checkpoint,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.judge import judge_failsafe
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer
from nemo_gym.telemetry.endpoints import traced_verify_endpoint


NEMO_GYM_MCP_SESSION_TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
NEMO_GYM_MCP_METADATA_KEY = "mcp"
# Salt namespacing the signed MCP session token, so it can't be confused with another signer
# that happens to share the same session-middleware secret.
_MCP_TOKEN_SALT = "nemo-gym-mcp-session-token"


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
RESERVED_MCP_TOOL_NAMES = frozenset({"verify", "seed_session", "aggregate_metrics", "mcp"})


class ReverifyMode(str, Enum):
    STATELESS = "stateless"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class BaseResourcesServerConfig(BaseRunServerInstanceConfig):
    # Opt in to serve this server's tool routes over MCP; default off.
    expose_tools_over_mcp: bool = False
    # A replacement process must remain paused until checkpoint restore completes.
    checkpoint_restore_expected: bool = False
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

    # Human-readable diagnosis of why `reward` may not reflect policy quality.
    # Machine-readable handling belongs to `mask_sample`/`failure_kind`.
    failure_reason: Optional[str] = None


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


class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    config: BaseResourcesServerConfig

    _CONTROL_COMPONENT = "resources_servers"
    _checkpoint_participant: Optional[ResourcesCheckpointParticipant] = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)
        # Starlette wraps middleware in reverse registration order.
        # Register this first so RolloutContext strips /ng-rollout/<id> before route admission.
        self.setup_resources_checkpoint(app)
        app.add_middleware(RolloutContextMiddleware)

        app.post("/seed_session")(self.seed_session)
        # Wrapped outside judge_failsafe so the span covers the failsafe's own handling too.
        app.post("/verify")(
            traced_verify_endpoint(
                judge_failsafe(self.verify),
                static_attributes={"nemo.gym.server.name": self.config.name},
            )
        )
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.get("/reverify_mode")(self.get_reverify_mode)
        self.setup_control_plane(app)

        return app

    def checkpoint_state_enabled(self) -> bool:
        """Whether this server implements logical session export and restore."""
        return False

    async def export_checkpoint_state(self, rollout_id: str, attempt_index: int) -> dict[str, Any]:
        raise NotImplementedError

    async def restore_checkpoint_states(self, snapshots: list[ResourceSnapshot]) -> None:
        """Validate and atomically activate all restored sessions."""
        raise NotImplementedError

    async def retire_checkpoint_state(self, rollout_id: str, attempt_index: int) -> None:
        """Remove one execution's adapter-owned session state."""
        raise NotImplementedError

    def checkpoint_participant(self) -> ResourcesCheckpointParticipant:
        if self._checkpoint_participant is None:
            self._checkpoint_participant = ResourcesCheckpointParticipant(
                export_state=self.export_checkpoint_state,
                restore_states=self.restore_checkpoint_states,
                retire_state=self.retire_checkpoint_state,
                restore_expected=self.config.checkpoint_restore_expected,
            )
        return self._checkpoint_participant

    def checkpoint_route_kind(self, path: str, method: str) -> Optional[ResourcesRouteKind]:
        """Classify checkpointed routes, defaulting unknown POST data routes to mutation."""
        if method != "POST":
            return None
        if path.startswith("/ng-control/") or path in {"/aggregate_metrics", "/mcp"} or path.startswith("/mcp/"):
            return None
        if path == "/seed_session":
            return "start"
        if path == "/verify":
            return "terminal"
        return "mutation"

    def checkpoint_control_auth_token(self) -> Optional[str]:
        global_config = getattr(self.server_client, "global_config_dict", None)
        return checkpoint_control_auth_token(global_config)

    def setup_resources_checkpoint(self, app: FastAPI) -> None:
        auth_token = self.checkpoint_control_auth_token()
        if not self.checkpoint_state_enabled():
            return
        if auth_token is None:
            if self.config.checkpoint_restore_expected:
                raise ValueError("checkpoint_restore_expected requires checkpoint control authentication")
            return
        install_resources_checkpoint(
            app,
            participant=self.checkpoint_participant(),
            fence=self.checkpoint_fence(),
            auth_token=auth_token,
            server_name=self.config.name,
            route_kind=self.checkpoint_route_kind,
        )

    def control_capabilities(self) -> ControlCapabilities:
        capabilities = super().control_capabilities()
        if self.checkpoint_state_enabled() and self.checkpoint_control_auth_token() is not None:
            capabilities.checkpoint_mode = "export_restore"
            capabilities.concurrency_contract = "serialized_per_session"
        return capabilities

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
        ``/seed_session``.
        """
        return None

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        return BaseSeedSessionResponse()

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Compute aggregate metrics from verify responses.

        RewardProfiler provides baseline stats. Override compute_metrics() and/or
        get_key_metrics() for benchmark-specific customization.
        """
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )

    async def get_reverify_mode(self) -> ReverifyMode:
        return self.config.REVERIFY_MODE
