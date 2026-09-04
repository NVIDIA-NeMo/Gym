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
import time
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel


if TYPE_CHECKING:
    # Type-only: importing MCPTool at runtime would be circular (mcp_auto_exposure imports this
    # module) and would pull the mcp SDK into agent/model processes that never need it.
    from nemo_gym.mcp_auto_exposure import MCPTool

from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.judge import judge_failsafe
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from nemo_gym.server_utils import SESSION_ID_KEY, BaseRunServerInstanceConfig, BaseServer, SimpleServer
from nemo_gym.telemetry.endpoints import traced_verify_endpoint


logger = logging.getLogger(__name__)


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
# Lifecycle endpoints, never model-callable: a policy that could call `close_session`
# could end its own episode's resources mid-rollout.
RESERVED_MCP_TOOL_NAMES = frozenset({"verify", "seed_session", "close_session", "aggregate_metrics", "mcp"})


class ReverifyMode(str, Enum):
    STATELESS = "stateless"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class BaseResourcesServerConfig(BaseRunServerInstanceConfig):
    # Opt in to serve this server's tool routes over MCP; default off.
    expose_tools_over_mcp: bool = False
    # Reclaim a session that has been idle this long, by calling close_session for it.
    # None (the default) keeps today's behavior: no sweeper, no background task. Set it
    # when the environment holds an external resource that a crashed or cancelled trainer
    # would otherwise leave behind.
    session_ttl_s: Optional[float] = None
    # How often the sweeper looks; only meaningful when session_ttl_s is set.
    session_sweep_interval_s: float = 60.0
    # The mode of reverification (for gym eval reverify) of this server.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNKNOWN


class BaseResourcesServer(BaseServer):
    config: BaseResourcesServerConfig


class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


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


class BaseCloseSessionRequest(BaseModel):
    # Which session to release. Served over HTTP the caller identifies itself by cookie and
    # the route fills this in; the sweeper has no request to read a cookie from and sets it
    # directly. Either way an override receives it, so both paths reach the same code.
    session_id: Optional[str] = None


class BaseCloseSessionResponse(BaseModel):
    pass


class MCPServerMetadata(BaseModel):
    """Metadata returned from /seed_session for per-rollout Gym MCP access."""

    server_name: str
    url_path: str = "/mcp"
    transport: str = "http"
    headers: dict[str, str]


class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    config: BaseResourcesServerConfig

    # Last activity per session, for the optional idle sweeper. Private (leading underscore)
    # so pydantic does not try to build a schema for it.
    _session_last_seen: dict[str, float] = {}

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._session_last_seen = {}

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)
        app.add_middleware(RolloutContextMiddleware)

        if self.config.session_ttl_s is not None:

            @app.on_event("startup")
            async def _start_session_sweeper() -> None:  # pragma: no cover - exercised via the task
                asyncio.create_task(self._sweep_idle_sessions())

        app.post("/seed_session")(self.seed_session)
        app.post("/close_session")(self._close_session_endpoint)
        # Wrapped outside judge_failsafe so the span covers the failsafe's own handling too.
        app.post("/verify")(
            traced_verify_endpoint(
                judge_failsafe(self.verify),
                static_attributes={"nemo.gym.server.name": self.config.name},
            )
        )
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.get("/reverify_mode")(self.get_reverify_mode)

        return app

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

    def touch_session(self, session_id: str) -> None:
        """Record activity for a session so the sweeper does not reclaim it.

        Environments that opt into ``session_ttl_s`` call this whenever a session does
        something; the sweeper reclaims the ones that stop.
        """
        self._session_last_seen[session_id] = time.monotonic()

    def forget_session(self, session_id: str) -> None:
        """Stop tracking a session that has already been released."""
        self._session_last_seen.pop(session_id, None)

    async def _sweep_idle_sessions(self) -> None:
        """Call ``close_session`` for sessions idle beyond ``session_ttl_s``.

        A backstop, not the primary path: the agent layer closes sessions in a finally.
        This covers what no call can reach - a killed trainer, a dropped connection.
        """
        ttl = self.config.session_ttl_s
        assert ttl is not None
        while True:
            await asyncio.sleep(self.config.session_sweep_interval_s)
            try:
                now = time.monotonic()
                expired = [sid for sid, seen in self._session_last_seen.items() if now - seen > ttl]
                for session_id in expired:
                    self.forget_session(session_id)
                    logger.warning("reclaiming environment session %s after %.0fs idle", session_id, ttl)
                    await self.close_session(BaseCloseSessionRequest(session_id=session_id))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("environment session sweep failed")

    async def _close_session_endpoint(
        self, request: Request, body: BaseCloseSessionRequest
    ) -> BaseCloseSessionResponse:
        """Resolve the cookie session before delegating, so ``close_session`` has one signature.

        Without this the sweeper, which holds a session id but no request, could not tell an
        override which session to release.
        """
        if body.session_id is None:
            body.session_id = request.session.get(SESSION_ID_KEY)
        return await self.close_session(body)

    async def close_session(self, body: BaseCloseSessionRequest) -> BaseCloseSessionResponse:
        """Release whatever this session allocated. Default no-op.

        Environments that hold an external resource - a container, a browser, a provider
        session - should override this and make it **idempotent**: the caller may time out
        and retry, and a second call must not produce a different outcome. This is the
        same contract as ``SandboxProvider.close()``.

        The agent layer calls it in a ``finally`` around the episode, so it also runs when
        the rollout failed, was cancelled, or timed out - the cases where ``verify()``, the
        usual place environments release things, never happens.
        """
        return BaseCloseSessionResponse()

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
