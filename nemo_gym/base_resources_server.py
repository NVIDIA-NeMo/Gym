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
import hashlib
import hmac
import json
import logging
import time
from abc import abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field


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
    # Reclaim a session this long after it was seeded regardless of activity. Idle time is
    # not a liveness signal on its own, so this bounds a session whose handler has hung.
    session_max_lifetime_s: Optional[float] = None
    # How long a released session id is remembered, so a seed delayed in the network cannot
    # recreate what was already torn down.
    session_tombstone_s: float = 900.0
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


class SessionCloseReason(str, Enum):
    """Why the caller ended the session. Diagnostic only — it never decides whether the
    resource is released."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ABANDONED = "abandoned"
    EXPIRED = "expired"
    SHUTDOWN = "shutdown"


class SessionCloseStatus(str, Enum):
    """Terminal states of a close operation.

    There is deliberately no ``not_found``: an unknown id and an already-released one
    answer identically, so the endpoint cannot be used to probe whether another rollout's
    session exists.
    """

    CLOSED = "closed"
    ALREADY_CLOSED = "already_closed"
    RELEASE_PENDING = "release_pending"
    RELEASE_FAILED = "release_failed"


class BaseSeedSessionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Created by the caller before the first seed transmission, so it survives a lost seed
    # response. The provider's own handle cannot serve here: it is learned *from* that
    # response, which is exactly what goes missing.
    ng_session_id: Optional[str] = Field(default=None, alias="_ng_session_id")
    # Authorizes releasing this one session without the signed cookie. Only its digest is
    # kept; the value must not reach logs, stored rows, error bodies or telemetry.
    ng_session_close_token: Optional[str] = Field(default=None, alias="_ng_session_close_token")


class BaseSeedSessionResponse(BaseModel):
    pass


class BaseCloseSessionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ng_session_id: Optional[str] = Field(default=None, alias="_ng_session_id")
    ng_session_close_token: Optional[str] = Field(default=None, alias="_ng_session_close_token")
    reason: SessionCloseReason = SessionCloseReason.COMPLETED


class BaseCloseSessionResponse(BaseModel):
    """Lifecycle only. Carries no reward, trajectory, verifier field or provider handle."""

    object: Literal["nemo_gym.session.close"] = "nemo_gym.session.close"
    status: SessionCloseStatus
    released: bool


class MCPServerMetadata(BaseModel):
    """Metadata returned from /seed_session for per-rollout Gym MCP access."""

    server_name: str
    url_path: str = "/mcp"
    transport: str = "http"
    headers: dict[str, str]


def _digest(value: str) -> str:
    """Hash a secret or payload for comparison without keeping the original."""
    return hashlib.sha256(value.encode()).hexdigest()


def _redact(session_id: Optional[str]) -> str:
    """A session id is not a secret, but full ids do not belong in logs either."""
    return f"{session_id[:8]}..." if session_id else "?"


@dataclass
class _SessionRecord:
    """What the server remembers about one resources allocation.

    It outlives the resource: after release the record stays as a tombstone so a seed
    request delayed in the network cannot recreate what was already torn down.
    """

    seed_payload_digest: str
    seed_response: Any
    created_at: float
    last_seen: float
    close_token_digest: Optional[str] = None
    # Requests currently executing against this session. Idle expiry ignores a session
    # while any are in flight, because a rollout can legitimately spend minutes in a model
    # call without touching the resources server.
    active_handlers: int = 0
    # One release runs per session. Concurrent closers await this task instead of each
    # calling into the environment, so provider cleanup happens once.
    release_task: Optional[Any] = None
    closed_at: Optional[float] = None

    @property
    def is_closed(self) -> bool:
        return self.closed_at is not None


class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    config: BaseResourcesServerConfig

    # One record per session, live or tombstoned. Private (leading underscore) so pydantic
    # does not try to build a schema for it.
    _sessions: dict[str, _SessionRecord] = {}

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._sessions = {}

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)
        app.add_middleware(RolloutContextMiddleware)

        if self.config.session_ttl_s is not None:

            @app.on_event("startup")
            async def _start_session_sweeper() -> None:  # pragma: no cover - exercised via the task
                asyncio.create_task(self._sweep_idle_sessions())

        app.post("/seed_session")(self._seed_session_endpoint)
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
        """Allocate whatever this session needs. Default allocates nothing."""
        return BaseSeedSessionResponse()

    async def release_session(self, session_id: str, reason: SessionCloseReason) -> None:
        """Release what this session allocated. Default no-op.

        Deliberately not named ``close_session``: ``GymnasiumServer``, TALES and OpenAir
        already define a method by that name with a different signature, and a base method
        sharing it would silently override theirs. A server migrating to the common route
        can bridge instead of reimplementing::

            async def release_session(self, session_id, reason):
                await self.close_session(session_id)

        Identity, conflict checks, idempotency and the response belong to the route
        adapter. This hook only releases, and only what this server owns. It should be
        idempotent: a caller that times out will retry.
        """
        return None

    # ----- session identity and bookkeeping -------------------------------- #
    def touch_session(self, session_id: str) -> None:
        """Record activity so idle expiry does not reclaim a session that is still in use."""
        record = self._sessions.get(session_id)
        if record is not None:
            record.last_seen = time.monotonic()

    def forget_session(self, session_id: str) -> None:
        """Drop a record outright. Expiry normally tombstones instead, so this is for tests
        and for servers that manage their own identity."""
        self._sessions.pop(session_id, None)

    @asynccontextmanager
    async def active_session(self, session_id: Optional[str]):
        """Mark a session busy for the duration of a handler.

        Two jobs: idle expiry skips a session while a handler is in flight, and a handler
        cannot start once release has begun — otherwise close, or the reaper, can tear a
        resource down while verification is still reading it.

        The base uses this around ``/verify``. An environment's tool routes should use it
        too; broad adoption is #3037.
        """
        record = self._sessions.get(session_id) if session_id else None
        if record is not None:
            if record.is_closed or record.release_task is not None:
                raise HTTPException(status_code=409, detail="session is closing")
            record.active_handlers += 1
            record.last_seen = time.monotonic()
        try:
            yield
        finally:
            if record is not None:
                record.active_handlers = max(0, record.active_handlers - 1)
                record.last_seen = time.monotonic()

    # ----- seed ------------------------------------------------------------ #
    async def _seed_session_endpoint(self, request: Request, body: BaseSeedSessionRequest) -> Any:
        """Deduplicate by the caller-created id, then delegate to ``seed_session``.

        The transport may repeat a seed it is not sure was delivered. Without this the
        repeat allocates a second resource and the first becomes unreachable, which is the
        leak the caller-created id exists to prevent.
        """
        session_id = body.ng_session_id or request.session.get(SESSION_ID_KEY)
        payload_digest = _digest(
            json.dumps(
                body.model_dump(mode="json", exclude={"ng_session_id", "ng_session_close_token"}),
                sort_keys=True,
            )
        )
        record = self._sessions.get(session_id) if session_id else None

        if record is not None:
            if record.is_closed:
                # A seed delayed in the network must not resurrect a torn-down resource.
                raise HTTPException(status_code=409, detail="session id has already been closed")
            if record.seed_payload_digest != payload_digest:
                raise HTTPException(status_code=409, detail="session id already seeded with a different payload")
            record.last_seen = time.monotonic()
            return record.seed_response

        response = await self.seed_session(body)
        if session_id:
            now = time.monotonic()
            self._sessions[session_id] = _SessionRecord(
                seed_payload_digest=payload_digest,
                seed_response=response,
                created_at=now,
                last_seen=now,
                close_token_digest=(_digest(body.ng_session_close_token) if body.ng_session_close_token else None),
            )
        return response

    # ----- close ----------------------------------------------------------- #
    async def _close_session_endpoint(
        self, request: Request, body: BaseCloseSessionRequest, response: Response
    ) -> BaseCloseSessionResponse:
        """Resolve identity, authorize, and run exactly one release.

        The adapter owns everything that is not resource release: which session the caller
        means, whether it may release it, whether release already happened, and what the
        wire says about it.
        """
        cookie_id = request.session.get(SESSION_ID_KEY)
        body_id = body.ng_session_id

        if cookie_id and body_id and cookie_id != body_id:
            # Releasing either one would act on a session the caller did not name.
            raise HTTPException(status_code=409, detail="cookie and session id identify different sessions")

        session_id = body_id or cookie_id
        record = self._sessions.get(session_id) if session_id else None

        if record is not None and cookie_id is None:
            # No cookie: the seed response was lost, so the close capability is the only
            # thing that authorizes release.
            if record.close_token_digest is None or not body.ng_session_close_token:
                raise HTTPException(status_code=403, detail="close capability required without a session cookie")
            if not hmac.compare_digest(record.close_token_digest, _digest(body.ng_session_close_token)):
                raise HTTPException(status_code=403, detail="close capability does not match")

        return await self._release_once(session_id, record, body.reason, response)

    async def _release_once(
        self,
        session_id: Optional[str],
        record: Optional[_SessionRecord],
        reason: SessionCloseReason,
        response: Optional[Response] = None,
    ) -> BaseCloseSessionResponse:
        """Run release at most once per session; concurrent callers await the same result.

        An unknown session and an already-released one answer identically, so the endpoint
        cannot be used to learn whether another rollout's session exists.
        """
        if record is None or record.is_closed:
            return BaseCloseSessionResponse(status=SessionCloseStatus.ALREADY_CLOSED, released=True)

        if record.release_task is None:
            record.release_task = asyncio.create_task(self.release_session(session_id, reason))

        try:
            await asyncio.shield(record.release_task)
        except asyncio.CancelledError:
            # The caller went away; the release itself keeps running.
            raise
        except Exception:
            logger.warning("releasing session %s failed", _redact(session_id), exc_info=True)
            record.release_task = None
            if response is not None:
                response.status_code = 503
            return BaseCloseSessionResponse(status=SessionCloseStatus.RELEASE_FAILED, released=False)

        record.closed_at = time.monotonic()
        record.release_task = None
        return BaseCloseSessionResponse(status=SessionCloseStatus.CLOSED, released=True)

    # ----- expiry ---------------------------------------------------------- #
    async def _sweep_idle_sessions(self) -> None:
        """Reclaim sessions no call will reach: a killed trainer, a dropped connection.

        Idle time alone is not a liveness signal — a healthy rollout can sit in a long model
        call — so a session with handlers in flight is skipped, and an absolute lifetime
        covers the case where a handler itself has hung.
        """
        ttl = self.config.session_ttl_s
        assert ttl is not None
        while True:
            await asyncio.sleep(self.config.session_sweep_interval_s)
            try:
                await self._sweep_once(time.monotonic(), ttl)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("environment session sweep failed")

    async def _sweep_once(self, now: float, ttl: float) -> None:
        max_lifetime = self.config.session_max_lifetime_s
        tombstone_s = self.config.session_tombstone_s
        for session_id, record in list(self._sessions.items()):
            if record.is_closed:
                if now - record.closed_at > tombstone_s:
                    self._sessions.pop(session_id, None)
                continue
            idle_expired = record.active_handlers == 0 and now - record.last_seen > ttl
            # Not suppressed by active handlers: a handler stuck for hours is the case this
            # bound exists for.
            lifetime_expired = max_lifetime is not None and now - record.created_at > max_lifetime
            if idle_expired or lifetime_expired:
                logger.warning("reclaiming environment session %s", _redact(session_id))
                await self._release_once(session_id, record, SessionCloseReason.EXPIRED)

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
