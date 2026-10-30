# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An Environment Server that serves each episode's tools itself, so it needs no Resources Server."""

import asyncio
import logging
import re
from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import Field, JsonValue, ValidationError, field_validator

from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    EpisodeRequestT,
    EpisodeResponseT,
)
from nemo_gym.episode_tokens import EPISODE_TOKEN_HEADER, EpisodeTokenClaims, EpisodeTokens
from nemo_gym.episode_tools import EpisodeToolset, ToolArgumentsError, ToolCallRecord, ToolSpec
from nemo_gym.episode_types import EpisodeId
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from nemo_gym.session_routing import SessionClaims, SessionKind
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess, ToolAccess


if TYPE_CHECKING:
    import mcp.types


LOGGER = logging.getLogger(__name__)

# An episode's tool calls name it only by the episode token header; no cookie, seed or close names it.
EPISODE_TOOL_SESSIONS = SessionKind(
    cookie_key=None,
    id_field=None,
    seed_path=None,
    close_path=None,
    cookie_session=False,
)

ToolTransport = Literal["direct_http", "mcp"]
# The path ``nemo_gym.mcp_auto_exposure.mount_mcp_endpoint`` serves. Kept here so that a server without MCP never
# imports the MCP SDK.
_MCP_PATH = "/mcp"
# MCP clients reject tool names outside this alphabet, such as a name with a dot.
_MCP_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]+")


class BaseEnvironmentServerWithToolsConfig(BaseEnvironmentServerConfig):
    """Configure how a ``BaseEnvironmentServerWithTools`` grants its episodes' tools."""

    tool_transports: list[ToolTransport] = Field(
        default_factory=lambda: ["direct_http"],
        min_length=1,
        description=(
            "The transports serve_tools grants, in order. direct_http calls POST /<tool name>, for agents such as "
            "simple_agent. mcp serves the episode's tools at /mcp, for agents that take only MCP grants, such as "
            "Codex, Pi and Hermes."
        ),
    )

    @field_validator("tool_transports")
    @classmethod
    def validate_unique_transports(cls, transports: list[ToolTransport]) -> list[ToolTransport]:
        if len(set(transports)) != len(transports):
            raise ValueError("tool_transports must not repeat a transport")
        return transports


@dataclass
class ServedTools:
    """The grants that give the agent one episode's tools, and the calls it has made so far."""

    tool_accesses: list[ToolAccess]
    calls: list[ToolCallRecord]


@dataclass
class _OpenEpisode:
    claims: EpisodeTokenClaims
    toolset: EpisodeToolset
    specs: dict[str, ToolSpec]
    calls: list[ToolCallRecord]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class _Refusal:
    status: int
    detail: str


@dataclass(frozen=True)
class _ToolOutcome:
    status: int
    output: JsonValue
    is_error: bool


class BaseEnvironmentServerWithTools(BaseEnvironmentServer[EpisodeRequestT, EpisodeResponseT]):
    """An Environment Server that serves each episode's tools itself, so it needs no Resources Server.

    Other Environment Servers grant the agent a Resources Server's tools; this one owns the tools, their state and
    verification.

    ``run_episode`` holds the episode's state, and ``serve_tools`` grants the agent the episode's tools until the
    episode ends, over each transport in ``tool_transports``. Every call carries the episode token in the
    ``X-NeMo-Gym-Episode-Token`` header:

    - direct HTTP: a separate request to ``POST /<tool name>`` at the server's root, as on a Resources Server;
    - MCP: ``tools/call`` on ``/mcp``, where ``tools/list`` shows only the episode's own tools.

    Both transports check the token the same way. Calls to one episode run one at a time, whatever their
    transport, and each is recorded in ``ServedTools.calls``.

    With several workers, the worker that runs the episode claims it (``nemo_gym.session_routing``), so a tool
    call that reaches another worker, over either transport, is forwarded there, by the episode its token names.

    When the server shuts down, it finishes the cleanups of every episode still running, such as stopping its
    sandbox.
    """

    config: BaseEnvironmentServerWithToolsConfig

    routes_sessions_to_owner = True
    _episode_tokens: Optional[EpisodeTokens] = None
    _open_episodes: Optional[dict[EpisodeId, _OpenEpisode]] = None
    # Keyed by id(): two attempts of one episode can run at once without checkpointing.
    _running_cleanups: Optional[dict[int, CleanupContext]] = None
    _reserved_tool_names: frozenset[str] = frozenset()
    # With several workers, which worker runs each episode (see nemo_gym.session_routing).
    _session_claims: Optional[SessionClaims] = None

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._episode_tokens = EpisodeTokens(self.get_session_middleware_key())
        self._open_episodes = {}
        self._running_cleanups = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        if "mcp" in self.config.tool_transports:
            self._mount_mcp_endpoint(app)
        # A tool call under /ng-rollout/<id>/ (an agent that correlates its calls with the rollout) routes as bare.
        app.add_middleware(RolloutContextMiddleware)
        # A tool cannot take the name of the server's own routes; the catch-all below comes after them.
        self._reserved_tool_names = frozenset(
            route.path.strip("/").split("/")[0] for route in app.routes if getattr(route, "path", "/") != "/"
        )
        app.post("/{tool_name}")(self._tool_call_endpoint)

        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.finish_running_cleanups()

        app.router.lifespan_context = lifespan
        return app

    async def run(self, request: EpisodeRequestT, cleanup: CleanupContext) -> EpisodeResponseT:
        key = id(cleanup)
        self._running_cleanups[key] = cleanup

        async def forget() -> None:
            self._running_cleanups.pop(key, None)

        # Registered first, so it runs last: the episode stays tracked until each of its cleanups has finished.
        cleanup.register_cleanup("shutdown tracking", forget)
        return await self.run_episode(request, cleanup)

    @abstractmethod
    async def run_episode(self, request: EpisodeRequestT, cleanup: CleanupContext) -> EpisodeResponseT:
        """Run one episode, as ``BaseEnvironmentServer.run`` does. Register each resource's release in ``cleanup``."""

    async def finish_running_cleanups(self) -> None:
        """Finish the cleanups of every episode still running; the server runs this when it shuts down.

        Shutdown cancels the running ``/run`` requests and does not wait for them, so a second cancellation can
        interrupt an episode's cleanup partway. Cleanups are idempotent, so this runs each one that has not
        finished, each within ``cleanup_timeout_seconds``.
        """
        await asyncio.gather(*(cleanup.aclose() for cleanup in list(self._running_cleanups.values())))

    def claimed_session_routing(self) -> dict[str, Any]:
        return {
            "kind": EPISODE_TOOL_SESSIONS,
            "session_header": EPISODE_TOKEN_HEADER,
            "session_from_header": self._episode_session_id,
            "session_holder": self,
        }

    def _episode_session_id(self, token: str) -> Optional[str]:
        # Routing needs only the episode the token names; the worker that runs the episode checks its nonce.
        claims = self._episode_tokens.read(token)
        return claims.episode_id.capture_key if claims is not None else None

    def holds_session(self, session_id: str) -> bool:
        try:
            return EpisodeId.from_capture_key(session_id) in self._open_episodes
        except (ValueError, ValidationError):
            return False

    def use_session_claims(self, claims: SessionClaims) -> None:
        self._session_claims = claims

    async def serve_tools(
        self, request: EpisodeRequestT, toolset: EpisodeToolset, cleanup: CleanupContext
    ) -> ServedTools:
        """Grant the agent ``toolset`` for this episode, until the episode's cleanup runs.

        Returns one grant for each of ``tool_transports``, for the agent's seed
        (``AgentSeedSessionRequest.tool_accesses``), and the list that records each call. A call after the episode
        ends is refused.
        """
        episode_id = request.episode_id
        if episode_id in self._open_episodes:
            raise ValueError(f"Episode {episode_id.capture_key} already serves its tools")
        listed = list(toolset.specs())
        specs = {spec.name: spec for spec in listed}
        if len(specs) != len(listed):
            raise ValueError("Episode tool names must be unique")
        reserved = sorted(specs.keys() & self._reserved_tool_names)
        if reserved:
            raise ValueError(f"Episode tools cannot take the names of the server's routes: {', '.join(reserved)}")
        if "mcp" in self.config.tool_transports:
            unnamable = sorted(name for name in specs if not _MCP_TOOL_NAME.fullmatch(name))
            if unnamable:
                raise ValueError(f"MCP tool names may use only letters, digits, _ and -: {', '.join(unnamable)}")

        session_claims = self._session_claims
        if session_claims is not None and session_claims.claim(episode_id.capture_key) != session_claims.routing_id:
            raise ValueError(f"Episode {episode_id.capture_key} already serves its tools on another worker")

        token, claims = self._episode_tokens.mint(episode_id)
        episode = _OpenEpisode(claims=claims, toolset=toolset, specs=specs, calls=[])
        self._open_episodes[episode_id] = episode

        async def stop_serving() -> None:
            if self._open_episodes.get(episode_id) is episode:
                del self._open_episodes[episode_id]
                if session_claims is not None:
                    session_claims.release(episode_id.capture_key)

        cleanup.register_cleanup("episode tools", stop_serving)
        base_url = self.server_client._resolve_base_url(self.config.name).rstrip("/")
        headers = {EPISODE_TOKEN_HEADER: token}
        grants: dict[ToolTransport, ToolAccess] = {
            "direct_http": DirectHTTPToolAccess(
                name=f"{self.config.name}.direct_http", required=True, base_url=base_url, headers=headers
            ),
            "mcp": MCPToolAccess(
                name=self.config.name,
                required=True,
                connection=MCPStreamableHTTPConnection(url=f"{base_url}{_MCP_PATH}", headers=headers),
            ),
        }
        return ServedTools(
            tool_accesses=[grants[transport] for transport in self.config.tool_transports], calls=episode.calls
        )

    async def _tool_call_endpoint(self, tool_name: str, request: Request) -> JSONResponse:
        episode = self._episode_for_token(request.headers.get(EPISODE_TOKEN_HEADER))
        if isinstance(episode, _Refusal):
            return JSONResponse(status_code=episode.status, content={"detail": episode.detail})
        try:
            arguments = await request.json()
        except ValueError:
            arguments = None
        outcome = await self._run_tool(episode, tool_name, arguments)
        return JSONResponse(status_code=outcome.status, content=outcome.output)

    def _mount_mcp_endpoint(self, app: FastAPI) -> None:
        # Imported here, so the MCP SDK loads only in a server that grants MCP.
        import mcp.types as types

        from nemo_gym.mcp_auto_exposure import mount_mcp_endpoint, to_call_tool_result

        def episode_for(request: Optional[Request]) -> "_OpenEpisode | _Refusal":
            return self._episode_for_token(request.headers.get(EPISODE_TOKEN_HEADER) if request is not None else None)

        async def list_tools(request: Optional[Request]) -> list["mcp.types.Tool"]:
            episode = episode_for(request)
            if isinstance(episode, _Refusal):
                return []
            return [
                types.Tool(name=spec.name, description=spec.description, inputSchema=spec.parameters)
                for spec in episode.specs.values()
            ]

        async def call_tool(request: Optional[Request], name: str, arguments: dict) -> "mcp.types.CallToolResult":
            episode = episode_for(request)
            if isinstance(episode, _Refusal):
                return to_call_tool_result(episode.detail, is_error=True)
            outcome = await self._run_tool(episode, name, arguments)
            return to_call_tool_result(outcome.output, is_error=outcome.is_error)

        mount_mcp_endpoint(app, self.config.name, list_tools, call_tool)

    def _episode_for_token(self, token: Optional[str]) -> "_OpenEpisode | _Refusal":
        """The open episode an episode token was granted for, or why the call is refused."""
        claims = self._episode_tokens.read(token or "")
        if claims is None:
            return _Refusal(401, f"A tool call needs a valid {EPISODE_TOKEN_HEADER} header")
        episode = self._open_episodes.get(claims.episode_id)
        if episode is None:
            return _Refusal(410, f"Episode {claims.episode_id.capture_key} is not running on this server")
        if not episode.claims.matches(claims):
            return _Refusal(403, "The episode token was not granted for this episode")
        return episode

    async def _run_tool(self, episode: _OpenEpisode, name: str, arguments: Any) -> _ToolOutcome:
        """Run one tool call on the episode, after the calls before it, and record it."""
        if name not in episode.specs:
            detail = f"Episode {episode.claims.episode_id.capture_key} has no tool {name!r}"
            return _ToolOutcome(status=404, output={"detail": detail}, is_error=True)
        async with episode.lock:
            started = perf_counter()
            status, is_error = 200, False
            try:
                if not isinstance(arguments, dict):
                    raise ToolArgumentsError("Tool arguments must be a JSON object")
                result = await episode.toolset.call(name, arguments)
                output, is_error = result.output, result.is_error
            except ToolArgumentsError as error:
                status, is_error, output = 422, True, {"detail": str(error)}
            except Exception as error:
                LOGGER.exception(f"Episode tool {name!r} failed: episode_id={episode.claims.episode_id}")
                status, is_error, output = 500, True, {"detail": f"Tool {name!r} failed: {type(error).__name__}"}
            episode.calls.append(
                ToolCallRecord(
                    name=name,
                    arguments=arguments,
                    output=output,
                    is_error=is_error,
                    duration_seconds=perf_counter() - started,
                )
            )
        return _ToolOutcome(status=status, output=output, is_error=is_error)
