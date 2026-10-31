# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An Environment Server that serves each episode's tools itself, so it needs no Resources Server."""

import asyncio
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    CleanupContext,
    EpisodeRequestT,
    EpisodeResponseT,
)
from nemo_gym.episode_tokens import EPISODE_TOKEN_HEADER, EpisodeTokenClaims, EpisodeTokens
from nemo_gym.episode_tools import EpisodeToolset, ToolArgumentsError, ToolCallRecord, ToolSpec
from nemo_gym.episode_types import EpisodeId
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from nemo_gym.tool_access import DirectHTTPToolAccess, ToolAccess


LOGGER = logging.getLogger(__name__)


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


class BaseEnvironmentServerWithTools(BaseEnvironmentServer[EpisodeRequestT, EpisodeResponseT]):
    """An Environment Server that serves each episode's tools itself, so it needs no Resources Server.

    Other Environment Servers grant the agent a Resources Server's tools; this one owns the tools, their state and
    verification.

    ``run`` holds the episode's state, and ``serve_tools`` grants the agent the episode's tools until the episode
    ends. A tool call is a separate request to ``POST /<tool name>`` at the server's root, as on a Resources
    Server, with the episode token in the ``X-NeMo-Gym-Episode-Token`` header. Calls to one episode run one at a
    time, and each is recorded in ``ServedTools.calls``.
    """

    _episode_tokens: Optional[EpisodeTokens] = None
    _open_episodes: Optional[dict[EpisodeId, _OpenEpisode]] = None
    _reserved_tool_names: frozenset[str] = frozenset()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._episode_tokens = EpisodeTokens(self.get_session_middleware_key())
        self._open_episodes = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # A tool call under /ng-rollout/<id>/ (an agent that correlates its calls with the rollout) routes as bare.
        app.add_middleware(RolloutContextMiddleware)
        # A tool cannot take the name of the server's own routes; the catch-all below comes after them.
        self._reserved_tool_names = frozenset(
            route.path.strip("/").split("/")[0] for route in app.routes if getattr(route, "path", "/") != "/"
        )
        app.post("/{tool_name}")(self._tool_call_endpoint)
        return app

    def multiple_workers_unsupported_reason(self) -> Optional[str]:
        return "a tool call reaches whichever worker accepts it, and only the worker that runs the episode holds it"

    async def serve_tools(
        self, request: EpisodeRequestT, toolset: EpisodeToolset, cleanup: CleanupContext
    ) -> ServedTools:
        """Grant the agent ``toolset`` for this episode, until the episode's cleanup runs.

        Returns the grants for the agent's seed (``AgentSeedSessionRequest.tool_accesses``) and the list that
        records each call. A call after the episode ends gets 410.
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

        token, claims = self._episode_tokens.mint(episode_id)
        episode = _OpenEpisode(claims=claims, toolset=toolset, specs=specs, calls=[])
        self._open_episodes[episode_id] = episode

        async def stop_serving() -> None:
            if self._open_episodes.get(episode_id) is episode:
                del self._open_episodes[episode_id]

        cleanup.register_cleanup("episode tools", stop_serving)
        base_url = self.server_client._resolve_base_url(self.config.name)
        grant = DirectHTTPToolAccess(
            name=f"{self.config.name}.direct_http",
            required=True,
            base_url=base_url,
            headers={EPISODE_TOKEN_HEADER: token},
        )
        return ServedTools(tool_accesses=[grant], calls=episode.calls)

    async def _tool_call_endpoint(self, tool_name: str, request: Request) -> JSONResponse:
        claims = self._episode_tokens.read(request.headers.get(EPISODE_TOKEN_HEADER, ""))
        if claims is None:
            return _refusal(401, f"A tool call needs a valid {EPISODE_TOKEN_HEADER} header")
        episode = self._open_episodes.get(claims.episode_id)
        if episode is None:
            return _refusal(410, f"Episode {claims.episode_id.capture_key} is not running on this server")
        if not episode.claims.matches(claims):
            return _refusal(403, "The episode token was not granted for this episode")
        if tool_name not in episode.specs:
            return _refusal(404, f"Episode {claims.episode_id.capture_key} has no tool {tool_name!r}")

        try:
            arguments = await request.json()
        except ValueError:
            arguments = None
        async with episode.lock:
            return await self._call_tool(episode, tool_name, arguments)

    async def _call_tool(self, episode: _OpenEpisode, name: str, arguments: Any) -> JSONResponse:
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
        return JSONResponse(status_code=status, content=output)


def _refusal(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})
