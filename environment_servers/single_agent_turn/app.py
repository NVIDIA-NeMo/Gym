# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources-backed single-agent environment server."""

from typing import Any, Literal
from uuid import uuid4

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from fastapi import Body
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym._checkpoint.steps import StepMode, seed_restarts, seed_verify_mode
from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    HandledEpisodeError,
)
from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.config_types import (
    TOKEN_CAPTURE_PATH_SEGMENT,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ResourcesServerRef,
)
from nemo_gym.global_config import (
    TOKEN_ID_CAPTURE_BLOCK,
    get_first_server_config_dict,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from nemo_gym.single_agent_turn_types import (
    SingleAgentTurnFailure,
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
    SingleAgentTurnResult,
)
from nemo_gym.tool_access import (
    DirectHTTPToolAccess,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    ToolAccess,
)


class SessionHandles(BaseModel):
    """The sessions an episode owns: enough to reuse them after a restore and to close them."""

    resources_session_id: str
    resources_cookies: dict[str, str] = Field(default_factory=dict)
    agent_session_id: str
    agent_cookies: dict[str, str] = Field(default_factory=dict)
    # Whether a checkpoint may replay the resources server's /verify, as its seed reply reported.
    resources_verify: StepMode = "wait"

    @classmethod
    def new(cls) -> "SessionHandles":
        return cls(
            resources_session_id=f"resources-session-{uuid4().hex}", agent_session_id=f"agent-session-{uuid4().hex}"
        )


class SingleAgentTurnEnvironmentServerConfig(BaseEnvironmentServerConfig):
    """Bind Resources and Agent Servers for one agent turn."""

    model_config = ConfigDict(extra="forbid")

    resources_server: ResourcesServerRef
    agent_server: AgentServerRef
    resources_tool_transports: list[Literal["direct_http", "mcp"]] = Field(default_factory=list)


class SingleAgentTurnEnvironmentServer(BaseEnvironmentServer[SingleAgentTurnRequest, SingleAgentTurnResponse]):
    """Run one agent turn followed by Resources verification and cleanup."""

    ray_enabled = False
    config: SingleAgentTurnEnvironmentServerConfig
    request_model = SingleAgentTurnRequest
    checkpoint_boundaries = True
    response_model = SingleAgentTurnResponse

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Forward to the resources server, which owns verification in this protocol."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def run(
        self,
        request: SingleAgentTurnRequest,
        cleanup: CleanupContext,
    ) -> SingleAgentTurnResponse:
        # Steps: seed → invoke the agent → close the agent → verify.
        # A boundary before each step names the next step, the sessions to reuse, and what that step needs,
        # so a replacement attempt that continues a checkpoint resumes at that step.
        continuation = self.checkpoint_continuation(request) or {}
        stage = continuation.get("next", "seed")
        handles = (
            SessionHandles.model_validate(continuation["handles"])
            if "handles" in continuation
            else SessionHandles.new()
        )
        agent_response = (
            NeMoGymResponse.model_validate(continuation["response"]) if continuation.get("response") else None
        )
        observations = (
            AgentObservationBundle.model_validate(continuation["observations"])
            if continuation.get("observations")
            else None
        )

        # Register cleanup before seeding so a lost seed reply cannot hide the caller-assigned session ID.
        # Final cleanup runs after run() returns, outside the episode deadline.
        resources_cleanup = cleanup.register_cleanup(
            "resources session", lambda: self._close_resources(request, handles)
        )
        closed_agent: list[AgentCloseSessionResponse] = []

        async def close_agent() -> None:
            closed_agent.append(await self._close_agent(request, handles))

        # Register the agent cleanup before seeding the agent,
        # so cancellation can close a remotely created session even if its reply is lost;
        # a continuation past seeding registers it at once.
        agent_cleanup = (
            cleanup.register_cleanup("agent session", close_agent)
            if stage in ("invoke_agent", "close_agent")
            else None
        )

        if stage == "seed":
            await self.checkpoint_boundary(request, {"next": "seed", "handles": handles.model_dump()})
            # Seeds are idempotent for a caller-assigned session ID, so a checkpoint need not wait for them.
            async with self.checkpoint_step(request, "replay"):
                seed = await self._seed_resources(request, handles)
                agent_cleanup = cleanup.register_cleanup("agent session", close_agent)
                await self._seed_agent(request, handles, seed)
            stage = "invoke_agent"

        if stage == "invoke_agent":
            await self.checkpoint_boundary(request, {"next": "invoke_agent", "handles": handles.model_dump()})
            # The agent parks at its own boundaries and continues from them after a restore.
            async with self.checkpoint_step(request, "replay"):
                agent_response = await self._invoke_agent(request, handles)
            # The agent's activation is over; park here, not inside the close below,
            # so a checkpoint that counted this episode at the boundary
            # before the activation never finds it inside a wait step afterwards.
            await self.checkpoint_boundary(
                request,
                lambda: {
                    "next": "close_agent",
                    "handles": handles.model_dump(),
                    "response": agent_response.model_dump(mode="json"),
                },
            )
            stage = "close_agent"

        if stage == "close_agent":
            # Closing returns the agent's observations and final resources cookies exactly once,
            # so a checkpoint waits for it and records them.
            async with self.checkpoint_step(request, "wait"):
                try:
                    await agent_cleanup.close()
                    close = closed_agent[-1]
                except Exception as error:
                    raise self._failure(
                        stage="cleanup",
                        failure_reason=str(error),
                        terminal=not _is_retryable_dependency_error(error),
                        partial_response=agent_response,
                    ) from error
            observations = close.agent_observations
            stage = "verify"

        def dumped_observations() -> dict | None:
            return observations.model_dump(mode="json") if observations else None

        if stage == "verify":
            await self.checkpoint_boundary(
                request,
                lambda: {
                    "next": "verify",
                    "handles": handles.model_dump(),
                    "response": agent_response.model_dump(mode="json"),
                    "observations": dumped_observations(),
                },
            )
            async with self.checkpoint_step(request, handles.resources_verify):
                verification = await self._verify(request, handles, agent_response)
            # Record the result so a restore never runs a state-changing verification twice.
            await self.checkpoint_boundary(
                request,
                lambda: {
                    "next": "return",
                    "handles": handles.model_dump(),
                    "result": verification.model_dump(mode="json"),
                    "observations": dumped_observations(),
                },
            )
        else:
            verification = SingleAgentTurnResult.model_validate(continuation["result"])

        # Keep one bounded retry in final unwind without erasing a completed verdict.
        cleanup.register_cleanup("post-verification resources session", resources_cleanup.close)
        return SingleAgentTurnResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=verification.model_copy(update={"ng_agent_observations": observations}),
        )

    async def _seed_resources(
        self, request: SingleAgentTurnRequest, handles: "SessionHandles"
    ) -> ResourcesSeedSessionResponse:
        task_input = request.task.task_input
        try:
            seed_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=ResourcesSeedSessionRequest(
                    resources_session_id=handles.resources_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    task_data=task_input.task_data,
                ).model_dump(mode="json"),
            )
            await raise_for_status(seed_http_response)
            handles.resources_cookies = _cookies(seed_http_response)
            handles.resources_verify = seed_verify_mode(seed_http_response.headers)
            if seed_restarts(seed_http_response.headers):
                await self.checkpoint_restart(request)
            if not handles.resources_cookies:
                raise ValueError("Resources seed did not establish a session cookie")
            seed = ResourcesSeedSessionResponse.model_validate(await get_response_json(seed_http_response))
            if seed.resources_session_id != handles.resources_session_id:
                raise ValueError("Resources seed returned a different resources_session_id")
        except Exception as error:
            raise self._failure(
                stage="seed",
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error
        return seed

    async def _seed_agent(
        self, request: SingleAgentTurnRequest, handles: "SessionHandles", seed: ResourcesSeedSessionResponse
    ) -> None:
        resources_base_url = self.server_client._resolve_base_url(self.config.resources_server.name).rstrip("/")
        tool_accesses: list[ToolAccess] = []
        if "direct_http" in self.config.resources_tool_transports:
            tool_accesses.append(
                DirectHTTPToolAccess(
                    name=f"{self.config.resources_server.name}.direct_http",
                    required=True,
                    base_url=resources_base_url,
                    cookies=handles.resources_cookies,
                )
            )
        if "mcp" in self.config.resources_tool_transports:
            if seed.resources_tools is None:
                raise self._failure(
                    stage="seed",
                    failure_reason="Resources seed did not return requested MCP metadata",
                    terminal=True,
                )
            if seed.resources_tools.transport != "http":
                raise self._failure(
                    stage="seed",
                    failure_reason=f"Unsupported resources MCP transport: {seed.resources_tools.transport}",
                    terminal=True,
                )
            url_path = seed.resources_tools.url_path.lstrip("/")
            tool_accesses.append(
                MCPToolAccess(
                    name=seed.resources_tools.server_name,
                    required=True,
                    connection=MCPStreamableHTTPConnection(
                        url=f"{resources_base_url}/{url_path}",
                        headers=seed.resources_tools.headers,
                    ),
                )
            )

        try:
            agent_create_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                json=AgentSeedSessionRequest(
                    agent_session_id=handles.agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    tool_accesses=tool_accesses,
                    sandbox_access=seed.sandbox_access,
                ).model_dump(mode="json"),
            )
            await raise_for_status(agent_create_http_response)
            agent_session = AgentSeedSessionResponse.model_validate(
                await get_response_json(agent_create_http_response)
            )
            if agent_session.agent_session_id != handles.agent_session_id:
                raise ValueError("Agent seed returned a different agent_session_id")
            handles.agent_cookies = _cookies(agent_create_http_response)
            if seed_restarts(agent_create_http_response.headers):
                await self.checkpoint_restart(request)
        except Exception as error:
            raise self._failure(
                stage="agent",
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

    async def _invoke_agent(self, request: SingleAgentTurnRequest, handles: "SessionHandles") -> NeMoGymResponse:
        try:
            agent_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path=self._agent_responses_path(request),
                json=request.task.task_input.responses_create_params,
                cookies=handles.agent_cookies,
            )
            await raise_for_status(agent_http_response)
            response_cookies = _cookies(agent_http_response)
            if response_cookies:
                handles.agent_cookies = response_cookies
            return NeMoGymResponse.model_validate(await get_response_json(agent_http_response))
        except Exception as error:
            raise self._failure(
                stage="agent",
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

    async def _close_agent(
        self, request: SingleAgentTurnRequest, handles: "SessionHandles"
    ) -> AgentCloseSessionResponse:
        close_http_response = await self.server_client.post(
            server_name=self.config.agent_server.name,
            url_path="/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(
                agent_session_id=handles.agent_session_id,
                episode_id=request.episode_id,
            ).model_dump(mode="json"),
            cookies=handles.agent_cookies,
        )
        await raise_for_status(close_http_response)
        close = AgentCloseSessionResponse.model_validate(await get_response_json(close_http_response))
        if close.agent_session_id != handles.agent_session_id:
            raise ValueError("Agent close returned a different agent_session_id")
        if close.resources_cookies is not None:
            handles.resources_cookies = close.resources_cookies
        return close

    async def _close_resources(self, request: SingleAgentTurnRequest, handles: "SessionHandles") -> None:
        close_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/close_session",
            json=ResourcesCloseSessionRequest(
                resources_session_id=handles.resources_session_id,
                episode_id=request.episode_id,
            ).model_dump(mode="json"),
            cookies=handles.resources_cookies,
        )
        await raise_for_status(close_response)

    async def _verify(
        self, request: SingleAgentTurnRequest, handles: "SessionHandles", agent_response: NeMoGymResponse
    ) -> SingleAgentTurnResult:
        task_input = request.task.task_input
        try:
            verify_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                # The Resources Server's own flat verify body, as an Agent's /run sends it; the session
                # cookie identifies the episode.
                json=task_input.task_data
                | {
                    "responses_create_params": task_input.responses_create_params.model_dump(
                        mode="json", exclude_unset=True
                    ),
                    "response": agent_response.model_dump(mode="json"),
                },
                cookies=handles.resources_cookies,
            )
            await raise_for_status(verify_http_response)
            return SingleAgentTurnResult.model_validate(await get_response_json(verify_http_response))
        except Exception as error:
            raise self._failure(
                stage="verification",
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
                partial_response=agent_response,
            ) from error

    def _agent_responses_path(self, request: SingleAgentTurnRequest) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        agent_config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.agent_server.name,
        )
        token_capture = bool(block.get("enabled", False)) and (
            bool(block.get("all_agents", False)) or bool(agent_config.get("token_id_capture", False))
        )
        capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
        return f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}/v1/responses"

    @staticmethod
    def _failure(
        *,
        stage: str,
        failure_reason: str,
        terminal: bool,
        partial_response: Any = None,
    ) -> HandledEpisodeError:
        return HandledEpisodeError(
            SingleAgentTurnFailure(
                stage=stage,
                failure_reason=failure_reason[:2000],
                terminal=terminal,
                partial_response=partial_response,
            )
        )


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


def _is_retryable_dependency_error(error: Exception) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in {408, 425, 429} or error.status >= 500
    # A dropped connection mid-body raises ClientPayloadError, which is transient like a refused connection.
    return isinstance(error, (ClientConnectionError, ClientPayloadError, TimeoutError))


if __name__ == "__main__":
    SingleAgentTurnEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SingleAgentTurnEnvironmentServer.run_webserver()  # noqa: F401
