# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources-backed single-agent environment server."""

from typing import Any, Literal
from uuid import uuid4

from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import Body
from pydantic import ConfigDict, Field

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
    ResourcesVerifyResponse,
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
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.single_agent_turn_types import (
    SingleAgentTurnFailure,
    SingleAgentTurnRequest,
    SingleAgentTurnResourcesVerifyRequest,
    SingleAgentTurnResponse,
    SingleAgentTurnResult,
    SingleAgentTurnVerificationInput,
)
from nemo_gym.tool_access import (
    DirectHTTPToolAccess,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    ToolAccess,
)


class SingleAgentTurnEnvironmentServerConfig(BaseEnvironmentServerConfig):
    """Bind Resources and Agent Servers for one agent turn."""

    model_config = ConfigDict(extra="forbid")

    resources_server: ResourcesServerRef
    agent_server: AgentServerRef
    resources_tool_transports: list[Literal["direct_http", "mcp"]] = Field(default_factory=list)


class SingleAgentTurnEnvironmentServer(BaseEnvironmentServer[SingleAgentTurnRequest, SingleAgentTurnResponse]):
    """Run one agent turn followed by Resources verification and cleanup."""

    config: SingleAgentTurnEnvironmentServerConfig
    request_model = SingleAgentTurnRequest
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
        task_input = request.task.task_input

        resources_session_id = f"resources-session-{uuid4().hex}"
        resources_cookies: dict[str, str] = {}

        async def close_resources() -> None:
            close_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close_session",
                json=ResourcesCloseSessionRequest(
                    resources_session_id=resources_session_id,
                    episode_id=request.episode_id,
                ),
                cookies=resources_cookies,
            )
            await raise_for_status(close_response)

        # Register cleanup before seed so a lost seed response cannot hide the caller-assigned session ID.
        resources_cleanup = cleanup.register_cleanup("resources session", close_resources)

        try:
            seed_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=ResourcesSeedSessionRequest(
                    resources_session_id=resources_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    task_data=task_input.task_data,
                ),
            )
            await raise_for_status(seed_http_response)
            resources_cookies = _cookies(seed_http_response)
            if not resources_cookies:
                raise ValueError("Resources seed did not establish a session cookie")
            seed = ResourcesSeedSessionResponse.model_validate(await get_response_json(seed_http_response))
            if seed.resources_session_id != resources_session_id:
                raise ValueError("Resources seed returned a different resources_session_id")
        except Exception as error:
            raise self._failure(
                stage="seed",
                message=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

        resources_base_url = self.server_client._resolve_base_url(self.config.resources_server.name).rstrip("/")
        tool_accesses: list[ToolAccess] = []
        if "direct_http" in self.config.resources_tool_transports:
            tool_accesses.append(
                DirectHTTPToolAccess(
                    name=f"{self.config.resources_server.name}.direct_http",
                    required=True,
                    base_url=resources_base_url,
                    cookies=resources_cookies,
                )
            )
        if "mcp" in self.config.resources_tool_transports:
            if seed.resources_tools is None:
                raise self._failure(
                    stage="seed",
                    message="Resources seed did not return requested MCP metadata",
                    terminal=True,
                )
            if seed.resources_tools.transport != "http":
                raise self._failure(
                    stage="seed",
                    message=f"Unsupported resources MCP transport: {seed.resources_tools.transport}",
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
        agent_session_id = f"agent-session-{uuid4().hex}"
        agent_cookies: dict[str, str] = {}

        # Cleanup callbacks return None.
        # Capture agent-owned observations and the final Resources Server cookie jar for the episode result.
        agent_close_response: AgentCloseSessionResponse | None = None

        async def close_agent() -> None:
            nonlocal agent_close_response, resources_cookies
            close_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions/close",
                json=AgentCloseSessionRequest(
                    agent_session_id=agent_session_id,
                    episode_id=request.episode_id,
                ),
                cookies=agent_cookies,
            )
            await raise_for_status(close_http_response)
            agent_close_response = AgentCloseSessionResponse.model_validate(
                await get_response_json(close_http_response)
            )
            if agent_close_response.agent_session_id != agent_session_id:
                raise ValueError("Agent close returned a different agent_session_id")
            if agent_close_response.resources_cookies is not None:
                resources_cookies = agent_close_response.resources_cookies

        # Register cleanup before seed so cancellation can close a remotely created session even if its response is lost.
        agent_cleanup = cleanup.register_cleanup("agent session", close_agent)

        try:
            agent_create_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                json=AgentSeedSessionRequest(
                    agent_session_id=agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    tool_accesses=tool_accesses,
                    sandbox_access=seed.sandbox_access,
                ),
            )
            await raise_for_status(agent_create_http_response)
            agent_session = AgentSeedSessionResponse.model_validate(
                await get_response_json(agent_create_http_response)
            )
            if agent_session.agent_session_id != agent_session_id:
                raise ValueError("Agent seed returned a different agent_session_id")
            agent_cookies = _cookies(agent_create_http_response)
        except Exception as error:
            raise self._failure(
                stage="agent",
                message=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

        agent_response = None
        try:
            agent_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path=self._agent_responses_path(request),
                json=task_input.responses_create_params,
                cookies=agent_cookies,
            )
            await raise_for_status(agent_http_response)
            response_cookies = _cookies(agent_http_response)
            if response_cookies:
                agent_cookies = response_cookies
            from nemo_gym.openai_utils import NeMoGymResponse

            agent_response = NeMoGymResponse.model_validate(await get_response_json(agent_http_response))
        except Exception as error:
            raise self._failure(
                stage="agent",
                message=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

        try:
            await agent_cleanup.close()
        except Exception as error:
            raise self._failure(
                stage="cleanup",
                message=str(error),
                terminal=True,
                partial_response=agent_response,
            ) from error
        try:
            verify_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=SingleAgentTurnResourcesVerifyRequest(
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    verification_input=SingleAgentTurnVerificationInput(
                        responses_create_params=task_input.responses_create_params,
                        response=agent_response,
                    ),
                ),
                cookies=resources_cookies,
            )
            await raise_for_status(verify_http_response)
            verification = ResourcesVerifyResponse.model_validate(await get_response_json(verify_http_response))
        except Exception as error:
            raise self._failure(
                stage="verification",
                message=str(error),
                terminal=not _is_retryable_dependency_error(error),
                partial_response=agent_response,
            ) from error

        try:
            await resources_cleanup.close()
        except Exception as error:
            raise self._failure(
                stage="cleanup",
                message=str(error),
                terminal=True,
                partial_response=agent_response,
            ) from error

        return SingleAgentTurnResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=SingleAgentTurnResult(
                verification=verification,
                agent_observations=agent_close_response.agent_observations
                if agent_close_response is not None
                else None,
            ),
        )

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
        message: str,
        terminal: bool,
        partial_response: Any = None,
    ) -> HandledEpisodeError:
        return HandledEpisodeError(
            SingleAgentTurnFailure(
                stage=stage,
                message=message[:2000],
                terminal=terminal,
                partial_response=partial_response,
            )
        )


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


def _is_retryable_dependency_error(error: Exception) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in {408, 425, 429} or error.status >= 500
    return isinstance(error, (ClientConnectionError, TimeoutError))


if __name__ == "__main__":
    SingleAgentTurnEnvironmentServer.run_webserver()
