# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA episode orchestration with task services retained through verification."""

import asyncio
from collections.abc import Mapping
from time import time
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
    _cookies,
    _is_retryable_dependency_error,
)
from nemo_gym.base_environment_server import CleanupContext
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
from nemo_gym.episode_types import MaterializedTask
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from nemo_gym.single_agent_turn_types import (
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
    SingleAgentTurnResult,
    SingleAgentTurnTaskInput,
)
from nemo_gym.tool_access import DirectHTTPToolAccess, ToolAccess


class NOOASingleAgentTurnTaskInput(SingleAgentTurnTaskInput):
    """Add an execution budget without changing other agents' task contract."""

    agent_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def accept_flat_task_input(cls, value: object) -> object:
        """Keep the NOOA budget outside benchmark task data in either input shape."""
        if not isinstance(value, Mapping):
            return value
        fields = dict(value)
        timeout = fields.pop("agent_timeout_seconds", None)
        normalized = super().accept_flat_task_input(fields)
        assert isinstance(normalized, Mapping)
        return {**normalized, "agent_timeout_seconds": timeout}


class NOOASingleAgentTurnRequest(SingleAgentTurnRequest):
    """One NOOA episode with an optional agent execution deadline."""

    task: MaterializedTask[NOOASingleAgentTurnTaskInput]


class NOOASingleAgentTurnEnvironmentServerConfig(SingleAgentTurnEnvironmentServerConfig):
    """Bind the NOOA session protocol to one Resources Server."""

    resources_tool_transports: list[Literal["direct_http"]] = Field(default_factory=list)


class NOOASingleAgentTurnEnvironmentServer(SingleAgentTurnEnvironmentServer):
    """Finish NOOA before grading, then stop its services before resource teardown.

    Admission, episode deadlines, cancellation and bounded final cleanup use the
    unchanged base environment. Only successful results carry agent observations;
    failure persistence follows Gym's existing partial-response contract.
    """

    config: NOOASingleAgentTurnEnvironmentServerConfig
    request_model = NOOASingleAgentTurnRequest

    async def run(
        self,
        request: NOOASingleAgentTurnRequest,
        cleanup: CleanupContext,
    ) -> SingleAgentTurnResponse:
        """Run one NOOA activation, retaining its services until verification ends."""
        task_input = request.task.task_input

        resources_session_id = f"resources-session-{uuid4().hex}"
        resources_cookies: dict[str, str] = {}
        verification_completed = False

        async def close_resources() -> None:
            close_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close_session",
                json=ResourcesCloseSessionRequest(
                    resources_session_id=resources_session_id,
                    episode_id=request.episode_id,
                ).model_dump(mode="json"),
                cookies=resources_cookies,
            )
            await raise_for_status(close_response)

        # Register cleanup before seed so a lost seed response cannot hide the caller-assigned session ID.
        # Final cleanup closes this session after run() returns, outside the episode deadline.
        resources_cleanup = cleanup.register_cleanup("resources session", close_resources)

        async def retry_resources_after_verification() -> None:
            if verification_completed:
                await resources_cleanup.close()

        # Register the resource retry below both agent attempts in LIFO order.
        # Even a failed first agent close must retry before its sandbox is destroyed.
        cleanup.register_cleanup("post-verification resources session", retry_resources_after_verification)

        try:
            seed_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=ResourcesSeedSessionRequest(
                    resources_session_id=resources_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    task_data=task_input.task_data,
                ).model_dump(mode="json"),
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
                failure_reason=str(error),
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
        agent_session_id = f"agent-session-{uuid4().hex}"
        agent_cookies: dict[str, str] = {}

        # Cleanup callbacks return None.
        # Capture agent-owned observations and the final Resources Server cookie jar for the episode result.
        agent_close_response: AgentCloseSessionResponse | None = None
        agent_response = None

        async def collect_agent_close(*, finish: bool = False) -> None:
            nonlocal agent_close_response, resources_cookies
            close_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions/finish" if finish else "/v1/agent_sessions/close",
                json=AgentCloseSessionRequest(
                    agent_session_id=agent_session_id,
                    episode_id=request.episode_id,
                ).model_dump(mode="json"),
                cookies=agent_cookies,
            )
            await raise_for_status(close_http_response)
            receipt = AgentCloseSessionResponse.model_validate(await get_response_json(close_http_response))
            if receipt.agent_session_id != agent_session_id:
                raise ValueError("Agent close returned a different agent_session_id")
            agent_close_response = receipt
            if agent_close_response.resources_cookies is not None:
                resources_cookies = agent_close_response.resources_cookies

        async def close_agent() -> None:
            await collect_agent_close()

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
                ).model_dump(mode="json"),
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
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

        agent_timed_out = False
        agent_deadline = asyncio.timeout(task_input.agent_timeout_seconds)
        try:
            async with agent_deadline:
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
                agent_response = NeMoGymResponse.model_validate(await get_response_json(agent_http_response))
        except Exception as error:
            # An HTTP/socket timeout is still an infrastructure failure. Only this
            # task's explicit agent deadline authorizes grading the stopped state.
            if isinstance(error, TimeoutError) and agent_deadline.expired():
                agent_timed_out = True
            else:
                raise self._failure(
                    stage="agent",
                    failure_reason=str(error),
                    terminal=not _is_retryable_dependency_error(error),
                ) from error

        # Finish collects final cookies/evidence while retaining successful task services.
        # A deadline still requires bounded close before grading the stopped state.
        # Session-capable agents replay evidence when ServerClient retries a lost reply.
        try:
            if agent_timed_out:
                await agent_cleanup.close()
            else:
                async with asyncio.timeout(self.config.cleanup_timeout_seconds):
                    await collect_agent_close(finish=True)
        except Exception as error:
            raise self._failure(
                stage="cleanup",
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
                partial_response=agent_response,
            ) from error
        if agent_timed_out:
            observations = agent_close_response.agent_observations if agent_close_response is not None else None
            for gap in observations.gaps if observations is not None else []:
                if gap.code == "infrastructure_error":
                    # A lost activation reply can hide an already failed run.
                    # Close carries its evidence, but not the exception's retry classification.
                    raise self._failure(
                        stage="agent",
                        failure_reason=gap.detail or "NOOA execution failed before the agent deadline",
                        terminal=False,
                    )
            # The unchanged close contract confirms stop but carries no partial
            # response. Artifact-based verifiers receive an explicit empty envelope.
            agent_response = NeMoGymResponse(
                id=f"agent-timeout-{request.episode_id.capture_key}",
                created_at=time(),
                model="unknown",
                object="response",
                output=[],
                status="incomplete",
                metadata={"agent_timed_out": "true", "response_source": "nooa_environment_timeout_envelope"},
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            )
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
                cookies=resources_cookies,
            )
            await raise_for_status(verify_http_response)
            verification = SingleAgentTurnResult.model_validate(await get_response_json(verify_http_response))
            verification = verification.model_copy(
                update={
                    "agent_timed_out": agent_timed_out,
                    "agent_timeout_seconds": task_input.agent_timeout_seconds,
                }
            )
        except Exception as error:
            raise self._failure(
                stage="verification",
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
                partial_response=agent_response,
            ) from error

        # Keep one bounded retry in final unwind without erasing a completed verdict.
        verification_completed = True
        # LIFO unwind must close the agent before destroying its borrowed sandbox.
        cleanup.register_cleanup("post-verification agent session", agent_cleanup.close)
        return SingleAgentTurnResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=verification.model_copy(
                update={
                    "ng_agent_observations": agent_close_response.agent_observations
                    if agent_close_response is not None
                    else None
                }
            ),
        )


if __name__ == "__main__":
    NOOASingleAgentTurnEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NOOASingleAgentTurnEnvironmentServer.run_webserver()  # noqa: F401
