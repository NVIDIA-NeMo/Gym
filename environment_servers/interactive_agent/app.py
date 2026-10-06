# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources-orchestrated native agent continuation without benchmark policy."""

from typing import Any
from uuid import uuid4

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
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
    ResourcesVerifyRequest,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.config_types import AgentServerRef, AggregateMetrics, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.interactive_agent_types import (
    AgentActivationRequest,
    AgentActivationResponse,
    InteractiveAgentFailure,
    InteractiveAgentRequest,
    InteractiveAgentResponse,
    InteractiveAgentResult,
    InteractiveResourcesSeedResponse,
    InteractiveVerificationInput,
    ResourcesStepRequest,
    ResourcesStepResponse,
)
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status


class InteractiveAgentEnvironmentServerConfig(BaseEnvironmentServerConfig):
    """Compose independent Resources and continuation-capable candidate definitions."""

    model_config = ConfigDict(extra="forbid")
    resources_server: ResourcesServerRef
    agent_server: AgentServerRef
    max_activations: int = Field(
        default=1000, ge=1, description="Safety fence; benchmark stopping belongs to Resources."
    )


class InteractiveAgentEnvironmentServer(BaseEnvironmentServer[InteractiveAgentRequest, InteractiveAgentResponse]):
    """Drive seed, ordered activations and steps, confirmed agent close, then verification."""

    config: InteractiveAgentEnvironmentServerConfig
    request_model = InteractiveAgentRequest
    response_model = InteractiveAgentResponse

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/aggregate_metrics", json=body
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def run(self, request: InteractiveAgentRequest, cleanup: CleanupContext) -> InteractiveAgentResponse:
        resources_session_id = f"resources-session-{uuid4().hex}"
        agent_session_id = f"agent-session-{uuid4().hex}"
        resources_cookies: dict[str, str] = {}
        agent_cookies: dict[str, str] = {}
        agent_close: AgentCloseSessionResponse | None = None
        activations: list[AgentActivationResponse] = []
        steps: list[ResourcesStepResponse] = []
        stage = "seed"

        async def close_resources() -> None:
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close_session",
                json=ResourcesCloseSessionRequest(
                    resources_session_id=resources_session_id, episode_id=request.episode_id
                ).model_dump(mode="json"),
                cookies=resources_cookies,
            )
            await raise_for_status(response)

        resources_cleanup = cleanup.register_cleanup("resources session", close_resources)

        async def close_agent() -> None:
            nonlocal agent_close
            response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions/close",
                json=AgentCloseSessionRequest(
                    agent_session_id=agent_session_id, episode_id=request.episode_id
                ).model_dump(mode="json"),
                cookies=agent_cookies,
            )
            await raise_for_status(response)
            receipt = AgentCloseSessionResponse.model_validate(await get_response_json(response))
            if receipt.agent_session_id != agent_session_id:
                raise ValueError("Agent close returned a different session ID")
            if not receipt.cleanup_confirmed:
                raise ValueError("Agent close did not confirm cleanup")
            agent_close = receipt
            if receipt.resources_cookies is not None:
                resources_cookies.update(receipt.resources_cookies)
            agent_cookies.update(_cookies(response))

        try:
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=ResourcesSeedSessionRequest(
                    resources_session_id=resources_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    task_data=request.task.task_input.task_data,
                ).model_dump(mode="json"),
            )
            await raise_for_status(response)
            resources_cookies.update(_cookies(response))
            seed = InteractiveResourcesSeedResponse.model_validate(await get_response_json(response))
            if seed.resources_session_id != resources_session_id or not resources_cookies:
                raise ValueError("Resources seed must return the requested session ID and a session cookie")
            # Registration precedes setup: a lost seed reply must not leak a remote candidate session.
            agent_cleanup = cleanup.register_cleanup("agent session", close_agent)
            stage = "agent"
            response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                json=AgentSeedSessionRequest(
                    agent_session_id=agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    sandbox_access=seed.sandbox_access,
                    continuation=seed.continuation,
                    runtime_policy=seed.runtime_policy,
                ).model_dump(mode="json"),
            )
            await raise_for_status(response)
            agent_cookies.update(_cookies(response))
            agent_seed = AgentSeedSessionResponse.model_validate(await get_response_json(response))
            if agent_seed.agent_session_id != agent_session_id or not agent_cookies:
                raise ValueError("Agent seed must return the requested session ID and a session cookie")
            capabilities = agent_seed.capabilities
            if capabilities is None or capabilities.mode != seed.continuation.mode:
                raise ValueError("Agent seed did not confirm native continuation support")
            if set(seed.continuation.observations) - set(capabilities.observations):
                raise ValueError("Agent seed did not confirm all required observation capabilities")
            next_input = seed.responses_create_params
            while True:
                stage = "agent"
                activation_id = len(activations)
                if activation_id >= self.config.max_activations:
                    raise ValueError("Resources did not stop before the environment activation safety limit")
                response = await self.server_client.post(
                    server_name=self.config.agent_server.name,
                    url_path=f"/ng-rollout/{request.episode_id.capture_key}/v1/agent_sessions/activate",
                    json=AgentActivationRequest(
                        agent_session_id=agent_session_id,
                        episode_id=request.episode_id,
                        activation_id=activation_id,
                        responses_create_params=next_input,
                    ).model_dump(mode="json"),
                    cookies=agent_cookies,
                )
                await raise_for_status(response)
                agent_cookies.update(_cookies(response))
                activation = AgentActivationResponse.model_validate(await get_response_json(response))
                if activation.activation_id != activation_id:
                    raise ValueError("Agent returned an unexpected activation ID")
                activations.append(activation)
                stage = "step"
                response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/step",
                    json=ResourcesStepRequest(
                        resources_session_id=resources_session_id,
                        episode_id=request.episode_id,
                        activation=activation,
                    ).model_dump(mode="json"),
                    cookies=resources_cookies,
                )
                await raise_for_status(response)
                resources_cookies.update(_cookies(response))
                step = ResourcesStepResponse.model_validate(await get_response_json(response))
                if step.activation_id != activation_id:
                    raise ValueError("Resources returned an unexpected activation ID")
                steps.append(step)
                if not step.continue_episode:
                    break
                assert step.responses_create_params is not None
                next_input = step.responses_create_params

            stage = "cleanup"
            await agent_cleanup.close()
            assert agent_close is not None
            if agent_close.activations != activations:
                raise ValueError("Agent cumulative close receipt disagrees with activation history")
            stage = "verification"
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=ResourcesVerifyRequest[InteractiveVerificationInput](
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    verification_input=InteractiveVerificationInput(
                        resources_session_id=resources_session_id,
                        responses_create_params=seed.responses_create_params,
                        activations=activations,
                        steps=steps,
                        agent_close=agent_close,
                    ),
                ).model_dump(mode="json"),
                cookies=resources_cookies,
            )
            await raise_for_status(response)
            verification = InteractiveAgentResult.model_validate(await get_response_json(response))
        except Exception as error:
            raise HandledEpisodeError(
                InteractiveAgentFailure(
                    stage=stage,
                    failure_reason=str(error)[:2000],
                    terminal=not _is_retryable_dependency_error(error),
                    partial_response=activations[-1].response if activations else None,
                    activations=activations,
                    steps=steps,
                    agent_close=agent_close,
                )
            ) from error
        cleanup.register_cleanup("post-verification resources session", resources_cleanup.close)
        return InteractiveAgentResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=verification.model_copy(
                update={
                    "ng_activations": activations,
                    "ng_steps": steps,
                    "ng_agent_close": agent_close,
                    "ng_agent_observations": agent_close.agent_observations,
                }
            ),
        )


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


def _is_retryable_dependency_error(error: Exception) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in {408, 425, 429} or error.status >= 500
    return isinstance(error, (ClientConnectionError, ClientPayloadError, TimeoutError))


if __name__ == "__main__":
    InteractiveAgentEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = InteractiveAgentEnvironmentServer.run_webserver()
