# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

from fastapi import Body, HTTPException, Request
from pydantic import ConfigDict

from nemo_gym.agents import (
    ActivationConflictError,
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    OrderedOperationLedger,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseUsage
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap


def response_output_items(policy_outputs: list[Any]) -> list[Any]:
    """Return only items that are valid members of a Responses API output."""
    input_roles = {"user", "system", "developer"}
    return [item for item in policy_outputs if getattr(item, "role", None) not in input_roles]


def _aggregate_usage_values(values: list[Any]) -> Any:
    """Recursively add numeric usage fields while preserving provider-specific details."""
    values = [value for value in values if value is not None]
    if not values:
        return None
    if all(isinstance(value, bool) for value in values):
        return values[-1]
    if all(isinstance(value, Number) and not isinstance(value, bool) for value in values):
        return sum(values)
    if all(isinstance(value, Mapping) for value in values):
        keys = dict.fromkeys(key for value in values for key in value)
        return {key: _aggregate_usage_values([value.get(key) for value in values]) for key in keys}
    if all(isinstance(value, list) for value in values):
        return [item for value in values for item in value]
    return values[-1]


def aggregate_response_usage(usages: list[NeMoGymResponseUsage | None]) -> NeMoGymResponseUsage | None:
    """Aggregate usage from all model calls represented by one agent response."""
    if not usages or any(usage is None for usage in usages):
        return None
    usage = _aggregate_usage_values([item.model_dump(mode="python") for item in usages])
    return NeMoGymResponseUsage.model_validate(usage)


class LangGraphAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef | None = None


class LangGraphRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class LangGraphVerifyResponse(BaseVerifyResponse):
    # extra="allow" so the verifier's passthrough fields survive — notably the
    # `_ng_failure_class` routing key, which rollout_collection needs to divert a
    # judge failure to the sidecar instead of scoring it. Matches the subagents.
    model_config = ConfigDict(extra="allow")


@dataclass
class _LangGraphSession(AgentSessionState):
    operations: OrderedOperationLedger[NeMoGymResponse] = field(default_factory=OrderedOperationLedger)
    activation_path: str | None = None


class LangGraphAgentAdapter(SimpleResponsesAPIAgent):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    config: LangGraphAgentConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = self.build_graph()

    @abstractmethod
    def build_graph(self) -> Any:
        pass

    @abstractmethod
    async def get_initial_state(self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict) -> dict:
        pass

    @abstractmethod
    def extract_outputs(self, final_state: dict) -> list:
        pass

    def extract_model_response(self, final_state: dict) -> NeMoGymResponse:
        if "last_policy_response" in final_state:
            return final_state["last_policy_response"]
        raise NotImplementedError("State must contain 'last_policy_response' or override extract_model_response()")

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        return await self._run_graph_session(
            request,
            body,
            operation=lambda: self._invoke_graph(
                body.model_copy(deep=True), model_url_path=self.url_path_for_request("/v1/responses", request)
            ),
        )

    async def _invoke_graph(
        self, body: NeMoGymResponseCreateParamsNonStreaming, *, model_url_path: str
    ) -> NeMoGymResponse:
        # Model cookies are independent of the agent and Resources Server session jars.
        initial_state = await self.get_initial_state(body, {})
        initial_state["model_url_path"] = model_url_path
        final_state = await self.graph.ainvoke(initial_state)

        model_response = self.extract_model_response(final_state)
        outputs = self.extract_outputs(final_state)
        model_response.output = outputs
        if "policy_usages" in final_state:
            model_response.usage = aggregate_response_usage(final_state["policy_usages"])
        return model_response

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("This agent does not support sandbox access")
        required = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required:
            raise AgentSeedRejectedError(f"This agent does not support required tool access: {', '.join(required)}")
        return _LangGraphSession(request=body)

    async def _run_graph_session(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        operation: Callable[[], Awaitable[NeMoGymResponse]],
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            return await operation()
        state = self._require_agent_session(session_id)
        if not isinstance(state, _LangGraphSession):
            raise TypeError("Expected LangGraph session state")
        rollout_id = request.path_params.get("rollout_id")
        if rollout_id != state.request.episode_id.capture_key:
            raise HTTPException(409, "Activation requires the seeded session's rollout path")
        if state.activation_path is None:
            state.activation_path = request.url.path
        elif state.activation_path != request.url.path:
            raise HTTPException(409, "Activation path does not match the session's first activation")
        try:
            return await state.operations.execute(index=1, request=body, operation=operation)
        except ActivationConflictError as error:
            raise HTTPException(409, str(error)) from error

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, _LangGraphSession):
            raise TypeError("Expected LangGraph session state")
        await state.operations.close(timeout=30.0)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(
                source=self.config.name,
                gaps=[
                    ObservationGap(
                        code="observation_capture_failed",
                        detail="This model-only adapter does not record native invocations or turns.",
                    )
                ],
            ),
        )

    async def run(self, request: Request, body: LangGraphRunRequest) -> LangGraphVerifyResponse:
        if self.config.resources_server is None:
            raise HTTPException(422, "Legacy /run requires resources_server; use agent sessions for composition")
        cookies = request.cookies

        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed)
        cookies = seed.cookies

        resp = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
        )
        await raise_for_status(resp)

        verify_request_dict = body.model_dump() | {"response": await get_response_json(resp)}

        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request_dict,
            cookies=cookies,
        )
        await raise_for_status(verify)
        return LangGraphVerifyResponse.model_validate(await get_response_json(verify))
