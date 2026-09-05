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

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Protocol

from nooa.runtime.hooks import hooks_scope

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentEpisode
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.gym_llm import GymResponsesLLM, PolicyCallBudgetExceeded, RolloutLLMState
from responses_api_agents.nooa_agent.observability import GymTraceHooks
from responses_api_agents.nooa_agent.resource_tools import (
    ResourceToolDispatcher,
    create_agent_class_with_resource_methods,
    validate_agent_resource_method_bindings,
)


@dataclass(slots=True)
class NOOARunRequest:
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    rollout_id: str
    model_url_path: str
    model_cookies: dict[str, str] = field(default_factory=dict)
    resource_cookies: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class NOOARunResult:
    episode: AgentEpisode
    return_value: Any
    model_cookies: dict[str, str]
    resource_cookies: dict[str, str]
    termination_reason: str | None = None
    termination_error: str | None = None


class NOOARunner(Protocol):
    async def run(self, request: NOOARunRequest) -> NOOARunResult: ...


class EmbeddedNOOARunner:
    """Construct and invoke one isolated NOOA agent instance per Gym rollout."""

    def __init__(
        self,
        *,
        invocation: NOOAInvocationConfig,
        server_client: ServerClient,
        model_server_name: str,
        resources_server_name: str,
        max_policy_calls: int,
    ) -> None:
        if invocation.execution_mode == "sandboxed":
            raise NotImplementedError("NOOA sandboxed execution is not implemented")
        self._invocation = invocation
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._resources_server_name = resources_server_name
        self._max_policy_calls = max_policy_calls
        self._agent_class, self._invocation_adapter = validate_invocation(invocation)

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        state = RolloutLLMState(max_policy_calls=self._max_policy_calls)
        trace = GymTraceHooks()
        llm = GymResponsesLLM(
            server_client=self._server_client,
            model_server_name=self._model_server_name,
            model_url_path=request.model_url_path,
            state=state,
            cookies=request.model_cookies,
            on_call=trace.on_model_call,
        )
        dispatcher = ResourceToolDispatcher(
            server_client=self._server_client,
            resources_server_name=self._resources_server_name,
            cookies=request.resource_cookies,
            trace_hooks=trace,
        )
        agent_class = create_agent_class_with_resource_methods(
            self._agent_class,
            dispatcher=dispatcher,
            tools=list(request.responses_create_params.tools),
        )
        agent = agent_class(llm=llm, **copy.deepcopy(self._invocation.init_kwargs))
        validate_agent_resource_method_bindings(agent)

        termination_reason = None
        termination_error = None
        return_value = None
        with hooks_scope(trace):
            try:
                return_value = await self._invocation_adapter(agent, request.responses_create_params)
            except PolicyCallBudgetExceeded as error:
                termination_reason = "policy_budget_exceeded"
                termination_error = str(error)
            except ValueError as error:
                message = str(error)
                if "Gym model returned invalid" in message:
                    termination_reason = "invalid_policy_output"
                    termination_error = message
                else:
                    raise

        episode = trace.project(
            create_params=request.responses_create_params,
            state=state,
            default_model=self._model_server_name,
        )
        return NOOARunResult(
            episode=episode,
            return_value=return_value,
            model_cookies=request.model_cookies,
            resource_cookies=request.resource_cookies,
            termination_reason=termination_reason,
            termination_error=termination_error,
        )
