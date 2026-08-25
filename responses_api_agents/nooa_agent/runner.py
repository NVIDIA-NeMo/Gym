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

from dataclasses import dataclass, field
from typing import Any, Protocol

from nooa import Agent

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.gym_llm import GymResponsesLLM
from responses_api_agents.nooa_agent.gym_tools import GymToolExecution, GymTools
from responses_api_agents.nooa_agent.mapping import materialize_arguments


@dataclass(slots=True)
class NOOARunRequest:
    row: Any
    rollout_id: str
    task_id: str
    model_url_path: str
    model_cookies: dict[str, str] = field(default_factory=dict)
    resource_cookies: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class NOOARunResult:
    return_value: Any
    agent: Agent
    model_responses: list[NeMoGymResponse]
    tool_executions: list[GymToolExecution]
    model_cookies: dict[str, str]
    resource_cookies: dict[str, str]


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
        max_steps: int,
    ) -> None:
        self._invocation = invocation
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._resources_server_name = resources_server_name
        self._max_steps = max_steps
        self._agent_class, _ = validate_invocation(invocation)

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        responses: list[NeMoGymResponse] = []
        executions: list[GymToolExecution] = []
        llm = GymResponsesLLM(
            server_client=self._server_client,
            model_server_name=self._model_server_name,
            model_url_path=request.model_url_path,
            max_steps=self._max_steps,
            response_collector=responses,
            cookies=request.model_cookies,
        )
        agent = self._agent_class(llm=llm, **self._invocation.init_kwargs)
        tools = GymTools(
            server_client=self._server_client,
            resources_server_name=self._resources_server_name,
            tools=list(request.row.responses_create_params.tools),
            cookies=request.resource_cookies,
            observations=executions,
        )
        agent.gym_tools = tools

        arguments = materialize_arguments(request.row, self._invocation.arguments)
        entrypoint = getattr(agent, self._invocation.entrypoint)
        return_value = await entrypoint(**arguments)
        return NOOARunResult(
            return_value=return_value,
            agent=agent,
            model_responses=responses,
            tool_executions=executions,
            model_cookies=request.model_cookies,
            resource_cookies=request.resource_cookies,
        )
