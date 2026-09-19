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

from nemo_gym.rollout_observability import ModelCallRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.gym_llm import GymResponsesLLM
from responses_api_agents.nooa_agent.mapping import materialize_arguments
from responses_api_agents.nooa_agent.resource_tools import (
    ResourceToolDispatcher,
    create_agent_class_with_resource_methods,
    validate_agent_resource_method_bindings,
)


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
    model_calls: list[ModelCallRef]
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
        model_calls: list[ModelCallRef] = []
        llm = GymResponsesLLM(
            server_client=self._server_client,
            model_server_name=self._model_server_name,
            model_url_path=request.model_url_path,
            max_steps=self._max_steps,
            model_call_collector=model_calls,
            cookies=request.model_cookies,
        )
        dispatcher = ResourceToolDispatcher(
            server_client=self._server_client,
            resources_server_name=self._resources_server_name,
            cookies=request.resource_cookies,
        )
        agent_class = create_agent_class_with_resource_methods(
            self._agent_class,
            dispatcher=dispatcher,
            tools=list(request.row.responses_create_params.tools),
        )
        agent = agent_class(llm=llm, **self._invocation.init_kwargs)
        validate_agent_resource_method_bindings(agent)

        arguments = materialize_arguments(request.row, self._invocation.arguments)
        entrypoint = getattr(agent, self._invocation.entrypoint)
        return_value = await entrypoint(**arguments)
        return NOOARunResult(
            return_value=return_value,
            agent=agent,
            model_calls=model_calls,
            model_cookies=request.model_cookies,
            resource_cookies=request.resource_cookies,
        )
