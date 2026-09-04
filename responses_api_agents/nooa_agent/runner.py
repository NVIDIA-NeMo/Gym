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

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

from nooa import Agent
from nooa.runtime.hooks import hooks_scope
from nooa.tracing import session_scope

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import ObservationGap
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.gym_llm import (
    GymResponsesLLM,
    InvalidPolicyOutputError,
    PolicyCallBudgetExceeded,
)
from responses_api_agents.nooa_agent.gym_tools import GymToolExecution, build_tool_namespace
from responses_api_agents.nooa_agent.mapping import materialize_arguments
from responses_api_agents.nooa_agent.observability import GymTraceHooks, NOOATraceSnapshot


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
    model_requests: list[NeMoGymResponseCreateParamsNonStreaming]
    model_responses: list[NeMoGymResponse]
    tool_executions: list[GymToolExecution]
    model_cookies: dict[str, str]
    resource_cookies: dict[str, str]
    trace: NOOATraceSnapshot
    completed: bool = False
    termination_reason: str | None = None
    termination_error: str | None = None
    observation_gaps: list[ObservationGap] = field(default_factory=list)


class NOOARunFailure(RuntimeError):
    """An execution failure carrying evidence collected before it failed."""

    def __init__(self, error: Exception, result: NOOARunResult) -> None:
        super().__init__(str(error))
        self.error = error
        self.result = result


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
        if hasattr(self._agent_class, invocation.tool_namespace):
            raise ValueError(f"tool_namespace {invocation.tool_namespace!r} collides with an existing agent attribute")

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        model_requests: list[NeMoGymResponseCreateParamsNonStreaming] = []
        responses: list[NeMoGymResponse] = []
        executions: list[GymToolExecution] = []
        observation_gaps: list[ObservationGap] = []
        trace_hooks = GymTraceHooks(ModelServerRef(type="responses_api_models", name=self._model_server_name))
        llm = GymResponsesLLM(
            server_client=self._server_client,
            model_server_name=self._model_server_name,
            model_url_path=request.model_url_path,
            max_steps=self._max_steps,
            request_collector=model_requests,
            response_collector=responses,
            cookies=request.model_cookies,
            trace_hooks=trace_hooks,
            observation_gaps=observation_gaps,
        )
        tool_namespace = self._invocation.tool_namespace
        tools = build_tool_namespace(
            namespace_name=tool_namespace,
            server_client=self._server_client,
            resources_server_name=self._resources_server_name,
            tools=list(request.row.responses_create_params.tools),
            allowed_tools=frozenset(self._invocation.allowed_tools),
            cookies=request.resource_cookies,
            observations=executions,
            trace_hooks=trace_hooks,
        )
        agent_class = type(
            self._agent_class.__name__,
            (self._agent_class,),
            {"__annotations__": {tool_namespace: type(tools)}},
        )
        agent = agent_class(llm=llm, **self._invocation.init_kwargs)
        if tool_namespace in vars(agent):
            raise ValueError(f"tool_namespace {tool_namespace!r} collides with an existing agent attribute")
        setattr(agent, tool_namespace, tools)

        arguments = materialize_arguments(request.row, self._invocation.arguments)
        entrypoint = getattr(agent, self._invocation.entrypoint)

        def snapshot(
            return_value: Any = None,
            *,
            completed: bool = False,
            termination_reason: str | None = None,
            termination_error: str | None = None,
        ) -> NOOARunResult:
            return NOOARunResult(
                return_value=return_value,
                agent=agent,
                model_requests=model_requests,
                model_responses=responses,
                tool_executions=executions,
                model_cookies=request.model_cookies,
                resource_cookies=request.resource_cookies,
                trace=trace_hooks.snapshot(),
                completed=completed,
                termination_reason=termination_reason,
                termination_error=termination_error,
                observation_gaps=observation_gaps,
            )

        try:
            with session_scope(request.rollout_id), hooks_scope(trace_hooks):
                return_value = await entrypoint(**arguments)
        except PolicyCallBudgetExceeded as error:
            return snapshot(
                termination_reason="policy_budget_exceeded",
                termination_error=str(error),
            )
        except InvalidPolicyOutputError as error:
            return snapshot(
                termination_reason="invalid_policy_output",
                termination_error=str(error),
            )
        except asyncio.CancelledError as error:
            error.nooa_run_result = snapshot()  # type: ignore[attr-defined]
            raise
        except Exception as error:
            raise NOOARunFailure(error, snapshot()) from error
        return snapshot(return_value, completed=True)
