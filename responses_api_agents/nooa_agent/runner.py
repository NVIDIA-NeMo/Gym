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
from uuid import uuid4

from nooa.runtime.hooks import hooks_scope

from nemo_gym.rollout_observability import AgentEpisode, TrajectoryRecord
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.gym_llm import (
    GymResponsesLLM,
    InvalidPolicyOutputError,
    PolicyCallBudgetExceeded,
    RolloutLLMState,
)
from responses_api_agents.nooa_agent.mapping import materialize_arguments
from responses_api_agents.nooa_agent.observability import GymTraceHooks
from responses_api_agents.nooa_agent.resource_tools import (
    ResourceToolDispatcher,
    create_agent_class_with_resource_methods,
    validate_agent_resource_method_bindings,
)


@dataclass(slots=True)
class NOOARunRequest:
    row: Any
    model_url_path: str
    model_cookies: dict[str, str] = field(default_factory=dict)
    resource_cookies: dict[str, str] = field(default_factory=dict)
    task_id: str = "unknown"
    rollout_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(slots=True)
class NOOARunResult:
    episode: AgentEpisode
    return_value: Any
    model_cookies: dict[str, str]
    resource_cookies: dict[str, str]
    termination_reason: str | None = None
    termination_error: str | None = None
    trajectory: TrajectoryRecord | None = None


class NOOARunFailure(RuntimeError):
    """A failure carrying the same episode representation as a successful run."""

    def __init__(self, error: BaseException, result: NOOARunResult) -> None:
        super().__init__(str(error))
        self.result = result


class NOOARunner(Protocol):
    async def run(self, request: NOOARunRequest) -> NOOARunResult: ...


class ArgumentMappingError(ValueError):
    """Raised when a Gym row cannot supply the configured NOOA entrypoint arguments."""


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
        state = RolloutLLMState(max_steps=self._max_steps)
        trace = GymTraceHooks()
        sampling_overrides = request.row.responses_create_params.model_dump(
            include={"temperature", "top_p", "max_output_tokens"}, exclude_unset=True, exclude_none=True
        )
        llm = GymResponsesLLM(
            server_client=self._server_client,
            model_server_name=self._model_server_name,
            model_url_path=request.model_url_path,
            state=state,
            cookies=request.model_cookies,
            on_call=trace.on_model_call,
            sampling_overrides=sampling_overrides,
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
            tools=list(request.row.responses_create_params.tools),
        )
        agent = agent_class(llm=llm, **self._invocation.init_kwargs)
        validate_agent_resource_method_bindings(agent)

        try:
            arguments = materialize_arguments(request.row, self._invocation.arguments)
        except ValueError as error:
            raise ArgumentMappingError(str(error)) from error
        entrypoint = getattr(agent, self._invocation.entrypoint)
        termination_reason = None
        termination_error = None
        return_value = None
        failure: BaseException | None = None
        try:
            with hooks_scope(trace):
                return_value = await entrypoint(**arguments)
        except BaseException as error:
            failure = error
            # NOOA may wrap a model error after its strategy retry loop.
            cause: BaseException | None = error
            seen: set[int] = set()
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                if isinstance(cause, (PolicyCallBudgetExceeded, InvalidPolicyOutputError)):
                    termination_reason = (
                        "policy_budget_exceeded"
                        if isinstance(cause, PolicyCallBudgetExceeded)
                        else "invalid_policy_output"
                    )
                    termination_error = str(cause)
                    break
                cause = cause.__cause__ or cause.__context__

        episode, trajectory = trace.project(
            create_params=request.row.responses_create_params,
            state=state,
            task_id=request.task_id,
            rollout_id=request.rollout_id,
        )
        result = NOOARunResult(
            episode=episode,
            return_value=return_value,
            model_cookies=request.model_cookies,
            resource_cookies=request.resource_cookies,
            termination_reason=termination_reason,
            termination_error=termination_error,
            trajectory=trajectory,
        )
        if failure is not None and termination_reason is None:
            if isinstance(failure, asyncio.CancelledError):
                # Preserve asyncio.timeout's conversion of the original cancellation.
                failure.nooa_result = result
                raise failure
            if not isinstance(failure, Exception):
                raise failure
            raise NOOARunFailure(failure, result) from failure
        return result
