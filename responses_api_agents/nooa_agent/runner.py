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
import uuid
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
    RolloutCallBudget,
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
    sandbox_handle: str | None = None


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

    def _build_alias_clients(
        self,
        *,
        request: NOOARunRequest,
        trace_hooks: GymTraceHooks,
        model_requests: list[NeMoGymResponseCreateParamsNonStreaming],
        responses: list[NeMoGymResponse],
        observation_gaps: list[ObservationGap],
        budget: RolloutCallBudget,
        prior_outputs: list[dict[str, Any]],
    ) -> dict[str, GymResponsesLLM]:
        """Build one Gym-backed client per configured NOOA model string.

        Alias clients share the rollout's call budget, request/response
        collectors, and trace hooks with the primary client; only the target
        model server and the NOOA model string differ. Each gets its own
        cookie-jar copy so one server's session cookies never leak to
        another; only the primary client's cookies flow back into the
        rollout result.

        Model strings that are *not* configured here still resolve through
        NOOA's own registry, outside Gym's boundary — keep that registry
        empty (or keyless) in Gym rollouts to preserve the trust boundary.
        """
        clients: dict[str, GymResponsesLLM] = {}
        for alias, server_name in self._invocation.model_aliases.items():
            clients[alias] = GymResponsesLLM(
                server_client=self._server_client,
                model_server_name=server_name,
                model_url_path=request.model_url_path,
                max_steps=self._max_steps,
                request_collector=model_requests,
                response_collector=responses,
                cookies=dict(request.model_cookies),
                trace_hooks=trace_hooks,
                observation_gaps=observation_gaps,
                model=alias,
                model_ref=ModelServerRef(type="responses_api_models", name=server_name),
                budget=budget,
                prior_outputs=prior_outputs,
            )
        return clients

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        model_requests: list[NeMoGymResponseCreateParamsNonStreaming] = []
        responses: list[NeMoGymResponse] = []
        executions: list[GymToolExecution] = []
        observation_gaps: list[ObservationGap] = []
        trace_hooks = GymTraceHooks(
            ModelServerRef(type="responses_api_models", name=self._model_server_name),
            task_id=request.task_id,
            rollout_id=request.rollout_id,
        )
        # One rollout-wide budget and one exact-output ledger: the primary
        # client and every alias client (per-method model strings) draw from the
        # same allowance and restore one another's prior outputs.
        budget = RolloutCallBudget(self._max_steps)
        prior_outputs: list[dict[str, Any]] = []
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
            budget=budget,
            prior_outputs=prior_outputs,
        )
        alias_clients = self._build_alias_clients(
            request=request,
            trace_hooks=trace_hooks,
            model_requests=model_requests,
            responses=responses,
            observation_gaps=observation_gaps,
            budget=budget,
            prior_outputs=prior_outputs,
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
        if request.sandbox_handle is not None:
            # Seeded-sandbox environments (e.g. SWE-bench) return the container
            # id from /seed_session. Attach an exec-only sandbox so agent classes
            # that opt in (via the ``_gym_sandbox`` class attribute) run and edit
            # inside the verifier-owned container.
            from responses_api_agents.nooa_agent.sandbox_attach import attach_docker_sandbox

            agent_class._gym_sandbox = attach_docker_sandbox(request.sandbox_handle)
        if alias_clients:
            # NOOA resolves per-method model strings (``@strategy(llm="<alias>")``,
            # call-site ``llm=``) through the agent's ``_strategy_llm_alias_cache``,
            # checked before its own registry. Seeding the per-rollout subclass
            # routes those strings to the Gym-backed clients above; delegate()
            # children are built as ``type(self)``, so subagents inherit the
            # mapping. The attribute is per-rollout (fresh subclass each run), and
            # on NOOA versions without per-method LLM strings it is simply unused.
            agent_class._strategy_llm_alias_cache = alias_clients
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

        # Unique per-run trace session: the rollout id is stable across attempts, so reusing
        # it as the viewer session merges every run of a task into one session. The rollout id
        # itself stays authoritative for the trajectory, turn identity, and capture keying.
        trace_session = f"{request.rollout_id}-{uuid.uuid4().hex[:8]}"
        try:
            with session_scope(trace_session), hooks_scope(trace_hooks):
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
