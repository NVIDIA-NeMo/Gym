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
import copy
import inspect
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Protocol
from uuid import uuid4

from nooa.errors import GenerationError
from nooa.runtime.hooks import hooks_scope
from nooa.strategies.codeact import CodeActSession, CodeActStrategy
from nooa.strategies.predict import PredictStrategy

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentEpisode, TrajectoryRecord
from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.gym_llm import (
    GymModelClient,
    GymResponsesLLM,
    InvalidPolicyOutputError,
    OutputTokenLimitExceeded,
    PolicyCallBudgetExceeded,
    RolloutLLMState,
)
from responses_api_agents.nooa_agent.observability import GymTraceHooks
from responses_api_agents.nooa_agent.resource_tools import (
    ResourceToolDispatcher,
    create_agent_class_with_resource_methods,
    validate_agent_resource_method_bindings,
)
from responses_api_agents.nooa_agent.termination import NOOATerminationReason


@dataclass(slots=True)
class NOOARunRequest:
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    model_url_path: str
    tool_access: DirectHTTPToolAccess | None = None
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
    termination_reason: NOOATerminationReason | None = None
    termination_error: str | None = None
    trajectory: TrajectoryRecord | None = None


class NOOARunFailure(RuntimeError):
    """A failure carrying the same episode representation as a successful run."""

    def __init__(self, error: BaseException, result: NOOARunResult) -> None:
        super().__init__(str(error))
        self.result = result


class NOOARunner(Protocol):
    async def run(self, request: NOOARunRequest) -> NOOARunResult: ...


def _validate_agent_llm_bindings(agent_class: type) -> None:
    # `_plan_llm` is not a publicly documented NOOA API; this is only an
    # additional safeguard check against method-level overrides bypassing Gym.
    overridden = sorted(
        name
        for name, method in inspect.getmembers_static(agent_class)
        if getattr(method, "_plan_llm", None) is not None
    )
    if overridden:
        names = ", ".join(overridden)
        raise ValueError(
            "NOOA method-level LLM overrides bypass Gym's model server and are unsupported; "
            f"remove llm= from @strategy on: {names}"
        )


def _generation_error_frame(error: BaseException) -> FrameType | None:
    # GenerationError is also available to arbitrary agent code. Recognize the
    # pinned strategies by the code that raised it, not its type or message alone.
    if type(error) is not GenerationError or error.__traceback__ is None:
        return None
    traceback = error.__traceback__
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    return traceback.tb_frame


def _codeact_failure_session(error: BaseException) -> CodeActSession | None:
    frame = _generation_error_frame(error)
    if frame is None or frame.f_code is not CodeActStrategy._run_generation.__code__:
        return None
    session = frame.f_locals.get("session")
    return session if isinstance(session, CodeActSession) else None


def _is_codeact_iteration_exhaustion(error: BaseException) -> bool:
    # The pinned NOOA uses the same error type for iteration and retry exhaustion.
    session = _codeact_failure_session(error)
    return (
        session is not None
        and session.max_iterations is not None
        and session.iteration >= session.max_iterations
        and session.error_count < session.max_retries
        and str(error) == str(session.build_failure_error())
    )


class InProcessNOOARunner:
    """Construct and invoke one isolated NOOA agent instance per Gym rollout."""

    def __init__(
        self,
        *,
        invocation: NOOAInvocationConfig,
        server_client: GymModelClient,
        model_server_name: str,
        max_policy_calls: int | None,
    ) -> None:
        self._invocation = invocation
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._max_policy_calls = max_policy_calls
        self._agent_class, self._invocation_adapter = validate_invocation(invocation)
        _validate_agent_llm_bindings(self._agent_class)

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        state = RolloutLLMState(max_policy_calls=self._max_policy_calls)
        trace = GymTraceHooks()
        sampling_overrides = request.responses_create_params.model_dump(
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
            tool_access=request.tool_access,
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
        failure: BaseException | None = None
        try:
            with hooks_scope(trace):
                return_value = await self._invocation_adapter(agent, request.responses_create_params)
        except BaseException as error:
            failure = error
            # Only known NOOA strategy wrappers deliberately chain model errors.
            # Cleanup failures can carry a preceding limit as context and must stay fatal.
            cause: BaseException | None = error
            seen: set[int] = set()
            while isinstance(error, Exception) and cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                if isinstance(cause, OutputTokenLimitExceeded):
                    termination_reason = NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED
                    termination_error = str(cause)
                    break
                if isinstance(cause, (PolicyCallBudgetExceeded, InvalidPolicyOutputError)):
                    termination_reason = (
                        NOOATerminationReason.POLICY_BUDGET_EXCEEDED
                        if isinstance(cause, PolicyCallBudgetExceeded)
                        else NOOATerminationReason.INVALID_POLICY_OUTPUT
                    )
                    termination_error = str(cause)
                    break
                if _is_codeact_iteration_exhaustion(cause):
                    termination_reason = NOOATerminationReason.MAX_ITERATIONS_EXCEEDED
                    termination_error = str(cause)
                    break
                frame = _generation_error_frame(cause)
                if frame is None or frame.f_code not in (
                    CodeActStrategy._run_generation.__code__,
                    PredictStrategy._execute_inner.__code__,
                ):
                    break
                cause = cause.__cause__

        fatal_error = dispatcher.fatal_error
        if state.fatal_error is not None:
            # A budget or output error after an unrecovered transport failure
            # still lacks a valid rollout to grade. Keep the terminal cause.
            termination_reason = None
            termination_error = None
        if fatal_error is None and failure is None:
            fatal_error = state.fatal_error
        if fatal_error is not None and not isinstance(failure, asyncio.CancelledError):
            failure = fatal_error
            termination_reason = None
            termination_error = None

        episode, trajectory = trace.project(
            create_params=request.responses_create_params,
            state=state,
            task_id=request.task_id,
            rollout_id=request.rollout_id,
            default_model=self._model_server_name,
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
