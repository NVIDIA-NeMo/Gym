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
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from aiohttp import ClientResponseError
from nooa.runtime.hooks import hooks_scope

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentEpisode, TrajectoryRecord
from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.nooa_agent.gym_llm import (
    GymModelClient,
    GymResponsesLLM,
    RolloutLLMState,
)
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.observability import GymTraceHooks
from responses_api_agents.nooa_agent.resource_tools import (
    ResourceToolDispatcher,
    create_agent_class_with_resource_methods,
    validate_agent_resource_method_bindings,
)
from responses_api_agents.nooa_agent.termination import NOOATerminationReason, cleanup_errors, policy_termination


logger = logging.getLogger(__name__)


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
    model_status_code: int | None = None


class NOOARunFailure(RuntimeError):
    """A failure carrying the same episode representation as a successful run."""

    def __init__(self, error: BaseException, result: NOOARunResult) -> None:
        message = str(error)
        retained = cleanup_errors(error)
        if retained:
            # Bound each cleanup detail so both stages fit in the worker's 2,000-character diagnostic.
            details = "; ".join(f"{type(cleanup).__name__}: {str(cleanup)[:500]}" for cleanup in retained)
            message = f"Cleanup failed: {details}"
            if all(cleanup is not error for cleanup in retained):
                message += f". Original task failure: {error}"
        super().__init__(message)
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


class InProcessNOOARunner:
    """Construct and invoke one isolated NOOA agent instance per Gym rollout."""

    def __init__(
        self,
        *,
        invocation: NOOAInvocationConfig,
        server_client: GymModelClient,
        model_server_name: str,
        max_policy_calls: int | None,
        on_progress: Callable[[NOOARunResult], None] | None = None,
        context_window: int | None = None,
        media_inputs: tuple[str, ...] = (),
    ) -> None:
        self._invocation = invocation
        self._server_client = server_client
        self._model_server_name = model_server_name
        self._max_policy_calls = max_policy_calls
        self._on_progress = on_progress
        self._context_window = context_window
        self._media_inputs = media_inputs
        self._agent_class, self._invocation_adapter = validate_invocation(invocation)
        _validate_agent_llm_bindings(self._agent_class)

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        state = RolloutLLMState(max_policy_calls=self._max_policy_calls)
        progress_lock = Lock()

        def checkpoint() -> None:
            if self._on_progress is None:
                return
            # NOOA's synchronous callbacks do not yield to sibling async calls.
            # Capture a consistent projection before sibling calls resume. The
            # sandbox callback detaches it on this loop, then queues serialization
            # and persistence; a writer thread must never inspect the live trace.
            with progress_lock:
                try:
                    episode, trajectory = trace.project(
                        create_params=request.responses_create_params,
                        state=state,
                        task_id=request.task_id,
                        rollout_id=request.rollout_id,
                        default_model=self._model_server_name,
                        partial=True,
                    )
                    self._on_progress(
                        NOOARunResult(
                            episode=episode,
                            return_value=None,
                            model_cookies=dict(request.model_cookies),
                            resource_cookies=dict(request.resource_cookies),
                            trajectory=trajectory,
                        )
                    )
                except Exception:
                    # Evidence persistence is best effort. NOOA swallows tool
                    # hook errors, so report all checkpoint failures consistently.
                    logger.exception("Failed to persist NOOA partial checkpoint")

        trace = GymTraceHooks(on_progress=checkpoint)
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
            on_response=checkpoint,
            sampling_overrides=sampling_overrides,
            context_window=self._context_window,
            media_inputs=self._media_inputs,
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
            termination_reason, termination_error = policy_termination(error, model_error=state.fatal_error)

        fatal_error = dispatcher.fatal_error
        if state.fatal_error is not None and termination_reason is not NOOATerminationReason.MODEL_REJECTED:
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
            root_status="incomplete"
            if termination_reason is not None and termination_reason is not NOOATerminationReason.MODEL_REJECTED
            else "failed"
            if failure is not None
            else "completed",
            root_error_type=type(failure).__name__ if failure is not None else None,
        )
        result = NOOARunResult(
            episode=episode,
            return_value=return_value,
            model_cookies=request.model_cookies,
            resource_cookies=request.resource_cookies,
            termination_reason=termination_reason,
            termination_error=termination_error,
            trajectory=trajectory,
            # Retain the upstream status only after the model identity and cleanup
            # guards established a terminal rejection; do not parse diagnostics.
            model_status_code=state.fatal_error.status
            if termination_reason is NOOATerminationReason.MODEL_REJECTED
            and isinstance(state.fatal_error, ClientResponseError)
            else None,
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
