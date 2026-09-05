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
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from pydantic_core import to_json

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryToolCall,
)
from responses_api_agents.nooa_agent.gym_llm import GymModelCall, RolloutLLMState


def _json_output(value: Any) -> str:
    return value if isinstance(value, str) else to_json(value, serialize_unknown=True).decode()


def _input_items(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [NeMoGymEasyInputMessage(role="user", content=value)]
    return list(value)


@dataclass(slots=True)
class _AgentCall:
    invocation_id: str
    started: float
    token: Token[str]


@dataclass(slots=True)
class _ToolCall:
    observation: TrajectoryToolCall
    arguments: dict[str, Any]
    started: float


class GymTraceHooks:
    """Capture public NOOA lifecycle events and Gym-owned external calls."""

    def __init__(self) -> None:
        self._current: ContextVar[str] = ContextVar("gym_nooa_invocation", default="root")
        self._invocations: dict[str, AgentInvocation] = {}
        self._events: list[GymModelCall | _ToolCall] = []

    def _invocation(self, identity: str) -> AgentInvocation:
        return self._invocations.setdefault(identity, AgentInvocation(invocation_id=identity))

    def on_model_call(self, call: GymModelCall) -> None:
        call.invocation_id = self._current.get()
        self._invocation(call.invocation_id)
        self._events.append(call)

    def before_agent_call(self, *, call_id: str, parent_call_id: str | None, **_: Any) -> _AgentCall:
        self._invocations[call_id] = AgentInvocation(
            invocation_id=call_id,
            parent_invocation_id=parent_call_id,
            status="incomplete",
        )
        return _AgentCall(call_id, perf_counter(), self._current.set(call_id))

    def after_agent_call(self, *, context: Any, exception: BaseException | None, **_: Any) -> None:
        if not isinstance(context, _AgentCall):
            return
        invocation = self._invocations[context.invocation_id]
        invocation.status = "failed" if exception is not None else "completed"
        invocation.error_type = type(exception).__name__ if exception is not None else None
        invocation.duration_ms = max(0.0, (perf_counter() - context.started) * 1000)
        self._current.reset(context.token)

    @contextmanager
    def activate_agent_call(self, context: Any) -> Iterator[None]:
        token = self._current.set(context.invocation_id)
        try:
            yield
        finally:
            self._current.reset(token)

    def _start_tool(self, name: str, arguments: dict[str, Any], call_id: str) -> _ToolCall:
        invocation_id = self._current.get()
        self._invocation(invocation_id)
        event = _ToolCall(
            TrajectoryToolCall(
                invocation_id=invocation_id,
                tool_call_id=call_id,
                tool_name=name,
                started_at=time(),
                timing_source="harness",
                status="incomplete",
            ),
            arguments,
            perf_counter(),
        )
        self._events.append(event)
        return event

    def _finish_tool(self, event: _ToolCall, exception: BaseException | None) -> None:
        record = event.observation
        record.completed_at = max(record.started_at, time())
        record.duration_ms = max(0.0, (perf_counter() - event.started) * 1000)
        if exception is not None:
            record.status = "cancelled" if isinstance(exception, asyncio.CancelledError) else "failed"
            record.error_type = type(exception).__name__
            record.output = {"error": str(exception), "error_type": record.error_type}
        elif record.status == "incomplete":
            record.status = "completed"

    @contextmanager
    def resource_call(self, name: str, arguments: dict[str, Any]) -> Iterator[TrajectoryToolCall]:
        event = self._start_tool(name, arguments, f"resource-{uuid4().hex}")
        try:
            yield event.observation
        except BaseException as error:
            self._finish_tool(event, error)
            raise
        else:
            self._finish_tool(event, None)

    def before_code_execution(self, *, code: str, execution_id: str, **kwargs: Any) -> _ToolCall:
        return self._start_tool("execute_python", {"code": code}, str(kwargs.get("tool_call_id") or execution_id))

    def after_code_execution(self, *, context: Any, result: Any, exception: BaseException | None, **_: Any) -> None:
        if isinstance(context, _ToolCall):
            context.observation.output = result
            self._finish_tool(context, exception)

    def before_tool_execution(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        execution_id: str,
        **kwargs: Any,
    ) -> _ToolCall | None:
        if tool_name in {"execute_python", "return_result"}:
            return None
        return self._start_tool(tool_name, arguments, str(kwargs.get("tool_call_id") or execution_id))

    after_tool_execution = after_code_execution

    def before_generation(self, **_: Any) -> None:
        pass

    def after_generation(self, **_: Any) -> None:
        pass

    def before_method_invocation(self, **_: Any) -> None:
        pass

    def after_method_invocation(self, **_: Any) -> None:
        pass

    def on_messages_built(self, **_: Any) -> None:
        pass

    def project(
        self,
        *,
        create_params: NeMoGymResponseCreateParamsNonStreaming,
        state: RolloutLLMState,
        default_model: str = "nooa",
    ) -> AgentEpisode:
        """Project captured facts without interpreting the invocation adapter."""

        invocations = {key: record.model_copy(deep=True) for key, record in self._invocations.items()}
        output: list[Any] = []
        tools: list[TrajectoryToolCall] = []
        initialized_inputs: set[str] = set()
        visible_outputs = {
            (call.invocation_id, item.call_id): item.output
            for call in state.calls
            for item in call.request.input
            if isinstance(item, NeMoGymFunctionCallOutput)
        }
        for event in self._events:
            if isinstance(event, GymModelCall):
                invocation_id = event.invocation_id or "root"
                invocation = invocations.setdefault(invocation_id, AgentInvocation(invocation_id=invocation_id))
                if invocation_id not in initialized_inputs:
                    initialized_inputs.add(invocation_id)
                    initial = []
                    if event.request.instructions:
                        initial.append(NeMoGymEasyInputMessage(role="system", content=event.request.instructions))
                    initial.extend(_input_items(event.request.input))
                    visible = {(item.type, getattr(item, "call_id", None)) for item in initial}
                    invocation.conversation = initial + [
                        item
                        for item in invocation.conversation
                        if (item.type, getattr(item, "call_id", None)) not in visible
                    ]
                if event.response is None:
                    continue
                reference = ModelCallRef(model_ref=event.model_ref, response_id=event.response.id)
                invocation.model_calls.append(reference)
                invocation.conversation.extend(event.response.output)
                output.extend(event.response.output)
            else:
                record = event.observation.model_copy(deep=True)
                invocation = invocations.setdefault(
                    record.invocation_id,
                    AgentInvocation(invocation_id=record.invocation_id),
                )
                record.output = visible_outputs.get((record.invocation_id, record.tool_call_id), record.output)
                if not any(
                    isinstance(item, NeMoGymResponseFunctionToolCall) and item.call_id == record.tool_call_id
                    for item in invocation.conversation
                ):
                    call_item = NeMoGymResponseFunctionToolCall(
                        id=record.tool_call_id,
                        call_id=record.tool_call_id,
                        name=record.tool_name,
                        arguments=_json_output(event.arguments),
                    )
                    invocation.conversation.append(call_item)
                    output.append(call_item)
                result_item = NeMoGymFunctionCallOutput(
                    call_id=record.tool_call_id,
                    output=_json_output(record.output),
                    status="completed" if record.status == "completed" else "incomplete",
                )
                record.output = result_item.output
                invocation.conversation.append(result_item)
                output.append(result_item)
                tools.append(record)

        if not invocations:
            invocations["root"] = AgentInvocation(invocation_id="root", status="completed")
        if not initialized_inputs:
            root = next((item for item in invocations.values() if item.parent_invocation_id is None), None)
            if root is not None:
                root.conversation = [*_input_items(create_params.input), *root.conversation]

        usages = [call.response.usage if call.response is not None else None for call in state.calls]
        return AgentEpisode(
            response=NeMoGymResponse(
                id="nooa-embedded",
                created_at=time(),
                model=create_params.model or default_model,
                object="response",
                output=output,
                tools=create_params.tools,
                tool_choice=create_params.tool_choice,
                parallel_tool_calls=create_params.parallel_tool_calls,
                usage=(
                    NeMoGymResponseUsage.sum_from_list(usages)
                    if usages and all(usage is not None for usage in usages)
                    else None
                ),
            ),
            observations=AgentObservationBundle(
                source="nooa",
                records=[
                    *invocations.values(),
                    *(ToolCallObservation.model_validate(tool.model_dump(exclude={"output"})) for tool in tools),
                ],
                gaps=list(state.gaps),
            ),
        )


def _ends_with_completed_assistant_message(response: NeMoGymResponse) -> bool:
    if not response.output:
        return False
    last = response.output[-1]
    return isinstance(last, NeMoGymResponseOutputMessage) and last.status == "completed"


def ensure_verifier_final_message(
    response: NeMoGymResponse,
    return_value: Any,
) -> tuple[NeMoGymResponse, list[ObservationGap]]:
    """Add a verifier-facing fallback message without mutating captured evidence."""

    gaps: list[ObservationGap] = []
    if _ends_with_completed_assistant_message(response) or return_value is None:
        return response, gaps

    # Verifiers commonly grade the last assistant message, so preserve the full ATIF trace and append the
    # entrypoint return only when that trace does not already end with a completed assistant message.
    output = list(response.output)
    output.append(
        NeMoGymResponseOutputMessage(
            id="nooa_fallback",
            content=[NeMoGymResponseOutputText(annotations=[], text=_json_output(return_value))],
        )
    )
    gaps.append(
        ObservationGap(
            code="non_trainable_fallback_output",
            detail="NOOA returned a value without a final model-authored message.",
        )
    )
    return response.model_copy(update={"output": output}), gaps


def finalize_observations(
    observations: AgentObservationBundle,
    *,
    extra_gaps: list[ObservationGap] | None = None,
    termination_reason: str | None = None,
    termination_error: str | None = None,
) -> AgentObservationBundle:
    """Merge lifecycle-only observation gaps onto the Gym projection."""

    gaps = [*observations.gaps, *(extra_gaps or [])]
    if termination_reason is not None:
        gaps.append(
            ObservationGap(
                code=termination_reason,
                detail=termination_error or f"NOOA execution terminated with {termination_reason}.",
            )
        )
    if not gaps:
        return observations
    return observations.model_copy(update={"gaps": gaps})
