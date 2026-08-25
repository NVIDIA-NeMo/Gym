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

import json
from dataclasses import dataclass
from typing import Any, Literal

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    accumulate_response_usage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)


@dataclass(slots=True)
class TraceEvent:
    kind: Literal["model", "tool"]
    value: Any
    invocation_id: str


class NOOAEventTracker:
    """Capture NOOA's non-persistent lifecycle events and current invocation."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self._stack: list[str] = []

    @property
    def invocation_id(self) -> str:
        return self._stack[-1] if self._stack else "root"

    def handle(self, event: Any) -> None:
        self.events.append(event)
        if event.event_type == "BeforeAgentCall":
            self._stack.append(event.call_id)
        elif event.event_type == "AfterAgentCall":
            if event.call_id in self._stack:
                self._stack = self._stack[: self._stack.index(event.call_id)]


def _input_items(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [NeMoGymEasyInputMessage(role="user", content=value)]
    return list(value)


def _json_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def project_nooa_result(
    *,
    responses_create_params: Any,
    return_value: Any,
    model_responses: list[NeMoGymResponse],
    tool_executions: list[Any],
    timeline: list[TraceEvent],
    nooa_events: list[Any],
    model_ref: ModelServerRef,
) -> tuple[NeMoGymResponse, AgentObservationBundle]:
    """Project embedded execution into Responses output and normalized observations."""

    output: list[Any] = []
    usage = None
    for event in timeline:
        if event.kind == "model":
            response = event.value
            output.extend(response.output)
            usage = accumulate_response_usage(usage, response.usage)
        else:
            execution = event.value
            output.extend(
                [
                    NeMoGymResponseFunctionToolCall(
                        id=execution.tool_call_id,
                        call_id=execution.tool_call_id,
                        name=execution.name,
                        arguments=json.dumps(execution.arguments),
                        status="completed",
                    ),
                    NeMoGymFunctionCallOutput(
                        call_id=execution.tool_call_id,
                        output=_json_output(execution.output),
                        status="completed" if execution.status == "completed" else "incomplete",
                    ),
                ]
            )

    gaps: list[ObservationGap] = []
    has_final_message = any(
        isinstance(item, NeMoGymResponseOutputMessage) and item.status == "completed" for item in output
    )
    if not has_final_message and return_value is not None:
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

    if model_responses:
        response = model_responses[-1].model_copy(deep=True, update={"output": output, "usage": usage})
    else:
        response = NeMoGymResponse(
            id="nooa_embedded",
            created_at=0.0,
            model="nooa",
            object="response",
            output=output,
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
            usage=usage,
        )

    before = {event.call_id: event for event in nooa_events if event.event_type == "BeforeAgentCall"}
    after = {event.call_id: event for event in nooa_events if event.event_type == "AfterAgentCall"}
    model_calls: dict[str, list[ModelCallRef]] = {}
    for trace_event in timeline:
        if trace_event.kind != "model":
            continue
        model_response = trace_event.value
        invocation_id = trace_event.invocation_id
        if model_response.id:
            model_calls.setdefault(invocation_id, []).append(
                ModelCallRef(model_ref=model_ref, response_id=model_response.id)
            )

    root_conversation = [*_input_items(responses_create_params.input), *output]
    invocations: list[AgentInvocation] = []
    for call_id, started in before.items():
        completed = after.get(call_id)
        start_time = started.timestamp.timestamp()
        end_time = completed.timestamp.timestamp() if completed is not None else None
        invocations.append(
            AgentInvocation(
                invocation_id=call_id,
                parent_invocation_id=started.parent_call_id,
                status=(
                    "completed"
                    if completed is not None and completed.success
                    else "failed"
                    if completed is not None
                    else "incomplete"
                ),
                duration_ms=max(0.0, (end_time - start_time) * 1000) if end_time is not None else None,
                error_type=completed.exception_type if completed is not None else None,
                model_calls=model_calls.get(call_id, []),
                conversation=root_conversation if started.is_top_level else [],
            )
        )
    if not invocations:
        invocations.append(
            AgentInvocation(
                invocation_id="root",
                status="completed",
                model_calls=model_calls.get("root", []),
                conversation=root_conversation,
            )
        )

    known_invocations = {invocation.invocation_id for invocation in invocations}
    tool_records = [
        ToolCallObservation(
            invocation_id=(
                execution.invocation_id
                if execution.invocation_id in known_invocations
                else invocations[0].invocation_id
            ),
            tool_call_id=execution.tool_call_id,
            tool_name=execution.name,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_ms=execution.duration_ms,
            timing_source="harness",
            status=execution.status,
            error_type=execution.error_type,
        )
        for execution in tool_executions
    ]
    return response, AgentObservationBundle(source="nooa", records=[*invocations, *tool_records], gaps=gaps)
