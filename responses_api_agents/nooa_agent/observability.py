# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

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
class _InvocationContext:
    invocation_id: str
    parent_invocation_id: str | None
    started_monotonic: float
    token: Token[str]


@dataclass(slots=True)
class NOOATraceSnapshot:
    """Immutable-by-convention projection inputs captured during one NOOA run."""

    output: list[Any] = field(default_factory=list)
    invocations: list[AgentInvocation] = field(default_factory=list)
    model_calls: dict[str, list[ModelCallRef]] = field(default_factory=dict)


class GymTraceHooks:
    """Project NOOA's typed lifecycle callbacks into Gym observation records.

    NOOA remains authoritative for call identity and nesting. Gym records only
    its own response/tool sidecars and the final schema projection.
    """

    def __init__(self, model_ref: ModelServerRef) -> None:
        self._model_ref = model_ref
        self._current_invocation: ContextVar[str] = ContextVar(f"gym_nooa_invocation_{id(self)}", default="root")
        self._snapshot = NOOATraceSnapshot()
        self._invocations: dict[str, AgentInvocation] = {}

    @property
    def invocation_id(self) -> str:
        return self._current_invocation.get()

    def snapshot(self) -> NOOATraceSnapshot:
        return NOOATraceSnapshot(
            output=list(self._snapshot.output),
            invocations=[invocation.model_copy(deep=True) for invocation in self._invocations.values()],
            model_calls={key: list(value) for key, value in self._snapshot.model_calls.items()},
        )

    def record_model_response(self, response: NeMoGymResponse) -> None:
        invocation_id = self.invocation_id
        self._snapshot.output.extend(response.output)
        if response.id:
            self._snapshot.model_calls.setdefault(invocation_id, []).append(
                ModelCallRef(model_ref=self._model_ref, response_id=response.id)
            )

    def record_tool_execution(self, execution: Any) -> None:
        self._snapshot.output.extend(
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

    def before_agent_call(
        self,
        agent: Any,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        call_id: str,
        parent_call_id: str | None,
        **extra_kwargs: Any,
    ) -> _InvocationContext:
        del agent, method_name, args, kwargs, extra_kwargs
        self._invocations[call_id] = AgentInvocation(
            invocation_id=call_id,
            parent_invocation_id=parent_call_id,
            status="unknown",
        )
        return _InvocationContext(
            invocation_id=call_id,
            parent_invocation_id=parent_call_id,
            started_monotonic=perf_counter(),
            token=self._current_invocation.set(call_id),
        )

    def after_agent_call(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: BaseException | None,
        context: Any,
        **kwargs: Any,
    ) -> None:
        del agent, method_name, result, kwargs
        if not isinstance(context, _InvocationContext):
            return
        self._current_invocation.reset(context.token)
        self._invocations[context.invocation_id] = AgentInvocation(
            invocation_id=context.invocation_id,
            parent_invocation_id=context.parent_invocation_id,
            status="failed" if exception is not None else "completed",
            duration_ms=max(0.0, (perf_counter() - context.started_monotonic) * 1000),
            error_type=type(exception).__name__ if exception is not None else None,
            model_calls=list(self._snapshot.model_calls.get(context.invocation_id, [])),
        )

    def before_generation(
        self,
        agent: Any,
        method_name: str,
        strategy: str,
        generation_id: str,
        parent_generation_id: str | None,
        **kwargs: Any,
    ) -> None:
        return None

    def after_generation(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: BaseException | None,
        context: Any,
        generation_id: str,
        **kwargs: Any,
    ) -> None:
        return None

    def before_code_execution(
        self,
        agent: Any,
        code: str,
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        return None

    def after_code_execution(
        self,
        agent: Any,
        code: str,
        result: Any,
        exception: BaseException | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        return None

    def before_method_invocation(
        self,
        agent: Any,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        invocation_id: str,
        **extra_kwargs: Any,
    ) -> None:
        return None

    def after_method_invocation(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: BaseException | None,
        context: Any,
        invocation_id: str,
        **kwargs: Any,
    ) -> None:
        return None

    def before_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        return None

    def after_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        exception: BaseException | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        return None

    def on_messages_built(
        self,
        agent: Any,
        method_name: str,
        messages: list[dict[str, Any]],
        generation_id: str,
        **kwargs: Any,
    ) -> None:
        return None


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


def _message_text(item: Any) -> str | None:
    """Return the text of a completed assistant output message."""

    if not isinstance(item, NeMoGymResponseOutputMessage) or item.status != "completed":
        return None
    return "\n".join(part.text for part in item.content if isinstance(part, NeMoGymResponseOutputText))


def project_nooa_result(
    *,
    responses_create_params: Any,
    return_value: Any,
    model_responses: list[NeMoGymResponse],
    tool_executions: list[Any],
    trace: NOOATraceSnapshot,
    result_present: bool = True,
    termination_reason: str | None = None,
    termination_error: str | None = None,
    observation_gaps: list[ObservationGap] | None = None,
) -> tuple[NeMoGymResponse, AgentObservationBundle]:
    """Project native NOOA lifecycle facts and Gym-owned sidecars."""

    output = list(trace.output)
    usage = None
    for model_response in model_responses:
        usage = accumulate_response_usage(usage, model_response.usage)

    gaps = list(observation_gaps or [])
    if termination_reason is not None:
        gaps.append(
            ObservationGap(
                code=termination_reason,
                detail=termination_error or f"NOOA execution terminated with {termination_reason}.",
            )
        )
    if result_present:
        terminal_text = _json_output(return_value)
        terminal_item_text = _message_text(output[-1]) if output else None
        if terminal_item_text != terminal_text:
            output.append(
                NeMoGymResponseOutputMessage(
                    id="nooa_terminal_result",
                    content=[NeMoGymResponseOutputText(annotations=[], text=terminal_text)],
                )
            )
            gaps.append(
                ObservationGap(
                    code="non_trainable_terminal_output",
                    detail="Gym projected NOOA's typed return value into the final assistant message.",
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

    response_input = (
        responses_create_params.get("input", [])
        if isinstance(responses_create_params, dict)
        else responses_create_params.input
    )
    root_conversation = [*_input_items(response_input), *output]
    invocations = [invocation.model_copy(deep=True) for invocation in trace.invocations]
    if invocations:
        for invocation in invocations:
            invocation.model_calls = list(trace.model_calls.get(invocation.invocation_id, []))
            if invocation.parent_invocation_id is None:
                invocation.conversation = root_conversation
    else:
        invocations.append(
            AgentInvocation(
                invocation_id="root",
                status="incomplete" if termination_reason is not None else "completed",
                error_type=termination_reason,
                model_calls=list(trace.model_calls.get("root", [])),
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
