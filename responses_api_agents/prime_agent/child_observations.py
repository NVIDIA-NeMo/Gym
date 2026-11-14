# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export evidence observed by the native Prime child SDK, without inferring missing calls."""

import json

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymEasyInputMessage
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)
from responses_api_agents.prime_agent.app import parse_prime_agent_events


def child_observations(
    capture: dict, *, system_prompt: str, model_ref: ModelServerRef, invocation_id: str
) -> AgentObservationBundle:
    events = capture["events"]
    gaps = []
    invocation = AgentInvocation(
        invocation_id=invocation_id,
        activation_id=1,
        in_session=True,
        status="failed"
        if capture["termination"] == "error"
        else "incomplete"
        if capture["termination"] == "turn_limit"
        else "completed",
        error_type="sdk_error" if capture["termination"] == "error" else None,
        stop_reason="max_iterations" if capture["termination"] == "turn_limit" else None,
    )
    if system_prompt:
        invocation.conversation.append(NeMoGymEasyInputMessage(role="system", content=system_prompt))
    tools = {}
    for event in events:
        kind = event.get("type")
        if kind == "message_end":
            message = event.get("message") or {}
            role = message.get("role")
            if role == "user":
                content = message.get("content")
                text = (
                    content
                    if isinstance(content, str)
                    else "".join(item.get("text", "") for item in content or [] if item.get("type") == "text")
                )
                invocation.conversation.append(NeMoGymEasyInputMessage(role="user", content=text))
            elif role in {"assistant", "toolResult"}:
                # An abort is a control event, not another model answer.
                if message.get("stopReason") == "aborted":
                    continue
                output, _ = parse_prime_agent_events(json.dumps(event))
                invocation.conversation.extend(output)
                if role == "assistant":
                    response_id = message.get("responseId")
                    if isinstance(response_id, str) and response_id:
                        invocation.model_calls.append(ModelCallRef(model_ref=model_ref, response_id=response_id))
                    else:
                        gaps.append(ObservationGap(code="model_response_id_unavailable", invocation_id=invocation_id))
                    if message.get("stopReason") == "error" or message.get("errorMessage"):
                        invocation.status = "failed"
                        invocation.error_type = "model_error"
        elif kind == "tool_execution_start":
            call_id = event.get("toolCallId")
            if not isinstance(call_id, str) or not call_id:
                gaps.append(ObservationGap(code="tool_call_id_unavailable", invocation_id=invocation_id))
                continue
            tools[call_id] = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=call_id,
                tool_name=event.get("toolName"),
                started_at=event.get("gymObservedAt"),
                status="incomplete",
                timing_source="harness",
            )
        elif kind == "tool_execution_end":
            call_id = event.get("toolCallId")
            if call_id not in tools:
                gaps.append(ObservationGap(code="tool_start_unavailable", invocation_id=invocation_id))
                continue
            result = event.get("result") or {}
            details = result.get("details") or {}
            failed = event.get("isError") or result.get("isError") or details.get("status") == "error"
            status = "failed" if failed else "completed" if details.get("status") == "ok" else "unknown"
            if details.get("status") == "timeout":
                status = "timeout"
            if status == "unknown":
                gaps.append(ObservationGap(code="tool_status_unavailable", invocation_id=invocation_id))
            observed = tools[call_id]
            observed.status = status
            observed.error_type = "kernel_error" if failed else "timeout" if status == "timeout" else None
            observed.completed_at = event.get("gymObservedAt")
            duration = details.get("durationMs")
            if isinstance(duration, (int, float)) and duration >= 0:
                observed.duration_ms = duration
            elif observed.started_at is not None and observed.completed_at is not None:
                observed.duration_ms = max(0, (observed.completed_at - observed.started_at) * 1000)
    if not invocation.model_calls:
        gaps.append(ObservationGap(code="model_call_ownership_unavailable", invocation_id=invocation_id))
    return AgentObservationBundle(source="prime_child", records=[invocation, *tools.values()], gaps=gaps)
