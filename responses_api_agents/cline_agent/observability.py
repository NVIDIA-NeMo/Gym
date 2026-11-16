# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent observations read from the `cline --json` event stream of a sandbox session."""

import json
from datetime import datetime
from typing import Any, Literal, Optional

from nemo_gym.core import NeMoGymFunctionCallOutput
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, ToolCallObservation


InvocationStatus = Literal["completed", "failed", "incomplete", "unknown"]
# How a tool call still running when Cline stopped ended: the deadline, a cancellation, or unknown.
UnfinishedToolStatus = Literal["timeout", "cancelled", "incomplete"]


def _timestamp(record: dict[str, Any]) -> Optional[float]:
    """Cline stamps every stream record with an ISO-8601 `ts`."""
    value = record.get("ts")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _tool_failed(event: dict[str, Any]) -> Optional[bool]:
    """Read a tool call's outcome: an `error`, or a `success` flag on each entry of its output (Cline 3.0.55)."""
    if event.get("error"):
        return True
    output = event.get("output")
    entries = output if isinstance(output, list) else [output]
    flags = [entry.get("success") for entry in entries if isinstance(entry, dict) and "success" in entry]
    if any(flag is False for flag in flags):
        return True
    if flags and all(flag is True for flag in flags):
        return False
    return None


def build_cline_observations(
    stdout: str,
    invocation_id: str,
    *,
    status: InvocationStatus,
    error_type: Optional[str],
    conversation: list[Any],
    sandbox_id: Optional[str] = None,
    unfinished_tool_status: UnfinishedToolStatus = "incomplete",
) -> AgentObservationBundle:
    """Build the invocation and tool timings of one Cline run; gaps name what the stream does not carry.

    Model calls are owned through the `x-session-id` header, which carries ``invocation_id``, so the
    invocation lists no model-call references of its own. A tool call Cline never finished gets
    ``unfinished_tool_status`` and, in the observed conversation only, an output saying it returned none.
    """

    def gap(code: str, detail: Optional[str] = None) -> ObservationGap:
        return ObservationGap(code=code, invocation_id=invocation_id, detail=detail)

    gaps = [gap("subagent_hierarchy_unavailable")]
    starts: dict[str, tuple[Optional[float], Optional[str]]] = {}
    tools: dict[str, ToolCallObservation] = {}
    duration_ms: Optional[float] = None
    stop_reason: Optional[str] = None
    for line in stdout.split("\n"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("type") == "run_result":
            duration = record.get("durationMs")
            if isinstance(duration, (int, float)) and duration >= 0:
                duration_ms = float(duration)
            reason = record.get("finishReason")
            stop_reason = reason if isinstance(reason, str) else None
            continue
        event = record.get("event")
        if record.get("type") != "agent_event" or not isinstance(event, dict) or event.get("contentType") != "tool":
            continue
        call_id = event.get("toolCallId")
        if not isinstance(call_id, str) or not call_id:
            continue
        name = event.get("toolName") if isinstance(event.get("toolName"), str) else None
        if event.get("type") == "content_start":
            starts[call_id] = (_timestamp(record), name)
        elif event.get("type") == "content_end":
            started_at, started_name = starts.pop(call_id, (None, None))
            completed_at = _timestamp(record)
            timed = started_at is not None and completed_at is not None and completed_at >= started_at
            failed = _tool_failed(event)
            tools[call_id] = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=call_id,
                sandbox_id=sandbox_id,
                tool_name=name or started_name,
                started_at=started_at if timed else None,
                completed_at=completed_at if timed else None,
                duration_ms=(completed_at - started_at) * 1000 if timed else None,
                timing_source="harness",
                status="unknown" if failed is None else "failed" if failed else "completed",
            )
            if not timed:
                gaps.append(gap("tool_timing_unavailable", call_id))
            if failed is None:
                gaps.append(gap("tool_outcome_unavailable", call_id))
    # A call the run never finished, such as one the deadline interrupted.
    conversation = list(conversation)
    for call_id, (started_at, name) in starts.items():
        tools[call_id] = ToolCallObservation(
            invocation_id=invocation_id,
            tool_call_id=call_id,
            sandbox_id=sandbox_id,
            tool_name=name,
            started_at=started_at,
            timing_source="harness",
            status=unfinished_tool_status,
        )
        gaps.append(gap("tool_timing_unavailable", call_id))
        conversation.append(
            NeMoGymFunctionCallOutput(
                type="function_call_output",
                call_id=call_id,
                output=f"[no output: Cline stopped before this tool call returned ({unfinished_tool_status})]",
                status="incomplete",
            )
        )
    if status == "unknown":
        gaps.append(gap("invocation_outcome_unavailable"))
    return AgentObservationBundle(
        source="cline",
        records=[
            AgentInvocation(
                invocation_id=invocation_id,
                status=status,
                stop_reason=stop_reason,
                duration_ms=duration_ms,
                error_type=error_type,
                conversation=conversation,
            ),
            *tools.values(),
        ],
        gaps=gaps,
    )
