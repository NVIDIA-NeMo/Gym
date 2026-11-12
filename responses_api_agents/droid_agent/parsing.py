# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert ``droid exec --output-format stream-json`` output into Gym response items and observations.

Droid 1.1.0 writes one JSON object per line: ``system`` (init), ``message`` (user and assistant text),
``reasoning``, ``tool_call``, ``tool_result``, ``completion`` (final text and the run's usage) and ``error``.
Subagents started with the Task tool run in their own Droid sessions: stdout shows only the Task call, its
result and the completion report that Droid injects as a user message.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.trajectory import ObservationGap, ToolCallObservation


# Droid's Execute tool reports the command's exit status at the end of its result text.
_EXIT_CODE = re.compile(r"\[Process exited with code (-?\d+)\]\s*$")


@dataclass
class DroidRun:
    """What one ``droid exec`` run reported on stdout."""

    items: list[Any] = field(default_factory=list)
    # The items plus the user messages Droid injects mid-run, which have no place in a response's output.
    conversation: list[Any] = field(default_factory=list)
    tools: list[ToolCallObservation] = field(default_factory=list)
    gaps: list[ObservationGap] = field(default_factory=list)
    usage: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None
    duration_ms: Optional[float] = None
    completed: bool = False
    error: Optional[str] = None
    unparseable_lines: int = 0


def _millis(event: dict[str, Any]) -> Optional[float]:
    timestamp = event.get("timestamp")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and timestamp >= 0:
        return timestamp
    return None


def _tool_status(event: dict[str, Any], output: str) -> Literal["completed", "failed"]:
    if event.get("isError") is True:
        return "failed"
    exit_code = _EXIT_CODE.search(output) if event.get("toolId") == "Execute" else None
    return "failed" if exit_code is not None and int(exit_code[1]) != 0 else "completed"


def parse_events(stdout: str, *, invocation_id: str) -> DroidRun:
    """Parse stdout in order, keeping every tool call and result and the text the model wrote."""
    run = DroidRun()
    calls: dict[str, NeMoGymResponseFunctionToolCall] = {}
    started: dict[str, tuple[Optional[float], Optional[str]]] = {}
    reasoning: dict[str, NeMoGymResponseReasoningItem] = {}
    prompt_seen = False
    injected_messages = 0

    def add(item: Any) -> None:
        run.items.append(item)
        run.conversation.append(item)

    # Only newlines end a record; splitlines() would also split at U+2028, U+2029 and U+0085 inside JSON strings.
    for line in stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            run.unparseable_lines += 1
            continue
        if not isinstance(event, dict):
            run.unparseable_lines += 1
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            session_id = event.get("session_id")
            run.session_id = session_id if isinstance(session_id, str) else None
        elif kind == "message" and event.get("role") == "assistant":
            text = event.get("text")
            if isinstance(text, str) and text:
                add(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{len(run.items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
        elif kind == "message" and event.get("role") == "user":
            # The first is the prompt; later ones are reports Droid injects, such as a finished subagent's.
            text = event.get("text")
            if prompt_seen and isinstance(text, str):
                injected_messages += 1
                run.conversation.append(NeMoGymEasyInputMessage(role="user", content=text))
            prompt_seen = True
        elif kind == "reasoning":
            text = event.get("text")
            if not isinstance(text, str) or not text:
                continue
            # Droid can repeat a reasoning event with the same id; keep one item, at its first position.
            reasoning_id = str(event.get("id") or len(run.items))
            if reasoning_id in reasoning:
                reasoning[reasoning_id].summary = [NeMoGymSummary(text=text, type="summary_text")]
                continue
            item = NeMoGymResponseReasoningItem(
                id=f"rs_{len(run.items)}", summary=[NeMoGymSummary(text=text, type="summary_text")], type="reasoning"
            )
            reasoning[reasoning_id] = item
            add(item)
        elif kind == "tool_call":
            call_id = str(event.get("id") or f"call_{len(run.items)}")
            parameters = event.get("parameters")
            name = str(event.get("toolName") or event.get("toolId") or "")
            call = NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(parameters) if isinstance(parameters, dict) else str(parameters or ""),
                call_id=call_id,
                name=name,
                type="function_call",
                id=call_id,
                status="incomplete",
            )
            calls[call_id] = call
            started[call_id] = (_millis(event), name or None)
            add(call)
        elif kind == "tool_result":
            call_id = str(event.get("id") or "")
            value = event.get("value")
            error = event.get("error")
            if value is None and isinstance(error, dict):
                # A failed tool has no value; the model sees the error message instead.
                value = error.get("message")
            output = value if isinstance(value, str) else json.dumps(value)
            if call_id in calls:
                calls[call_id].status = "completed"
            add(
                NeMoGymFunctionCallOutput(
                    type="function_call_output", call_id=call_id, output=output, status="completed"
                )
            )
            started_ms, name = started.pop(call_id, (None, None))
            completed_ms = _millis(event)
            timed = started_ms is not None and completed_ms is not None and completed_ms >= started_ms
            run.tools.append(
                ToolCallObservation(
                    invocation_id=invocation_id,
                    tool_call_id=call_id,
                    tool_name=name or event.get("toolId"),
                    started_at=started_ms / 1000 if timed else None,
                    completed_at=completed_ms / 1000 if timed else None,
                    duration_ms=completed_ms - started_ms if timed else None,
                    timing_source="harness",
                    status=_tool_status(event, output),
                )
            )
            if not timed:
                run.gaps.append(
                    ObservationGap(code="tool_timing_unavailable", invocation_id=invocation_id, detail=call_id)
                )
        elif kind == "completion":
            run.completed = True
            usage = event.get("usage")
            run.usage = usage if isinstance(usage, dict) else None
            duration = event.get("durationMs")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0:
                run.duration_ms = float(duration)
        elif kind == "error":
            message = event.get("message")
            # The CLI repeats the agent loop's error; keep the first.
            run.error = run.error or (message if isinstance(message, str) else json.dumps(message))

    for call_id, (started_ms, name) in started.items():
        run.tools.append(
            ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=call_id,
                tool_name=name,
                started_at=started_ms / 1000 if started_ms is not None else None,
                timing_source="harness",
                status="incomplete",
            )
        )
    if any(call.name == "Task" for call in calls.values()):
        run.gaps.append(ObservationGap(code="subagent_hierarchy_unavailable", invocation_id=invocation_id))
    if injected_messages:
        run.gaps.append(
            ObservationGap(
                code="injected_user_messages_omitted",
                invocation_id=invocation_id,
                detail=f"{injected_messages} harness-injected user messages are in the conversation, not the output",
            )
        )
    if run.unparseable_lines:
        run.gaps.append(
            ObservationGap(
                code="agent_stdout_unparseable", invocation_id=invocation_id, detail=f"{run.unparseable_lines} lines"
            )
        )
    return run
