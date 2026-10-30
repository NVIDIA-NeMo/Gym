# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert persisted Pool events, retaining actual tool IDs rather than stdout ordering."""

import json
from dataclasses import dataclass, field

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)


@dataclass
class PoolTrajectory:
    """Root trajectory and explicit evidence gaps; missing tool results are never fabricated."""

    output: list[NeMoGymResponseOutputItem] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    exit_reason: str | None = None


def parse_trajectory(text: str) -> PoolTrajectory:
    """Read Forge's trajectory.Event NDJSON, including interrupted final lines."""
    parsed = PoolTrajectory()
    calls: set[str] = set()
    results: set[str] = set()
    synthetic: set[str] = set()
    seen: set[str] = set()
    # NDJSON records end at LF; Unicode line separators can occur inside JSON strings.
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                parsed.gaps.append("truncated_trajectory_event")
                break
            raise ValueError(f"Invalid Pool trajectory JSON at line {index + 1}") from None
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError(f"Invalid Pool trajectory event at line {index + 1}")
        event_id = event.get("id")
        if isinstance(event_id, str):
            if event_id in seen:
                continue
            seen.add(event_id)
        kind = event["type"]
        payload = event.get(kind.replace(".", "_")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid Pool {kind} payload")
        if kind == "assistant_message.end":
            parsed.output.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{index}",
                    role="assistant",
                    status="completed",
                    content=[NeMoGymResponseOutputText(text=payload["assistant_message"], annotations=[])],
                )
            )
        elif kind == "thought.end":
            parsed.output.append(
                NeMoGymResponseReasoningItem(
                    id=f"rs_{index}",
                    summary=[NeMoGymSummary(type="summary_text", text=payload["thought"])],
                    status="completed",
                )
            )
        elif kind == "tool_call.parsed":
            call_id = payload["id"]
            if not isinstance(call_id, str) or not call_id or call_id in calls:
                raise ValueError("Pool trajectory contains an invalid or duplicate tool call ID")
            calls.add(call_id)
            if payload.get("is_synthetic"):
                synthetic.add(call_id)
                continue
            parsed.output.append(
                NeMoGymResponseFunctionToolCall(
                    id=f"fc_{index}",
                    call_id=call_id,
                    name=payload["name"],
                    arguments=payload.get("raw_args") or json.dumps(payload.get("args") or {}),
                    status="completed",
                )
            )
            validation_error = payload.get("validation_error")
            if isinstance(validation_error, dict) and isinstance(validation_error.get("detail"), str):
                # Pool emits no result event for a call rejected before execution.
                results.add(call_id)
                parsed.output.append(
                    NeMoGymFunctionCallOutput(call_id=call_id, output=validation_error["detail"], status="completed")
                )
        elif kind == "tool_call.result":
            call_id = payload["id"]
            if call_id not in calls or call_id in results:
                raise ValueError("Pool trajectory contains an unmatched or duplicate tool result")
            results.add(call_id)
            if call_id in synthetic:
                continue
            parsed.output.append(
                NeMoGymFunctionCallOutput(
                    call_id=call_id,
                    output=payload.get("observation", ""),
                    status="completed",
                )
            )
        elif kind == "session.error":
            parsed.errors.append(payload.get("error") or "Pool session failed")
        elif kind == "session.exit":
            parsed.exit_reason = payload.get("reason")
    if calls - results:
        parsed.gaps.append("tool_result_unavailable")
    if parsed.exit_reason is None:
        parsed.gaps.append("session_exit_unavailable")
    return parsed
