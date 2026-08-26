# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated projections for rollout records written before ``ng_trajectory``."""

from __future__ import annotations

from typing import Any

from nemo_gym.health.checks import _item_has_tool_call, _nonempty


RESPONSE_OUTPUT_SOURCE = "response.output"
MODEL_CALL_CAPTURE_SOURCE = "ng_model_call_capture.calls"

_AGENT_TURN_BOUNDARY_TYPES = frozenset(
    {
        "function_call_output",
        "tool_call_output",
        "tool_result",
        "computer_call_output",
        "custom_tool_call_output",
        "local_shell_call_output",
        "mcp_approval_response",
    }
)
_INCOMPLETE_CAPTURE_GAPS = frozenset(
    {
        "model_call_capture_no_records",
        "model_call_capture_incomplete",
        "model_call_capture_records_unreadable",
        "model_call_capture_unreadable",
    }
)


def _item_is_agent_content(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return item.get("role") in {"assistant", "agent"} or item.get("type") == "reasoning" or _item_has_tool_call(item)


def _item_ends_agent_turn(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return item.get("role") == "user" or item.get("type") in _AGENT_TURN_BOUNDARY_TYPES


def response_output_trajectory(record: dict[str, Any]) -> dict[str, Any] | None:
    """Project Responses output items into coarse turns without call bindings."""
    response = record.get("response")
    output = response.get("output") if isinstance(response, dict) else None
    if not isinstance(output, list):
        return None

    turns: list[dict[str, Any]] = []
    has_message = False
    has_tool_calls = False
    has_agent_content = False

    def flush() -> None:
        nonlocal has_message, has_tool_calls, has_agent_content
        if has_agent_content:
            answer: list[dict[str, Any]] = []
            if has_message:
                answer.append({"type": "message", "content": "legacy message content"})
            if has_tool_calls:
                answer.append({"type": "tool_call"})
            turns.append(
                {
                    "turn_no": len(turns),
                    "answer": answer,
                    "model_calls": [],
                }
            )
        has_message = False
        has_tool_calls = False
        has_agent_content = False

    for item in output:
        if _item_ends_agent_turn(item):
            flush()
            continue
        if not _item_is_agent_content(item):
            continue
        has_agent_content = True
        has_tool_calls = has_tool_calls or _item_has_tool_call(item)
        has_message = has_message or _nonempty(item)
    flush()
    return {"turns": turns}


def embedded_model_calls(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return complete embedded legacy capture calls, never sidecar evidence."""
    capture = record.get("ng_model_call_capture")
    if not isinstance(capture, dict):
        return None
    raw_calls = capture.get("calls")
    if not isinstance(raw_calls, list) or not raw_calls or any(not isinstance(call, dict) for call in raw_calls):
        return None
    gaps = capture.get("gaps")
    if isinstance(gaps, list) and any(
        isinstance(gap, dict) and gap.get("code") in _INCOMPLETE_CAPTURE_GAPS for gap in gaps
    ):
        return None
    return [dict(call) for call in raw_calls]
