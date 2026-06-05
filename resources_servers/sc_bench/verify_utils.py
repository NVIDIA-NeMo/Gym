# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any, Dict, List, Optional

from resources_servers.sc_bench.evaluation import compute_reward_from_tool_trace


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def strip_thinking_traces(text: str) -> str:
    """Remove thinking-model reasoning blocks before parsing tool arguments."""
    text = _THINK_TAG_RE.sub("", text)
    text = _THINKING_TAG_RE.sub("", text)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_trade_order_id(question: str) -> Optional[str]:
    if not question:
        return None
    match = re.search(r"T\d+", question.upper())
    return match.group(0) if match else None


def parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    cleaned = strip_thinking_traces(arguments)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_tool_trace_from_response(response_output: List[Any]) -> List[Dict[str, Any]]:
    """Pair function_call items with their function_call_output by call_id."""
    pending_calls: Dict[str, Dict[str, Any]] = {}
    trace: List[Dict[str, Any]] = []

    for item in response_output:
        item_type = item.type if hasattr(item, "type") else item.get("type")
        if item_type == "function_call":
            call_id = item.call_id if hasattr(item, "call_id") else item.get("call_id")
            name = item.name if hasattr(item, "name") else item.get("name")
            arguments = item.arguments if hasattr(item, "arguments") else item.get("arguments")
            pending_calls[call_id] = {"name": name, "arguments": parse_tool_arguments(arguments)}
        elif item_type == "function_call_output":
            call_id = item.call_id if hasattr(item, "call_id") else item.get("call_id")
            output_raw = item.output if hasattr(item, "output") else item.get("output")
            if call_id not in pending_calls:
                continue
            call = pending_calls.pop(call_id)
            if isinstance(output_raw, str):
                try:
                    output = json.loads(strip_thinking_traces(output_raw))
                except (json.JSONDecodeError, TypeError):
                    output = {"raw_output": output_raw}
            else:
                output = output_raw
            trace.append(
                {
                    "step": len(trace) + 1,
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "output": output,
                }
            )

    return trace


def get_verifier_fields(verifier_metadata: Optional[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = verifier_metadata or {}
    gt_lines = meta.get("gt_lines") or []
    expected_result = meta.get("expected_result") or {}
    if not isinstance(gt_lines, list):
        gt_lines = []
    if not isinstance(expected_result, dict):
        expected_result = {}
    return gt_lines, expected_result


def compute_episode_reward(
    response_output: List[Any],
    gt_lines: List[Dict[str, Any]],
) -> float:
    tool_trace = extract_tool_trace_from_response(response_output)
    if not tool_trace:
        return 0.0
    return compute_reward_from_tool_trace(tool_trace, gt_lines)
