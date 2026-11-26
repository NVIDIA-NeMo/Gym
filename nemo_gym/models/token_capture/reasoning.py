# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning equivalence for capture on the vLLM Chat tool path."""

import hashlib
import json


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate tool argument key")
    return value


def _tool_turn_digest(calls: list[dict]) -> str | None:
    """Bind tool IDs, names and parsed arguments in their rendered order."""
    try:
        ids = set()
        ordered = []
        for call in calls:
            if (
                not isinstance(call, dict)
                or call.keys() != {"id", "type", "function"}
                or call.get("type") != "function"
            ):
                return None
            call_id = call.get("id")
            function = call.get("function")
            if not isinstance(call_id, str) or not call_id or call_id in ids or not isinstance(function, dict):
                return None
            ids.add(call_id)
            if (
                function.keys() != {"name", "arguments"}
                or not isinstance(function["name"], str)
                or not function["name"]
            ):
                return None
            raw = function.get("arguments")
            if not isinstance(raw, str):
                return None
            arguments = json.loads(raw, object_pairs_hook=_unique_json_object)
            if not isinstance(arguments, dict):
                return None
            ordered.append((call_id, function["name"], arguments))
        encoded = json.dumps(ordered, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, RecursionError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _capture_tool_turn(message: dict) -> dict:
    if message.get("role") != "assistant" or message.get("type", "message") != "message":
        return message
    if any(
        message.get(key) is not None for key in ("reasoning", "function_call", "refusal", "audio", "thinking_blocks")
    ):
        return message
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return message
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if (
        reasoning in (None, "")
        and isinstance(content, str)
        and content.startswith("<think>")
        and content.endswith("</think>")
    ):
        reasoning = content[len("<think>") : -len("</think>")]
    elif content != "":
        return message
    if not isinstance(reasoning, str) or not reasoning.strip() or "<think>" in reasoning or "</think>" in reasoning:
        return message
    digest = _tool_turn_digest(calls)
    if digest is None:
        return message
    result = dict(message)
    result.pop("reasoning_content", None)
    # The assistant content participates in both the strict and the argument-relaxed lookup.
    # Include the ordered arguments here so the relaxed lookup cannot widen this equivalence.
    result["content"] = [{"type": "thinking", "thinking": reasoning, "tool_turn_digest": digest}]
    return result


def capture_chat_tool_turns(items: list[dict]) -> list[dict]:
    """Compare a single inline thinking block with its exact echo in the reasoning field.

    The model server enables this only when its Chat preprocessor renders both
    tool-only forms as empty visible text and the same separate reasoning. The
    returned items are for capture lookup only. Visible text, competing aliases,
    literal tags within thinking, and other shapes retain their original evidence.
    """
    return [_capture_tool_turn(item) for item in items]
