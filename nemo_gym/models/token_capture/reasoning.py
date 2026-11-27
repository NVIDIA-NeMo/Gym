# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning equivalence for capture on the vLLM Chat tool path."""

import hashlib
import json
import re
from collections import Counter


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
    visible_text = message.get("content")
    if not isinstance(visible_text, str):
        return message
    reasoning = message.get("reasoning_content")
    if reasoning in (None, "") and visible_text.startswith("<think>"):
        reasoning, closing_tag, visible_text = visible_text[len("<think>") :].partition("</think>")
        if not closing_tag:
            return message
    if not isinstance(reasoning, str) or not reasoning.strip():
        return message
    if any(tag in text for tag in ("<think>", "</think>") for text in (reasoning, visible_text)):
        return message
    digest = _tool_turn_digest(calls)
    if digest is None:
        return message
    result = dict(message)
    result.pop("reasoning_content", None)
    # The assistant content participates in both the strict and the argument-relaxed lookup.
    # Include the ordered arguments here so the relaxed lookup cannot widen this equivalence.
    result["content"] = [{"type": "thinking", "thinking": reasoning, "tool_turn_digest": digest}]
    if visible_text:
        result["content"].append({"type": "text", "text": visible_text})
    return result


def _hermes_id_aliases(ids: list[str]) -> list[str] | None:
    """Return the exact aliases Hermes uses for complete parallel vLLM batches."""
    if len(ids) < 2 or any(not isinstance(call_id, str) for call_id in ids):
        return None
    if all(call_id.startswith("chatcmpl-tool-") for call_id in ids):
        return [f"call_{hashlib.sha256(call_id.encode()).hexdigest()[:12]}" for call_id in ids]
    if all(re.fullmatch(r"call_[0-9a-f]{12}", call_id) for call_id in ids):
        return ids
    return None


def _capture_hermes_ids(items: list[dict], normalized: list[dict], *, served_reply: bool) -> list[dict]:
    if served_reply and len(items) != 1:
        return normalized
    batches = {}
    projected_ids: Counter[str] = Counter()
    for index, item in enumerate(items):
        calls = item.get("tool_calls")
        if item.get("role") != "assistant" or not isinstance(calls, list):
            continue
        ids = [call.get("id") for call in calls if isinstance(call, dict)]
        aliases = _hermes_id_aliases(ids) if normalized[index] is not item and len(ids) == len(calls) else None
        if aliases is not None:
            batches[index] = (ids, aliases)
        projected_ids.update(call_id for call_id in (aliases or ids) if isinstance(call_id, str))

    for index, (ids, aliases) in batches.items():
        if any(projected_ids[alias] != 1 for alias in aliases):
            continue
        end = index + 1 + len(ids)
        results = items[index + 1 : end]
        if not served_reply:
            if len(results) != len(ids) or any(
                item.get("role") != "tool" or item.get("tool_call_id") != call_id
                for item, call_id in zip(results, ids, strict=True)
            ):
                continue
            if end < len(items) and items[end].get("role") == "tool":
                continue

        item = items[index]
        calls = [{**call, "id": alias} for call, alias in zip(item["tool_calls"], aliases, strict=True)]
        turn = _capture_tool_turn({**item, "tool_calls": calls})
        # A marked echo is only eligible when it binds every result in order.
        turn["content"][0]["verified_parallel_tool_ids"] = True
        normalized[index] = turn
        if not served_reply:
            normalized[index + 1 : end] = [
                {**result, "tool_call_id": alias} for result, alias in zip(results, aliases, strict=True)
            ]
    return normalized


def capture_chat_tool_turns(
    items: list[dict], *, match_hermes_parallel_tool_ids: bool = False, served_reply: bool = False
) -> list[dict]:
    """Compare a single inline thinking block with its exact echo in the reasoning field.

    The model server enables this only when its Chat preprocessor renders both
    forms as the same separate reasoning with identical visible text. The
    returned items are for capture lookup only. Competing aliases, nested tags,
    and other shapes retain their original evidence.
    """
    normalized = [_capture_tool_turn(item) for item in items]
    if match_hermes_parallel_tool_ids:
        return _capture_hermes_ids(items, normalized, served_reply=served_reply)
    return normalized
