# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project upstream Hermes messages into the semantic action trajectory."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def normalize_hermes_chat_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Expose Hermes dispatcher calls as the actual tool calls they represent."""
    normalized = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue
        message = deepcopy(raw_message)
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue

            arguments = function.get("arguments", "")
            parsed_arguments = arguments
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass

            if function.get("name") == "tool_call" and isinstance(parsed_arguments, dict):
                target_name = parsed_arguments.get("name")
                if isinstance(target_name, str) and target_name and "arguments" in parsed_arguments:
                    function["name"] = target_name
                    arguments = parsed_arguments["arguments"]
            function["arguments"] = _json_arguments(arguments)
        normalized.append(message)
    return normalized


def project_hermes_response_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Remove Hermes-internal dispatch steps from the external action trajectory."""
    messages = [deepcopy(message) for message in messages if isinstance(message, dict)]
    omitted_call_ids = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict) and function.get("name") in {"tool_describe", "tool_search"}:
                call_id = call.get("id")
                if call_id:
                    omitted_call_ids.add(call_id)

        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, str) and (
                "does not exist. Available tools:" in content or "The tool was NOT invoked." in content
            ):
                call_id = message.get("tool_call_id")
                if call_id:
                    omitted_call_ids.add(call_id)

    projected = []
    for message in messages:
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            message["tool_calls"] = [
                call for call in calls if not isinstance(call, dict) or call.get("id") not in omitted_call_ids
            ]
        if message.get("role") == "tool" and message.get("tool_call_id") in omitted_call_ids:
            continue
        if message.get("role") == "assistant" and not message.get("content") and not message.get("tool_calls"):
            continue
        projected.append(message)
    return normalize_hermes_chat_messages(projected)
