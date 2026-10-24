# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers over Responses API items held as plain dicts: their text, call arguments, and calls paired with
their outputs."""

from __future__ import annotations

import json
from typing import Any, Iterator, Sequence


# Responses items as plain dicts.
Items = Sequence[dict[str, Any]]

# Argument keys with which rows mark a tool call whose arguments were not a JSON object: the sandbox harness's
# marker, and the convention docs/data.md recommends.
MALFORMED_ARGS_KEY = "__malformedToolArguments"
UNPARSEABLE_ARGS_KEY = "__unparseable_arguments__"


def as_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    return dict(item)


def content_text(content: Any) -> str:
    """A message's text: a string as it is, or its content parts' texts run together."""
    if isinstance(content, list):
        return "".join(c.get("text", "") or "" for c in content if isinstance(c, dict))
    return content or ""


def json_object(text: Any) -> dict[str, Any] | None:
    """The JSON object a text holds, or None."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError, RecursionError):  # also an over-long integer or very deep nesting
        return None
    return value if isinstance(value, dict) else None


def parse_args(call: dict[str, Any]) -> dict[str, Any] | None:
    """A function call's arguments ({} when it has none), or None when they are not a JSON object."""
    return json_object(call.get("arguments") or "{}")


def calls_and_outputs(items: Items, tools: Sequence[str]) -> Iterator[tuple[dict[str, Any], str]]:
    """(call, output text) for every answered call of these tools; a non-text output reads as ""."""
    calls = {x.get("call_id"): x for x in items if x.get("type") == "function_call" and x.get("name") in tools}
    for item in items:
        if item.get("type") == "function_call_output" and item.get("call_id") in calls:
            output = item.get("output")
            yield calls[item["call_id"]], output if isinstance(output, str) else ""
