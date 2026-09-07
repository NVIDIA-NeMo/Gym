# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe parsing and normalization for visual-browser computer-use actions."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.parse import urlparse

from nemo_gym.web.models import WebAction


NANO_OMNI_TOOL_NAMES = frozenset({"computer", "navigate", "tabs_create", "tabs_focus", "terminate"})
COMPUTER_ACTIONS = frozenset(
    {
        "double_click",
        "key_down",
        "key_press",
        "key_up",
        "left_click",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "middle_click",
        "mouse_move",
        "right_click",
        "scroll",
        "triple_click",
        "type",
        "wait",
    }
)
CLICK_ACTIONS = frozenset(
    {
        "double_click",
        "left_click",
        "middle_click",
        "right_click",
        "triple_click",
    }
)
MAX_SCROLL_AMOUNT = 50


class ActionParseError(ValueError):
    """Raised when a policy adapter emits an unsafe or unsupported action."""


def _native_number(value: Any, *, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionParseError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ActionParseError(f"{field} must be in [{minimum:g}, {maximum:g}]")
    return number


def _native_coordinate(value: Any, *, field: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ActionParseError(f"{field} must contain normalized x and y")
    _native_number(value[0], field=f"{field}[0]", minimum=0, maximum=1)
    _native_number(value[1], field=f"{field}[1]", minimum=0, maximum=1)


def _validate_native_computer_action(
    action: Any,
    index: int,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise ActionParseError(f"native computer action[{index}] must be an object")
    normalized = dict(action)
    name = action.get("action")
    if name not in COMPUTER_ACTIONS:
        raise ActionParseError(f"unsupported native computer action[{index}]: {name!r}")
    prefix = f"native computer action[{index}] ({name})"
    if name in CLICK_ACTIONS or name == "mouse_move":
        _native_coordinate(action.get("coordinate"), field=f"{prefix}.coordinate")
    elif name == "left_click_drag":
        if action.get("start_coordinate") is not None:
            _native_coordinate(action.get("start_coordinate"), field=f"{prefix}.start_coordinate")
        _native_coordinate(action.get("coordinate"), field=f"{prefix}.coordinate")
    elif name == "type":
        if not isinstance(action.get("text"), str):
            raise ActionParseError(f"{prefix}.text must be a string")
    elif name in {"key_press", "key_down", "key_up"}:
        keys = action.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(key, str) and key for key in keys):
            raise ActionParseError(f"{prefix}.keys must be a non-empty string list")
    elif name in {"left_mouse_down", "left_mouse_up"}:
        coordinate = action.get("coordinate")
        if coordinate is not None:
            _native_coordinate(coordinate, field=f"{prefix}.coordinate")
    elif name == "wait":
        _native_number(action.get("duration"), field=f"{prefix}.duration", minimum=0, maximum=30)
    elif name == "scroll":
        coordinate = action.get("coordinate")
        if coordinate is not None:
            _native_coordinate(coordinate, field=f"{prefix}.coordinate")
        parameters = action.get("scroll_parameters")
        if not isinstance(parameters, dict):
            raise ActionParseError(f"{prefix}.scroll_parameters must be an object")
        direction = parameters.get("scroll_direction")
        if direction not in {"up", "down", "left", "right"}:
            raise ActionParseError(f"{prefix}.scroll_direction is unsupported")
        amount = parameters.get("scroll_amount")
        if isinstance(amount, bool) or not isinstance(amount, int) or not 0 <= amount <= MAX_SCROLL_AMOUNT:
            raise ActionParseError(f"{prefix}.scroll_amount must be an integer in [0, {MAX_SCROLL_AMOUNT}]")
    return normalized


def _validate_native_tool_arguments(
    name: str,
    arguments: dict[str, Any],
    *,
    max_computer_actions: int,
) -> tuple[dict[str, Any], int]:
    normalized = dict(arguments)
    if name == "computer":
        actions = arguments.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ActionParseError("native computer tool requires a non-empty actions list")
        if len(actions) > max_computer_actions:
            raise ActionParseError(f"native computer tool exceeded the {max_computer_actions}-action batch limit")
        validated_actions: list[dict[str, Any]] = []
        for index, action in enumerate(actions):
            validated_actions.append(_validate_native_computer_action(action, index))
        normalized["actions"] = validated_actions
        return normalized, len(actions)
    if name == "navigate":
        url = arguments.get("url")
        if not isinstance(url, str) or not url:
            raise ActionParseError("native navigate.url must be a non-empty string")
        if url not in {"back", "forward"} and urlparse(url).scheme not in {"http", "https"}:
            raise ActionParseError("native navigate.url must use http(s), back, or forward")
        tab_id = arguments.get("tab_id")
        if tab_id is not None and (isinstance(tab_id, bool) or not isinstance(tab_id, int) or tab_id < 0):
            raise ActionParseError("native navigate.tab_id must be a non-negative integer or null")
    elif name == "tabs_create":
        url = arguments.get("url", "about:blank")
        if not isinstance(url, str) or (url != "about:blank" and urlparse(url).scheme not in {"http", "https"}):
            raise ActionParseError("native tabs_create.url must be about:blank or use http(s)")
    elif name == "tabs_focus":
        tab_id = arguments.get("tab_id")
        if isinstance(tab_id, bool) or not isinstance(tab_id, int) or tab_id < 0:
            raise ActionParseError("native tabs_focus.tab_id must be a non-negative integer")
    elif name == "terminate":
        if arguments.get("status") not in {"success", "failure"}:
            raise ActionParseError("native terminate.status must be success or failure")
        answer = arguments.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise ActionParseError("native terminate.answer must be a string or null")
    return normalized, 0


def parse_nano_omni_tool_calls(
    items: list[Any],
    *,
    max_calls: int = 8,
    max_computer_actions: int = 20,
) -> WebAction:
    """Validate parser-produced Nano Omni calls without repairing their contents."""

    calls: list[dict[str, Any]] = []
    parse_records: list[dict[str, Any]] = []
    for item in items:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "function_call":
            continue
        name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        raw_arguments = item.get("arguments") if isinstance(item, dict) else getattr(item, "arguments", None)
        call_id = item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"invalid JSON arguments for Nano Omni tool {name!r}") from exc
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ActionParseError(f"Nano Omni tool {name!r} arguments must be an object")
        if name not in NANO_OMNI_TOOL_NAMES:
            raise ActionParseError(f"unsupported Nano Omni browser tool: {name!r}")
        arguments, computer_actions = _validate_native_tool_arguments(
            name,
            arguments,
            max_computer_actions=max_computer_actions,
        )
        calls.append({"id": call_id, "name": name, "arguments": arguments})
        parse_records.append(
            {
                "call_id": call_id,
                "tool": name,
                "computer_actions": computer_actions,
            }
        )

    if not calls:
        raise ActionParseError("model response did not contain a Nano Omni function call")
    if len(calls) > max_calls:
        raise ActionParseError(f"Nano Omni response exceeded the {max_calls}-call limit")
    terminal_indices = [index for index, call in enumerate(calls) if call["name"] == "terminate"]
    if terminal_indices and terminal_indices != [len(calls) - 1]:
        raise ActionParseError("Nano Omni terminate must be the final tool call")

    terminal = bool(terminal_indices)
    terminal_args = calls[-1]["arguments"] if terminal else {}
    answer = terminal_args.get("answer") if terminal else None
    return WebAction(
        name=calls[0]["name"] if len(calls) == 1 else "computer_use_tool_calls",
        script="",
        arguments={"calls": calls},
        terminal=terminal,
        answer=None if answer is None else str(answer),
        raw_model_output=json.dumps(calls, ensure_ascii=False),
        metadata={"nano_omni_parse": {"calls": parse_records}},
    )


__all__ = [
    "ActionParseError",
    "MAX_SCROLL_AMOUNT",
    "parse_nano_omni_tool_calls",
]
