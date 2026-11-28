# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai.types.responses import FunctionToolParam

from resources_servers.webarena.key_names import normalize_key_names


logger = logging.getLogger(__name__)


def _coordinate_schema(description: str) -> dict[str, Any]:
    return {
        "anyOf": [
            {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "prefixItems": [
                    {"type": "number", "minimum": 0, "maximum": 1},
                    {"type": "number", "minimum": 0, "maximum": 1},
                ],
            },
            {"type": "null"},
        ],
        "default": None,
        "description": description,
    }


WEB_ARENA_TOOLS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "navigate",
        "strict": False,
        "description": "Navigate to a URL, or go forward/back in browser history.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": 'The URL to navigate to. Use "forward" to go forward in history or "back" to go back in history.',
                },
                "tab_id": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "description": "Tab ID to navigate.",
                },
            },
            "required": ["url"],
        },
    },
    {
        "type": "function",
        "name": "computer",
        "strict": False,
        "description": "Interact with the web browser with a sequence of computer actions.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": "List of actions to perform sequentially.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "The action to perform: `left_click`, `middle_click`, `right_click`, `double_click`, `triple_click`, `mouse_move` (coordinate), `type` (text), `key_press` (list of keys to press), `scroll` (direction + amount, optional coordinate), `left_click_drag` (start_coordinate to coordinate), or `wait` (duration in seconds).",
                                "enum": [
                                    "left_click",
                                    "middle_click",
                                    "right_click",
                                    "double_click",
                                    "triple_click",
                                    "mouse_move",
                                    "type",
                                    "key_press",
                                    "wait",
                                    "scroll",
                                    "left_click_drag",
                                ],
                            },
                            "coordinate": _coordinate_schema(
                                "(x, y) relative coordinates in the [0, 1] range, where (0, 0) is the top-left of the viewport and (1, 1) is the bottom-right. Required for click actions and `mouse_move`. For `scroll`, defaults to the screen center when omitted. For `left_click_drag`, this is the end position."
                            ),
                            "start_coordinate": _coordinate_schema(
                                "(x, y) relative starting coordinates in the [0, 1] range for `left_click_drag`."
                            ),
                            "duration": {
                                "anyOf": [
                                    {"type": "integer", "minimum": 0, "maximum": 30},
                                    {"type": "null"},
                                ],
                                "default": None,
                                "description": "The number of seconds to wait. Required for `wait`. Maximum 30 seconds.",
                            },
                            "keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": 'List of keys to press for the `key_press` action. Use platform modifier keys such as "cmd" on Mac or "ctrl" on Windows/Linux, e.g., ["ctrl", "a"] for select all.',
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to type. Only used for the `type` action.",
                            },
                            "scroll_parameters": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "scroll_amount": {
                                                "type": "integer",
                                                "minimum": 0,
                                                "default": 1,
                                                "description": "Number of mouse wheel clicks to scroll in the requested direction. This value is uncapped.",
                                            },
                                            "scroll_direction": {
                                                "type": "string",
                                                "enum": ["up", "down", "left", "right"],
                                                "default": "down",
                                                "description": "The direction to scroll in.",
                                            },
                                        },
                                        "required": ["scroll_direction", "scroll_amount"],
                                    },
                                    {"type": "null"},
                                ],
                                "default": None,
                                "description": "The parameters to scroll with. Required for `scroll`.",
                            },
                        },
                        "required": ["action"],
                    },
                }
            },
            "required": ["actions"],
        },
    },
    {
        "type": "function",
        "name": "tabs_create",
        "strict": False,
        "description": "Creates a new empty tab in the current tab group",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "default": "about:blank",
                    "description": "Start URL for new tab. Default about:blank.",
                }
            },
        },
    },
    {
        "type": "function",
        "name": "tabs_focus",
        "strict": False,
        "description": "Focus an existing tab in the current tab group.",
        "parameters": {
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "integer",
                    "description": "Tab ID to focus.",
                }
            },
            "required": ["tab_id"],
        },
    },
    {
        "type": "function",
        "name": "terminate",
        "strict": False,
        "description": "Terminate the current task and report its completion status.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "The status of the task.",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer of the task.",
                },
            },
            "required": ["status"],
        },
    },
]

TOOL_NAMES = {tool["name"] for tool in WEB_ARENA_TOOLS}
POINTER_ACTIONS = {
    "left_click",
    "middle_click",
    "right_click",
    "double_click",
    "triple_click",
    "mouse_move",
}


def decode_tool_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        raise TypeError(f"Tool arguments must be JSON text or an object, got {type(raw).__name__}")
    parsed = json.loads(raw or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must decode to an object")
    return parsed


def _relative_point(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-element [x, y] array")
    x, y = float(value[0]), float(value[1])
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        raise ValueError(f"{field} coordinates must be in [0, 1]")
    return x, y


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOL_NAMES:
        raise ValueError(f"Unknown WebArena tool: {name}")
    args = dict(arguments)
    if name == "navigate":
        if not isinstance(args.get("url"), str) or not args["url"].strip():
            raise ValueError("navigate.url must be a non-empty string")
        if args.get("tab_id") is not None and int(args["tab_id"]) < 0:
            raise ValueError("navigate.tab_id must be non-negative")
    elif name == "tabs_focus":
        if "tab_id" not in args or int(args["tab_id"]) < 0:
            raise ValueError("tabs_focus.tab_id must be non-negative")
    elif name == "tabs_create":
        if args.get("url") is not None and not isinstance(args["url"], str):
            raise ValueError("tabs_create.url must be a string")
    elif name == "terminate":
        if args.get("status") not in {"success", "failure"}:
            raise ValueError("terminate.status must be success or failure")
        if args.get("answer") is not None and not isinstance(args["answer"], str):
            raise ValueError("terminate.answer must be a string")
    elif name == "computer":
        actions = args.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("computer.actions must be a non-empty list")
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"computer.actions[{index}] must be an object")
            kind = action.get("action")
            if kind in POINTER_ACTIONS:
                _relative_point(action.get("coordinate"), f"actions[{index}].coordinate")
            elif kind == "left_click_drag":
                _relative_point(action.get("start_coordinate"), f"actions[{index}].start_coordinate")
                _relative_point(action.get("coordinate"), f"actions[{index}].coordinate")
            elif kind == "type":
                if not isinstance(action.get("text"), str):
                    raise ValueError(f"actions[{index}].text must be a string")
            elif kind == "key_press":
                if not isinstance(action.get("keys"), list) or not action["keys"]:
                    raise ValueError(f"actions[{index}].keys must be a non-empty list")
            elif kind == "wait":
                duration = float(action.get("duration", 0))
                if not 0 <= duration <= 30:
                    raise ValueError(f"actions[{index}].duration must be in [0, 30]")
            elif kind == "scroll":
                params = action.get("scroll_parameters")
                if not isinstance(params, dict):
                    raise ValueError(f"actions[{index}].scroll_parameters is required")
                if params.get("scroll_direction") not in {"up", "down", "left", "right"}:
                    raise ValueError(f"actions[{index}] has invalid scroll direction")
                if int(params.get("scroll_amount", -1)) < 0:
                    raise ValueError(f"actions[{index}] has invalid scroll amount")
            else:
                raise ValueError(f"Unsupported computer action at index {index}: {kind!r}")
    return args


def _absolute_point(value: Any, width: int, height: int) -> tuple[int, int]:
    x, y = _relative_point(value, "coordinate")
    return round(x * width), round(y * height)


def execute_computer_actions(
    pyautogui: Any,
    arguments: dict[str, Any],
    *,
    width: int,
    height: int,
    typer: Any | None = None,
    normalize_keys: bool = False,
) -> None:
    """Run a computer tool call with pyautogui.

    ``typer`` (an ``x11_typing.XKeymapTyper``) replaces ``pyautogui.write`` for the
    ``type`` action when given; ``normalize_keys`` maps ``key_press`` names the way the
    real eval harness does (``key_names.py``). The defaults keep the pre-2026-10-17 behaviour.
    """
    args = validate_tool_arguments("computer", arguments)
    for action in args["actions"]:
        kind = action["action"]
        if kind in POINTER_ACTIONS:
            x, y = _absolute_point(action["coordinate"], width, height)
            if kind == "mouse_move":
                pyautogui.moveTo(x, y)
            else:
                button = {
                    "middle_click": "middle",
                    "right_click": "right",
                }.get(kind, "left")
                clicks = {"double_click": 2, "triple_click": 3}.get(kind, 1)
                pyautogui.click(x, y, button=button, clicks=clicks)
        elif kind == "type":
            if typer is None:
                pyautogui.write(action["text"], interval=0.02)
            else:
                result = typer.type_text(action["text"], interval=0.02)
                if result.dropped or result.released_modifiers or result.inserted:
                    logger.info(
                        "type action: %d keyed, inserted=%r, dropped=%r, released modifier keycodes %s",
                        result.typed,
                        result.inserted,
                        result.dropped,
                        result.released_modifiers,
                    )
        elif kind == "key_press":
            keys = normalize_key_names(action["keys"]) if normalize_keys else [str(key) for key in action["keys"]]
            pyautogui.hotkey(*keys)
        elif kind == "wait":
            time.sleep(float(action.get("duration", 0)))
        elif kind == "scroll":
            params = action["scroll_parameters"]
            amount = int(params["scroll_amount"])
            direction = params["scroll_direction"]
            x, y = _absolute_point(action.get("coordinate") or [0.5, 0.5], width, height)
            if direction in {"up", "down"}:
                pyautogui.scroll(amount if direction == "up" else -amount, x=x, y=y)
            else:
                pyautogui.moveTo(x, y)
                pyautogui.hscroll(amount if direction == "right" else -amount)
        elif kind == "left_click_drag":
            start_x, start_y = _absolute_point(action["start_coordinate"], width, height)
            end_x, end_y = _absolute_point(action["coordinate"], width, height)
            pyautogui.moveTo(start_x, start_y)
            pyautogui.dragTo(end_x, end_y, duration=0.5, button="left")
        time.sleep(0.3)
