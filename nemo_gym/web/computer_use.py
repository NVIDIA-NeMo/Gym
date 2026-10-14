# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned Nano Omni prompt and tool contract for visual web tasks.

This module contains no browser implementation and no benchmark data adapter.
It is selected by the agent's ``nano_omni_toolcall`` policy protocol; Qwen and
future policies provide their own model-facing adapters while sharing the same
visual-browser environment and normalized ``WebAction`` wire contract.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nemo_gym.web.actions import MAX_SCROLL_AMOUNT


NANO_OMNI_SYSTEM_PROMPT = """You are a GUI agent controlling a web browser. You are given a task instruction, a screenshot of the browser, and your previous interactions. You need to perform a series of actions to complete the task. The browser is already open and logged into the required websites.

<tool_guidelines>
- Operate via x,y coordinates from the latest screenshot using the `computer` tool.
- Coordinates are relative to the viewport in [0, 1], with (0, 0) at the top-left.
- Use `tabs_create` and `tabs_focus` to manage tabs.
- Use `navigate` to go to URLs or use "back"/"forward" for browser history.
- When the task is complete, call `terminate` with status and answer.
</tool_guidelines>"""


_COORDINATE_SCHEMA: dict[str, Any] = {
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
}


NANO_OMNI_TOOLS: list[dict[str, Any]] = [
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
                    "description": (
                        'The URL to navigate to. Use "forward" to go forward in history '
                        'or "back" to go back in history.'
                    ),
                },
                "tab_id": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "description": "Tab ID to navigate.",
                    "default": None,
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
                                "description": (
                                    "The action to perform: `left_click`, `middle_click`, `right_click`, "
                                    "`double_click`, `triple_click`, `mouse_move` (coordinate), `type` "
                                    "(text), `key_press` (list of keys to press), `scroll` (direction + "
                                    "amount, optional coordinate), `left_click_drag` (start_coordinate "
                                    "to coordinate), or `wait` (duration in seconds)."
                                ),
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
                            "coordinate": _COORDINATE_SCHEMA
                            | {
                                "description": (
                                    "(x, y) relative coordinates in the [0, 1] range, where (0, 0) is "
                                    "the top-left of the viewport and (1, 1) is the bottom-right. Required "
                                    "for click actions and `mouse_move`. For `scroll`, defaults to the "
                                    "screen center when omitted. For `left_click_drag`, this is the end "
                                    "position."
                                )
                            },
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
                                "description": (
                                    "List of keys to press for the `key_press` action. Use platform modifier "
                                    'keys such as "cmd" on Mac or "ctrl" on Windows/Linux, e.g., '
                                    '["ctrl", "a"] for select all.'
                                ),
                            },
                            "scroll_parameters": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "scroll_amount": {
                                                "type": "integer",
                                                "minimum": 0,
                                                "maximum": MAX_SCROLL_AMOUNT,
                                                "default": 1,
                                                "description": (
                                                    "Number of mouse wheel clicks to scroll in the requested direction."
                                                ),
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
                            "start_coordinate": _COORDINATE_SCHEMA
                            | {
                                "description": (
                                    "(x, y) relative starting coordinates in the [0, 1] range for `left_click_drag`."
                                )
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to type. Only used for the `type` action.",
                            },
                        },
                        "required": ["action"],
                    },
                },
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
                    "description": "Start URL for new tab. Default about:blank.",
                    "default": "about:blank",
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
            "properties": {"tab_id": {"type": "integer", "description": "Tab ID to focus."}},
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
                "answer": {"type": "string", "description": "The answer of the task."},
            },
            "required": ["status"],
        },
    },
]


def nano_omni_tools() -> list[dict[str, Any]]:
    """Return a mutation-safe copy of Nano Omni's Responses tool schema."""

    return deepcopy(NANO_OMNI_TOOLS)


__all__ = ["NANO_OMNI_SYSTEM_PROMPT", "NANO_OMNI_TOOLS", "nano_omni_tools"]
