# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned Nano Omni native-visual prompt, tools, and WebVoyager adapter.

This module contains no browser implementation.  It is the stable contract
shared by dataset preparation, the Responses agent and native web runtimes.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from nemo_gym.web.actions import MAX_NATIVE_SCROLL_AMOUNT


NATIVE_VISUAL_SYSTEM_PROMPT = """You are a GUI agent controlling a web browser. You are given a task instruction, a screenshot of the browser, and your previous interactions. You need to perform a series of actions to complete the task. The browser is already open and logged into the required websites.

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


NATIVE_VISUAL_TOOLS: list[dict[str, Any]] = [
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
                                                "maximum": MAX_NATIVE_SCROLL_AMOUNT,
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


def native_visual_tools() -> list[dict[str, Any]]:
    """Return a mutation-safe copy for one Responses request."""

    return deepcopy(NATIVE_VISUAL_TOOLS)


def adapt_native_visual_record(
    record: Mapping[str, Any],
    *,
    benchmark: str,
    verifier_profile: str,
    task_id: str | int | None = None,
) -> dict[str, Any]:
    """Build one native-visual wire row without importing Gym runtime deps.

    Batch preparation runs before the Gym virtual environments exist.  Keep
    this adapter next to the prompt/tool contract so a stdlib-only controller
    can produce the exact same row consumed by the full runtime.
    """

    if benchmark != "webvoyager":
        raise ValueError(f"unsupported native web benchmark: {benchmark!r}")
    source_id = record.get("id", record.get("task_id"))
    if source_id is None:
        raise ValueError(f"native {benchmark} record requires id or task_id")
    normalized_task_id = task_id if task_id is not None else source_id
    start_url = record.get("web") or record.get("start_url")
    if isinstance(start_url, str):
        start_urls = [part.strip() for part in start_url.split(" |AND| ") if part.strip()]
    elif isinstance(start_url, list):
        start_urls = [str(part) for part in start_url if part]
    elif start_url:
        start_urls = [str(start_url)]
    else:
        start_urls = []
    site_value = record.get("web_name") or record.get("sites") or []
    if isinstance(site_value, str):
        sites = [site_value]
    else:
        sites = [str(site) for site in site_value if site]
    image_value = record.get("image") or record.get("images") or []
    if isinstance(image_value, str):
        input_images = [image_value]
    else:
        input_images = [str(image) for image in image_value if image]
    web_task = {
        "benchmark": benchmark,
        "task_id": str(normalized_task_id),
        "intent": str(record.get("ques") or record.get("intent") or ""),
        "start_urls": start_urls,
        "sites": sites,
        "input_images": input_images,
        "runtime_profile": "native_visual",
        "observation_profile": "screenshot",
        "action_profile": "native_toolcall",
        "verifier_profile": verifier_profile,
        "auth_profile": None,
        "seed": 0,
        "task_kwargs": {},
        "original_metadata": dict(record),
    }
    return {
        "responses_create_params": {
            "input": [],
            "metadata": {"benchmark": benchmark, "task_id": str(normalized_task_id)},
            "instructions": NATIVE_VISUAL_SYSTEM_PROMPT,
            "tools": native_visual_tools(),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
        "web_task": web_task,
    }


def adapt_native_webvoyager_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return adapt_native_visual_record(
        record,
        benchmark="webvoyager",
        verifier_profile="native_webvoyager_gemini",
    )
