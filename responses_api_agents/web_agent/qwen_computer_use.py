# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen XML computer-use policy adapter for the visual-browser runtime.

This module owns only model protocol: prompt construction, screenshot folding,
and conversion of Qwen XML calls into Gym's model-independent ``WebAction``.
Browser launch, proxy/CAPTCHA handling, input execution, and evaluation remain
in the shared visual-browser resources server.
"""

from __future__ import annotations

import ast
import base64
import datetime
import json
import math
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Iterable

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse
from nemo_gym.web.actions import MAX_SCROLL_AMOUNT, ActionParseError
from nemo_gym.web.models import WebAction, WebObservation


COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and hold a single key without releasing it.
* `key_up`: Release a previously held single key.
* `left_mouse_down`: Press and hold the left mouse button.
* `left_mouse_up`: Release the left mouse button.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified coordinate.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified coordinate.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `triple_click`: Triple-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `screenshot`: Capture a new screenshot of the current screen.
* `call_user`: Ask user for information or confirmation.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status. You MUST include `answer` for both `success` and `failure`. For `success`, `answer` must answer the task instruction. For `failure`, `answer` must briefly explain why the task failed.
""".strip()


def _description_prompt(width: int, height: int, coordinate_type: str) -> str:
    displayed_resolution = f"{width}x{height}" if coordinate_type == "absolute" else "1000x1000"
    return "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            f"* The screen's resolution is {displayed_resolution}.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
    )


def build_tool_definition(width: int, height: int, coordinate_type: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": _description_prompt(width, height, coordinate_type),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": ACTION_DESCRIPTION_PROMPT,
                        "enum": [
                            "key",
                            "key_down",
                            "key_up",
                            "left_mouse_down",
                            "left_mouse_up",
                            "left_click",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "triple_click",
                            "mouse_move",
                            "left_click_drag",
                            "type",
                            "scroll",
                            "hscroll",
                            "screenshot",
                            "call_user",
                            "wait",
                            "terminate",
                        ],
                    },
                    "coordinate": {
                        "type": "array",
                        "description": "(x, y) coordinates. Required only by `action=mouse_move` and `action=left_click_drag`, optional for `action=left_mouse_down` and `action=left_mouse_up`.",
                    },
                    "keys": {
                        "type": "array",
                        "description": "Required only by `action=key`, `action=key_down`, or `action=key_up`.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Required only by `action=type` and `action=call_user`.",
                    },
                    "pixels": {
                        "type": "number",
                        "description": "Scroll amount for `scroll` or `hscroll`.",
                    },
                    "time": {
                        "type": "number",
                        "description": "Seconds to wait. Required only by `action=wait`.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Task status for `terminate`.",
                        "enum": ["success", "failure"],
                    },
                    "answer": {
                        "type": "string",
                        "description": "Required when `action=terminate`. For `status=success`, provide final answer text that answers the task instruction. For `status=failure`, briefly summarize why the task failed.",
                    },
                },
            },
        },
    }


def build_system_prompt(width: int, height: int, coordinate_type: str) -> str:
    tool = json.dumps(build_tool_definition(width, height, coordinate_type), ensure_ascii=False)
    date = datetime.datetime.today().strftime("%A, %B %d, %Y")
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        f"{tool}\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
        "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
        "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
        "<IMPORTANT>\nReminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {date}.\n"
        f"- Collapsed screenshots appear as text: {COLLAPSED_SCREENSHOT_TEXT}\n"
        "</IMPORTANT>\n\n# Response format\n\nFor normal UI interaction steps:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "For terminal steps, you may either:\n- output a final natural-language response with no tool call, or\n"
        "- use a terminal tool call such as call_user or terminate.\n\nRules:\n"
        "- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n- Do not output anything after a tool call.\n"
        "- Use call_user when you need user information or confirmation.\n"
        "- Use terminate when you want to explicitly end the task with a success or failure status.\n"
        "- When terminating (success or failure), include `answer`. For success, it must directly answer the task instruction. For failure, it must briefly explain why the task failed.\n"
        "- If the task is infeasible, say so explicitly in the response."
    )


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 16 * 16 * 4 * 12800,
    max_long_side: int = 8192,
) -> tuple[int, int]:
    if height < 2 or width < 2:
        raise ValueError("image dimensions must both be at least 2")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must not exceed 200")
    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)
    resized_height = _round_by_factor(height, factor)
    resized_width = _round_by_factor(width, factor)
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def _process_data_url(data_url: str) -> tuple[str, tuple[int, int], tuple[int, int]]:
    from PIL import Image

    try:
        encoded = data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded)
    except (IndexError, ValueError) as exc:
        raise ValueError("Qwen visual policy requires an inline screenshot data URL") from exc
    image = Image.open(BytesIO(raw))
    original = image.size
    resized_height, resized_width = smart_resize(height=image.height, width=image.width)
    image = image.resize((resized_width, resized_height))
    output = BytesIO()
    image.save(output, format="PNG")
    processed = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{processed}", original, (resized_width, resized_height)


def ensure_empty_think_prefix(response: str) -> str:
    if re.match(r"^\s*<think>.*?</think>\s*", response or "", re.DOTALL):
        return response
    return "<think>\n\n</think>\n\n" + (response or "").lstrip("\n")


def response_text(response: NeMoGymResponse) -> str:
    reasoning: list[str] = []
    content: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) == "reasoning":
            reasoning.extend(str(getattr(summary, "text", "")) for summary in getattr(item, "summary", []))
        elif getattr(item, "type", None) == "message":
            for block in getattr(item, "content", []) or []:
                if getattr(block, "type", None) == "output_text":
                    content.append(str(getattr(block, "text", "")))
    final = "".join(content)
    thought = "\n".join(part for part in reasoning if part).strip()
    return f"<think>\n{thought}\n</think>\n\n{final.lstrip()}" if thought else final


def _update_folding_state(total: int, folded: int, image_max: int, fold_size: int) -> int:
    while total - folded > image_max:
        folded += fold_size
    return min(folded, total)


@dataclass
class QwenPolicyState:
    instruction: str
    max_image_history: int = 20
    fold_size: int = 10
    history_n: int = 100
    coordinate_type: str = "relative"
    screenshots: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    folded_prefix: int = 0
    original_size: tuple[int, int] = (1920, 1080)
    processed_size: tuple[int, int] = (1920, 1080)

    def append_observation(self, observation: WebObservation) -> None:
        screenshot = observation.screenshot
        if screenshot is None or not screenshot.data_url:
            raise ValueError("Qwen visual policy requires an inline browser screenshot")
        processed, self.original_size, self.processed_size = _process_data_url(screenshot.data_url)
        self.screenshots.append(processed)
        self.folded_prefix = _update_folding_state(
            len(self.screenshots), self.folded_prefix, self.max_image_history, self.fold_size
        )

    def record_response(self, text: str, action: WebAction) -> None:
        self.responses.append(text)
        self.actions.append(str(action.metadata.get("natural_language_action") or action.name))

    def messages(self) -> list[NeMoGymEasyInputMessage]:
        total = len(self.screenshots)
        if total == 0:
            raise ValueError("Qwen policy history has no screenshot")
        start = max(1, total - self.history_n)
        result = [
            NeMoGymEasyInputMessage(
                role="system",
                content=[
                    {
                        "type": "input_text",
                        "text": build_system_prompt(*self.processed_size, self.coordinate_type),
                    }
                ],
            )
        ]
        previous = [f"Step {index + 1}: {self.actions[index]}" for index in range(min(start - 1, len(self.actions)))]
        previous_text = "\n".join(previous) if previous else "None"
        for step in range(start, total + 1):
            first = step == start
            collapsed = step <= self.folded_prefix
            if first:
                prompt = (
                    "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
                    f"Instruction: {self.instruction}\n\nPrevious actions:\n{previous_text}"
                )
                parts = [{"type": "input_text", "text": prompt}]
                if not collapsed:
                    parts.insert(0, {"type": "input_image", "image_url": self.screenshots[step - 1], "detail": "high"})
            else:
                observation_parts: list[dict[str, Any]]
                if collapsed:
                    observation_parts = [{"type": "input_text", "text": COLLAPSED_SCREENSHOT_TEXT}]
                else:
                    observation_parts = [
                        {"type": "input_image", "image_url": self.screenshots[step - 1], "detail": "high"}
                    ]
                parts = (
                    [{"type": "input_text", "text": "<tool_response>\n"}]
                    + observation_parts
                    + [{"type": "input_text", "text": "\n</tool_response>"}]
                )
            result.append(NeMoGymEasyInputMessage(role="user", content=parts))
            response_index = step - 1
            if step <= total - 1 and response_index < len(self.responses):
                result.append(
                    NeMoGymEasyInputMessage(
                        role="assistant",
                        content=[
                            {
                                "type": "input_text",
                                "text": ensure_empty_think_prefix(self.responses[response_index]),
                            }
                        ],
                    )
                )
        return result


def _decode_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return stripped
    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _tool_calls(text: str) -> Iterable[dict[str, Any]]:
    for block in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        inner = block.group(1)
        function = re.search(r"<function=([^>]+)>", inner)
        if function is None or function.group(1) != "computer_use":
            continue
        params: dict[str, Any] = {}
        for match in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", inner, re.DOTALL):
            params[match.group(1)] = _decode_jsonish(match.group(2).strip())
        if params:
            yield params


def _parse_coordinate(value: Any) -> tuple[float, float]:
    value = _decode_jsonish(value)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ActionParseError("Qwen computer_use action requires coordinate=[x, y]")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ActionParseError("Qwen coordinate values must be numeric") from exc


def _normalized_coordinate(
    value: Any,
    *,
    coordinate_type: str,
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
) -> list[float]:
    x, y = _parse_coordinate(value)
    if coordinate_type == "relative":
        return [max(0.0, min(1.0, x / 999.0)), max(0.0, min(1.0, y / 999.0))]
    original_width, original_height = original_size
    processed_width, processed_height = processed_size
    pixel_x = x * original_width / processed_width
    pixel_y = y * original_height / processed_height
    return [
        max(0.0, min(1.0, pixel_x / original_width)),
        max(0.0, min(1.0, pixel_y / original_height)),
    ]


def _parse_keys(value: Any) -> list[str]:
    if isinstance(value, str):
        original = value.strip()
        try:
            value = json.loads(original)
        except (json.JSONDecodeError, TypeError, RecursionError):
            try:
                value = ast.literal_eval(original)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                value = original
        if isinstance(value, str):
            parts = re.split(r"\s*\+\s*", value)
            if len(parts) > 1:
                value = parts
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        keys: list[str] = []
        for item in value:
            keys.extend(_parse_keys(item))
        return keys
    token = str(value).strip().strip("[](){}\"'").lower()
    aliases = {
        "cmd": "ctrl",
        "command": "ctrl",
        "control": "ctrl",
        "return": "enter",
        "escape": "esc",
        "option": "alt",
    }
    return [aliases.get(token, token)] if token else []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_coordinate(value: Any) -> str:
    try:
        x, y = _parse_coordinate(value)
    except ActionParseError:
        return "(unknown)"
    return f"({x:.0f}, {y:.0f})"


def _format_computer_action(parameters: dict[str, Any]) -> str:
    """Match the maintained runner's compact previous-action summaries."""

    name = str(parameters.get("action") or "")
    if name in {"left_click", "right_click", "middle_click", "double_click", "triple_click", "mouse_move"}:
        return f"{name.replace('_', ' ')} at {_format_coordinate(parameters.get('coordinate'))}"
    if name == "left_click_drag":
        return (
            f"drag from {_format_coordinate(parameters.get('start_coordinate'))} "
            f"to {_format_coordinate(parameters.get('coordinate'))}"
        )
    if name == "type":
        text = str(parameters.get("text") or "").replace("\n", "\\n")
        return f'type "{text[:77] + "..." if len(text) > 80 else text}"'
    if name == "key":
        keys = _parse_keys(parameters.get("keys"))
        return f"press {'+'.join(keys) if keys else '(no keys)'}"
    if name in {"key_down", "key_up"}:
        keys = _parse_keys(parameters.get("keys"))
        return f"{name.replace('_', ' ')} {'+'.join(keys) if keys else '(no keys)'}"
    if name in {"scroll", "hscroll"}:
        return f"{name} by {parameters.get('pixels', 0)}"
    if name in {"left_mouse_down", "left_mouse_up"}:
        return f"{name.replace('_', ' ')} at {_format_coordinate(parameters.get('coordinate'))}"
    if name == "screenshot":
        return "capture screenshot"
    if name == "call_user":
        text = str(parameters.get("text") or "").replace("\n", "\\n")
        return f"call user {text[:117] + '...' if len(text) > 120 else text}"
    if name == "wait":
        return f"wait {parameters.get('time', 'default')} seconds"
    if name == "terminate":
        return f"terminate with status {parameters.get('status', 'success')}"
    return name or "unknown action"


def _action_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def _explicitly_declines_task(text: str) -> bool:
    """Recognize only an explicit policy refusal in a tool-less response.

    WebVoyager's judge, rather than a broad phrase list, decides whether a
    natural-language answer satisfies the task.  Keeping this fallback narrow
    avoids importing desktop-task assumptions such as account, extension, or
    hardware requirements into a web benchmark.
    """

    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in (
            r"\[infeasible\]",
            r"\b(?:this|the) task is (?:infeasible|impossible)\b",
            r"\b(?:i|we) (?:cannot|can't|am unable to|are unable to) "
            r"(?:complete|finish|perform) (?:this|the) task\b",
        )
    )


def parse_qwen_action(
    text: str,
    *,
    coordinate_type: str,
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
) -> WebAction:
    """Normalize one Qwen XML response into the visual-browser action contract."""

    computer_actions: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    payloads: list[dict[str, Any]] = []
    descriptions: list[str] = []
    status = "error"
    for params in _tool_calls(text):
        name = str(params.get("action") or "")
        if not name:
            continue
        payloads.append({"name": "computer_use", "arguments": params})
        descriptions.append(_format_computer_action(params))

        def coordinate(key: str = "coordinate") -> list[float]:
            return _normalized_coordinate(
                params.get(key),
                coordinate_type=coordinate_type,
                original_size=original_size,
                processed_size=processed_size,
            )

        if name in {"left_click", "right_click", "middle_click", "double_click", "triple_click", "mouse_move"}:
            computer_actions.append({"action": name, "coordinate": coordinate()})
            status = "action"
        elif name == "left_click_drag":
            action = {"action": name, "coordinate": coordinate()}
            if params.get("start_coordinate") is not None:
                action["start_coordinate"] = coordinate("start_coordinate")
            computer_actions.append(action)
            status = "action"
        elif name == "type":
            computer_actions.append({"action": "type", "text": str(params.get("text") or "")})
            status = "action"
        elif name in {"key", "key_down", "key_up"}:
            keys = _parse_keys(params.get("keys"))
            if not keys:
                continue
            computer_actions.append({"action": "key_press" if name == "key" else name, "keys": keys})
            status = "action"
        elif name in {"left_mouse_down", "left_mouse_up"}:
            action = {"action": name}
            if params.get("coordinate") is not None:
                action["coordinate"] = coordinate()
            computer_actions.append(action)
            status = "action"
        elif name in {"scroll", "hscroll"}:
            requested = int(_number(params.get("pixels"), 0.0))
            amount = min(abs(requested), MAX_SCROLL_AMOUNT)
            if name == "scroll":
                direction = "up" if requested >= 0 else "down"
            else:
                direction = "right" if requested >= 0 else "left"
            computer_actions.append(
                {
                    "action": "scroll",
                    "scroll_parameters": {"scroll_direction": direction, "scroll_amount": amount},
                }
            )
            status = "action"
        elif name == "screenshot":
            computer_actions.append({"action": "wait", "duration": 0.0})
            status = "action"
        elif name == "wait":
            computer_actions.append(
                {"action": "wait", "duration": max(0.0, min(_number(params.get("time"), 5), 30.0))}
            )
            status = "action"
        elif name == "call_user":
            # Benchmark rollouts cannot supply an interactive user reply.
            terminal = {"status": "failure", "answer": str(params.get("text") or "") or None}
            status = "failure"
        elif name == "terminate":
            requested_status = str(params.get("status") or "success").lower()
            answer = params.get("answer")
            if answer is None:
                answer = params.get("text")
            terminal = {
                "status": "success" if requested_status == "success" else "failure",
                "answer": None if answer is None else str(answer),
            }
            status = terminal["status"]

    if status == "error":
        terminal = {
            "status": "failure" if _explicitly_declines_task(text) else "success",
            "answer": text.strip() or None,
        }
        status = terminal["status"]

    # The maintained runner parses all calls but skips every UI action when
    # the final parsed status is terminal. Preserve that observable behavior
    # instead of executing a click that happened to precede terminate.
    calls: list[dict[str, Any]] = []
    if status == "action" and computer_actions:
        calls.append({"id": None, "name": "computer", "arguments": {"actions": computer_actions}})
        terminal = None
    elif terminal is not None:
        calls.append({"id": None, "name": "terminate", "arguments": terminal})
    if not calls:
        raise ActionParseError("Qwen response did not contain an executable computer_use action")
    natural_action = _action_line(text)
    if not natural_action and descriptions:
        natural_action = descriptions[0]
    if not natural_action:
        natural_action = "Task failed" if terminal and terminal["status"] == "failure" else "Task completed"
    return WebAction(
        name=calls[0]["name"] if len(calls) == 1 else "computer_use_calls",
        script="",
        arguments={"calls": calls},
        terminal=terminal is not None,
        answer=terminal.get("answer") if terminal is not None else None,
        raw_model_output=text,
        metadata={
            "policy_protocol": "qwen_xml_computer_use",
            "natural_language_action": natural_action,
            "parsed_action": "\n".join(descriptions) if descriptions else natural_action,
            "source_calls": payloads,
        },
    )


__all__ = [
    "COLLAPSED_SCREENSHOT_TEXT",
    "QwenPolicyState",
    "build_system_prompt",
    "build_tool_definition",
    "ensure_empty_think_prefix",
    "parse_qwen_action",
    "response_text",
    "smart_resize",
]
