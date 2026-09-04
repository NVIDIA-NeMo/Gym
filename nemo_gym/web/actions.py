# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe parsing and normalization for visual-browser computer-use actions."""

from __future__ import annotations

import json
import math
from typing import Any, Literal
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
NanoOmniActionRecovery = Literal["strict", "decode_string", "repair_single_closing_bracket"]
NanoOmniToolAliasRecovery = Literal["strict", "webvoyager_v3"]


class ActionParseError(ValueError):
    """Raised when a policy adapter emits an unsafe or unsupported action."""


def _json_container_balance(value: str) -> tuple[int, int, bool]:
    """Return square/curly balance while respecting JSON string literals."""

    square = 0
    curly = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            square += 1
        elif character == "]":
            square -= 1
        elif character == "{":
            curly += 1
        elif character == "}":
            curly -= 1
        if square < 0 or curly < 0:
            return square, curly, in_string
    return square, curly, in_string


def _decode_native_actions(value: Any, recovery: NanoOmniActionRecovery) -> tuple[Any, str]:
    if not isinstance(value, str):
        return value, "strict"
    if recovery == "strict":
        return value, "strict"
    try:
        return json.loads(value), "decoded_inner_string"
    except json.JSONDecodeError as first_error:
        if recovery != "repair_single_closing_bracket":
            raise ActionParseError("native computer actions string is invalid JSON") from first_error
        square, curly, in_string = _json_container_balance(value)
        if not value.lstrip().startswith("[") or square != 1 or curly != 0 or in_string:
            raise ActionParseError(
                "native computer actions string is not eligible for one-bracket recovery"
            ) from first_error
        try:
            return json.loads(value + "]"), "closed_one_missing_bracket"
        except json.JSONDecodeError as recovery_error:
            raise ActionParseError("native computer actions string remains invalid after recovery") from recovery_error


def _native_number(value: Any, *, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionParseError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ActionParseError(f"{field} must be in [{minimum:g}, {maximum:g}]")
    return number


def _native_clamped_number(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
    allow_string: bool = False,
) -> tuple[float, dict[str, Any] | None]:
    """Clamp a finite numeric alias while preserving an audit record."""

    decoded = value
    if allow_string and isinstance(value, str):
        try:
            decoded = float(value.strip())
        except ValueError as exc:
            raise ActionParseError(f"{field} must be a number") from exc
    if isinstance(decoded, bool) or not isinstance(decoded, (int, float)):
        raise ActionParseError(f"{field} must be a number")
    number = float(decoded)
    if not math.isfinite(number):
        raise ActionParseError(f"{field} must be finite")
    normalized = min(max(number, minimum), maximum)
    if normalized == number:
        return normalized, None
    return normalized, {
        "field": field,
        "original": value,
        "normalized": normalized,
        "minimum": minimum,
        "maximum": maximum,
    }


def _native_coordinate(value: Any, *, field: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ActionParseError(f"{field} must contain normalized x and y")
    _native_number(value[0], field=f"{field}[0]", minimum=0, maximum=1)
    _native_number(value[1], field=f"{field}[1]", minimum=0, maximum=1)


def _validate_native_computer_action(
    action: Any,
    index: int,
    *,
    alias_recovery: NanoOmniToolAliasRecovery,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    if not isinstance(action, dict):
        raise ActionParseError(f"native computer action[{index}] must be an object")
    normalized = dict(action)
    name = action.get("action")
    alias_modes: list[str] = []
    alias_details: list[dict[str, Any]] = []
    if name == "click" and alias_recovery == "webvoyager_v3":
        name = "left_click"
        normalized["action"] = name
        alias_modes.append("computer.click_to_left_click")
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
        if alias_recovery == "webvoyager_v3":
            duration, detail = _native_clamped_number(
                action.get("duration"),
                field=f"computer.actions[{index}].duration",
                minimum=0,
                maximum=30,
            )
            if detail is not None:
                normalized["duration"] = duration
                alias_modes.append("computer.wait_duration_clamped")
                alias_details.append(detail)
        else:
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
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ActionParseError(f"{prefix}.scroll_amount must be a non-negative integer")
    return normalized, alias_modes, alias_details


def _native_alias_number(value: Any, *, field: str, minimum: float, maximum: float) -> float:
    decoded = value
    if isinstance(value, str):
        try:
            decoded = float(value.strip())
        except ValueError as exc:
            raise ActionParseError(f"{field} must be a number") from exc
    return _native_number(decoded, field=field, minimum=minimum, maximum=maximum)


def _native_alias_coordinate(value: Any, *, field: str) -> list[float]:
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"{field} must be a JSON coordinate array") from exc
    if not isinstance(decoded, (list, tuple)) or len(decoded) != 2:
        raise ActionParseError(f"{field} must contain normalized x and y")
    return [
        _native_alias_number(decoded[0], field=f"{field}[0]", minimum=0, maximum=1),
        _native_alias_number(decoded[1], field=f"{field}[1]", minimum=0, maximum=1),
    ]


def _normalize_native_tool_alias(
    name: Any,
    arguments: dict[str, Any],
    *,
    alias_recovery: NanoOmniToolAliasRecovery,
) -> tuple[Any, dict[str, Any], list[str], list[dict[str, Any]]]:
    """Normalize only unambiguous public-v3 tool aliases.

    The native benchmark contract remains strict by default. The opt-in mode
    accepts shapes observed in the public Nano Omni v3 transport logs and
    rejects fields whose intended browser semantics cannot be proven.
    """

    if alias_recovery == "strict":
        return name, arguments, [], []

    keys = set(arguments)
    if name in {"click", "left_click"}:
        coordinate: list[float] | None = None
        mode: str | None = None
        if keys == {"x", "y"}:
            x = _native_alias_number(arguments["x"], field="native click.x", minimum=0, maximum=1)
            y = _native_alias_number(arguments["y"], field="native click.y", minimum=0, maximum=1)
            coordinate = [x, y]
            mode = f"tool.{name}_xy_to_computer_left_click"
        elif keys <= {"action", "coordinate"} and "coordinate" in arguments:
            declared_action = arguments.get("action")
            if declared_action not in {None, "click", "left_click"}:
                return name, arguments, [], []
            coordinate = _native_alias_coordinate(arguments["coordinate"], field="native click.coordinate")
            mode = f"tool.{name}_coordinate_to_computer_left_click"
        if coordinate is not None and mode is not None:
            return (
                "computer",
                {"actions": [{"action": "left_click", "coordinate": coordinate}]},
                [mode],
                [],
            )

    if name == "type" and keys <= {"action", "text"}:
        if arguments.get("action") in {None, "type"} and isinstance(arguments.get("text"), str):
            return (
                "computer",
                {"actions": [{"action": "type", "text": arguments["text"]}]},
                ["tool.type_to_computer_type"],
                [],
            )

    if name == "wait" and keys <= {"action", "duration"}:
        if arguments.get("action") in {None, "wait"}:
            duration, detail = _native_clamped_number(
                arguments.get("duration"),
                field="tool.wait.duration",
                minimum=0,
                maximum=30,
                allow_string=True,
            )
            modes = ["tool.wait_to_computer_wait"]
            details: list[dict[str, Any]] = []
            if detail is not None:
                modes.append("tool.wait_duration_clamped")
                details.append(detail)
            return (
                "computer",
                {"actions": [{"action": "wait", "duration": duration}]},
                modes,
                details,
            )

    return name, arguments, [], []


def _validate_native_tool_arguments(
    name: str,
    arguments: dict[str, Any],
    *,
    recovery: NanoOmniActionRecovery,
    alias_recovery: NanoOmniToolAliasRecovery,
    max_computer_actions: int,
) -> tuple[dict[str, Any], str, int, list[str], list[dict[str, Any]]]:
    normalized = dict(arguments)
    alias_modes: list[str] = []
    alias_details: list[dict[str, Any]] = []
    if name == "computer":
        actions, recovery_mode = _decode_native_actions(arguments.get("actions"), recovery)
        if not isinstance(actions, list) or not actions:
            raise ActionParseError("native computer tool requires a non-empty actions list")
        if len(actions) > max_computer_actions:
            raise ActionParseError(f"native computer tool exceeded the {max_computer_actions}-action batch limit")
        validated_actions: list[dict[str, Any]] = []
        for index, action in enumerate(actions):
            validated, action_aliases, action_details = _validate_native_computer_action(
                action,
                index,
                alias_recovery=alias_recovery,
            )
            validated_actions.append(validated)
            alias_modes.extend(action_aliases)
            alias_details.extend(action_details)
        normalized["actions"] = validated_actions
        return normalized, recovery_mode, len(actions), alias_modes, alias_details
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
    return normalized, "strict", 0, alias_modes, alias_details


def parse_nano_omni_tool_calls(
    items: list[Any],
    *,
    max_calls: int = 8,
    max_computer_actions: int = 20,
    recovery: NanoOmniActionRecovery = "strict",
    alias_recovery: NanoOmniToolAliasRecovery = "strict",
) -> WebAction:
    """Validate Nano Omni function calls without executing arbitrary code."""

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
        original_name = name
        name, arguments, alias_modes, alias_details = _normalize_native_tool_alias(
            name,
            arguments,
            alias_recovery=alias_recovery,
        )
        if name not in NANO_OMNI_TOOL_NAMES:
            raise ActionParseError(f"unsupported Nano Omni browser tool: {name!r}")
        (
            arguments,
            recovery_mode,
            computer_actions,
            action_alias_modes,
            action_alias_details,
        ) = _validate_native_tool_arguments(
            name,
            arguments,
            recovery=recovery,
            alias_recovery=alias_recovery,
            max_computer_actions=max_computer_actions,
        )
        alias_modes.extend(action_alias_modes)
        alias_details.extend(action_alias_details)
        calls.append({"id": call_id, "name": name, "arguments": arguments})
        parse_records.append(
            {
                "call_id": call_id,
                "tool": name,
                "original_tool": original_name,
                "recovery_mode": recovery_mode,
                "alias_recovery_modes": alias_modes,
                "alias_recovery_details": alias_details,
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
        metadata={
            "nano_omni_parse": {
                "calls": parse_records,
                "recovered": any(
                    record["recovery_mode"] != "strict" or record["alias_recovery_modes"] for record in parse_records
                ),
            }
        },
    )


__all__ = [
    "ActionParseError",
    "MAX_SCROLL_AMOUNT",
    "NanoOmniActionRecovery",
    "NanoOmniToolAliasRecovery",
    "parse_nano_omni_tool_calls",
]
