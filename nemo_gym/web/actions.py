# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe parsing for BrowserGym and legacy WebVoyager model actions."""

from __future__ import annotations

import ast
import re
from typing import Any

from nemo_gym.web.models import WebAction, WebActionProfile


ALLOWED_ACTIONS = frozenset(
    {
        "clear",
        "click",
        "dblclick",
        "drag_and_drop",
        "fill",
        "focus",
        "go_back",
        "go_forward",
        "goto",
        "hover",
        "keyboard_press",
        "new_tab",
        "noop",
        "press",
        "report_infeasible",
        "scroll",
        "select_option",
        "send_msg_to_user",
        "tab_close",
        "tab_focus",
        "upload_file",
    }
)
TERMINAL_ACTIONS = frozenset({"send_msg_to_user", "report_infeasible"})


class ActionParseError(ValueError):
    """Raised when model output does not contain a safe supported action."""


def _strip_model_scaffolding(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[-1].strip()

    action_match = re.search(r"(?:^|\n)\s*Action\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if action_match:
        return action_match.group(1).strip()
    return text.strip()


def _literal(value: ast.AST) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ActionParseError("action arguments must be Python literals") from exc


def parse_browsergym_action(text: str, *, max_calls: int = 2) -> WebAction:
    candidate = _strip_model_scaffolding(text)
    if not candidate:
        raise ActionParseError("model output did not contain an action")
    try:
        tree = ast.parse(candidate, mode="exec")
    except SyntaxError as exc:
        raise ActionParseError(f"invalid action syntax: {exc.msg}") from exc

    if not 1 <= len(tree.body) <= max_calls:
        raise ActionParseError(f"expected between 1 and {max_calls} action calls")

    calls: list[tuple[str, list[Any], dict[str, Any], ast.Call]] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ActionParseError("each action must be a direct function call")
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id not in ALLOWED_ACTIONS:
            raise ActionParseError("unsupported browser action")
        if any(keyword.arg is None for keyword in call.keywords):
            raise ActionParseError("expanded keyword arguments are not allowed")
        args = [_literal(arg) for arg in call.args]
        kwargs = {keyword.arg: _literal(keyword.value) for keyword in call.keywords if keyword.arg is not None}
        calls.append((call.func.id, args, kwargs, call))

    names = [name for name, *_ in calls]
    terminal_names = [name for name in names if name in TERMINAL_ACTIONS]
    if terminal_names and names[-1] not in TERMINAL_ACTIONS:
        raise ActionParseError("a terminal action must be the final call")

    answer = None
    if terminal_names:
        _, args, kwargs, _ = calls[-1]
        value = args[0] if args else kwargs.get("text", kwargs.get("reason"))
        answer = "" if value is None else str(value)

    script = "\n".join(ast.unparse(call) for *_, call in calls)
    arguments: dict[str, Any]
    if len(calls) == 1:
        _, args, kwargs, _ = calls[0]
        arguments = {"args": args, "kwargs": kwargs}
    else:
        arguments = {"calls": [{"name": name, "args": args, "kwargs": kwargs} for name, args, kwargs, _ in calls]}
    return WebAction(
        name=names[0] if len(names) == 1 else "multi_action",
        script=script,
        arguments=arguments,
        terminal=bool(terminal_names),
        answer=answer,
        raw_model_output=text,
    )


def _legacy_webvoyager_action(text: str) -> WebAction:
    candidate = _strip_model_scaffolding(text).strip().rstrip(".")

    answer_match = re.fullmatch(r"ANSWER\s*[;:]?\s*\[?(.*?)\]?", candidate, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        return parse_browsergym_action(f"send_msg_to_user({answer!r})")

    click_match = re.fullmatch(r"Click\s*\[([^\]]+)\]", candidate, flags=re.IGNORECASE)
    if click_match:
        return parse_browsergym_action(f"click({click_match.group(1).strip()!r})")

    type_match = re.fullmatch(
        r"Type\s*\[([^\]]+)\]\s*;\s*\[?(.*?)\]?",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if type_match:
        bid, value = type_match.group(1).strip(), type_match.group(2).strip()
        return parse_browsergym_action(f"fill({bid!r}, {value!r})\nkeyboard_press('Enter')")

    scroll_match = re.fullmatch(r"Scroll(?:\s*\[[^\]]+\])?\s*;?\s*(up|down)", candidate, flags=re.IGNORECASE)
    if scroll_match:
        dy = -500 if scroll_match.group(1).lower() == "up" else 500
        return parse_browsergym_action(f"scroll(0, {dy})")

    if re.fullmatch(r"Wait", candidate, flags=re.IGNORECASE):
        return parse_browsergym_action("noop()")
    if re.fullmatch(r"GoBack", candidate, flags=re.IGNORECASE):
        return parse_browsergym_action("go_back()")
    if re.fullmatch(r"Google", candidate, flags=re.IGNORECASE):
        return parse_browsergym_action("goto('https://www.google.com/')")

    return parse_browsergym_action(text)


def parse_model_action(text: str, profile: WebActionProfile | str) -> WebAction:
    profile = WebActionProfile(profile)
    if profile == WebActionProfile.WEBVOYAGER_LEGACY:
        action = _legacy_webvoyager_action(text)
        return action.model_copy(update={"raw_model_output": text})
    return parse_browsergym_action(text)
