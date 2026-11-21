# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate the pinned converter's canonical actions into desktop commands.

convert_procua_sft.py:assistant_plain/tool_call_xml and
sft_coordinate_normalization.py:normalize_pyautogui_code/_normalize_pixel_axis.
Independent wire decoder; reference validation is called without copying it.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .coordinate_compat import STEP1500_COORDINATES, STRICT_COORDINATES, compile_compatible
from .reference import Reference


PROTOCOL_ID = "nemotron-v4-prompt-format-35d35a7c-rfc0015-feedback-v2"
_THINK = re.compile(r"\A\s*<think>(.*?)</think>", re.DOTALL)
_TOOL = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_.]*)>\s*"
    r"(.*?)\s*</function>\s*</tool_call>\s*\Z",
    re.DOTALL,
)
_PARAMETER = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate tool argument: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict[str, Any]
    reasoning: str
    summary: str
    projected_source: str = ""
    coordinate_policy: str = STRICT_COORDINATES

    @property
    def model_source(self) -> str:
        return self.arguments.get("code_block", "") if self.name == "exec_py" else ""

    @property
    def desktop_action(self) -> str:
        if self.name == "computer.wait":
            return "WAIT"
        if self.name == "computer.terminate":
            return "DONE" if self.arguments["status"] == "success" else "FAIL"
        return self.projected_source


def project_coordinates(
    code: str, screen_size: tuple[int, int], reference: Reference, coordinate_policy: str = STRICT_COORDINATES
) -> str:
    """Validate with the producer and invert its closed pixel-index convention."""
    if coordinate_policy == STEP1500_COORDINATES:
        return compile_compatible(code, screen_size, reference.coordinates.PYAUTOGUI_PARAM_NAMES)
    if coordinate_policy != STRICT_COORDINATES:
        raise ValueError(f"Unknown v4 coordinate policy: {coordinate_policy}")
    reference.coordinates.normalize_pyautogui_code(code, screen_size=None)
    tree = ast.parse(code, mode="exec")
    if not tree.body:
        raise ValueError("exec_py requires nonempty executable Python source")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "pyautogui":
            continue
        parameters = reference.coordinates.PYAUTOGUI_PARAM_NAMES.get(node.func.attr)
        if not parameters:
            continue
        for axis, size in zip(("x", "y"), screen_size, strict=True):
            position = parameters.index(axis)
            keyword = next((kw for kw in node.keywords if kw.arg == axis), None)
            value_node = node.args[position] if len(node.args) > position else keyword.value if keyword else None
            if value_node is None:
                continue
            # The reference validator has already rejected aliases, expressions,
            # booleans, missing axes, and non-unit/unsupported coordinate calls.
            value = ast.literal_eval(value_node)
            replacement = ast.copy_location(ast.Constant(round(value * (size - 1))), value_node)
            if len(node.args) > position:
                node.args[position] = replacement
            else:
                keyword.value = replacement
    projected = ast.unparse(tree)
    compile(projected, "<nemotron-v4-exec-py>", "exec")
    return projected


def parse_action(
    response: Any, *, reference: Reference, screen_size: tuple[int, int], coordinate_policy: str = STRICT_COORDINATES
) -> Action:
    finish = _field(response, "finish_reason")
    if finish is not None and finish not in {"stop", "tool_calls"}:
        raise ValueError(f"Incomplete model response: finish_reason={finish!r}")
    content = response if isinstance(response, str) else _field(response, "content", "") or ""
    if not isinstance(content, str):
        raise ValueError("Expected text model content")
    reasoning = _field(response, "reasoning_content") or _field(response, "reasoning") or ""
    inline = _THINK.match(content)
    if inline:
        reasoning = reasoning or inline.group(1)
        content = content[inline.end() :]
    calls = _field(response, "tool_calls")
    if calls:
        if len(calls) != 1:
            raise ValueError("The canonical converter emits one tool call per action")
        function = _field(calls[0], "function")
        name = _field(function, "name")
        encoded = _field(function, "arguments")
        arguments = json.loads(encoded, object_pairs_hook=_unique_object) if isinstance(encoded, str) else encoded
        summary = content.strip()
    else:
        match = _TOOL.search(content)
        if not match or "<tool_call>" in content[: match.start()]:
            raise ValueError("Expected the reference function/parameter XML action")
        name, body = match.group(1, 2)
        arguments = {}
        consumed = 0
        for parameter in _PARAMETER.finditer(body):
            if body[consumed : parameter.start()].strip() or parameter.group(1) in arguments:
                raise ValueError("Malformed or duplicate XML tool parameter")
            arguments[parameter.group(1)] = parameter.group(2)
            consumed = parameter.end()
        if body[consumed:].strip():
            raise ValueError("Malformed XML tool parameters")
        summary = content[: match.start()].strip()
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    if name == "exec_py":
        if (
            set(arguments) != {"code_block"}
            or not isinstance(arguments["code_block"], str)
            or not arguments["code_block"].strip()
        ):
            raise ValueError("exec_py requires a nonempty string code_block")
        projected = project_coordinates(arguments["code_block"], screen_size, reference, coordinate_policy)
    elif name == "computer.wait":
        if arguments:
            raise ValueError("The pinned argument_free_v1 profile accepts no wait arguments")
        projected = ""
    elif name == "computer.terminate":
        if set(arguments) - {"status", "answer"} or arguments.get("status") not in {"success", "failure"}:
            raise ValueError("computer.terminate requires status success or failure and optional answer")
        if "answer" in arguments and not isinstance(arguments["answer"], str):
            raise ValueError("computer.terminate answer must be a string")
        projected = ""
    else:
        raise ValueError(f"Tool is not in the pinned reference schema: {name!r}")
    return Action(str(name), dict(arguments), str(reasoning).strip(), summary, projected, coordinate_policy)
