# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Super 3.5 VL tool-XML protocol implemented against Gym's agent contract.

The wire grammar is documented in the public model's chat template:
https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/blob/
fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe/chat_template.jinja

This is an independent Gym scaffold; no external OSWorld agent implementation
is loaded or vendored. Its prompt and protocol identity are specific to Gym.
"""

from __future__ import annotations

import ast
import base64
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from responses_api_agents.osworld_agent.model_protocol import OSWorldModelProtocol, ProtocolMessageContext


SUPER35_PROTOCOL_ID = "nemotron-super35-tool-xml-v1"
_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_CALL = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_.-]*)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>", re.DOTALL)
_XY_ARGUMENTS = {
    method: (0, 1)
    for method in (
        "click",
        "rightClick",
        "middleClick",
        "doubleClick",
        "tripleClick",
        "moveTo",
        "dragTo",
        "mouseDown",
        "mouseUp",
    )
}
_XY_ARGUMENTS.update({method: (1, 2) for method in ("scroll", "hscroll", "vscroll")})

SYSTEM_PROMPT = """You operate a desktop through screenshots. Complete the user's task by choosing one action per turn.
The desktop password is {password}.

Use exec_py to run Python source for mouse and keyboard control. For PyAutoGUI mouse positions, use literal
normalized x and y coordinates in [0, 1], measured from the screenshot's top-left corner. Do not put expressions
or variables in coordinate arguments. Use computer.wait to wait for the next observation, and computer.terminate
to finish. Report success only after checking the requested outcome. Report failure only when evidence shows
that the task is infeasible; an unfamiliar UI, a failed action, or a low remaining step budget is insufficient.

<tools>
<function>
<name>exec_py</name>
<description>Execute Python for desktop control.</description>
<parameters>
<parameter><name>code_block</name><type>string</type></parameter>
<required>["code_block"]</required>
</parameters>
</function>
<function>
<name>computer.wait</name>
<description>Wait for the next desktop observation. No arguments.</description>
<parameters><required>[]</required></parameters>
</function>
<function>
<name>computer.terminate</name>
<description>Finish the task with an explicit status and optional answer.</description>
<parameters>
<parameter><name>status</name><type>string</type><enum>["success", "failure"]</enum></parameter>
<parameter><name>answer</name><type>string</type></parameter>
<required>["status"]</required>
</parameters>
</function>
</tools>

Give a short action description followed by exactly one tool call in this format, with nothing after it:
<tool_call>
<function=exec_py>
<parameter=code_block>
pyautogui.click(0.5, 0.5)
</parameter>
</function>
</tool_call>
Reasoning, when enabled, belongs in <think>...</think> before the action description.
""".strip()


def _field(response: object, name: str) -> Any:
    return response.get(name) if isinstance(response, Mapping) else getattr(response, name, None)


def _tool_xml(name: str, arguments: Mapping[str, str]) -> str:
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in arguments.items():
        lines.extend([f"<parameter={key}>", value, "</parameter>"])
    return "\n".join([*lines, "</function>", "</tool_call>"])


def _arguments(value: object) -> dict[str, str]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in decoded.items()
    ):
        raise ValueError("Tool arguments must be a JSON object with string values")
    return dict(decoded)


def _tool_call(response: object, content: str) -> tuple[str, dict[str, str], str]:
    calls = _field(response, "tool_calls")
    if calls:
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("One tool call is required per desktop step")
        function = _field(calls[0], "function")
        name = _field(function, "name")
        if not isinstance(name, str):
            raise ValueError("Tool call has no function name")
        return name, _arguments(_field(function, "arguments")), content.strip()

    matches = list(_CALL.finditer(content))
    if len(matches) == 1:
        match = matches[0]
        if content[match.end() :].strip() or "<tool_call>" in content[: match.start()]:
            raise ValueError("Only one tool call is allowed, with no text after it")
        parameters: dict[str, str] = {}
        end = 0
        body = match[2]
        for parameter in _PARAMETER.finditer(body):
            if body[end : parameter.start()].strip() or parameter[1] in parameters:
                raise ValueError("Malformed or duplicate tool parameter")
            parameters[parameter[1]] = parameter[2].removeprefix("\n").removesuffix("\n")
            end = parameter.end()
        if body[end:].strip():
            raise ValueError("Malformed tool parameters")
        return match[1], parameters, content[: match.start()].strip()

    # The existing Gym transport also accepts native OpenAI tool_calls and
    # serializes them into a JSON tool_call block for text-based agents.
    wrapped = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>\s*$", content, re.DOTALL)
    if wrapped and "<tool_call>" not in content[: wrapped.start()]:
        value = json.loads(wrapped[1])
        if isinstance(value, Mapping) and isinstance(value.get("name"), str):
            return value["name"], _arguments(value.get("arguments")), content[: wrapped.start()].strip()
    raise ValueError("Expected one complete function/parameter XML tool call")


def _project_coordinates(code: str, screen_size: tuple[int, int], coordinate_type: str) -> str:
    if coordinate_type != "relative":
        raise ValueError("Super 3.5 tool-XML requires normalized relative coordinates")
    tree = ast.parse(code, mode="exec")
    if not tree.body:
        raise ValueError("exec_py requires executable Python source")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "pyautogui":
            continue
        positions = _XY_ARGUMENTS.get(node.func.attr)
        if positions is None:
            continue
        for axis, position, size in zip(("x", "y"), positions, screen_size, strict=True):
            keyword = next((entry for entry in node.keywords if entry.arg == axis), None)
            argument = node.args[position] if len(node.args) > position else keyword.value if keyword else None
            if argument is None:
                continue
            try:
                value = ast.literal_eval(argument)
            except (ValueError, TypeError) as exc:
                raise ValueError("Mouse coordinates must be literal normalized numbers") from exc
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Mouse coordinates must be finite normalized numbers")
            if not 0 <= value <= 1:
                raise ValueError("Mouse coordinates must be in [0, 1]")
            pixel = ast.copy_location(ast.Constant(round(value * size)), argument)
            if len(node.args) > position:
                node.args[position] = pixel
            elif keyword is not None:
                keyword.value = pixel
    return ast.unparse(tree)


def parse_super35_response(
    response: object,
    *,
    screen_size: tuple[int, int],
    coordinate_type: str,
    thinking: bool,
) -> tuple[str, list[str], dict[str, Any]]:
    """Convert raw tool XML or native OpenAI tool calls into validated actions."""

    content = response if isinstance(response, str) else _field(response, "content") or ""
    if not isinstance(content, str):
        raise ValueError("Super 3.5 response content must be text")
    reasoning = _field(response, "reasoning_content") or _field(response, "reasoning") or ""
    inline_thought = _THINK.search(content)
    if inline_thought:
        reasoning = reasoning or inline_thought[1]
        content = content[: inline_thought.start()] + content[inline_thought.end() :]
    name, arguments, summary = _tool_call(response, content)
    if name == "exec_py":
        if set(arguments) != {"code_block"} or not arguments["code_block"].strip():
            raise ValueError("exec_py requires only a nonempty code_block")
        code = _project_coordinates(arguments["code_block"], screen_size, coordinate_type)
        commands = [code]
        summary = summary or "Execute the desktop action."
    elif name == "computer.wait":
        if arguments:
            raise ValueError("computer.wait takes no arguments")
        commands = ["WAIT"]
        summary = summary or "Wait for the next observation."
    elif name == "computer.terminate":
        if set(arguments) - {"status", "answer"} or arguments.get("status") not in {"success", "failure"}:
            raise ValueError("computer.terminate requires status success or failure and an optional answer")
        commands = ["DONE" if arguments["status"] == "success" else "FAIL"]
        summary = summary or arguments.get("answer") or "Finish the task."
    else:
        raise ValueError(f"Unknown Super 3.5 desktop tool: {name!r}")
    return (
        summary,
        commands,
        {
            "thought": str(reasoning).strip() if thinking else "",
            "action": summary,
            "code": commands[0],
            "original_code": arguments.get("code_block", ""),
            "tool_call_xml": _tool_xml(name, arguments),
        },
    )


def render_super35_messages(context: ProtocolMessageContext) -> list[dict[str, Any]]:
    """Keep the task first, then user observations and complete assistant tool turns."""

    messages: list[dict[str, Any]] = [{"role": "system", "content": context.system_prompt}]
    observations = [*context.observations, context.current_observation]
    for turn, observation in enumerate(observations):
        text = (
            f"Task: {context.instruction}\n"
            if turn == 0
            else 'Computer observation:\n<tool_response>\nresult:\n{"status": "unknown"}\n'
        )
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if turn in context.image_turns:
            encoded = base64.b64encode(observation["screenshot"]).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
        if turn:
            parts.append({"type": "text", "text": "\n</tool_response>"})
        if turn == len(context.observations) and context.guidance:
            parts.append({"type": "text", "text": "\n" + context.guidance + "\n"})
        messages.append({"role": "user", "content": parts})
        if turn < len(context.assistant_history):
            messages.append({"role": "assistant", "content": context.assistant_history[turn]})
    return messages


SUPER35_PROTOCOL = OSWorldModelProtocol(
    protocol_id=SUPER35_PROTOCOL_ID,
    parser_id="nemotron-super35-tool-xml-parser-v1",
    instruction_template="Task: {instruction}\n",
    step_template="Computer observation:\n<tool_response>\nresult:\n",
    text_history_template="{tool_call_xml}",
    system_prompt_thinking=SYSTEM_PROMPT,
    system_prompt_non_thinking=SYSTEM_PROMPT,
    assistant_history_template_thinking="<think>\n{thought}\n</think>\n{action}\n{tool_call_xml}",
    assistant_history_template_non_thinking="{action}\n{tool_call_xml}",
    parser=parse_super35_response,
    message_renderer=render_super35_messages,
    message_renderer_id="nemotron-super35-user-observations-v1",
    chat_template_kwargs_thinking={"enable_thinking": True, "truncate_history_thinking": False},
    chat_template_kwargs_non_thinking={"enable_thinking": False, "truncate_history_thinking": False},
    parse_retry_template=(
        "The previous tool call could not be executed: {error}\n"
        "Return one corrected <tool_call> with a <function=...> and its <parameter=...> values. "
        "Use exec_py, computer.wait, or computer.terminate. Do not terminate merely because parsing failed."
    ),
)
