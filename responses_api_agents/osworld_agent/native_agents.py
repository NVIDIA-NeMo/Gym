# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-owned Nemotron OSWorld agents.

The internal OSWorld ``nemotron-v3`` branch carried model-specific prompt,
history, parsing, and coordinate logic in ``mm_agents``. Keeping that logic
here lets Gym use an unmodified OSWorld dependency: the adapter injects the
actual model call at runtime while OSWorld continues to own DesktopEnv and
task evaluation. ``NemotronOmniAgent`` specializes the same protocol for the
hosted Nemotron 3 Nano Omni endpoint, which accepts only one image per prompt.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Mapping, Tuple


LOG = logging.getLogger("nemo_gym.osworld_agent.native_agents")


def _jsonable(value: Any) -> Any:
    """Convert parser evidence to JSON without changing rollout behavior."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _append_agent_io(event: Dict[str, Any]) -> None:
    """Append parser evidence beside the agent model-I/O events."""

    path = os.environ.get("OSWORLD_MODEL_IO_LOG", "").strip()
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(event), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        LOG.exception("Failed to append OSWorld parser evidence to %s", path)


INSTRUCTION_TEMPLATE = (
    "# Task Instruction:\n{instruction}\n\n"
    "Please generate the next move according to the screenshot, task instruction "
    "and previous steps (if provided).\n"
)
STEP_TEMPLATE = "# Step {step_num}:\n"
TEXT_HISTORY_TEMPLATE = "## Thought:\n{thought}\n\n## Action:\n{action}\n## Code:\n```python\n{code}\n```\n"
ASSISTANT_HISTORY_TEMPLATE_THINKING = (
    "<think>\n{thought}\n</think>\n## Action:\n{action}\n## Code:\n```python\n{code}\n```\n"
)
ASSISTANT_HISTORY_TEMPLATE_NON_THINKING = (
    "## Thought:\n{thought}\n\n## Action:\n{action}\n## Code:\n```python\n{code}\n```\n"
)

SYSTEM_PROMPT_THINKING = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The password of the computer is {password}.

For each step, provide your response in this format:
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}
""".strip()

SYSTEM_PROMPT_NON_THINKING = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The password of the computer is {password}.

For each step, provide your response in this format:
## Thought
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}
""".strip()

OMNI_MINI_SYSTEM_PROMPT = """
You are a GUI agent. You are given an instruction, a screenshot of the screen,
and your previous interactions with the computer. You need to perform a series
of actions to complete the task. The password of the computer is __PASSWORD__.

For each step, provide your response in exactly this format:
{thought}
## Action:
{a concise description of the next action}
## Code:
```python
{code}
```

In the code section, return either pyautogui code or exactly one of the two
function calls below. Use absolute pixel coordinates from the 1920x1080
screenshot. Do not use pyautogui.screenshot() or image matching. Prefer one
short, verifiable action per turn. Every code block must be self-contained.

- Wait: {"name":"computer.wait","arguments":{}}
- Finish: {"name":"computer.terminate","arguments":{"status":"success","answer":"optional answer"}}
- Give up only when impossible: {"name":"computer.terminate","arguments":{"status":"failure","answer":"reason"}}

Do not claim success until the visible UI confirms the requested final state.
""".strip()

_CODE_BLOCK_RE = re.compile(r"```(?:code|python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think(?:ing)?>\s*(.*?)\s*</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_PYAUTOGUI_CALL_RE = re.compile(
    r"pyautogui\.(click|rightClick|middleClick|doubleClick|tripleClick|moveTo|dragTo)\(([^()\n]*)\)"
)
_XY_POSITIONAL_METHODS = {
    "click",
    "rightClick",
    "middleClick",
    "doubleClick",
    "tripleClick",
    "moveTo",
    "dragTo",
}


def _encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


def _response_parts(response: Any) -> tuple[str, str]:
    """Return ``(content, reasoning)`` for OpenAI-style strings or messages."""

    if isinstance(response, str):
        match = _THINK_RE.search(response)
        return response, match.group(1).strip() if match else ""
    if isinstance(response, Mapping):
        content = response.get("content") or ""
        reasoning = response.get("reasoning_content") or response.get("reasoning") or ""
        return str(content), str(reasoning)

    content = getattr(response, "content", "") or ""
    reasoning = getattr(response, "reasoning_content", "") or getattr(response, "reasoning", "") or ""
    return str(content), str(reasoning)


def normalize_response_content(content: str) -> str:
    """Normalize serialized newlines outside fenced code blocks.

    The public BF16 checkpoint can emit literal ``\\n`` around Action/Code
    headings. Preserve escapes inside Python fences while restoring the prose
    structure expected by the internal Nemotron parser.
    """

    if not isinstance(content, str):
        return ""
    segments = content.split("```")
    for index in range(0, len(segments), 2):
        segments[index] = segments[index].replace("\\r\\n", "\n").replace("\\n", "\n")
    return "```".join(segments)


def normalize_python_code_newlines(code: str) -> str:
    """Restore structural ``\\n`` escapes while preserving string literals.

    The model occasionally emits a fenced action such as
    ``\\npyautogui.click(...)\\n``.  Decoding every escape would corrupt valid
    payloads like ``pyautogui.write('first\\nsecond')``.  This small lexical
    scanner restores newline escapes only outside Python string literals.
    Syntax validation later in the pipeline remains the final safety check.
    """

    if not isinstance(code, str) or "\\n" not in code:
        return code

    output: List[str] = []
    quote = ""
    in_comment = False
    index = 0
    while index < len(code):
        if quote:
            if code.startswith(quote, index):
                output.append(quote)
                index += len(quote)
                quote = ""
            elif code[index] == "\\" and index + 1 < len(code):
                output.append(code[index : index + 2])
                index += 2
            else:
                output.append(code[index])
                index += 1
            continue

        if code.startswith("\\r\\n", index):
            output.append("\n")
            index += 4
            in_comment = False
            continue
        if code.startswith("\\n", index):
            output.append("\n")
            index += 2
            in_comment = False
            continue
        if code[index] == "\n":
            output.append("\n")
            index += 1
            in_comment = False
            continue
        if in_comment:
            output.append(code[index])
            index += 1
            continue
        if code[index] == "#":
            in_comment = True
            output.append(code[index])
            index += 1
            continue
        if code.startswith("'''", index) or code.startswith('"""', index):
            quote = code[index : index + 3]
            output.append(quote)
            index += 3
            continue
        if code[index] in {"'", '"'}:
            quote = code[index]
        output.append(code[index])
        index += 1
    return "".join(output)


def _literal_number(node: ast.AST) -> float | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _project_xy(
    x: float,
    y: float,
    *,
    screen_width: int,
    screen_height: int,
    coordinate_type: str,
) -> tuple[int, int]:
    if coordinate_type == "absolute":
        return round(x), round(y)
    if coordinate_type == "qwen25":
        return round(x / 1000 * screen_width), round(y / 1000 * screen_height)
    if abs(x) <= 1.0 and abs(y) <= 1.0:
        return round(x * screen_width), round(y * screen_height)
    return round(x), round(y)


def project_pyautogui_coordinates(
    code: str,
    *,
    screen_width: int,
    screen_height: int,
    coordinate_type: str = "relative",
) -> str:
    """Project literal x/y pairs in common pyautogui calls to screen pixels."""

    if coordinate_type not in {"relative", "absolute", "qwen25"}:
        raise ValueError(f"Unsupported coordinate_type: {coordinate_type}")

    def replace(match: re.Match[str]) -> str:
        method, arguments = match.groups()
        try:
            parsed = ast.parse(f"f({arguments})", mode="eval").body
        except SyntaxError:
            return match.group(0)
        if not isinstance(parsed, ast.Call) or method not in _XY_POSITIONAL_METHODS:
            return match.group(0)

        x_node = parsed.args[0] if len(parsed.args) > 0 else None
        y_node = parsed.args[1] if len(parsed.args) > 1 else None
        x_keyword = next((kw for kw in parsed.keywords if kw.arg == "x"), None)
        y_keyword = next((kw for kw in parsed.keywords if kw.arg == "y"), None)
        x_node = x_node or (x_keyword.value if x_keyword else None)
        y_node = y_node or (y_keyword.value if y_keyword else None)
        if x_node is None or y_node is None:
            return match.group(0)

        x_value = _literal_number(x_node)
        y_value = _literal_number(y_node)
        if x_value is None or y_value is None:
            return match.group(0)
        projected_x, projected_y = _project_xy(
            x_value,
            y_value,
            screen_width=screen_width,
            screen_height=screen_height,
            coordinate_type=coordinate_type,
        )

        if len(parsed.args) > 0:
            parsed.args[0] = ast.Constant(projected_x)
        elif x_keyword is not None:
            x_keyword.value = ast.Constant(projected_x)
        if len(parsed.args) > 1:
            parsed.args[1] = ast.Constant(projected_y)
        elif y_keyword is not None:
            y_keyword.value = ast.Constant(projected_y)

        rendered = [ast.unparse(arg) for arg in parsed.args]
        rendered.extend(f"{kw.arg}={ast.unparse(kw.value)}" for kw in parsed.keywords if kw.arg)
        return f"pyautogui.{method}({', '.join(rendered)})"

    return _PYAUTOGUI_CALL_RE.sub(replace, code)


def parse_nemotron_response(
    response: Any,
    *,
    screen_size: tuple[int, int],
    coordinate_type: str,
    thinking: bool,
) -> tuple[str, List[str], Dict[str, Any]]:
    """Parse the internal Nemotron GUI protocol into OSWorld actions."""

    content, reasoning = _response_parts(response)
    content = normalize_response_content(content).lstrip()
    sections: Dict[str, Any] = {"thought": reasoning if thinking else ""}
    if not thinking:
        thought_match = re.search(
            r"^##\s*Thought\s*:?\s*[\n\r]+(.*?)(?=^##\s*Action\s*:|^##|\Z)",
            content,
            re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        if thought_match:
            sections["thought"] = thought_match.group(1).strip()

    action_match = re.search(
        r"^\s*##\s*Action\s*:?\s*[\n\r]+(.*?)(?=^\s*##|\Z)",
        content,
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    action = action_match.group(1).strip() if action_match else ""
    sections["action"] = action

    code_blocks = _CODE_BLOCK_RE.findall(content)
    if code_blocks:
        raw_code = code_blocks[-1].strip()
    else:
        # Some OpenAI-compatible deployments omit the fence while preserving
        # the requested ``## Code`` section. Accept that narrow variant, but
        # never fall back to executing arbitrary prose.
        code_match = re.search(
            r"^\s*##\s*Code\s*:?\s*[\n\r]+(.*?)(?=^\s*##|\Z)",
            content,
            re.DOTALL | re.MULTILINE | re.IGNORECASE,
        )
        if not code_match:
            error = "<Error>: no Code section found"
            return error, ["FAIL"], sections
        raw_code = code_match.group(1).strip()
    original_code = normalize_python_code_newlines(raw_code).strip()
    if original_code != raw_code:
        sections["raw_code"] = raw_code
    sections["original_code"] = original_code
    lowered = original_code.lower()
    if "computer.wait" in lowered:
        sections["code"] = "WAIT"
        return action or "Wait for the computer.", ["WAIT"], sections
    if "computer.terminate" in lowered:
        status_match = re.search(
            r"[\"']?status[\"']?\s*[:=]\s*[\"'](success|failure)[\"']",
            original_code,
            re.IGNORECASE,
        )
        if not status_match:
            error = "<Error>: computer.terminate is missing a success/failure status"
            return error, ["FAIL"], sections
        terminal = "DONE" if status_match.group(1).lower() == "success" else "FAIL"
        sections["code"] = terminal
        return action or original_code, [terminal], sections

    projected = project_pyautogui_coordinates(
        original_code,
        screen_width=screen_size[0],
        screen_height=screen_size[1],
        coordinate_type=coordinate_type,
    )
    sections["code"] = projected
    if not action or not projected:
        error = "<Error>: response is missing an Action or Code section"
        return error, ["FAIL"], sections
    return action, [projected], sections


def _validate_python_actions(actions: List[str]) -> None:
    """Reject malformed Python before OSWorld silently discards its return code."""

    for action in actions:
        if action in {"DONE", "FAIL", "WAIT"}:
            continue
        try:
            compile(action, "<osworld-action>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"Invalid Python action: {exc.msg} (line {exc.lineno}, offset {exc.offset})") from exc


class NemotronV3Agent:
    """Nemotron GUI scaffold with a Gym-injected model transport."""

    def __init__(
        self,
        model: str,
        max_steps: int,
        max_image_history_length: int = 3,
        platform: str = "ubuntu",
        max_tokens: int = 16384,
        top_p: float | None = 0.95,
        temperature: float = 1.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        screen_size: Tuple[int, int] = (1920, 1080),
        coordinate_type: str = "relative",
        client_password: str = "password",  # pragma: allowlist secret
        thinking: bool = True,
        parse_retries: int = 5,
        **_kwargs: Any,
    ) -> None:
        if coordinate_type not in {"relative", "absolute", "qwen25"}:
            raise ValueError(f"Unsupported coordinate_type: {coordinate_type}")
        if action_space != "pyautogui":
            raise ValueError("NemotronV3Agent only supports pyautogui")
        if observation_type != "screenshot":
            raise ValueError("NemotronV3Agent only supports screenshot observations")

        self.model = model
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.coordinate_type = coordinate_type
        self.screen_size = screen_size
        self.max_image_history_length = max(1, max_image_history_length)
        self.max_steps = max_steps
        self.client_password = client_password
        self.thinking = thinking
        self.parse_retries = max(1, parse_retries)
        prompt = SYSTEM_PROMPT_THINKING if thinking else SYSTEM_PROMPT_NON_THINKING
        self.system_prompt = prompt.replace("{password}", client_password)
        self.assistant_history_template = (
            ASSISTANT_HISTORY_TEMPLATE_THINKING if thinking else ASSISTANT_HISTORY_TEMPLATE_NON_THINKING
        )
        self.reset()

    def reset(self, _logger: logging.Logger | None = None, **_kwargs: Any) -> None:
        self.logger = _logger or LOG
        self.observations: List[Dict[str, Any]] = []
        self.actions: List[str] = []
        self.cots: List[Dict[str, Any]] = []

    def call_llm(self, payload: Dict[str, Any], _model: str | None = None) -> Any:
        """Injected by ``client.run_osworld_task`` before the first prediction."""

        raise RuntimeError("NemotronV3Agent requires an injected Gym model transport")

    def _assistant_history(self, cot: Dict[str, Any]) -> str:
        return self.assistant_history_template.format(
            thought=cot.get("thought", ""),
            action=cot.get("action", ""),
            code=cot.get("original_code", cot.get("code", "")),
        )

    def _messages(self, instruction: str, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        instruction_prompt = INSTRUCTION_TEMPLATE.format(instruction=instruction)
        image_history = min(len(self.actions), self.max_image_history_length - 1)
        image_window_start = len(self.actions) - image_history

        text_history = ""
        if image_window_start > 0:
            history_parts = []
            for index in range(image_window_start):
                history_parts.append(
                    STEP_TEMPLATE.format(step_num=index + 1)
                    + TEXT_HISTORY_TEMPLATE.format(
                        thought=self.cots[index].get("thought", ""),
                        action=self.cots[index].get("action", ""),
                        code=self.cots[index].get("original_code", self.cots[index].get("code", "")),
                    )
                )
            text_history = "# Previous History Actions:\n" + "\n".join(history_parts)

        for index in range(image_window_start, len(self.actions)):
            user_text = instruction_prompt
            if index == image_window_start and text_history:
                user_text += text_history + "\n"
            user_text += f"You are currently on Step {index + 1}.\n"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_encode_image(self.observations[index]['screenshot'])}"
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            )
            messages.append({"role": "assistant", "content": self._assistant_history(self.cots[index])})

        current_text = instruction_prompt
        if image_history == 0 and text_history:
            current_text += text_history + "\n"
        current_text += f"You are currently on Step {len(self.actions) + 1}.\n"
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_encode_image(obs['screenshot'])}"},
                    },
                    {"type": "text", "text": current_text},
                ],
            }
        )
        return messages

    def _scale_windows_scroll(self, code: str, factor: int = 50) -> str:
        if self.platform.lower() != "windows":
            return code
        pattern = re.compile(r"(pyautogui\.scroll\()\s*([-+]?\d+)\s*\)")
        return pattern.sub(lambda match: f"{match.group(1)}{int(match.group(2)) * factor})", code)

    def predict(self, instruction: str, obs: Dict[str, Any], **_kwargs: Any) -> tuple[str, List[str], Dict[str, Any]]:
        messages = self._messages(instruction, obs)
        last_error = "No response"
        parsed_info: Dict[str, Any] = {}

        for attempt in range(self.parse_retries):
            response: Any = None
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature if attempt == 0 else max(0.2, self.temperature),
                "_nemo_gym_return_message": True,
                "_nemo_gym_require_stop": True,
            }
            if self.top_p is not None:
                payload["top_p"] = self.top_p
            try:
                response = self.call_llm(payload, self.model)
                content, _reasoning = _response_parts(response)
                if not content:
                    raise ValueError("model response has no content")
                low_level, actions, parsed_info = parse_nemotron_response(
                    response,
                    screen_size=self.screen_size,
                    coordinate_type=self.coordinate_type,
                    thinking=self.thinking,
                )
                if low_level.startswith("<Error>"):
                    raise ValueError(low_level)
                _validate_python_actions(actions)
                if os.environ.get("OSWORLD_MODEL_IO_LOG", "").strip():
                    _append_agent_io(
                        {
                            "schema_version": 1,
                            "event": "agent_parse",
                            "timestamp_unix_ns": time.time_ns(),
                            "pid": os.getpid(),
                            "step": len(self.actions) + 1,
                            "attempt": attempt + 1,
                            "normalized_model_response": response,
                            "parsed_low_level": low_level,
                            "parsed_actions": actions,
                            "parsed_info": parsed_info,
                        }
                    )
                break
            except Exception as exc:  # noqa: BLE001 - malformed model output is retryable.
                last_error = str(exc)
                if os.environ.get("OSWORLD_MODEL_IO_LOG", "").strip():
                    _append_agent_io(
                        {
                            "schema_version": 1,
                            "event": "agent_parse_error",
                            "timestamp_unix_ns": time.time_ns(),
                            "pid": os.getpid(),
                            "step": len(self.actions) + 1,
                            "attempt": attempt + 1,
                            "normalized_model_response": response,
                            "error_type": type(exc).__name__,
                            "error": repr(exc),
                        }
                    )
                self.logger.warning(
                    "Nemotron response attempt %d/%d failed: %s",
                    attempt + 1,
                    self.parse_retries,
                    exc,
                )
        else:
            return last_error, ["FAIL"], parsed_info

        actions = [self._scale_windows_scroll(action) for action in actions]
        self.observations.append(obs)
        self.actions.append(low_level)
        self.cots.append(parsed_info)

        if len(self.actions) >= self.max_steps and not any(action in {"DONE", "FAIL"} for action in actions):
            parsed_info["code"] = "FAIL"
            return content, ["FAIL"], parsed_info
        return content, actions, parsed_info


class NemotronOmniAgent(NemotronV3Agent):
    """Nemotron 3 Nano Omni scaffold with single-image prompt semantics.

    The hosted reasoning endpoint rejects a request containing more than one
    image. Previous interactions are therefore rendered as bounded text in
    the system message; the current user message is the only image-bearing
    turn. This is deliberately separate from Qwen3-Omni's ``Qwen3VLAgent``:
    Qwen uses a different ``<tool_call>`` protocol and image-history policy.
    """

    def __init__(
        self,
        *args: Any,
        max_text_history_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = OMNI_MINI_SYSTEM_PROMPT.replace("__PASSWORD__", self.client_password)
        configured_length = (
            self.max_image_history_length if max_text_history_length is None else max_text_history_length
        )
        self.max_text_history_length = max(0, int(configured_length))

    @staticmethod
    def _render_history_entry(cot: Dict[str, Any]) -> str:
        parts = []
        for label, key in (("Thought", "thought"), ("Action", "action"), ("Code", "code")):
            value = str(cot.get(key, "") or "").strip()
            if value:
                parts.append(f"{label}: {value}")
        return "\n".join(parts) or "No valid action was recorded."

    def _messages(self, instruction: str, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        history = self.cots[-self.max_text_history_length :] if self.max_text_history_length else []
        system_prompt = self.system_prompt
        if history:
            rendered = []
            first_step = len(self.cots) - len(history) + 1
            for step_num, cot in enumerate(history, first_step):
                # Bound each record so a long reasoning trace cannot crowd the
                # current screenshot and instruction out of the context.
                entry = self._render_history_entry(cot)[-4000:]
                rendered.append(f"Previous interaction {step_num}:\n{entry}")
            system_prompt += (
                "\n\nPrevious interactions (text only; the user message below contains "
                "the sole current screenshot):\n" + "\n\n".join(rendered)
            )

        current_text = INSTRUCTION_TEMPLATE.format(instruction=instruction)
        current_text += f"You are currently on Step {len(self.actions) + 1}.\n"
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_encode_image(obs['screenshot'])}"},
                    },
                    {"type": "text", "text": current_text},
                ],
            },
        ]
