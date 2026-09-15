# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym adapter-owned model scaffolds for OSWorld.

The adapter owns model-specific prompting, history, parsing, and coordinate
projection while OSWorld continues to own ``DesktopEnv`` and task evaluation.
``NemotronV3NanoOmniAgent`` supports both image-history and single-image
endpoints through configuration.

This module is intentionally named ``adapter_agents`` because the model-specific
scaffold is implemented by the Gym adapter rather than upstream OSWorld's
``mm_agents`` package. OSWorld's environment and evaluator remain unmodified.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Tuple

from responses_api_agents.osworld_agent.history_policy import (
    HistoryPolicySpec,
    HistoryPolicyState,
    plan_history,
)
from responses_api_agents.osworld_agent.trajectory import stable_id


LOG = logging.getLogger("nemo_gym.osworld_agent.adapter_agents")


def _jsonable(value: Any) -> Any:
    """Convert parser details to JSON without changing rollout behavior."""

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
    """Append parser details beside the agent model-I/O events."""

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
        LOG.exception("Failed to append OSWorld parser log to %s", path)


INSTRUCTION_TEMPLATE = (
    "# Task Instruction:\n{instruction}\n\n"
    "Please generate the next move according to the screenshot, task instruction "
    "and previous steps (if provided).\n"
)
PARSE_RETRY_TEMPLATE = """Your previous response could not be executed because its Code section was invalid:
{error}

Return a corrected response for the same screenshot and task. Keep the required
## Action / ## Code format, emit syntactically valid Python, and do not repeat
the invalid code. Do not claim success merely because parsing failed."""
PRE_DONE_CHECKLIST = """Before returning computer.terminate with status=success, re-read the exact task instruction and verify the current state:
- the exact target app, page, slide, sheet, column, file, and setting scope are correct;
- every negative constraint (for example, "do not" or "without") is satisfied;
- any requested output file visibly exists and is non-empty.
If any item is not verified, take one corrective or verification action instead of terminating."""
STEP_TEMPLATE = "# Step {step_num}:\n"
TEXT_HISTORY_TEMPLATE = "## Thought:\n{thought}\n\n## Action:\n{action}\n"
ASSISTANT_HISTORY_TEMPLATE_THINKING = "<think>\n{thought}\n</think>\n## Action:\n{action}\n"
ASSISTANT_HISTORY_TEMPLATE_NON_THINKING = "## Thought:\n{thought}\n\n## Action:\n{action}\n"

# Keep the validated checkpoint prompt spelling stable. Although "passoword"
# is misspelled, changing prompt bytes here would make maintained runs differ
# from the best-known Nano Omni OSWorld recipe.
SYSTEM_PROMPT_THINKING = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

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
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

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

_CODE_BLOCK_RE = re.compile(r"```(?:code|python|py|json)?[ \t]*\r?\n?(.*?)\s*```", re.DOTALL | re.IGNORECASE)
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


# A rejected request and a malformed sample are different defects with
# different remedies, and the old code funnelled both into one bucket named
# after only one of them.  Classify explicitly so telemetry can tell a server
# 4xx apart from the model emitting a stop token mid-string.
_CONTEXT_OVERFLOW_MARKERS = (
    "maximum context length",
    "context_length_exceeded",
    "exceeds model",
    "reduce the length",
    "input length",
    "prompt is too long",
    "please reduce",
)

# One bucket named "model_response_invalid" made a server 4xx and a
# malformed sample indistinguishable in telemetry. Keep the family name for
# consumers that aggregate, but report the specific defect.
MODEL_FAILURE_OUTCOMES = {
    "context_overflow": "model_context_overflow",
    "transport_error": "model_call_failed",
    "output_truncated": "model_output_truncated",
    "empty_response": "model_response_empty",
    "unparseable": "model_response_unparseable",
}

MODEL_FAILURE_KINDS = (
    "context_overflow",
    "transport_error",
    "output_truncated",
    "empty_response",
    "unparseable",
)


def _failure_status_code(exc: BaseException) -> int | None:
    for holder in (exc, getattr(exc, "response", None)):
        code = getattr(holder, "status_code", None)
        if isinstance(code, int):
            return code
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def classify_model_failure(exc: BaseException, *, model_call_completed: bool) -> str:
    """Name the defect behind one failed sampling attempt.

    ``context_overflow`` is the only kind a retry cannot fix on its own: the
    request is deterministically rejected, so resending it unchanged burns the
    whole retry budget. Callers are expected to shrink the prompt instead.
    """

    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _CONTEXT_OVERFLOW_MARKERS):
        return "context_overflow"
    if not model_call_completed:
        return "transport_error"
    if "finish_reason=" in text and "length" in text:
        # The finish-reason guard runs before the parser, so this is the only
        # kind that really means "the sampler hit its token budget".
        return "output_truncated"
    if "has no content" in text:
        return "empty_response"
    return "unparseable"


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
    structure expected by the Nemotron response parser.
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


# Model output dialects can differ even when prompts request the same schema.
# When onboarding or upgrading a model, capture its lossless raw response and
# add a regression case for each newly observed, unambiguous section layout
# before widening this parser. Keep ``## Code`` as the explicit execution
# boundary: compatibility must never fall back to extracting code from
# arbitrary prose or unrelated Markdown fences.


def _extract_markdown_section(content: str, name: str) -> str | None:
    """Return an explicit ``## <name>`` section in common model formats.

    Both of these unambiguous forms are accepted::

        ## Action:\nClick the target.
        ## Action: Click the target.

    The section remains bounded by the next level-two Markdown heading. Plain
    prose labels such as ``Action:`` are deliberately not accepted, so parser
    flexibility does not turn arbitrary response text into executable input.
    """

    match = re.search(
        rf"^[ \t]*##[ \t]+{re.escape(name)}[ \t]*:?[ \t]*(?:\r?\n)?(.*?)(?=^[ \t]*##(?:[ \t]+|$)|\Z)",
        content,
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


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
    """Parse the Nemotron GUI protocol into OSWorld actions."""

    content, reasoning = _response_parts(response)
    content = normalize_response_content(content).lstrip()
    sections: Dict[str, Any] = {"thought": reasoning if thinking else ""}
    if not thinking:
        thought = _extract_markdown_section(content, "Thought")
        if thought is not None:
            sections["thought"] = thought

    action = _extract_markdown_section(content, "Action") or ""
    sections["action"] = action

    # Code is the execution authority. Require an explicit Code section, but
    # accept its payload on the heading line, the following line, or in a
    # supported Markdown fence. Never execute an unrelated global code block.
    code_section = _extract_markdown_section(content, "Code")
    if code_section is None:
        error = "<Error>: no explicit ## Code section found"
        return error, [], sections
    code_blocks = _CODE_BLOCK_RE.findall(code_section)
    if code_blocks:
        raw_code = code_blocks[-1].strip()
    else:
        if code_section.startswith("```"):
            error = "<Error>: unsupported or unterminated Code fence"
            return error, [], sections
        raw_code = code_section
    original_code = normalize_python_code_newlines(raw_code).strip()
    if not original_code:
        error = "<Error>: the ## Code section is empty"
        return error, [], sections
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
            return error, [], sections
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
    if not projected:
        error = "<Error>: the ## Code section is empty"
        return error, [], sections
    # Action and Thought are descriptive metadata; a missing description must
    # not discard an explicit, validated Code section. Keep the inference
    # visible in parser logs and textual history.
    if not action:
        action = "Execute the provided code."
        sections["action"] = action
        sections["action_inferred"] = True
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


DEFAULT_NEMOTRON_PROTOCOL_ID = "nano-omni-v3-osworld-v1"
NEMOTRON_AGENT_OPTION_DEFAULTS: Dict[str, Any] = {
    "coordinate_type": "relative",
    "thinking": True,
    "parse_retries": 5,
    "parse_error_feedback": False,
    "parse_retry_temperature": None,
    "pre_done_checklist": False,
    "repeated_action_warning_threshold": 0,
    "repeated_action_window": 12,
}


def normalize_nemotron_agent_options(options: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the constructor's behavioral defaults and normalizations once."""

    normalized = dict(NEMOTRON_AGENT_OPTION_DEFAULTS)
    normalized.update({field: options[field] for field in normalized if field in options})
    coordinate_type = normalized["coordinate_type"]
    if coordinate_type not in {"relative", "absolute", "qwen25"}:
        raise ValueError(f"Unsupported coordinate_type: {coordinate_type}")
    retry_temperature = normalized["parse_retry_temperature"]
    return {
        "coordinate_type": coordinate_type,
        "thinking": bool(normalized["thinking"]),
        "parse_retries": max(1, int(normalized["parse_retries"])),
        "parse_error_feedback": bool(normalized["parse_error_feedback"]),
        "parse_retry_temperature": (None if retry_temperature is None else max(0.0, float(retry_temperature))),
        "pre_done_checklist": bool(normalized["pre_done_checklist"]),
        "repeated_action_warning_threshold": max(0, int(normalized["repeated_action_warning_threshold"])),
        "repeated_action_window": max(1, int(normalized["repeated_action_window"])),
    }


@dataclass(frozen=True)
class NemotronModelProtocol:
    """Frozen model-facing wire format, independent of history selection."""

    protocol_id: str
    parser_id: str
    instruction_template: str
    step_template: str
    text_history_template: str
    system_prompt_thinking: str
    system_prompt_non_thinking: str
    assistant_history_template_thinking: str
    assistant_history_template_non_thinking: str
    parser: Callable[..., tuple[str, List[str], Dict[str, Any]]]
    schema_version: int = 1

    def system_prompt(self, *, thinking: bool, password: str) -> str:
        template = self.system_prompt_thinking if thinking else self.system_prompt_non_thinking
        return template.replace("{password}", password)

    def assistant_history_template(self, *, thinking: bool) -> str:
        return self.assistant_history_template_thinking if thinking else self.assistant_history_template_non_thinking

    def to_contract(self, *, thinking: bool) -> Dict[str, Any]:
        wire_material = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "parser_id": self.parser_id,
            "thinking": thinking,
            "instruction_template": self.instruction_template,
            "step_template": self.step_template,
            "text_history_template": self.text_history_template,
            "system_prompt_template": (self.system_prompt_thinking if thinking else self.system_prompt_non_thinking),
            "assistant_history_template": self.assistant_history_template(thinking=thinking),
        }
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "parser_id": self.parser_id,
            "thinking": thinking,
            "wire_contract_id": stable_id("osworld-model-wire", wire_material),
        }


NEMOTRON_PROTOCOL_REGISTRY: Dict[str, NemotronModelProtocol] = {
    DEFAULT_NEMOTRON_PROTOCOL_ID: NemotronModelProtocol(
        protocol_id=DEFAULT_NEMOTRON_PROTOCOL_ID,
        parser_id="nemotron-v3-nano-omni-parser-v1",
        instruction_template=INSTRUCTION_TEMPLATE,
        step_template=STEP_TEMPLATE,
        text_history_template=TEXT_HISTORY_TEMPLATE,
        system_prompt_thinking=SYSTEM_PROMPT_THINKING,
        system_prompt_non_thinking=SYSTEM_PROMPT_NON_THINKING,
        assistant_history_template_thinking=ASSISTANT_HISTORY_TEMPLATE_THINKING,
        assistant_history_template_non_thinking=ASSISTANT_HISTORY_TEMPLATE_NON_THINKING,
        parser=parse_nemotron_response,
    )
}


def resolve_nemotron_protocol(protocol_id: str | None = None) -> NemotronModelProtocol:
    """Resolve a named model protocol and fail closed on unknown identities."""

    resolved_id = protocol_id or DEFAULT_NEMOTRON_PROTOCOL_ID
    try:
        return NEMOTRON_PROTOCOL_REGISTRY[resolved_id]
    except KeyError as exc:
        allowed = ", ".join(sorted(NEMOTRON_PROTOCOL_REGISTRY))
        raise ValueError(f"Unknown model_protocol_id={resolved_id!r}. Allowed: {allowed}") from exc


class NemotronV3NanoOmniAgent:
    """Nemotron Nano Omni scaffold with a Gym-injected model transport."""

    def __init__(
        self,
        model: str,
        max_steps: int,
        max_image_history_length: int | None = None,
        max_live_images: int | None = None,
        history_policy: Mapping[str, Any] | HistoryPolicySpec | None = None,
        model_protocol_id: str | None = None,
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
        parse_error_feedback: bool = False,
        parse_retry_temperature: float | None = None,
        pre_done_checklist: bool = False,
        repeated_action_warning_threshold: int = 0,
        repeated_action_window: int = 12,
        log_context: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        behavior_options = normalize_nemotron_agent_options(
            {
                "coordinate_type": coordinate_type,
                "thinking": thinking,
                "parse_retries": parse_retries,
                "parse_error_feedback": parse_error_feedback,
                "parse_retry_temperature": parse_retry_temperature,
                "pre_done_checklist": pre_done_checklist,
                "repeated_action_warning_threshold": repeated_action_warning_threshold,
                "repeated_action_window": repeated_action_window,
            }
        )
        if action_space != "pyautogui":
            raise ValueError("NemotronV3NanoOmniAgent only supports pyautogui")
        if observation_type != "screenshot":
            raise ValueError("NemotronV3NanoOmniAgent only supports screenshot observations")

        self.model = model
        self.platform = platform
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.coordinate_type = behavior_options["coordinate_type"]
        self.screen_size = screen_size
        if history_policy is not None:
            if max_image_history_length is not None or max_live_images is not None:
                raise ValueError(
                    "history_policy cannot be combined with legacy max_image_history_length/max_live_images"
                )
            self.history_policy = (
                history_policy
                if isinstance(history_policy, HistoryPolicySpec)
                else HistoryPolicySpec.from_mapping(history_policy)
            )
        else:
            self.history_policy = HistoryPolicySpec.from_legacy(
                keep_images=3 if max_image_history_length is None else max_image_history_length,
                max_live_images=max_live_images,
            )
        # Compatibility aliases remain observable for callers which inspected
        # these attributes, but the policy object is now the authority.
        self.max_image_history_length = self.history_policy.low_water
        self.max_live_images = self.history_policy.high_water
        self.model_protocol = resolve_nemotron_protocol(model_protocol_id)
        self.model_protocol_id = self.model_protocol.protocol_id
        self.max_steps = max_steps
        self.client_password = client_password
        self.thinking = behavior_options["thinking"]
        self.parse_retries = behavior_options["parse_retries"]
        removed_options = sorted(option for option in ("training_mode", "training_turn_strategy") if option in _kwargs)
        if removed_options:
            raise ValueError(
                "OSWorld no longer accepts training-specific agent switches: " + ", ".join(removed_options)
            )
        self.parse_error_feedback = behavior_options["parse_error_feedback"]
        self.parse_retry_temperature = behavior_options["parse_retry_temperature"]
        self.pre_done_checklist = behavior_options["pre_done_checklist"]
        self.repeated_action_warning_threshold = behavior_options["repeated_action_warning_threshold"]
        self.repeated_action_window = behavior_options["repeated_action_window"]
        self.behavior_options = behavior_options
        self.log_context = {
            str(key): value for key, value in dict(log_context or {}).items() if value is not None and value != ""
        }
        self.system_prompt = self.model_protocol.system_prompt(
            thinking=self.thinking,
            password=client_password,
        )
        self.assistant_history_template = self.model_protocol.assistant_history_template(thinking=self.thinking)
        self.reset()

    def reset(self, _logger: logging.Logger | None = None, **_kwargs: Any) -> None:
        self.logger = _logger or LOG
        self.observations: List[Dict[str, Any]] = []
        self.actions: List[str] = []
        self.cots: List[Dict[str, Any]] = []
        self.history_policy_state = HistoryPolicyState()
        # Turns folded into the text summary. Replaces the former
        # ``compacted_before`` scalar, which assumed the live images were one
        # contiguous suffix and is meaningless once a policy keeps a sink.
        self.compacted_turns: tuple[int, ...] = ()
        self.snapshot_window: Dict[str, Any] = {
            "prompt_snapshot_count": 0,
            "snapshot_window_start": 0,
            "snapshot_image_intervals": [],
            "snapshot_sink_size": self.history_policy.sink,
            "snapshot_image_budget_clamped": False,
            "snapshot_compaction_triggered": False,
            "snapshot_window_min": self.max_image_history_length,
            "snapshot_window_max": self.max_live_images,
            "history_policy_id": self.history_policy.policy_id,
            "history_policy_name": self.history_policy.name,
            "history_policy_schema_version": self.history_policy.schema_version,
            "history_policy_compaction_epoch": 0,
        }

    @property
    def agent_contract(self) -> Dict[str, Any]:
        """Observed model/history contract used by this agent instance."""

        return {
            "schema_version": 1,
            "model_protocol": self.model_protocol.to_contract(thinking=self.thinking),
            "history_policy": self.history_policy.to_contract(),
            "agent_options": dict(self.behavior_options),
        }

    def call_llm(self, payload: Dict[str, Any], _model: str | None = None) -> Any:
        """Injected by ``client.run_osworld_task`` before the first prediction."""

        raise RuntimeError("NemotronV3NanoOmniAgent requires an injected Gym model transport")

    def _log_event_context(self, *, step: int, parse_attempt: int) -> Dict[str, Any]:
        context = dict(self.log_context)
        context.update({"step": step, "parse_attempt": parse_attempt})
        return context

    def _assistant_history(self, cot: Dict[str, Any]) -> str:
        return self.assistant_history_template.format(
            thought=cot.get("thought", ""),
            action=cot.get("action", ""),
        )

    def _parse_retry_messages(
        self,
        messages: List[Dict[str, Any]],
        response: Any,
        error: str,
    ) -> List[Dict[str, Any]]:
        """Add bounded parser feedback without duplicating the screenshot."""

        content, _reasoning = _response_parts(response)
        invalid_response = content[-8000:] if content else "<empty model response>"
        return [
            *messages,
            {"role": "assistant", "content": invalid_response},
            {"role": "user", "content": PARSE_RETRY_TEMPLATE.format(error=error[-2000:])},
        ]

    @staticmethod
    def _is_trivial_repeat(code: str) -> bool:
        compact = re.sub(r"\s+", "", code).lower()
        return compact in {
            "wait",
            "pyautogui.sleep(0.5)",
            "pyautogui.keydown('return')pyautogui.keyup('return')",
            'pyautogui.keydown("return")pyautogui.keyup("return")',
        }

    def _repeated_action_guidance(self) -> str:
        """Warn after a non-trivial executable action repeats in a short window."""

        threshold = self.repeated_action_warning_threshold
        if threshold <= 1 or not self.cots:
            return ""
        recent = self.cots[-self.repeated_action_window :]
        signatures = [str(cot.get("code", "") or "").strip() for cot in recent]
        current = signatures[-1]
        if not current or current in {"DONE", "FAIL", "WAIT"} or self._is_trivial_repeat(current):
            return ""
        occurrences = sum(signature == current for signature in signatures)
        if occurrences < threshold:
            return ""
        return (
            "Recovery check: the same executable action appeared "
            f"{occurrences} times in the last {len(signatures)} steps. Re-read the screenshot and task. "
            "Do not repeat the same strategy unless the visible state is progressing; choose a different "
            "verifiable action or terminate with failure if the task is genuinely impossible."
        )

    def _step_guidance(self) -> str:
        guidance = []
        repeated_action = self._repeated_action_guidance()
        if repeated_action:
            guidance.append(repeated_action)
        if self.pre_done_checklist:
            guidance.append(PRE_DONE_CHECKLIST)
        return "\n\n".join(guidance)

    def _messages(
        self,
        instruction: str,
        obs: Dict[str, Any],
        *,
        max_images: int | None = None,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        instruction_prompt = self.model_protocol.instruction_template.format(instruction=instruction)
        action_count = len(self.actions)
        plan = plan_history(
            self.history_policy,
            self.history_policy_state,
            completed_turns=action_count,
            max_images=max_images,
        )
        self.history_policy_state = plan.next_state
        self.compacted_turns = plan.text_turns
        self.snapshot_window = plan.telemetry()

        # Walk the plan in turn order and flush a text block whenever a run of
        # folded turns ends.  For a single leading text run -- every policy
        # shipped before sink_window -- this reproduces the previous rendering
        # byte for byte.  It is also the only ordering that stays correct once
        # a policy keeps a sink, because there the summary of the folded middle
        # must land *after* the sink images and *before* the recent window,
        # not attached to whichever image happens to come first.
        pending_text: List[int] = []

        def _flush_text() -> str:
            if not pending_text:
                return ""
            history_parts = [
                self.model_protocol.step_template.format(step_num=index + 1)
                + self.model_protocol.text_history_template.format(
                    thought=self.cots[index].get("thought", ""),
                    action=self.cots[index].get("action", ""),
                )
                for index in pending_text
            ]
            pending_text.clear()
            return "# Previous History Actions:\n" + "\n".join(history_parts) + "\n"

        image_history = 0
        for decision in plan.decisions:
            index = decision.turn_index
            if index >= action_count:
                break
            if decision.disposition != "live_image":
                pending_text.append(index)
                continue
            image_history += 1
            user_text = instruction_prompt + _flush_text()
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

        current_text = instruction_prompt + _flush_text()
        current_text += f"You are currently on Step {len(self.actions) + 1}.\n"
        guidance = self._step_guidance()
        if guidance:
            current_text += f"\n{guidance}\n"
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
        # Re-planning after a context overflow must start from the same policy
        # state as the first attempt, otherwise the window would advance once
        # per retry and the trajectory would stop being replayable.
        entry_policy_state = self.history_policy_state
        image_budget: int | None = None
        messages = self._messages(instruction, obs)
        request_messages = messages
        repeated_action_warning = bool(self._repeated_action_guidance())
        last_error = "No response"
        last_error_type = "RuntimeError"
        last_failure_stage = "model_call"
        last_failure_kind = "transport_error"
        failure_kind_counts: Dict[str, int] = {}
        prompt_shrink_events: List[Dict[str, Any]] = []
        completed_model_calls = 0
        model_calls: List[Dict[str, Any]] = []
        parsed_info: Dict[str, Any] = {"model_calls": model_calls}

        for attempt in range(self.parse_retries):
            response: Any = None
            attempt_actions: List[str] = []
            step_number = len(self.actions) + 1
            parse_attempt = attempt + 1
            call_log_context = self._log_event_context(step=step_number, parse_attempt=parse_attempt)
            call_log_context.update(
                {
                    "parse_feedback_injected": request_messages is not messages,
                    "pre_done_checklist_injected": self.pre_done_checklist,
                    "repeated_action_warning_injected": repeated_action_warning,
                    **self.snapshot_window,
                }
            )
            retry_temperature = (
                max(0.2, self.temperature) if self.parse_retry_temperature is None else self.parse_retry_temperature
            )
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": request_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature if attempt == 0 else retry_temperature,
                "_nemo_gym_return_message": True,
                "_nemo_gym_require_stop": True,
                "_osworld_log_context": call_log_context,
            }
            if self.top_p is not None:
                payload["top_p"] = self.top_p
            model_call_record: Dict[str, Any] = {
                "parse_attempt": parse_attempt,
                "prompt_messages": _jsonable(request_messages),
                "response": None,
                "accepted": False,
                "parse_error": None,
                "parsed_actions": [],
                **self.snapshot_window,
            }
            model_call_completed = False
            try:
                response = self.call_llm(payload, self.model)
                model_call_completed = True
                completed_model_calls += 1
                model_call_record["response"] = _jsonable(response)
                finish_reason = response.get("finish_reason") if isinstance(response, Mapping) else None
                if finish_reason is not None and finish_reason not in {"stop", "tool_calls"}:
                    raise ValueError(f"Model response did not finish cleanly: finish_reason={finish_reason!r}")
                content, _reasoning = _response_parts(response)
                if not content:
                    raise ValueError("model response has no content")
                low_level, actions, response_info = self.model_protocol.parser(
                    response,
                    screen_size=self.screen_size,
                    coordinate_type=self.coordinate_type,
                    thinking=self.thinking,
                )
                attempt_actions = list(actions)
                parsed_info.update(response_info)
                if low_level.startswith("<Error>"):
                    raise ValueError(low_level)
                _validate_python_actions(actions)
                model_call_record["accepted"] = True
                model_call_record["parsed_actions"] = attempt_actions
                model_calls.append(model_call_record)
                if os.environ.get("OSWORLD_MODEL_IO_LOG", "").strip():
                    _append_agent_io(
                        {
                            **call_log_context,
                            "schema_version": 2,
                            "event": "agent_parse",
                            "timestamp_unix_ns": time.time_ns(),
                            "pid": os.getpid(),
                            "step": step_number,
                            "attempt": parse_attempt,
                            "parse_attempt": parse_attempt,
                            "normalized_model_response": response,
                            "parsed_low_level": low_level,
                            "parsed_actions": actions,
                            "parsed_info": parsed_info,
                        }
                    )
                break
            except Exception as exc:  # noqa: BLE001 - malformed model output is retryable.
                last_error = str(exc)
                last_error_type = type(exc).__name__
                last_failure_stage = "response_parse" if model_call_completed else "model_call"
                last_failure_kind = classify_model_failure(exc, model_call_completed=model_call_completed)
                failure_kind_counts[last_failure_kind] = failure_kind_counts.get(last_failure_kind, 0) + 1
                model_call_record["failure_stage"] = last_failure_stage
                model_call_record["failure_kind"] = last_failure_kind
                model_call_record["parse_error"] = last_error
                model_call_record["parsed_actions"] = attempt_actions
                model_calls.append(model_call_record)
                will_retry = attempt + 1 < self.parse_retries
                if last_failure_kind == "context_overflow" and will_retry:
                    # Resending an over-long prompt cannot succeed; parse
                    # feedback would only make it longer. Shrink the live-image
                    # set and rebuild instead, replanning from the entry state.
                    previous_images = int(self.snapshot_window.get("prompt_snapshot_count") or 1)
                    next_budget = max(1, min(previous_images, image_budget or previous_images) - 1)
                    if next_budget < previous_images:
                        image_budget = next_budget
                        self.history_policy_state = entry_policy_state
                        messages = self._messages(instruction, obs, max_images=image_budget)
                        request_messages = messages
                        prompt_shrink_events.append(
                            {
                                "parse_attempt": parse_attempt,
                                "from_images": previous_images,
                                "to_images": int(self.snapshot_window.get("prompt_snapshot_count") or 0),
                                "reason": "context_overflow",
                            }
                        )
                        self.logger.warning(
                            "Context overflow at step %d attempt %d: shrinking live images %d -> %d",
                            step_number,
                            parse_attempt,
                            previous_images,
                            self.snapshot_window.get("prompt_snapshot_count"),
                        )
                    feedback_next = False
                else:
                    feedback_next = self.parse_error_feedback and will_retry
                if os.environ.get("OSWORLD_MODEL_IO_LOG", "").strip():
                    _append_agent_io(
                        {
                            **call_log_context,
                            "schema_version": 2,
                            "event": "agent_parse_error",
                            "timestamp_unix_ns": time.time_ns(),
                            "pid": os.getpid(),
                            "step": step_number,
                            "attempt": parse_attempt,
                            "parse_attempt": parse_attempt,
                            "normalized_model_response": response,
                            "error_type": type(exc).__name__,
                            "error": repr(exc),
                            "will_retry": will_retry,
                            "retry_feedback_injected_next": feedback_next,
                            "retry_temperature_next": retry_temperature if will_retry else None,
                        }
                    )
                self.logger.warning(
                    "Nemotron response attempt %d/%d failed: %s",
                    attempt + 1,
                    self.parse_retries,
                    exc,
                )
                if feedback_next:
                    request_messages = self._parse_retry_messages(messages, response, last_error)
        else:
            # Report facts only.  The runner owns rollout termination and the
            # runtime-admission policy decides whether the result is masked.
            # In particular, do not turn malformed sampled output into a
            # synthetic model-authored FAIL action.
            parsed_info.update(
                {
                    "agent_outcome": MODEL_FAILURE_OUTCOMES[last_failure_kind],
                    "agent_outcome_family": "model_response_invalid",
                    "stop_rollout": True,
                    "model_call_completed": completed_model_calls > 0,
                    "parse_failure": {
                        "attempt_count": len(model_calls),
                        "completed_model_call_count": completed_model_calls,
                        "last_failure_stage": last_failure_stage,
                        "last_failure_kind": last_failure_kind,
                        "failure_kind_counts": dict(failure_kind_counts),
                        "prompt_shrink_events": list(prompt_shrink_events),
                        "last_error_type": last_error_type,
                        "last_error": last_error,
                    },
                }
            )
            return last_error, [], parsed_info

        # A step that recovered after shrinking its prompt still consumed extra
        # sampling and rendered a smaller history than the policy asked for.
        # Record that on the successful step too, otherwise the only trace of a
        # server rejection would be on rollouts that died -- which is exactly
        # how twenty HTTP 400s went missing from a whole benchmark release.
        # Emitted only when non-empty, so the clean path keeps its byte shape.
        if prompt_shrink_events:
            parsed_info["prompt_shrink_events"] = list(prompt_shrink_events)
        if failure_kind_counts:
            parsed_info["failure_kind_counts"] = dict(failure_kind_counts)

        actions = [self._scale_windows_scroll(action) for action in actions]
        self.observations.append(obs)
        self.actions.append(low_level)
        # The returned evidence may contain several full prompts and images.
        # Keep only parser semantics in the agent's rolling text history.
        self.cots.append({key: value for key, value in parsed_info.items() if key != "model_calls"})

        return content, actions, parsed_info
