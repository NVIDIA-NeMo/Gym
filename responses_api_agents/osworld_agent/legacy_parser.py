# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy OSWorld response parsing.

``parse_response_to_cot_and_action`` and ``project_coordinate_to_absolute_scale``
are copied from the pinned OSWorld ``mm_agents.nvidia.nemotron_agent`` module;
``normalize_vendor_parser_input`` and ``is_invalid_legacy_parse`` add the input
normalization and invalid-parse rule used with that parser. Keep them
byte-for-byte in behavior.
"""

from __future__ import annotations

import ast
import logging
import re
import traceback
from typing import Any, Dict, List, Mapping, Tuple


LOG = logging.getLogger(__name__)
LEGACY_PARSER_ID = "mm-agents-nemotron-parse-response-to-cot-and-action"


def normalize_vendor_parser_input(response: Mapping[str, str]) -> Dict[str, str]:
    """Restore the Action header omitted by some Responses generations."""

    response = dict(response)
    content = response["content"]
    if re.search(r"^\s*##\s*Action\b", content, re.MULTILINE):
        return response
    code_match = re.search(r"^\s*##\s*Code\s*:?", content, re.MULTILINE)
    if code_match is None:
        return response
    implicit_action = content[: code_match.start()].strip()
    if not implicit_action:
        return response
    return {
        **response,
        "content": (f"## Action:\n{implicit_action}\n\n{content[code_match.start() :]}"),
    }


def is_invalid_legacy_parse(low_level_instruction: Any, actions: List[str], cot: Any) -> bool:
    """Parses that are replaced with a FAIL action.

    An ``<Error>`` parse that still produced code is not in this set: its
    returned ``FAIL`` action is executed as an ordinary policy action.
    """

    return (
        not actions or str(low_level_instruction).startswith(":") or not isinstance(cot, dict) or not cot.get("code")
    )


def parse_response_to_cot_and_action(
    response, screen_size, coordinate_type, thinking: bool
) -> Tuple[str, List[str], dict]:
    """Parse response including Thought (via <think> tags or ## Thought), Action and code block"""
    input_string = response["content"].lstrip()

    sections = {}
    try:
        if thinking:
            if "reasoning_content" in response:
                sections["thought"] = response["reasoning_content"].strip()
            else:
                sections["thought"] = response.get("reasoning", "").strip()
            m = re.search(r"^##\s*Action\b", input_string, flags=re.MULTILINE)
            if m:
                input_string = input_string[m.start() :]
        else:
            thought = re.search(
                r"^##\s*Thought\s*:?[\n\r]+(.*?)(?=^##\s*Action:|^##|\Z)", input_string, re.DOTALL | re.MULTILINE
            )
            if thought:
                sections["thought"] = thought.group(1).strip()
            else:
                sections["thought"] = ""

        action_match = re.search(
            r"^\s*##\s*Action\s*:?\s*[\n\r]+(.*?)(?=^\s*##|\Z)", input_string, re.DOTALL | re.MULTILINE
        )
        if action_match:
            action = action_match.group(1).strip()
            sections["action"] = action.strip()

        code_blocks = re.findall(r"```(?:code|python)?\s*(.*?)\s*```", input_string, re.DOTALL | re.IGNORECASE)
        if not code_blocks:
            return f"<Error>: no code blocks found in the input string: {input_string}", ["FAIL"], sections

        code_block = code_blocks[-1].strip()
        sections["original_code"] = code_block

        if "computer.wait" in code_block.lower():
            sections["code"] = "WAIT"
            return sections["action"], ["WAIT"], sections
        elif "computer.terminate" in code_block.lower():
            lower_block = code_block.lower()
            if ("failure" in lower_block) or ("fail" in lower_block):
                sections["code"] = "FAIL"
                return code_block, ["FAIL"], sections
            elif "success" in lower_block:
                sections["code"] = "DONE"
                return code_block, ["DONE"], sections
            else:
                return (
                    f"<Error>: terminate action found but no specific status provided in code block: {input_string}",
                    ["FAIL"],
                    sections,
                )

        corrected_code = code_block
        sections["code"] = corrected_code
        sections["code"] = project_coordinate_to_absolute_scale(
            corrected_code, screen_width=screen_size[0], screen_height=screen_size[1], coordinate_type=coordinate_type
        )

        if ("code" not in sections or sections["code"] is None or sections["code"] == "") or (
            "action" not in sections or sections["action"] is None or sections["action"] == ""
        ):
            return f"<Error>: no code parsed: {input_string}", ["FAIL"], sections

        return sections["action"], [sections["code"]], sections

    except Exception as e:
        error_message = (
            f"<Error>: parsing response: {str(e)}\nTraceback:\n{traceback.format_exc()}\nInput string: {input_string}"
        )
        LOG.warning(error_message)
        return error_message, ["FAIL"], sections


def project_coordinate_to_absolute_scale(
    pyautogui_code_relative_coordinates, screen_width, screen_height, coordinate_type="relative"
):
    """
    Convert the relative coordinates in the pyautogui code to absolute coordinates based on the logical screen size.
    """

    def _coordinate_projection(x, y, screen_width, screen_height, coordinate_type):
        if x <= 1.0 and y <= 1.0:
            return int(round(x * screen_width)), int(round(y * screen_height))
        else:
            return int(round(x)), int(round(y))

    pattern = r"(pyautogui\.\w+\([^\)]*\))"
    matches = re.findall(pattern, pyautogui_code_relative_coordinates)

    new_code = pyautogui_code_relative_coordinates

    for full_call in matches:
        func_name_pattern = r"(pyautogui\.\w+)\((.*)\)"
        func_match = re.match(func_name_pattern, full_call, re.DOTALL)
        if not func_match:
            continue

        func_name = func_match.group(1)
        args_str = func_match.group(2)

        try:
            parsed = ast.parse(f"func({args_str})").body[0].value
            parsed_args = parsed.args
            parsed_keywords = parsed.keywords

        except SyntaxError:
            return pyautogui_code_relative_coordinates

        function_parameters = {
            "click": ["x", "y", "clicks", "interval", "button", "duration", "pause"],
            "rightClick": ["x", "y", "duration", "tween", "pause"],
            "middleClick": ["x", "y", "duration", "tween", "pause"],
            "doubleClick": ["x", "y", "interval", "button", "duration", "pause"],
            "tripleClick": ["x", "y", "interval", "button", "duration", "pause"],
            "moveTo": ["x", "y", "duration", "tween", "pause"],
            "dragTo": ["x", "y", "duration", "button", "mouseDownUp", "pause"],
        }

        func_base_name = func_name.split(".")[-1]

        param_names = function_parameters.get(func_base_name, [])

        args = {}
        for idx, arg in enumerate(parsed_args):
            if idx < len(param_names):
                param_name = param_names[idx]
                arg_value = ast.literal_eval(arg)
                args[param_name] = arg_value

        try:
            for kw in parsed_keywords:
                param_name = kw.arg
                arg_value = ast.literal_eval(kw.value)
                args[param_name] = arg_value
        except Exception as e:
            LOG.warning(f"Error parsing keyword arguments: {e}")
            return pyautogui_code_relative_coordinates

        updated = False
        if "x" in args and "y" in args:
            try:
                x_rel = float(args["x"])
                y_rel = float(args["y"])
                x_abs, y_abs = _coordinate_projection(x_rel, y_rel, screen_width, screen_height, coordinate_type)
                args["x"] = x_abs
                args["y"] = y_abs
                updated = True
            except ValueError:
                pass

        if updated:
            reconstructed_args = []
            for idx, param_name in enumerate(param_names):
                if param_name in args:
                    arg_value = args[param_name]
                    if isinstance(arg_value, str):
                        arg_repr = f"'{arg_value}'"
                    else:
                        arg_repr = str(arg_value)
                    reconstructed_args.append(arg_repr)
                else:
                    break

            used_params = set(param_names[: len(reconstructed_args)])
            for kw in parsed_keywords:
                if kw.arg not in used_params:
                    arg_value = args[kw.arg]
                    if isinstance(arg_value, str):
                        arg_repr = f"{kw.arg}='{arg_value}'"
                    else:
                        arg_repr = f"{kw.arg}={arg_value}"
                    reconstructed_args.append(arg_repr)

            new_args_str = ", ".join(reconstructed_args)
            new_full_call = f"{func_name}({new_args_str})"
            new_code = new_code.replace(full_call, new_full_call)

    return new_code
