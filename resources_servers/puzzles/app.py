# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Puzzles resources server.

Port of NeMo-RLVR's `nemo_rl/environments/puzzles_environment.py`. Verifies a
model's puzzle solution against a ground-truth answer using one of several
extraction/comparison strategies (json_block, boxed, python_block,
double_brackets, etc.). Pure scoring — no tools, no external dependencies.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Callable, Optional, Union

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Extraction helpers (kept identical in behavior to the RLVR originals).
# ----------------------------------------------------------------------------


def extract_json(text: str) -> Optional[Any]:
    """Extract JSON from a single ```json ... ``` markdown code block."""
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError as e:
            logger.debug("Failed to parse JSON block: %s", e)
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract answer from \boxed{...} format."""
    pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def extract_python_list(text: str) -> Optional[list]:
    """Extract a Python 2D list from model output."""
    pattern = r"\[\s*\[.*?\]\s*\]"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in reversed(matches):
        try:
            result = ast.literal_eval(match)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                return result
        except (ValueError, SyntaxError) as e:
            logger.debug("Failed to parse 2D list '%s...': %s", match[:50], e)
            continue
    return None


def extract_1d_list(text: str) -> Optional[list]:
    """Extract a 1D list from model output (e.g. scheduling puzzles)."""
    pattern = r"\[[\d\s,]+\]"
    matches = re.findall(pattern, text)
    if matches:
        for match in reversed(matches):
            try:
                result = ast.literal_eval(match)
                if isinstance(result, list) and len(result) > 0 and not isinstance(result[0], list):
                    return result
            except (ValueError, SyntaxError) as e:
                logger.debug("Failed to parse 1D list '%s': %s", match, e)
                continue
    return None


def extract_json_answer(text: str) -> Optional[Any]:
    """Extract answer from JSON response with `result[0].answer` shape."""
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict) and "result" in parsed:
                results = parsed["result"]
                if isinstance(results, list) and len(results) > 0:
                    return results[0].get("answer")
        except json.JSONDecodeError as e:
            logger.debug("Failed to parse JSON answer block: %s", e)

    json_obj = extract_json(text)
    if json_obj and isinstance(json_obj, dict):
        if "answer" in json_obj:
            return json_obj["answer"]
        if "result" in json_obj:
            results = json_obj["result"]
            if isinstance(results, list) and len(results) > 0:
                return results[0].get("answer")
    return None


def extract_double_brackets(text: str) -> Optional[str]:
    """Extract content from [[...]] format (last match)."""
    pattern = r"\[\[(.*?)\]\]"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def extract_bold_text(text: str) -> Optional[str]:
    """Extract text from **...** format (last match)."""
    pattern = r"\*\*([^*]+)\*\*"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_python_block(text: str) -> Optional[str]:
    """Extract content from ```python block."""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_tuple_list(text: str) -> Optional[list]:
    """Extract list of tuples like [(0,1), (2,3)]."""
    pattern = r"\[[\s\(\)\d,\s]+\]"
    matches = re.findall(pattern, text)
    for match in reversed(matches):
        try:
            result = ast.literal_eval(match)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], tuple):
                return result
        except (ValueError, SyntaxError) as e:
            logger.debug("Failed to parse tuple list '%s...': %s", match[:50], e)
            continue
    return None


def parse_ground_truth(ground_truth: Any) -> Any:
    """Parse a ground-truth string into a Python object (literal_eval, then json)."""
    if not isinstance(ground_truth, str):
        return ground_truth
    try:
        return ast.literal_eval(ground_truth)
    except (ValueError, SyntaxError, TypeError):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            return ground_truth


# ----------------------------------------------------------------------------
# Comparison helpers.
# ----------------------------------------------------------------------------


def compare_2d_structures(model_data: Any, gt_data: Any, case_insensitive: bool = True) -> bool:
    """Compare two 2D list/tuple structures."""
    if len(model_data) != len(gt_data):
        return False
    for row_model, row_gt in zip(model_data, gt_data):
        if len(row_model) != len(row_gt):
            return False
        for cell_model, cell_gt in zip(row_model, row_gt):
            if case_insensitive:
                if str(cell_model).strip().lower() != str(cell_gt).strip().lower():
                    return False
            else:
                if cell_model != cell_gt:
                    return False
    return True


def smart_compare(model_value: Any, gt_value: Any) -> bool:
    """Smart comparison that handles different types automatically."""
    if model_value is None:
        return False

    if isinstance(gt_value, bool):
        # bool must be checked before int (bool is a subclass of int).
        return bool(model_value) == gt_value

    if isinstance(gt_value, int):
        try:
            return int(model_value) == gt_value
        except (ValueError, TypeError):
            return False

    if isinstance(gt_value, float):
        try:
            return abs(float(model_value) - gt_value) < 0.01
        except (ValueError, TypeError):
            return False

    if isinstance(gt_value, (list, tuple)):
        if not isinstance(model_value, (list, tuple)):
            return False
        if len(model_value) != len(gt_value):
            return False
        if len(gt_value) > 0 and isinstance(gt_value[0], (list, tuple)):
            return compare_2d_structures(model_value, gt_value)
        return all(smart_compare(m, g) for m, g in zip(model_value, gt_value))

    if isinstance(gt_value, dict):
        if not isinstance(model_value, dict):
            return False
        for key, val in gt_value.items():
            if key not in model_value:
                return False
            if not smart_compare(model_value[key], val):
                return False
        return True

    return str(model_value).strip().lower() == str(gt_value).strip().lower()


# ----------------------------------------------------------------------------
# Verification handlers (one per verification_type).
# ----------------------------------------------------------------------------


def verify_dict_match(model_output: str, ground_truth: Any) -> bool:
    """dict_match: extract JSON, expect a 'solution' key matching ground_truth dict."""
    if isinstance(ground_truth, str):
        try:
            ground_truth_dict = json.loads(ground_truth)
        except json.JSONDecodeError:
            return False
    else:
        ground_truth_dict = ground_truth

    output = extract_json(model_output)
    if not output or "solution" not in output:
        return False
    output = output["solution"]

    for key, value in ground_truth_dict.items():
        if key not in output:
            return False
        for attr, val in value.items():
            if attr not in output[key]:
                return False
            if str(output[key][attr]).strip().lower() != str(val).strip().lower():
                return False
    return True


def verify_json_block(model_output: str, ground_truth: Any) -> bool:
    """json_block: any JSON in ```json block."""
    gt_value = parse_ground_truth(ground_truth)
    model_json = extract_json(model_output)

    if model_json is None:
        model_json = extract_json_answer(model_output)
        if model_json is None:
            logger.debug("Failed to extract JSON. Output: %s...", model_output[:200])
            return False
        return smart_compare(model_json, gt_value)

    if isinstance(model_json, dict) and "solution" in model_json:
        model_json = model_json["solution"]

    if isinstance(gt_value, dict) and "rows" in gt_value:
        if not isinstance(model_json, dict) or "rows" not in model_json:
            return False
        return compare_2d_structures(model_json["rows"], gt_value["rows"])

    return smart_compare(model_json, gt_value)


def verify_boxed(model_output: str, ground_truth: Any) -> bool:
    r"""boxed: any value in \boxed{} format."""
    gt_value = parse_ground_truth(ground_truth)
    boxed = extract_boxed_answer(model_output)
    if boxed is None:
        logger.debug("Failed to extract boxed answer. Output: %s...", model_output[:200])
        return False
    return smart_compare(boxed, gt_value)


def verify_python_block(model_output: str, ground_truth: Any) -> bool:
    """python_block: any value in ```python block."""
    gt_value = parse_ground_truth(ground_truth)
    code_block = extract_python_block(model_output)
    if code_block is None:
        logger.debug("Failed to extract python block. Output: %s...", model_output[:200])
        return False

    try:
        model_value = ast.literal_eval(code_block)
        return smart_compare(model_value, gt_value)
    except (ValueError, SyntaxError) as e:
        logger.debug("Failed to parse python block as literal: %s", e)

    # Fall back to evaluating as a pure-arithmetic expression in a sandbox.
    try:
        model_result = eval(code_block, {"__builtins__": {}}, {})
        return smart_compare(model_result, gt_value)
    except Exception as e:
        logger.debug("Failed to evaluate python block as expression: %s", e)

    return code_block.strip().lower() == str(gt_value).strip().lower()


def verify_double_brackets(model_output: str, ground_truth: Any) -> bool:
    """double_brackets: any value in [[...]] format."""
    gt_value = parse_ground_truth(ground_truth)
    content = extract_double_brackets(model_output)
    if content is None:
        logger.debug("Failed to extract double brackets. Output: %s...", model_output[:200])
        return False

    try:
        if content.startswith("["):
            model_value = ast.literal_eval(content)
        else:
            model_value = ast.literal_eval(f"[{content}]")
        return smart_compare(model_value, gt_value)
    except (ValueError, SyntaxError) as e:
        logger.debug("Failed to parse double brackets content as literal: %s", e)

    return content.strip().lower() == str(gt_value).strip().lower()


def verify_python_2d_list(model_output: str, ground_truth: Any) -> bool:
    """python_2d_list: 2D list in Python format (no code block)."""
    gt_list = parse_ground_truth(ground_truth)
    model_list = extract_python_list(model_output)
    if model_list is None:
        logger.debug("Failed to extract 2D list. Output: %s...", model_output[:200])
        return False
    return compare_2d_structures(model_list, gt_list)


def verify_bracket_list(model_output: str, ground_truth: Any) -> bool:
    """bracket_list: 1D list in [v1, v2] format."""
    gt_list = parse_ground_truth(ground_truth)
    model_list = extract_1d_list(model_output)
    if model_list is None:
        logger.debug("Failed to extract 1D list. Output: %s...", model_output[:200])
        return False
    return smart_compare(model_list, gt_list)


def verify_plain_number(model_output: str, ground_truth: Any) -> bool:
    """plain_number: just output the number; pick any number in the response that matches."""
    gt_value = parse_ground_truth(ground_truth)
    numbers = re.findall(r"-?\d+\.?\d*", model_output)
    if not numbers:
        logger.debug("Failed to find number in output: %s...", model_output[:200])
        return False
    for num_str in reversed(numbers):
        try:
            if isinstance(gt_value, int):
                if int(float(num_str)) == gt_value:
                    return True
            elif isinstance(gt_value, float):
                if abs(float(num_str) - gt_value) < 0.01:
                    return True
        except ValueError as e:
            logger.debug("Failed to parse number '%s': %s", num_str, e)
            continue
    return False


def verify_tuple_list(model_output: str, ground_truth: Any) -> bool:
    """tuple_list: list of tuples like [(0,1), (2,3)]; order-insensitive."""
    gt_list = parse_ground_truth(ground_truth)
    model_list = extract_tuple_list(model_output)
    if model_list is None:
        logger.debug("Failed to extract tuple list. Output: %s...", model_output[:200])
        return False
    if not isinstance(gt_list, list) or len(model_list) != len(gt_list):
        return False
    try:
        return sorted(model_list) == sorted(gt_list)
    except TypeError as e:
        logger.debug("Cannot sort lists for comparison, using direct compare: %s", e)
        return model_list == gt_list


def verify_bold_yes_no(model_output: str, ground_truth: Any) -> bool:
    """bold_yes_no: **yes, no, yes** format."""
    bold_content = extract_bold_text(model_output)
    if bold_content is None:
        logger.debug("Failed to extract bold text. Output: %s...", model_output[:200])
        return False
    return bold_content.strip().lower() == str(ground_truth).strip().lower()


def verify_plain_sequence(model_output: str, ground_truth: Any) -> bool:
    """plain_sequence: bracket/paren sequence anywhere in the response."""
    gt_str = str(ground_truth).strip()
    model_clean = model_output.strip()
    lines = model_clean.split("\n")
    for line in reversed(lines):
        line_clean = line.strip()
        if line_clean and all(c in "()[]{}<>" for c in line_clean):
            if line_clean == gt_str:
                return True
    if gt_str in model_output:
        return True
    logger.debug("Failed to find sequence '%s' in output: %s...", gt_str, model_output[:200])
    return False


def verify_plain_string(model_output: str, ground_truth: Any) -> bool:
    """plain_string: case-insensitive substring match."""
    gt_str = str(ground_truth).strip().lower()
    if gt_str in model_output.lower():
        return True
    logger.debug("Failed to find string in output: %s...", model_output[:200])
    return False


VERIFICATION_HANDLERS: dict[str, Callable[[str, Any], bool]] = {
    "dict_match": verify_dict_match,
    "json_block": verify_json_block,
    "boxed": verify_boxed,
    "python_block": verify_python_block,
    "double_brackets": verify_double_brackets,
    "python_2d_list": verify_python_2d_list,
    "bracket_list": verify_bracket_list,
    "plain_number": verify_plain_number,
    "tuple_list": verify_tuple_list,
    "bold_yes_no": verify_bold_yes_no,
    "plain_sequence": verify_plain_sequence,
    "plain_string": verify_plain_string,
    # Legacy aliases preserved from RLVR for dataset compatibility.
    "exact_match": verify_plain_string,
    "json_block_rows": verify_json_block,
    "json_response": verify_json_block,
    "boxed_int": verify_boxed,
    "boxed_string": verify_boxed,
    "python_block_tuple": verify_python_block,
    "python_block_list": verify_python_block,
    "python_block_expr": verify_python_block,
    "double_brackets_list": verify_double_brackets,
    "double_brackets_string": verify_double_brackets,
}


def strip_thinking_tokens(text: str) -> str:
    """Strip <think>...</think> content. Unclosed <think> -> empty string (invalid)."""
    if "<think>" in text and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def verify_puzzles_sample(
    model_output: str, ground_truth: Any, verification_type: str = "dict_match"
) -> bool:
    """Verify a puzzle sample given a ground_truth and verification_type."""
    model_output = strip_thinking_tokens(model_output)
    if not model_output:
        return False
    handler = VERIFICATION_HANDLERS.get(verification_type)
    if handler is None:
        logger.warning(
            "Unknown verification type: %s, falling back to plain_string", verification_type
        )
        handler = verify_plain_string
    return handler(model_output, ground_truth)


# ----------------------------------------------------------------------------
# Server.
# ----------------------------------------------------------------------------


class PuzzlesResourcesServerConfig(BaseResourcesServerConfig):
    pass


class PuzzlesVerifyRequest(BaseVerifyRequest):
    # The expected answer. May be a raw string, JSON-encoded literal, or any
    # Python-literal-compatible value depending on `verification_type`.
    ground_truth: Union[str, dict, list, int, float, bool]
    # Selects how the answer is extracted from the model output and compared
    # against `ground_truth`. See `VERIFICATION_HANDLERS` for accepted values.
    verification_type: str = "dict_match"


class PuzzlesVerifyResponse(BaseVerifyResponse):
    verification_type: str
    verification_failed: bool


class PuzzlesResourcesServer(SimpleResourcesServer):
    config: PuzzlesResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: PuzzlesVerifyRequest) -> PuzzlesVerifyResponse:
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                final_response_text = last_output.content[0].text

        verification_failed = False
        reward = 0.0
        try:
            verified = verify_puzzles_sample(
                final_response_text, body.ground_truth, body.verification_type
            )
            reward = 1.0 if verified else 0.0
        except Exception as e:
            logger.warning("Puzzles verification failed: %s", e)
            verification_failed = True
            reward = 0.0

        return PuzzlesVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verification_failed=verification_failed,
        )


if __name__ == "__main__":
    PuzzlesResourcesServer.run_webserver()
