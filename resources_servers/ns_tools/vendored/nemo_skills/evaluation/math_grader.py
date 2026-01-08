# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re

from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify

from ..utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def _additional_normalization(expr):
    # Remove % and \\% from the number
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    # Remove . corresponding to the end of sentence
    expr = expr.rstrip(".\\")
    return expr


def math_equal(gt_answer, predicted_answer, take_modulo: int | None = None, **kwargs):
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    # if we are sure that gt is always integer
    if take_modulo is not None:
        gt_answer = int(gt_answer) % take_modulo
        try:
            predicted_answer = int(predicted_answer) % take_modulo
        except Exception:
            predicted_answer = None
        # no need to simpy call in this case
        return predicted_answer == gt_answer

    # Try to compare as MCQ options
    mcq_options = "ABCDEFGHIJ"
    norm_gt_mcq = gt_answer.strip()

    is_mcq = re.fullmatch("|".join(mcq_options), norm_gt_mcq)
    parsed_gt = parse(gt_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    parsed_pred = parse(predicted_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    if is_mcq and verify(parsed_gt, parsed_pred):
        return verify(parsed_gt, parsed_pred)

    # Additional normalization step
    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)

    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_normalized_equal = normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    # Fast path: if normalized strings are equal, no need for symbolic comparison
    if is_normalized_equal:
        return True

    # For TEXT literals (not numeric), use direct string comparison
    text_literal_pattern = r"[a-zA-Z ,]+"
    is_text_literal = re.fullmatch(text_literal_pattern, normalized_gt) and re.fullmatch(
        text_literal_pattern, normalized_pred
    )
    if is_text_literal:
        return False  # Already checked is_normalized_equal above

    # Fallback to symbolic comparison via math_verify
    # This handles leading zeros ("016" == "16"), fractions, expressions, etc.
    current_gt_answer = gt_answer
    current_predicted_answer = predicted_answer

    # math_verify.parse expects input to be in latex environment, e.g. $...$
    latex_env_search_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_env_search_pattern, current_gt_answer, re.DOTALL):
        current_gt_answer = f"${current_gt_answer}$"
    if not re.search(latex_env_search_pattern, current_predicted_answer, re.DOTALL):
        current_predicted_answer = f"${current_predicted_answer}$"

    parsed_gt = parse(current_gt_answer, [LatexExtractionConfig()])
    parsed_pred = parse(current_predicted_answer, [LatexExtractionConfig()])

    return verify(parsed_gt, parsed_pred, **kwargs)


def extract_answer(
    string: str, extract_from_boxed: bool = True, extract_regex: str = r"The final answer is (.+)$", relaxed=False
):
    """Extract Answer String from \\boxed expression or based on regex
    If relaxed=True: try both methods, regex first.
    If relaxed=False: use only one method based on extract_from_boxed flag.
    """
    if relaxed:
        return search_regex(string, extract_regex) or search_boxed(string)

    if extract_from_boxed:
        return search_boxed(string)
    return search_regex(string, extract_regex)


def search_regex(string: str, regex: str):
    match = re.findall(regex, string)
    if match:
        return match[-1]
    return None


def search_boxed(string: str):
    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None
