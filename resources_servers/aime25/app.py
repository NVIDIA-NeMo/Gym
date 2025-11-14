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

import re
from typing import Union

from fastapi import FastAPI
from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class AIME25ResourcesServerConfig(BaseResourcesServerConfig):
    pass


class AIME25RunRequest(BaseRunRequest):
    expected_answer: str


class AIME25VerifyRequest(AIME25RunRequest, BaseVerifyRequest):
    pass


def extract_answer(string):
    """Extract Answer String from \\boxed expression."""
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
        except:
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

    # Try literal comparison
    literal_pattern = r"[a-zA-Z ,]+|[0-9 ]+"
    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_literal = re.fullmatch(literal_pattern, normalized_gt) and re.fullmatch(literal_pattern, normalized_pred)
    is_normalized_equal = normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    if is_literal or is_normalized_equal:
        return is_normalized_equal

    # Fallback to symbolic comparison
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


def extract_math_answer(response: str) -> str:
    """Extract math answer from response (alias for extract_answer)."""
    return extract_answer(response)


class AIME25ResourcesServer(SimpleResourcesServer):
    config: AIME25ResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: AIME25VerifyRequest) -> BaseVerifyResponse:
        expected_answer = body.expected_answer

        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        prediction = extract_math_answer(combined_response)

        is_correct = prediction and math_equal(expected_answer, prediction)

        reward = 1.0 if is_correct else 0.0

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    AIME25ResourcesServer.run_webserver()
