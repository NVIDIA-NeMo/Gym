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

"""Multi-Step Tool Orchestration Benchmark resource server.

Tests LLM ability to plan and sequence multiple tool calls in the correct order
to accomplish complex tasks that require data from multiple sources. Evaluates
across 33 tools and 70 test queries at 4 difficulty levels (easy, medium, hard,
very_hard) with sequences ranging from 2-step to 6+ step chains.
"""

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ToolCallMultistepConfig(BaseResourcesServerConfig):
    pass


class ToolCallMultistepVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class ToolCallMultistepVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    sequence_exact_match: bool = False
    function_recall: float = 0.0
    function_precision: float = 0.0
    order_correct: bool = False
    expected_functions: List[str] = []
    predicted_functions: List[str] = []
    missing_functions: List[str] = []
    extra_functions: List[str] = []
    expected_length: int = 0
    predicted_length: int = 0
    difficulty: str = "unknown"
    query_id: Optional[int] = None


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json_array(text: str) -> Optional[List[Dict]]:
    """Extract a JSON array from model output text."""
    text = _strip_thinking(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON array in markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to find any JSON array
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _check_order_correct(pred_functions: List[str], expected_functions: List[str]) -> bool:
    """Check if relative ordering of matched functions is preserved."""
    if len(pred_functions) < 2 or len(expected_functions) < 2:
        return True
    for i, exp_func in enumerate(expected_functions[:-1]):
        for j, exp_func2 in enumerate(expected_functions[i + 1 :], i + 1):
            if exp_func in pred_functions and exp_func2 in pred_functions:
                pred_i = pred_functions.index(exp_func)
                pred_j = pred_functions.index(exp_func2)
                if pred_i > pred_j:
                    return False
    return True


class ToolCallMultistepServer(SimpleResourcesServer):
    config: ToolCallMultistepConfig

    async def verify(self, body: ToolCallMultistepVerifyRequest) -> ToolCallMultistepVerifyResponse:
        metadata = body.verifier_metadata or {}
        expected_sequence = metadata.get("expected_sequence", [])
        difficulty = metadata.get("difficulty", "unknown")
        query_id = metadata.get("query_id")

        expected_functions = [s["function"] for s in expected_sequence]

        pred_functions = []
        if body.response and hasattr(body.response, "output_text"):
            text = body.response.output_text or ""
            prediction = _extract_json_array(text)
            if prediction:
                for step in prediction:
                    if isinstance(step, dict) and "function" in step:
                        pred_functions.append(step["function"])

        sequence_exact_match = pred_functions == expected_functions

        pred_set = set(pred_functions)
        expected_set = set(expected_functions)
        correct_functions = pred_set & expected_set
        missing_functions = expected_set - pred_set
        extra_functions = pred_set - expected_set

        function_recall = len(correct_functions) / len(expected_set) if expected_set else 0.0
        function_precision = len(correct_functions) / len(pred_set) if pred_set else 0.0

        order_correct = _check_order_correct(pred_functions, expected_functions)

        reward = 1.0 if sequence_exact_match else 0.0

        return ToolCallMultistepVerifyResponse(
            **body.model_dump(),
            reward=reward,
            sequence_exact_match=sequence_exact_match,
            function_recall=function_recall,
            function_precision=function_precision,
            order_correct=order_correct,
            expected_functions=expected_functions,
            predicted_functions=pred_functions,
            missing_functions=list(missing_functions),
            extra_functions=list(extra_functions),
            expected_length=len(expected_functions),
            predicted_length=len(pred_functions),
            difficulty=difficulty,
            query_id=query_id,
        )


if __name__ == "__main__":
    ToolCallMultistepServer.run_webserver()
