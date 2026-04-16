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

"""Enterprise Function Calling Benchmark resource server.

Tests LLM function calling with enterprise-grade tool schemas featuring complex
JSON Schema constructs: anyOf types, regex patterns, enums, nullable fields, and
detailed business descriptions. Evaluates across 10 function clusters (requirements,
projects, engineering changes, quality, configuration, suppliers, documents, assets,
compliance, workforce) with 100 total functions scaled at 20/40/60/80/100.
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


class ToolCallEnterpriseConfig(BaseResourcesServerConfig):
    pass


class ToolCallEnterpriseVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class ToolCallEnterpriseVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    function_correct: bool = False
    predicted_function: Optional[str] = None
    expected_function: Optional[str] = None
    confused_with: Optional[str] = None
    function_count: int = 0
    params_total: int = 0
    params_correct: int = 0
    param_details: List[Dict[str, Any]] = []
    predicted_params: Dict[str, Any] = {}
    expected_params: Dict[str, Any] = {}
    exact_match: bool = False


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _values_match(predicted: Any, expected: Any) -> bool:
    if predicted == expected:
        return True
    if str(predicted).lower() == str(expected).lower():
        return True
    return False


def _evaluate_params(predicted: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    if not expected:
        return {"total": 0, "correct": 0, "details": []}

    details = []
    correct = 0
    for key, exp_val in expected.items():
        pred_val = predicted.get(key)
        is_correct = _values_match(pred_val, exp_val)
        if is_correct:
            correct += 1
        details.append({"param": key, "expected": exp_val, "predicted": pred_val, "correct": is_correct})

    return {"total": len(expected), "correct": correct, "details": details}


def _extract_tool_call(response_output: list) -> tuple:
    """Extract function name and arguments from the model response output items."""
    for item in response_output:
        item_dict = item if isinstance(item, dict) else item.model_dump() if hasattr(item, "model_dump") else {}
        if item_dict.get("type") == "function_call":
            name = item_dict.get("name")
            args_str = item_dict.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, TypeError):
                args = {}
            return name, args
    return None, {}


class ToolCallEnterpriseServer(SimpleResourcesServer):
    config: ToolCallEnterpriseConfig

    async def verify(self, body: ToolCallEnterpriseVerifyRequest) -> ToolCallEnterpriseVerifyResponse:
        metadata = body.verifier_metadata or {}
        expected_func = metadata.get("expected_function")
        expected_params = metadata.get("expected_params", {})
        confusion_candidates = metadata.get("confusion_candidates", [])
        function_count = metadata.get("function_count", 0)

        pred_func, pred_params = None, {}
        if body.response and hasattr(body.response, "output") and body.response.output:
            pred_func, pred_params = _extract_tool_call(body.response.output)

        if pred_func is None and body.response and hasattr(body.response, "output_text"):
            text = _strip_thinking(body.response.output_text or "")
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    pred_func = parsed.get("function") or parsed.get("function_name") or parsed.get("name")
                    pred_params = parsed.get("parameters") or parsed.get("params") or parsed.get("arguments", {})
            except (json.JSONDecodeError, TypeError):
                pass

        func_correct = pred_func == expected_func
        confused_with = None
        if not func_correct and pred_func and pred_func in confusion_candidates:
            confused_with = pred_func

        param_eval = _evaluate_params(pred_params, expected_params)
        exact_match = func_correct and param_eval["total"] == param_eval["correct"]

        reward = 1.0 if exact_match else 0.0

        return ToolCallEnterpriseVerifyResponse(
            **body.model_dump(),
            reward=reward,
            function_correct=func_correct,
            predicted_function=pred_func,
            expected_function=expected_func,
            confused_with=confused_with,
            function_count=function_count,
            params_total=param_eval["total"],
            params_correct=param_eval["correct"],
            param_details=param_eval["details"],
            predicted_params=pred_params,
            expected_params=expected_params,
            exact_match=exact_match,
        )


if __name__ == "__main__":
    ToolCallEnterpriseServer.run_webserver()
