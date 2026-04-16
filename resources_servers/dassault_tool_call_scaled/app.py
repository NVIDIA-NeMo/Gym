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

"""Scaled Function Calling Benchmark resource server.

Tests LLM function calling at scale (20-100 functions) with deliberate semantic
overlap to stress-test model discrimination. Evaluates across 10 function clusters
(orders, inventory, customers, products, shipping, analytics, support, payments,
marketing, employees) with 100 total functions and 250 test queries.
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


class ToolCallScaledConfig(BaseResourcesServerConfig):
    pass


class ToolCallScaledVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class ToolCallScaledVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    function_correct: bool = False
    predicted_function: Optional[str] = None
    expected_function: Optional[str] = None
    confused_with: Optional[str] = None
    confusion_candidates: List[str] = []
    function_count: int = 0
    params_total: int = 0
    params_correct: int = 0
    param_details: Dict[str, Any] = {}
    predicted_params: Dict[str, Any] = {}
    expected_params: Dict[str, Any] = {}
    exact_match: bool = False
    query_id: Optional[int] = None


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    str_val = str(value).lower().strip()
    if str_val in ("true", "1", "yes"):
        return True
    if str_val in ("false", "0", "no"):
        return False
    try:
        if "." in str_val:
            return float(str_val)
        return int(str_val)
    except (ValueError, TypeError):
        return str_val


def _compare_params(expected_params: Dict[str, Any], predicted_params: Dict[str, Any]) -> Dict[str, Any]:
    if not expected_params:
        return {"total": 0, "correct": 0, "details": {}}

    predicted_params = predicted_params or {}
    details = {}
    total = 0
    correct = 0

    for param_name, expected_value in expected_params.items():
        total += 1
        predicted_value = predicted_params.get(param_name)
        norm_expected = _normalize_value(expected_value)
        norm_predicted = _normalize_value(predicted_value)
        is_correct = norm_expected == norm_predicted
        if is_correct:
            correct += 1
        details[param_name] = {"expected": expected_value, "predicted": predicted_value, "correct": is_correct}

    return {"total": total, "correct": correct, "details": details}


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


class ToolCallScaledServer(SimpleResourcesServer):
    config: ToolCallScaledConfig

    async def verify(self, body: ToolCallScaledVerifyRequest) -> ToolCallScaledVerifyResponse:
        metadata = body.verifier_metadata or {}
        expected_func = metadata.get("expected_function")
        expected_params = metadata.get("expected_params", {})
        confusion_candidates = metadata.get("confusion_candidates", [])
        function_count = metadata.get("function_count", 0)
        query_id = metadata.get("query_id")

        pred_func, pred_params = None, {}
        if body.response and hasattr(body.response, "output") and body.response.output:
            pred_func, pred_params = _extract_tool_call(body.response.output)

        if pred_func is None and body.response and hasattr(body.response, "output_text"):
            text = _strip_thinking(body.response.output_text or "")
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    pred_func = parsed.get("function_name") or parsed.get("function") or parsed.get("name")
                    pred_params = parsed.get("parameters") or parsed.get("params") or parsed.get("arguments", {})
            except (json.JSONDecodeError, TypeError):
                pass

        func_correct = pred_func == expected_func

        confused_with = None
        if not func_correct and pred_func and pred_func in confusion_candidates:
            confused_with = pred_func

        # Only evaluate params when function is correct (matching original eval logic)
        if func_correct and expected_params:
            param_eval = _compare_params(expected_params, pred_params)
        else:
            param_eval = {"total": 0, "correct": 0, "details": {}}

        # Exact match: function correct + all params correct (or no params expected)
        if func_correct:
            if param_eval["total"] > 0:
                exact_match = param_eval["total"] == param_eval["correct"]
            else:
                exact_match = True
        else:
            exact_match = False

        reward = 1.0 if exact_match else 0.0

        return ToolCallScaledVerifyResponse(
            **body.model_dump(),
            reward=reward,
            function_correct=func_correct,
            predicted_function=pred_func,
            expected_function=expected_func,
            confused_with=confused_with,
            confusion_candidates=confusion_candidates,
            function_count=function_count,
            params_total=param_eval["total"],
            params_correct=param_eval["correct"],
            param_details=param_eval["details"],
            predicted_params=pred_params,
            expected_params=expected_params,
            exact_match=exact_match,
            query_id=query_id,
        )


if __name__ == "__main__":
    ToolCallScaledServer.run_webserver()
