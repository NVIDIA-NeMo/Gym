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

import json
from typing import Any, Dict, List

from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class XlamFcResourcesServerConfig(BaseResourcesServerConfig):
    pass


class XlamFcVerifyRequest(BaseVerifyRequest):
    expected_answers: List[Dict[str, Any]] = Field(default_factory=list)


class XlamFcVerifyResponse(BaseVerifyResponse):
    num_expected: int = 0
    num_predicted: int = 0
    num_correct: int = 0
    predicted_calls: List[Dict[str, Any]] = Field(default_factory=list)


class XlamFcResourcesServer(SimpleResourcesServer):
    config: XlamFcResourcesServerConfig

    @staticmethod
    def _normalize_arguments(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {}
        elif isinstance(arguments, dict):
            return arguments
        else:
            return {}

    @staticmethod
    def _function_calls_match(predicted: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        if predicted.get("name") != expected.get("name"):
            return False

        predicted_args = XlamFcResourcesServer._normalize_arguments(predicted.get("arguments", {}))
        expected_args = XlamFcResourcesServer._normalize_arguments(expected.get("arguments", {}))

        for key, expected_value in expected_args.items():
            if key not in predicted_args:
                return False
            if predicted_args[key] != expected_value:
                return False

        return True

    def _extract_function_calls_from_response(self, body: BaseVerifyRequest) -> List[Dict[str, Any]]:
        function_calls = []

        for output_item in body.response.output:
            if output_item.type == "function_call":
                function_call = {
                    "name": output_item.name,
                    "arguments": self._normalize_arguments(output_item.arguments),
                }
                function_calls.append(function_call)

        return function_calls

    def _calculate_reward(
        self, predicted_calls: List[Dict[str, Any]], expected_answers: List[Dict[str, Any]]
    ) -> tuple[float, int]:
        if not expected_answers:
            return (1.0, 0) if not predicted_calls else (0.0, 0)

        num_correct = 0
        matched_predicted_indices = set()

        for expected_call in expected_answers:
            for i, predicted_call in enumerate(predicted_calls):
                if i in matched_predicted_indices:
                    continue

                if self._function_calls_match(predicted_call, expected_call):
                    num_correct += 1
                    matched_predicted_indices.add(i)
                    break

        reward = 1.0 if num_correct == len(expected_answers) == len(predicted_calls) else 0.0
        return reward, num_correct

    async def verify(self, body: XlamFcVerifyRequest) -> XlamFcVerifyResponse:
        predicted_calls = self._extract_function_calls_from_response(body)

        reward, num_correct = self._calculate_reward(predicted_calls, body.expected_answers)

        return XlamFcVerifyResponse(
            **body.model_dump(),
            reward=reward,
            num_expected=len(body.expected_answers),
            num_predicted=len(predicted_calls),
            num_correct=num_correct,
            predicted_calls=predicted_calls,
        )


if __name__ == "__main__":
    XlamFcResourcesServer.run_webserver()
