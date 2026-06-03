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
"""Single-tool-call environment scored on three decoupled reward components.

This is a multi-reward environment intended for GDPO
(https://arxiv.org/abs/2601.05242). A single expected tool call is graded on three
independent {0, 1} components:

- ``correctness``  : a predicted call matches the expected name + expected arguments.
- ``schema_valid`` : the call's arguments parse as a JSON object containing every
                     required parameter of the tool.
- ``format``       : exactly one tool call was emitted and no extra assistant text.

These decouple: a response can be well-formed but wrong, correct but malformed, etc.
GDPO normalizes each component independently, so two responses with the same total
reward but different composition receive different advantages (whereas GRPO, which
normalizes the summed reward, collapses them).

The components are returned in ``reward_components``; NeMo-RL exposes them as reward1,
reward2, ... ordered by component name (correctness, format, schema_valid). ``reward``
is set to the sum so a single-reward (GRPO) baseline reads the same aggregate.
"""

import json
from typing import Any, Dict, List

from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ToolCallMultiRewardResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ToolCallMultiRewardVerifyRequest(BaseVerifyRequest):
    # The single tool call the model is expected to make, e.g.
    # {"name": "get_weather", "arguments": {"city": "San Francisco"}}.
    expected_call: Dict[str, Any] = Field(default_factory=dict)


class ToolCallMultiRewardVerifyResponse(BaseVerifyResponse):
    # Per-component scores are also surfaced as top-level fields so the aggregate
    # metrics endpoint profiles each one in addition to the combined reward.
    correctness: float = 0.0
    schema_valid: float = 0.0
    format: float = 0.0
    predicted_calls: List[Dict[str, Any]] = Field(default_factory=list)


class ToolCallMultiRewardResourcesServer(SimpleResourcesServer):
    config: ToolCallMultiRewardResourcesServerConfig

    @staticmethod
    def _parse_arguments(arguments: Any) -> tuple[Dict[str, Any], bool]:
        """Return (parsed_dict, is_valid_json_object)."""
        if isinstance(arguments, dict):
            return arguments, True
        if isinstance(arguments, str):
            try:
                value = json.loads(arguments)
            except json.JSONDecodeError:
                return {}, False
            return (value, True) if isinstance(value, dict) else ({}, False)
        return {}, False

    def _extract_function_calls(self, body: BaseVerifyRequest) -> List[Dict[str, Any]]:
        calls = []
        for output_item in body.response.output:
            if output_item.type == "function_call":
                calls.append({"name": output_item.name, "arguments": output_item.arguments})
        return calls

    @staticmethod
    def _has_assistant_text(body: BaseVerifyRequest) -> bool:
        return any(getattr(item, "type", None) == "message" for item in body.response.output)

    def _required_params(self, body: BaseVerifyRequest, tool_name: str) -> List[str]:
        tools = getattr(body.responses_create_params, "tools", None) or []
        for tool in tools:
            tool_dict = tool if isinstance(tool, dict) else tool.model_dump()
            if tool_dict.get("name") == tool_name:
                return tool_dict.get("parameters", {}).get("required", []) or []
        return []

    @staticmethod
    def _call_matches(predicted: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        if not expected:
            return False
        if predicted.get("name") != expected.get("name"):
            return False
        predicted_args, _ = ToolCallMultiRewardResourcesServer._parse_arguments(
            predicted.get("arguments", {})
        )
        expected_args = expected.get("arguments", {}) or {}
        for key, value in expected_args.items():
            if predicted_args.get(key) != value:
                return False
        return True

    async def verify(
        self, body: ToolCallMultiRewardVerifyRequest
    ) -> ToolCallMultiRewardVerifyResponse:
        predicted_calls = self._extract_function_calls(body)

        # format: exactly one tool call and no extra assistant prose.
        format_score = (
            1.0 if len(predicted_calls) == 1 and not self._has_assistant_text(body) else 0.0
        )

        # schema_valid: the first call's arguments are a valid object with all required params.
        schema_score = 0.0
        if predicted_calls:
            first = predicted_calls[0]
            parsed, is_valid = self._parse_arguments(first["arguments"])
            required = self._required_params(body, first["name"])
            schema_score = 1.0 if is_valid and all(k in parsed for k in required) else 0.0

        # correctness: some predicted call matches the expected name + arguments.
        correctness_score = (
            1.0 if any(self._call_matches(c, body.expected_call) for c in predicted_calls) else 0.0
        )

        reward_components = {
            "correctness": correctness_score,
            "schema_valid": schema_score,
            "format": format_score,
        }

        return ToolCallMultiRewardVerifyResponse(
            **body.model_dump(),
            reward=sum(reward_components.values()),
            reward_components=reward_components,
            correctness=correctness_score,
            schema_valid=schema_score,
            format=format_score,
            predicted_calls=[
                {"name": c["name"], "arguments": self._parse_arguments(c["arguments"])[0]}
                for c in predicted_calls
            ],
        )


if __name__ == "__main__":
    ToolCallMultiRewardResourcesServer.run_webserver()
