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
from unittest.mock import MagicMock

from app import (
    ToolCallMultiRewardResourcesServer,
    ToolCallMultiRewardResourcesServerConfig,
    ToolCallMultiRewardVerifyRequest,
)

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


WEATHER_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": ""}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

EXPECTED = {"name": "get_weather", "arguments": {"city": "San Francisco"}}


def _function_call(name: str, arguments: str) -> dict:
    return {"call_id": "call_1", "name": name, "arguments": arguments, "type": "function_call"}


def _assistant_text(text: str) -> dict:
    return {
        "id": "msg_1",
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _response(output: list) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _server() -> ToolCallMultiRewardResourcesServer:
    return ToolCallMultiRewardResourcesServer(
        config=ToolCallMultiRewardResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name=""
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def _request(output: list, expected_call: dict = EXPECTED) -> ToolCallMultiRewardVerifyRequest:
    return ToolCallMultiRewardVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "weather in San Francisco?"}],
            "tools": WEATHER_TOOLS,
        },
        response=_response(output),
        expected_call=expected_call,
    )


class TestApp:
    def test_sanity(self) -> None:
        _server()

    async def test_perfect_all_components(self) -> None:
        result = await _server().verify(
            _request([_function_call("get_weather", json.dumps({"city": "San Francisco"}))])
        )
        assert result.reward_components == {"correctness": 1.0, "schema_valid": 1.0, "format": 1.0}
        assert result.reward == 3.0

    async def test_wrong_city_keeps_schema_and_format(self) -> None:
        # Correct tool, valid schema, but wrong argument value -> only correctness drops.
        result = await _server().verify(
            _request([_function_call("get_weather", json.dumps({"city": "New York"}))])
        )
        assert result.reward_components == {"correctness": 0.0, "schema_valid": 1.0, "format": 1.0}
        assert result.reward == 2.0

    async def test_missing_required_param_drops_schema(self) -> None:
        result = await _server().verify(
            _request([_function_call("get_weather", json.dumps({}))])
        )
        assert result.correctness == 0.0
        assert result.schema_valid == 0.0
        assert result.format == 1.0

    async def test_invalid_json_arguments_drops_schema(self) -> None:
        result = await _server().verify(
            _request([_function_call("get_weather", "{not valid json")])
        )
        assert result.schema_valid == 0.0
        assert result.format == 1.0

    async def test_extra_text_drops_format(self) -> None:
        # Correct, valid call but accompanied by assistant prose -> format drops only.
        result = await _server().verify(
            _request(
                [
                    _function_call("get_weather", json.dumps({"city": "San Francisco"})),
                    _assistant_text("Here you go!"),
                ]
            )
        )
        assert result.correctness == 1.0
        assert result.schema_valid == 1.0
        assert result.format == 0.0

    async def test_two_calls_drops_format(self) -> None:
        result = await _server().verify(
            _request(
                [
                    _function_call("get_weather", json.dumps({"city": "San Francisco"})),
                    _function_call("get_weather", json.dumps({"city": "San Francisco"})),
                ]
            )
        )
        assert result.format == 0.0

    async def test_same_total_different_composition(self) -> None:
        # The GDPO motivation: two responses, same summed reward (2.0), different
        # component composition -> GRPO would collapse them, GDPO would not.
        wrong_city = await _server().verify(
            _request([_function_call("get_weather", json.dumps({"city": "New York"}))])
        )
        with_extra_text = await _server().verify(
            _request(
                [
                    _function_call("get_weather", json.dumps({"city": "San Francisco"})),
                    _assistant_text("Done."),
                ]
            )
        )
        assert wrong_city.reward == with_extra_text.reward == 2.0
        assert wrong_city.reward_components != with_extra_text.reward_components
