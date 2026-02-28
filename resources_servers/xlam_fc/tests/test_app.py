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
    XlamFcResourcesServer,
    XlamFcResourcesServerConfig,
    XlamFcVerifyRequest,
)

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        config = XlamFcResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        XlamFcResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_verify_exact_match_single_call(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco"}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 1

    async def test_verify_exact_match_multiple_calls(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "live_giveaways_by_type",
                    "arguments": json.dumps({"type": "beta"}),
                    "type": "function_call",
                },
                {
                    "call_id": "call_2",
                    "name": "live_giveaways_by_type",
                    "arguments": json.dumps({"type": "game"}),
                    "type": "function_call",
                },
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "Where can I find live giveaways?"},
                ],
            },
            response=response,
            expected_answers=[
                {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
                {"name": "live_giveaways_by_type", "arguments": {"type": "game"}},
            ],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 2
        assert result.num_predicted == 2
        assert result.num_correct == 2

    async def test_verify_partial_match(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "live_giveaways_by_type",
                    "arguments": json.dumps({"type": "beta"}),
                    "type": "function_call",
                },
                {
                    "call_id": "call_2",
                    "name": "live_giveaways_by_type",
                    "arguments": json.dumps({"type": "loot"}),
                    "type": "function_call",
                },
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "Where can I find live giveaways?"},
                ],
            },
            response=response,
            expected_answers=[
                {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
                {"name": "live_giveaways_by_type", "arguments": {"type": "game"}},
            ],
        )

        result = await server.verify(verify_request)
        assert result.reward == 0.0
        assert result.num_expected == 2
        assert result.num_predicted == 2
        assert result.num_correct == 1

    async def test_verify_no_match(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "wrong_function",
                    "arguments": json.dumps({"city": "New York"}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 0.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 0

    async def test_verify_wrong_arguments(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "New York"}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 0.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 0

    async def test_verify_no_predictions(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_1",
                    "content": [
                        {
                            "annotations": [],
                            "text": "I don't know how to help with that.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 0.0
        assert result.num_expected == 1
        assert result.num_predicted == 0
        assert result.num_correct == 0

    async def test_verify_extra_arguments(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco", "units": "celsius"}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 1

    async def test_verify_no_expected_no_predicted(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_1",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Here's some information.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "Tell me something."},
                ],
            },
            response=response,
            expected_answers=[],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 0
        assert result.num_predicted == 0
        assert result.num_correct == 0

    async def test_verify_multiple_arguments(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "t3ma",
                    "arguments": json.dumps({"symbol": "ETH/BTC", "interval": "1h", "time_period": 14}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "Get T3MA for ETH/BTC"},
                ],
            },
            response=response,
            expected_answers=[
                {"name": "t3ma", "arguments": {"symbol": "ETH/BTC", "interval": "1h", "time_period": 14}}
            ],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 1
