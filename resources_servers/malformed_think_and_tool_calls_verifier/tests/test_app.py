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
from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.malformed_think_and_tool_calls_verifier.app import (
    MalformedThinkAndToolCallsVerifierRequest,
    MalformedThinkAndToolCallsVerifierResourcesServer,
    MalformedThinkAndToolCallsVerifierResourcesServerConfig,
)


def _make_server():
    config = MalformedThinkAndToolCallsVerifierResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
    )
    return MalformedThinkAndToolCallsVerifierResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_request(response: NeMoGymResponse, label: str = "test_label"):
    return MalformedThinkAndToolCallsVerifierRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "Hello"}],
        },
        response=response,
        label=label,
    )


def _clean_assistant_response(**overrides) -> NeMoGymResponse:
    defaults = dict(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": "This is a clean response.", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return NeMoGymResponse(**(defaults | overrides))


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    async def test_verify_clean_response(self) -> None:
        server = _make_server()
        result = await server.verify(_make_request(_clean_assistant_response()))
        assert result.reward == 1.0
        assert result.response_error_type is None

    async def test_verify_incomplete(self) -> None:
        server = _make_server()
        response = _clean_assistant_response(incomplete_details={"reason": "max_output_tokens"})
        result = await server.verify(_make_request(response))
        assert result.reward == 0.0
        assert result.response_error_type == "incomplete"

    async def test_verify_malformed_thinking(self) -> None:
        server = _make_server()
        response = _clean_assistant_response(
            output=[
                {
                    "id": "rs_test",
                    "summary": [{"text": "Let me use <tool_call> here", "type": "summary_text"}],
                    "type": "reasoning",
                },
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "Clean assistant text.", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ]
        )
        result = await server.verify(_make_request(response))
        assert result.reward == 0.0
        assert result.response_error_type == "malformed_thinking"

    async def test_verify_malformed_thinking_closing_tag(self) -> None:
        server = _make_server()
        response = _clean_assistant_response(
            output=[
                {
                    "id": "rs_test",
                    "summary": [{"text": "some text </tool_call> leftover", "type": "summary_text"}],
                    "type": "reasoning",
                },
            ]
        )
        result = await server.verify(_make_request(response))
        assert result.reward == 0.0
        assert result.response_error_type == "malformed_thinking"

    async def test_verify_malformed_tool_call(self) -> None:
        server = _make_server()
        response = _clean_assistant_response(
            output=[
                {
                    "id": "msg_test",
                    "content": [
                        {
                            "annotations": [],
                            "text": 'Here is the result: <tool_call>{"name": "get_weather"}</tool_call>',
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
        )
        result = await server.verify(_make_request(response))
        assert result.reward == 0.0
        assert result.response_error_type == "malformed_tool_call"
