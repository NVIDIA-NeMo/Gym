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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

from fastapi.testclient import TestClient
from pydantic import ValidationError
from pytest import fixture, raises

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.tool_simulation_agent.app import ToolSimulationAgent, ToolSimulationAgentConfig


class TestApp:
    @fixture
    def agent_config(self) -> ToolSimulationAgentConfig:
        return ToolSimulationAgentConfig(
            host="localhost",
            port=10001,
            entrypoint="",
            name="tool_agent",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="tool_resource_server",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="model_server",
            ),
        )

    def _set_server_client_post_responses(
        self,
        server_client_post_mock: AsyncMock,
        first_response: dict[str, Any],
        *additional_responses: dict[str, Any],
    ) -> None:
        server_client_post_mock.reset_mock(
            return_value=True,
            side_effect=True,
        )
        responses = [first_response]
        additional_responses_present = len(additional_responses) > 0
        if additional_responses_present:
            responses.extend(additional_responses)

        post_responses = []
        for response in responses:
            post_response_mock = AsyncMock()
            post_response_mock.json.return_value = response
            post_responses.append(post_response_mock)

        if additional_responses_present:
            server_client_post_mock.side_effect = post_responses
        else:
            server_client_post_mock.return_value = post_responses[0]

    async def test_responses(self, agent_config: ToolSimulationAgentConfig) -> None:
        server_client_post_mock = AsyncMock()
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = server_client_post_mock
        agent_server = ToolSimulationAgent(
            config=agent_config,
            server_client=server_client_mock,
        )
        webserver = agent_server.setup_webserver()
        test_client = TestClient(webserver)

        no_time_response_object = {
            "id": "no_created_at",
        }
        self._set_server_client_post_responses(server_client_post_mock, no_time_response_object)
        with raises(RuntimeError, match="Received an invalid response from the model server: "):
            test_client.post(
                "/v1/responses",
                json={
                    "input": [],
                },
            )

        server_client_post_mock.assert_called_once_with(
            server_name="model_server",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
            ),
        )

        chat_response_object = {
            "id": "chat_response_id",
            "created_at": 1,
            "model": "response_model",
            "object": "response",
            "output": [
                {
                    "id": "output_message_1_id",
                    "content": [
                        {
                            "annotations": [],
                            "text": "What is the question?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        self._set_server_client_post_responses(server_client_post_mock, chat_response_object)
        chat_response = test_client.post(
            "/v1/responses",
            json={
                "input": [
                    {
                        "role": "user",
                        "content": "I'd like to ask a question.",
                    }
                ]
            },
        )
        assert chat_response.status_code == 200
        expected_chat_response_json = {
            "id": "chat_response_id",
            "created_at": 1,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "response_model",
            "object": "response",
            "output": [
                {
                    "id": "output_message_1_id",
                    "content": [
                        {
                            "annotations": [],
                            "text": "What is the question?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "conversation": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "prompt_cache_key": None,
            "reasoning": None,
            "safety_identifier": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
        }
        assert chat_response.json() == expected_chat_response_json
        server_client_post_mock.assert_called_once_with(
            server_name="model_server",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    NeMoGymEasyInputMessage(
                        role="user",
                        content="I'd like to ask a question.",
                    )
                ],
            ),
        )

    async def test_run(self, agent_config: ToolSimulationAgentConfig) -> None:
        server_client_post_mock = AsyncMock()
        server_client_mock = MagicMock(spec=ServerClient)
        server_client_mock.post = server_client_post_mock
        agent_server = ToolSimulationAgent(
            config=agent_config,
            server_client=server_client_mock,
        )
        webserver = agent_server.setup_webserver()
        test_client = TestClient(webserver)

        no_model_response_object = {
            "id": "no_model",
        }
        self._set_server_client_post_responses(server_client_post_mock, no_model_response_object)
        with raises(ValidationError, match="ToolSimulationAgentVerifyRequest"):
            test_client.post(
                "/run",
                json={
                    "responses_create_params": {
                        "input": [],
                    }
                },
            )

        server_client_post_mock.assert_called_once_with(
            server_name="tool_agent",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[],
            ),
        )

        tools = [
            {
                "name": "acknowledge",
                "parameters": None,
                "strict": None,
                "type": "function",
                "description": None,
            }
        ]
        tool_call_response_object = {
            "id": "tool_call_response_id",
            "created_at": 2,
            "model": "run_model",
            "object": "response",
            "output": [
                {
                    "arguments": "",
                    "call_id": "function_tool_call_1_id",
                    "name": "acknowledge",
                    "type": "function_call",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": tools,
        }
        invalid_verify_response_object = {
            "reward": 0.5,
        }
        self._set_server_client_post_responses(
            server_client_post_mock, tool_call_response_object, invalid_verify_response_object
        )
        with raises(ValidationError, match="ToolSimulationAgentVerifyResponse"):
            test_client.post(
                "/run",
                json={
                    "responses_create_params": {
                        "input": [
                            {
                                "role": "user",
                                "content": "Please provide an acknowledgment.",
                            }
                        ],
                        "tools": tools,
                    }
                },
            )

        full_tool_call_response = {
            "id": "tool_call_response_id",
            "created_at": 2,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "run_model",
            "object": "response",
            "output": [
                {
                    "arguments": "",
                    "call_id": "function_tool_call_1_id",
                    "name": "acknowledge",
                    "type": "function_call",
                    "id": None,
                    "status": None,
                }
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": tools,
            "top_p": None,
            "background": None,
            "conversation": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "prompt_cache_key": None,
            "reasoning": None,
            "safety_identifier": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
        }
        expected_invalid_verify_response_calls = [
            call(
                server_name="tool_agent",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        NeMoGymEasyInputMessage(
                            role="user",
                            content="Please provide an acknowledgment.",
                        )
                    ],
                    tools=tools,
                ),
            ),
            call(
                server_name="tool_resource_server",
                url_path="/verify",
                json={
                    "responses_create_params": {
                        "background": None,
                        "include": None,
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": "Please provide an acknowledgment.",
                            }
                        ],
                        "instructions": None,
                        "max_output_tokens": None,
                        "max_tool_calls": None,
                        "metadata": None,
                        "model": None,
                        "parallel_tool_calls": True,
                        "previous_response_id": None,
                        "prompt": None,
                        "reasoning": None,
                        "service_tier": None,
                        "store": None,
                        "temperature": None,
                        "text": None,
                        "tool_choice": "auto",
                        "tools": tools,
                        "top_logprobs": None,
                        "top_p": None,
                        "truncation": None,
                        "user": None,
                        "stream": None,
                    },
                    "response": full_tool_call_response,
                },
            ),
        ]
        assert server_client_post_mock.call_args_list == expected_invalid_verify_response_calls

        valid_verify_response_object = {
            "responses_create_params": {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Could you acknowledge this query?",
                    }
                ],
                "tools": tools,
            },
            "response": tool_call_response_object,
            "reward": 1,
        }
        self._set_server_client_post_responses(
            server_client_post_mock, tool_call_response_object, valid_verify_response_object
        )
        valid_verify_response = test_client.post(
            "/run",
            json={
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": "Please provide an acknowledgment.",
                        }
                    ],
                    "tools": tools,
                }
            },
        )
        assert valid_verify_response.status_code == 200
        expected_valid_verify_response_json = {
            "responses_create_params": {
                "background": None,
                "include": None,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Could you acknowledge this query?",
                    }
                ],
                "instructions": None,
                "max_output_tokens": None,
                "max_tool_calls": None,
                "metadata": None,
                "model": None,
                "parallel_tool_calls": True,
                "previous_response_id": None,
                "prompt": None,
                "reasoning": None,
                "service_tier": None,
                "store": None,
                "temperature": None,
                "text": None,
                "tool_choice": "auto",
                "tools": tools,
                "top_logprobs": None,
                "top_p": None,
                "truncation": None,
                "user": None,
                "stream": None,
            },
            "response": full_tool_call_response,
            "reward": 1,
        }
        assert valid_verify_response.json() == expected_valid_verify_response_json
        assert server_client_post_mock.call_args_list == expected_invalid_verify_response_calls
