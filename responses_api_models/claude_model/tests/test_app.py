# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import json
from contextlib import nullcontext
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_models.claude_model.app import ClaudeConverter, ClaudeModel, ClaudeModelConfig


class TestClaudeConverter:
    def test_responses_to_anthropic_maps_messages_tools_and_thinking(self) -> None:
        converter = ClaudeConverter()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "message",
                    "role": "developer",
                    "content": "Be concise.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is the weather?"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [{"type": "summary_text", "text": "Need weather data."}],
                    "encrypted_content": "signature_123",
                },
                {
                    "type": "function_call",
                    "call_id": "toolu_123",
                    "name": "get_weather",
                    "arguments": '{"city": "San Francisco"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "toolu_123",
                    "output": '{"temperature": 65}',
                },
            ],
            instructions="You are helpful.",
            max_output_tokens=512,
            temperature=0.2,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "strict": True,
                }
            ],
            tool_choice={"type": "function", "name": "get_weather"},
        )

        actual = converter.responses_to_anthropic(
            body=body,
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            thinking_budget_tokens=1024,
            extra_body={"metadata": {"request_id": "abc"}},
        )

        assert actual == {
            "metadata": {"request_id": "abc"},
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the weather?"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Need weather data.",
                            "signature": "signature_123",
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "get_weather",
                            "input": {"city": "San Francisco"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": '{"temperature": 65}',
                        }
                    ],
                },
            ],
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            "temperature": 0.2,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }

    def test_anthropic_to_responses_maps_text_thinking_tools_and_usage(self) -> None:
        converter = ClaudeConverter()
        request_body = NeMoGymResponseCreateParamsNonStreaming(input="hello")

        response = converter.anthropic_to_responses(
            anthropic_response={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I should call a tool.",
                        "signature": "signature_123",
                    },
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "San Francisco"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 3},
            },
            request_body=request_body,
            model="claude-sonnet-4-20250514",
        )

        assert response.model == "claude-sonnet-4-20250514"
        assert response.output[0].type == "reasoning"
        assert response.output[0].summary[0].text == "I should call a tool."
        assert response.output[0].encrypted_content == "signature_123"
        assert response.output[1].type == "message"
        assert response.output[1].content[0].text == "Let me check."
        assert response.output[2].type == "function_call"
        assert response.output[2].call_id == "toolu_123"
        assert response.output[2].name == "get_weather"
        assert json.loads(response.output[2].arguments) == {"city": "San Francisco"}
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30
        assert response.usage.input_tokens_details.cached_tokens == 3


class TestClaudeModel:
    def _setup_server(
        self,
        max_concurrent_requests=None,
        thinking_budget_tokens=None,
        anthropic_model="claude-sonnet-4-20250514",
        max_tokens=4096,
        extra_body=None,
        anthropic_base_url="https://api.anthropic.com/v1",
    ) -> ClaudeModel:
        config = ClaudeModelConfig(
            host="0.0.0.0",
            port=8081,
            anthropic_base_url=anthropic_base_url,
            anthropic_api_key="dummy_key",  # pragma: allowlist secret
            anthropic_model=anthropic_model,
            max_tokens=max_tokens,
            entrypoint="",
            name="",
            max_concurrent_requests=max_concurrent_requests,
            thinking_budget_tokens=thinking_budget_tokens,
            extra_body=extra_body or {},
        )
        return ClaudeModel(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_sanity(self) -> None:
        self._setup_server()

    def test_messages_url_accepts_host_or_v1_base_url(self) -> None:
        assert self._setup_server(anthropic_base_url="https://api.anthropic.com")._messages_url() == (
            "https://api.anthropic.com/v1/messages"
        )
        assert self._setup_server(anthropic_base_url="https://api.anthropic.com/v1")._messages_url() == (
            "https://api.anthropic.com/v1/messages"
        )

    def test_responses_endpoint_round_trip(self) -> None:
        server = self._setup_server(thinking_budget_tokens=1024)
        app = server.setup_webserver()
        client = TestClient(app)

        called_body = {}

        async def mock_messages_create(body, cookies):
            nonlocal called_body
            called_body = body
            return {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "Hello from Claude."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 5},
            }

        server._messages_create = mock_messages_create

        response = client.post(
            "/v1/responses",
            json={
                "input": "hello",
                "tools": [
                    {
                        "type": "function",
                        "name": "finish",
                        "description": "Finish task.",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    }
                ],
            },
        )

        assert response.status_code == 200
        assert called_body["model"] == "claude-sonnet-4-20250514"
        assert called_body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        assert called_body["tools"] == [
            {
                "name": "finish",
                "description": "Finish task.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        assert called_body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert response.json()["output"][0]["content"][0]["text"] == "Hello from Claude."

    def test_responses_endpoint_sends_curl_shaped_anthropic_request_fields(self) -> None:
        server = self._setup_server(
            anthropic_model="claude-opus-4-6",
            max_tokens=1024,
            extra_body={
                "thinking": {"type": "adaptive"},
            },
        )
        app = server.setup_webserver()
        client = TestClient(app)

        called_body = {}

        async def mock_messages_create(body, cookies):
            nonlocal called_body
            called_body = body
            return {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [
                    {"type": "thinking", "thinking": "Consider the greeting.", "signature": "signature_123"},
                    {"type": "text", "text": "Hello!"},
                    {"type": "tool_use", "id": "toolu_123", "name": "name", "input": {"location": "NYC"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }

        server._messages_create = mock_messages_create

        response = client.post(
            "/v1/responses",
            json={
                "input": [{"content": "Hello, world", "role": "user", "type": "message"}],
                "instructions": "Today's date is 2024-06-01.",
                "temperature": 1,
                "tools": [
                    {
                        "type": "function",
                        "name": "name",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                        "strict": True,
                    }
                ],
                "top_p": 0.95,
            },
        )

        assert response.status_code == 200
        assert called_body == {
            "thinking": {"type": "adaptive"},
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello, world"}]}],
            "system": [{"type": "text", "text": "Today's date is 2024-06-01."}],
            "temperature": 1.0,
            "top_p": 0.95,
            "tools": [
                {
                    "name": "name",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }
            ],
        }
        response_body = response.json()
        assert response_body["output"][0]["type"] == "reasoning"
        assert response_body["output"][0]["summary"][0]["text"] == "Consider the greeting."
        assert response_body["output"][1]["content"][0]["text"] == "Hello!"
        assert response_body["output"][2]["type"] == "function_call"
        assert response_body["output"][2]["name"] == "name"
        assert json.loads(response_body["output"][2]["arguments"]) == {"location": "NYC"}
        assert response_body["usage"]["input_tokens"] == 11
        assert response_body["usage"]["output_tokens"] == 7

    def test_semaphore_disabled_by_default(self) -> None:
        server = self._setup_server()
        assert isinstance(server._semaphore, type(nullcontext()))

    async def test_semaphore_caps_concurrency(self) -> None:
        server = self._setup_server(max_concurrent_requests=2)
        assert isinstance(server._semaphore, asyncio.Semaphore)

        in_flight = 0
        peak = 0

        async def worker() -> None:
            nonlocal in_flight, peak
            async with server._semaphore:
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(8)))
        assert peak == 2
