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
"""Egress-direction tests for the shared AnthropicConverter (Responses -> Anthropic request,
Anthropic response -> Responses). Mirrors the converter coverage that the egress anthropic_model
server's test suite provides on the #1546 branch, kept here so the shared converter module is
fully covered on this (ingress) branch where the egress server is absent."""

import json

import pytest

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


class TestAnthropicConverter:
    def test_responses_to_anthropic_maps_messages_tools_and_thinking(self) -> None:
        converter = AnthropicConverter()
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
            thinking=None,
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
        converter = AnthropicConverter()
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

    def test_anthropic_to_responses_maps_stop_reasons_to_incomplete_details(self) -> None:
        converter = AnthropicConverter()
        request_body = NeMoGymResponseCreateParamsNonStreaming(input="hello")

        base_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Partial response."}],
        }

        max_tokens_response = converter.anthropic_to_responses(
            anthropic_response=base_response | {"stop_reason": "max_tokens"},
            request_body=request_body,
            model="claude-sonnet-4-20250514",
        )
        assert max_tokens_response.incomplete_details.reason == "max_output_tokens"

        context_response = converter.anthropic_to_responses(
            anthropic_response=base_response | {"stop_reason": "model_context_window_exceeded"},
            request_body=request_body,
            model="claude-sonnet-4-20250514",
        )
        assert context_response.incomplete_details.reason == "max_output_tokens"

        refusal_response = converter.anthropic_to_responses(
            anthropic_response=base_response | {"stop_reason": "refusal"},
            request_body=request_body,
            model="claude-sonnet-4-20250514",
        )
        assert refusal_response.incomplete_details.reason == "content_filter"

        tool_use_response = converter.anthropic_to_responses(
            anthropic_response=base_response | {"stop_reason": "tool_use"},
            request_body=request_body,
            model="claude-sonnet-4-20250514",
        )
        assert tool_use_response.incomplete_details is None

    def test_responses_to_anthropic_maps_typed_adaptive_thinking(self) -> None:
        converter = AnthropicConverter()
        body = NeMoGymResponseCreateParamsNonStreaming(input="Hello")

        actual = converter.responses_to_anthropic(
            body=body,
            model="claude-opus-4-8",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            thinking_budget_tokens=None,
            extra_body={},
        )

        assert actual["thinking"] == {"type": "adaptive"}

    def test_responses_to_anthropic_maps_input_image_data_url(self) -> None:
        converter = AnthropicConverter()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is in this image?"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "high",
                        },
                    ],
                }
            ]
        )

        actual = converter.responses_to_anthropic(
            body=body,
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            thinking=None,
            thinking_budget_tokens=None,
            extra_body={},
        )

        assert actual["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                    },
                ],
            }
        ]

    def test_responses_to_anthropic_rejects_remote_image_url(self) -> None:
        converter = AnthropicConverter()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.png",
                            "detail": "high",
                        }
                    ],
                }
            ]
        )

        with pytest.raises(ValueError, match="base64 data URLs"):
            converter.responses_to_anthropic(
                body=body,
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                thinking=None,
                thinking_budget_tokens=None,
                extra_body={},
            )

    def test_responses_to_anthropic_rejects_invalid_image_data_url(self) -> None:
        converter = AnthropicConverter()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,not valid base64",
                            "detail": "high",
                        }
                    ],
                }
            ]
        )

        with pytest.raises(ValueError, match="invalid base64"):
            converter.responses_to_anthropic(
                body=body,
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                thinking=None,
                thinking_budget_tokens=None,
                extra_body={},
            )

    def test_responses_to_anthropic_rejects_ambiguous_thinking_config(self) -> None:
        converter = AnthropicConverter()
        body = NeMoGymResponseCreateParamsNonStreaming(input="Hello")

        with pytest.raises(ValueError, match="Configure Anthropic thinking in only one place"):
            converter.responses_to_anthropic(
                body=body,
                model="claude-opus-4-8",
                max_tokens=1024,
                thinking={"type": "adaptive"},
                thinking_budget_tokens=1024,
                extra_body={},
            )

    def test_responses_to_anthropic_rejects_opus_4_8_sampling_params(self) -> None:
        converter = AnthropicConverter()

        with pytest.raises(ValueError, match="does not support configurable sampling"):
            converter.responses_to_anthropic(
                body=NeMoGymResponseCreateParamsNonStreaming(input="Hello", temperature=0.2),
                model="claude-opus-4-8",
                max_tokens=1024,
                thinking={"type": "adaptive"},
                thinking_budget_tokens=None,
                extra_body={},
            )

        with pytest.raises(ValueError, match="does not support configurable sampling"):
            converter.responses_to_anthropic(
                body=NeMoGymResponseCreateParamsNonStreaming(input="Hello"),
                model="us/aws/anthropic/eccn-claude-opus-4-8",
                max_tokens=1024,
                thinking={"type": "adaptive"},
                thinking_budget_tokens=None,
                extra_body={"top_k": 5},
            )
