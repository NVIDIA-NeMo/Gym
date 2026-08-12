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
            extra_body={"metadata": {"user_id": "abc"}},
        )

        assert actual == {
            "metadata": {"user_id": "abc"},
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
                    "strict": True,
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
        assert response.usage.input_tokens == 13
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 33
        assert response.usage.input_tokens_details.cached_tokens == 3
        rebuilt = converter.responses_to_anthropic_response(response, "claude-sonnet-4-20250514")
        assert rebuilt["usage"] == {
            "cache_read_input_tokens": 3,
            "input_tokens": 10,
            "output_tokens": 20,
        }

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

        for stop_reason in ("pause_turn", "stop_sequence", "model_context_window_exceeded"):
            with pytest.raises(NotImplementedError, match="stop_reason"):
                converter.anthropic_to_responses(
                    anthropic_response=base_response | {"stop_reason": stop_reason},
                    request_body=request_body,
                    model="claude-sonnet-4-20250514",
                )

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
                            "detail": "auto",
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

    def test_responses_to_anthropic_maps_remote_image_url(self) -> None:
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
                            "detail": "auto",
                        }
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
        assert actual["messages"][0]["content"] == [
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/image.png"},
            }
        ]

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
                            "detail": "auto",
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

    def test_responses_to_anthropic_preserves_structured_tool_result(self) -> None:
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "toolu_1",
                    "output": [
                        {"type": "input_text", "text": "first"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.png",
                            "detail": "auto",
                        },
                        {"type": "input_file", "file_url": "https://example.com/file.pdf"},
                        {
                            "type": "input_file",
                            "file_data": "data:application/pdf;base64,aGVsbG8=",
                        },
                    ],
                }
            ]
        )
        actual = AnthropicConverter().responses_to_anthropic(body, "claude-sonnet-4-6", 100, None, None, {})
        assert actual["messages"][0]["content"][0]["content"] == [
            {"type": "text", "text": "first"},
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/image.png"},
            },
            {
                "type": "document",
                "source": {"type": "url", "url": "https://example.com/file.pdf"},
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "aGVsbG8=",
                },
            },
        ]

    def test_responses_to_anthropic_rejects_incomplete_tool_result(self) -> None:
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "toolu_1",
                    "output": "partial",
                    "status": "in_progress",
                }
            ]
        )

        with pytest.raises(NotImplementedError, match="function_call_output status"):
            AnthropicConverter().responses_to_anthropic(body, "claude-sonnet-4-6", 100, None, None, {})

    def test_redacted_thinking_round_trip(self) -> None:
        converter = AnthropicConverter()
        response = converter.anthropic_to_responses(
            {
                "content": [{"type": "redacted_thinking", "data": "encrypted"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
            NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            "claude-sonnet-4-6",
        )
        actual = converter.responses_to_anthropic_response(response, "claude-sonnet-4-6")
        assert actual["content"] == [{"type": "redacted_thinking", "data": "encrypted"}]

    def test_request_user_service_tier_and_parallel_settings_round_trip(self) -> None:
        converter = AnthropicConverter()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input="hi",
            user="user-1",
            service_tier="auto",
            tools=[
                {
                    "type": "function",
                    "name": "f",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            tool_choice="required",
            parallel_tool_calls=False,
        )
        actual = converter.responses_to_anthropic(body, "claude-sonnet-4-6", 100, None, None, {})
        assert actual["metadata"] == {"user_id": "user-1"}
        assert actual["service_tier"] == "auto"
        assert actual["tools"][0]["strict"] is True
        assert actual["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}

    def test_hosted_tools_and_media_metadata_are_rejected(self) -> None:
        converter = AnthropicConverter()
        with pytest.raises(NotImplementedError, match="tool type"):
            converter._copy_tools({"tools": [{"type": "web_search_preview"}]}, {})
        with pytest.raises(NotImplementedError, match="detail=high"):
            converter._input_image_to_anthropic_block(
                {
                    "type": "input_image",
                    "image_url": "https://example.com/image.png",
                    "detail": "high",
                }
            )
        with pytest.raises(NotImplementedError, match="filename"):
            converter._input_file_to_anthropic_block(
                {
                    "type": "input_file",
                    "file_url": "https://example.com/file.pdf",
                    "filename": "file.pdf",
                }
            )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("background", False),
            ("include", []),
            ("max_tool_calls", 1),
            ("metadata", {"key": "value"}),
            ("previous_response_id", "resp_1"),
            ("prompt", {"id": "pmpt_1"}),
            ("reasoning", {"effort": "high"}),
            ("store", False),
            ("text", {"verbosity": "low"}),
            ("top_logprobs", 1),
            ("truncation", "auto"),
        ],
    )
    def test_unrepresentable_responses_request_fields_are_rejected(self, field: str, value: object) -> None:
        body = NeMoGymResponseCreateParamsNonStreaming(input="hi", **{field: value})
        with pytest.raises(NotImplementedError, match=field):
            AnthropicConverter().responses_to_anthropic(body, "claude-sonnet-4-6", 100, None, None, {})

    def test_output_annotations_logprobs_and_usage_details_are_rejected(self) -> None:
        converter = AnthropicConverter()
        request = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        response = converter.anthropic_to_responses(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
            request,
            "claude-sonnet-4-6",
        )
        output = response.output[0].model_dump()
        output["content"][0]["annotations"] = [{"type": "url_citation", "url": "https://example.com"}]
        annotated = response.model_copy(update={"output": [output]})
        with pytest.raises(NotImplementedError, match="annotations"):
            converter.responses_to_anthropic_response(annotated, "claude-sonnet-4-6")

        output["content"][0]["annotations"] = []
        output["content"][0]["logprobs"] = []
        with pytest.raises(NotImplementedError, match="logprobs"):
            converter.responses_to_anthropic_response(
                response.model_copy(update={"output": [output]}), "claude-sonnet-4-6"
            )

        usage = response.usage.model_copy(
            update={
                "output_tokens_details": response.usage.output_tokens_details.model_copy(
                    update={"reasoning_tokens": 1}
                )
            }
        )
        with pytest.raises(NotImplementedError, match="reasoning_tokens"):
            converter.responses_to_anthropic_response(
                response.model_copy(update={"usage": usage}), "claude-sonnet-4-6"
            )

    @pytest.mark.parametrize(
        "usage_field,value",
        [
            ("cache_creation_input_tokens", 1),
            ("inference_geo", "us"),
            ("service_tier", "priority"),
            ("server_tool_use", {"web_search_requests": 1}),
        ],
    )
    def test_unrepresentable_anthropic_usage_is_rejected(self, usage_field: str, value: object) -> None:
        with pytest.raises(NotImplementedError, match=usage_field):
            AnthropicConverter().anthropic_to_responses(
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 2, usage_field: value},
                },
                NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                "claude-sonnet-4-6",
            )
