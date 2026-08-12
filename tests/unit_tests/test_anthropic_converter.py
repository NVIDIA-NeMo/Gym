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
"""Tests for the ingress (inverse) direction of the shared AnthropicConverter.

The egress direction (Responses -> Anthropic request, Anthropic response -> Responses) is
covered by responses_api_models/anthropic_model/tests/test_app.py. These tests cover the new
inverse direction used by an Anthropic Messages ingress proxy, plus round-trips that guard the
two directions against drift.
"""

import json
from typing import get_args, get_type_hints

import pytest
from anthropic.types import ContentBlockParam, Message, ToolUnionParam
from anthropic.types.message_create_params import MessageCreateParamsBase

from nemo_gym.anthropic_converter import (
    IGNORED_ANTHROPIC_REQUEST_FIELDS,
    MAPPED_ANTHROPIC_CONTENT_BLOCK_TYPES,
    MAPPED_ANTHROPIC_REQUEST_FIELDS,
    MAPPED_ANTHROPIC_STOP_REASONS,
    MAPPED_ANTHROPIC_TOOL_VARIANTS,
    MAPPED_RESPONSES_INPUT_ITEM_TYPES,
    MAPPED_RESPONSES_REQUEST_FIELDS,
    REJECTED_ANTHROPIC_CONTENT_BLOCK_TYPES,
    REJECTED_ANTHROPIC_REQUEST_FIELDS,
    REJECTED_ANTHROPIC_STOP_REASONS,
    REJECTED_ANTHROPIC_TOOL_VARIANTS,
    REJECTED_RESPONSES_INPUT_ITEM_TYPES,
    REJECTED_RESPONSES_REQUEST_FIELDS,
    AnthropicConverter,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseInputItem


PNG_DATA_URL = "data:image/png;base64,aGVsbG8="  # "hello"


def _converter() -> AnthropicConverter:
    return AnthropicConverter()


class TestAnthropicRequestToResponses:
    def test_system_string_and_user_text(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "model": "m",
                "system": "Be concise.",
                "max_tokens": 256,
                "temperature": 0.5,
                "top_p": 0.9,
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
        assert params.instructions == "Be concise."
        assert params.model == "m"
        assert params.max_output_tokens == 256
        assert params.temperature == 0.5
        assert params.top_p == 0.9
        assert len(params.input) == 1
        assert params.input[0].role == "user"
        assert params.input[0].content == "Hello"

    def test_system_block_list_is_joined(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "system": [
                    {"type": "text", "text": "Answer concisely."},
                    {"type": "text", "text": "Use JSON."},
                ],
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert params.instructions == "Answer concisely.\nUse JSON."

    def test_no_system_leaves_instructions_unset(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
        )
        assert params.instructions is None

    def test_system_list_without_text_leaves_instructions_unset(self) -> None:
        # A system list that contributes no usable text (empty-text blocks) yields no instructions.
        params = _converter().anthropic_request_to_responses(
            {
                "system": [{"type": "text", "text": ""}],
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert params.instructions is None

    def test_system_role_message_is_rejected(self) -> None:
        with pytest.raises(NotImplementedError, match="message role"):
            _converter().anthropic_request_to_responses(
                {
                    "max_tokens": 10,
                    "messages": [
                        {"role": "system", "content": "stay terse"},
                        {"role": "user", "content": "hi"},
                    ],
                }
            )

    def test_user_text_and_image_blocks(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
                            },
                        ],
                    }
                ],
            }
        )
        content = params.input[0].content
        assert content[0] == {"type": "input_text", "text": "What is this?"}
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"] == PNG_DATA_URL

    def test_assistant_tool_use_becomes_function_call(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "calling"},
                            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "Paris"}},
                        ],
                    }
                ],
            }
        )
        # text message, then function_call
        assert params.input[0].role == "assistant"
        assert params.input[0].content == "calling"
        fc = params.input[1]
        assert fc.type == "function_call"
        assert fc.call_id == "toolu_1"
        assert fc.name == "lookup"
        assert json.loads(fc.arguments) == {"city": "Paris"}

    def test_tool_result_becomes_function_call_output(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "Sunny",
                                "is_error": False,
                            }
                        ],
                    }
                ],
            }
        )
        out = params.input[0]
        assert out.type == "function_call_output"
        assert out.call_id == "toolu_1"
        assert out.output == "Sunny"

    def test_tool_result_block_list_content_stays_structured(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                            }
                        ],
                    }
                ],
            }
        )
        assert params.input[0].output == [
            {"type": "input_text", "text": "a"},
            {"type": "input_text", "text": "b"},
        ]

    def test_structured_tool_result_preserves_media_and_documents(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [
                                    {"type": "text", "text": "a"},
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
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        assert params.input[0].output == [
            {"type": "input_text", "text": "a"},
            {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "auto"},
            {"type": "input_file", "file_url": "https://example.com/file.pdf"},
            {"type": "input_file", "file_data": "data:application/pdf;base64,aGVsbG8="},
        ]

    def test_thinking_block_becomes_reasoning_item(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "hmm", "signature": "sig-1"}],
                    }
                ],
            }
        )
        item = params.input[0]
        assert item.type == "reasoning"
        assert item.summary[0].text == "hmm"
        assert item.encrypted_content == "sig-1"

    def test_redacted_thinking_becomes_encrypted_reasoning(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "encrypted"}],
                    }
                ],
            }
        )
        item = params.input[0]
        assert item.summary == []
        assert item.encrypted_content == "encrypted"

    def test_tools_and_tool_choice_variants(self) -> None:
        conv = _converter()
        params = conv.anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
                "tools": [
                    {
                        "name": "f",
                        "description": "d",
                        "input_schema": {"type": "object", "properties": {}},
                        "strict": True,
                    }
                ],
                "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
            }
        )
        assert params.tools[0]["type"] == "function"
        assert params.tools[0]["name"] == "f"
        assert params.tools[0]["parameters"] == {"type": "object", "properties": {}}
        assert params.tools[0]["strict"] is True
        assert params.tool_choice == "required"
        assert params.parallel_tool_calls is False

        assert conv._anthropic_tool_choice_to_responses({"type": "auto"}) == "auto"
        assert conv._anthropic_tool_choice_to_responses({"type": "none"}) == "none"
        assert conv._anthropic_tool_choice_to_responses({"type": "tool", "name": "f"}) == {
            "type": "function",
            "name": "f",
        }
        assert conv._anthropic_tool_choice_to_responses(None) is None

    def test_unsupported_block_raises(self) -> None:
        import pytest

        with pytest.raises(NotImplementedError):
            _converter().anthropic_request_to_responses(
                {
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": [{"type": "video", "data": "x"}]}],
                }
            )

    def test_unsupported_tool_choice_raises(self) -> None:
        import pytest

        with pytest.raises(NotImplementedError):
            _converter()._anthropic_tool_choice_to_responses({"type": "weird"})

    def test_unsupported_image_source_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _converter()._anthropic_image_to_input_part({"source": {"type": "file", "file_id": "file_1"}})

    def test_unsupported_image_media_type_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            _converter()._anthropic_image_to_input_part(
                {"source": {"type": "base64", "media_type": "image/tiff", "data": "x"}}
            )

    def test_unsupported_tool_result_block_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _converter()._anthropic_tool_result_content_to_responses([{"type": "search_result"}])

    @pytest.mark.parametrize(
        "field,value",
        [
            ("top_k", 3),
            ("stop_sequences", ["stop"]),
            ("container", "container_1"),
            ("inference_geo", "us"),
            ("thinking", {"type": "adaptive"}),
        ],
    )
    def test_unrepresentable_request_fields_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(NotImplementedError, match=field):
            _converter().anthropic_request_to_responses(
                {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}], field: value}
            )

    @pytest.mark.parametrize(
        "body,match",
        [
            (
                {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "x", "citations": []}]}],
                    "max_tokens": 10,
                },
                "citations",
            ),
            (
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {"type": "url", "url": "https://example.com/file.pdf"},
                                    "title": "report",
                                }
                            ],
                        }
                    ],
                    "max_tokens": 10,
                },
                "title",
            ),
            (
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": "bad",
                                    "is_error": True,
                                }
                            ],
                        }
                    ],
                    "max_tokens": 10,
                },
                "is_error",
            ),
            (
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "max_tokens": 10,
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                },
                "hosted tool",
            ),
        ],
    )
    def test_nested_unrepresentable_metadata_is_rejected(self, body: dict, match: str) -> None:
        with pytest.raises(NotImplementedError, match=match):
            _converter().anthropic_request_to_responses(body)

    def test_metadata_and_auto_service_tier_are_mapped(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"user_id": "user-1"},
                "service_tier": "auto",
            }
        )
        assert params.user == "user-1"
        assert params.service_tier == "auto"

        with pytest.raises(NotImplementedError, match="standard_only"):
            _converter().anthropic_request_to_responses(
                {
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                    "service_tier": "standard_only",
                }
            )

    def test_output_config_effort_is_mapped(self) -> None:
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"effort": "high"},
            }
        )
        assert params.reasoning["effort"] == "high"

        with pytest.raises(NotImplementedError, match="format"):
            _converter().anthropic_request_to_responses(
                {
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                    "output_config": {"format": {"type": "json_schema"}},
                }
            )

        with pytest.raises(NotImplementedError, match="max"):
            _converter().anthropic_request_to_responses(
                {
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                    "output_config": {"effort": "max"},
                }
            )

    def test_prompt_cache_hints_do_not_block_ingress(self) -> None:
        cache_control = {"type": "ephemeral"}
        params = _converter().anthropic_request_to_responses(
            {
                "cache_control": cache_control,
                "max_tokens": 10,
                "system": [{"type": "text", "text": "Be concise.", "cache_control": cache_control}],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hi", "cache_control": cache_control}],
                    }
                ],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Look up a value.",
                        "input_schema": {"type": "object", "properties": {}},
                        "cache_control": cache_control,
                    }
                ],
            }
        )
        assert params.instructions == "Be concise."
        assert params.input[0].content == "hi"
        assert params.tools[0]["name"] == "lookup"


class TestResponsesToAnthropicResponse:
    def _response_from_anthropic(self, anthropic_response: dict):
        conv = _converter()
        request_body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        return conv, conv.anthropic_to_responses(anthropic_response, request_body=request_body, model="m")

    def test_text_and_tool_use_and_stop_reason(self) -> None:
        conv, resp = self._response_from_anthropic(
            {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 5, "output_tokens": 7},
            }
        )
        out = conv.responses_to_anthropic_response(resp, model="m")
        assert out["role"] == "assistant"
        assert out["model"] == "m"
        assert out["content"][0] == {"type": "text", "text": "Hello"}
        tool_use = out["content"][1]
        assert tool_use["type"] == "tool_use"
        assert tool_use["id"] == "toolu_1"
        assert tool_use["input"] == {"a": 1}
        assert out["stop_reason"] == "tool_use"
        assert out["usage"] == {"cache_read_input_tokens": 0, "input_tokens": 5, "output_tokens": 7}

    def test_reasoning_becomes_thinking_block(self) -> None:
        conv, resp = self._response_from_anthropic(
            {
                "content": [
                    {"type": "thinking", "thinking": "step", "signature": "sig"},
                    {"type": "text", "text": "ok"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }
        )
        out = conv.responses_to_anthropic_response(resp, model="m")
        thinking = out["content"][0]
        assert thinking["type"] == "thinking"
        assert thinking["thinking"] == "step"
        assert thinking["signature"] == "sig"
        assert out["stop_reason"] == "end_turn"

    def test_max_tokens_stop_reason(self) -> None:
        conv, resp = self._response_from_anthropic(
            {
                "content": [{"type": "text", "text": "x"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        out = conv.responses_to_anthropic_response(resp, model="m")
        assert out["stop_reason"] == "max_tokens"

    def test_refusal_stop_reason(self) -> None:
        conv, resp = self._response_from_anthropic(
            {
                "content": [{"type": "text", "text": "x"}],
                "stop_reason": "refusal",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        out = conv.responses_to_anthropic_response(resp, model="m")
        assert out["stop_reason"] == "refusal"

    def test_missing_usage_defaults_to_zero(self) -> None:
        conv = _converter()
        request_body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        resp = conv.anthropic_to_responses(
            {"content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn"},
            request_body=request_body,
            model="m",
        )
        out = conv.responses_to_anthropic_response(resp, model="m")
        assert out["usage"] == {"cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    def test_reasoning_without_signature_defaults_to_empty(self) -> None:
        # Open-model reasoning carries no Anthropic signature, but the typed Message build
        # requires one — default it to "" rather than dropping the block or crashing.
        conv, resp = self._response_from_anthropic(
            {
                "content": [{"type": "thinking", "thinking": "step"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        out = conv.responses_to_anthropic_response(resp, model="m")
        assert out["content"][0] == {"type": "thinking", "thinking": "step", "signature": ""}

    def test_output_validates_as_anthropic_message(self) -> None:
        # Regression guard: the builder must emit an object the Anthropic SDK accepts as a Message
        # (this is what the internal Message.model_validate enforces on every response).
        conv, resp = self._response_from_anthropic(
            {
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }
        )
        out = conv.responses_to_anthropic_response(resp, model="m")
        message = Message.model_validate(out)  # raises if our output drifts from the SDK schema
        assert message.stop_reason == "tool_use"
        assert message.content[1].input == {"a": 1}

    def test_empty_output_yields_empty_content(self) -> None:
        # Defensive: a downstream response carrying no output items maps to empty content,
        # which is still a valid Anthropic Message. (Realistic empty responses arrive as an
        # empty message item and are rendered as a single empty text block instead — see
        # TestSharedHelperBranches.test_empty_anthropic_content_yields_empty_message.)
        conv, resp = self._response_from_anthropic(
            {
                "content": [{"type": "text", "text": "x"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
        )
        resp = resp.model_copy(update={"output": []})
        out = conv.responses_to_anthropic_response(resp, model="m")
        assert out["content"] == []
        assert out["stop_reason"] == "end_turn"
        Message.model_validate(out)  # empty content is still a valid Message


class TestAnthropicResponseToSSE:
    def _events(self, anthropic_response: dict):
        raw = list(_converter().anthropic_response_to_sse(anthropic_response))
        parsed = []
        for chunk in raw:
            lines = chunk.strip().split("\n")
            event_type = lines[0].removeprefix("event: ")
            data = json.loads(lines[1].removeprefix("data: "))
            parsed.append((event_type, data))
        return parsed

    def test_event_ordering_and_framing(self) -> None:
        events = self._events(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        )
        types = [t for t, _ in events]
        assert types == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        # message_start carries an empty content list
        assert events[0][1]["message"]["content"] == []
        # text delta
        assert events[2][1]["delta"] == {"type": "text_delta", "text": "hi"}
        # tool_use input arrives as input_json_delta
        assert events[5][1]["delta"]["type"] == "input_json_delta"
        assert json.loads(events[5][1]["delta"]["partial_json"]) == {"a": 1}
        # message_delta carries stop_reason + output usage
        assert events[7][1]["delta"]["stop_reason"] == "tool_use"
        assert events[7][1]["usage"] == {"output_tokens": 4}

    def test_thinking_block_delta(self) -> None:
        events = self._events(
            {
                "id": "msg_1",
                "role": "assistant",
                "model": "m",
                "content": [{"type": "thinking", "thinking": "ponder", "signature": "s"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        delta_events = [d for t, d in events if t == "content_block_delta"]
        assert [event["delta"] for event in delta_events] == [
            {"type": "thinking_delta", "thinking": "ponder"},
            {"type": "signature_delta", "signature": "s"},
        ]

    def test_unsupported_block_for_sse_raises(self) -> None:
        import pytest

        with pytest.raises(NotImplementedError):
            list(_converter().anthropic_response_to_sse({"content": [{"type": "image"}], "usage": {}}))


class TestRoundTrips:
    def test_request_round_trip_preserves_messages_system_tools(self) -> None:
        conv = _converter()
        original = {
            "model": "claude-sonnet-4-6",
            "system": "Be helpful.",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "Paris"}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"}]},
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look up weather.",
                    "input_schema": {"type": "object", "properties": {}},
                    "strict": False,
                }
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": False},
        }
        params = conv.anthropic_request_to_responses(original)
        rebuilt = conv.responses_to_anthropic(
            body=params,
            model="claude-sonnet-4-6",
            max_tokens=100,
            thinking=None,
            thinking_budget_tokens=None,
            extra_body={},
        )
        assert rebuilt["system"] == [{"type": "text", "text": "Be helpful."}]
        assert rebuilt["messages"] == original["messages"]
        assert rebuilt["tools"] == [
            {
                "name": "lookup",
                "description": "Look up weather.",
                "input_schema": {"type": "object", "properties": {}},
                "strict": False,
            }
        ]
        assert rebuilt["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": False}

    def test_structured_tool_result_round_trip_preserves_order(self) -> None:
        converter = _converter()
        original_content = [
            {"type": "text", "text": "first"},
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/image.png"},
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
        params = converter.anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": original_content,
                            }
                        ],
                    }
                ],
            }
        )
        rebuilt = converter.responses_to_anthropic(params, "claude-sonnet-4-6", 10, None, None, {})
        assert rebuilt["messages"][0]["content"][0]["content"] == original_content

    def test_response_round_trip_preserves_content(self) -> None:
        conv = _converter()
        request_body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        anthropic_response = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
        resp = conv.anthropic_to_responses(anthropic_response, request_body=request_body, model="m")
        rebuilt = conv.responses_to_anthropic_response(resp, model="m")
        assert rebuilt["content"] == anthropic_response["content"]
        assert rebuilt["stop_reason"] == "tool_use"

    def test_response_round_trip_preserves_adjacent_text_blocks(self) -> None:
        converter = _converter()
        anthropic_response = {
            "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        response = converter.anthropic_to_responses(
            anthropic_response,
            request_body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            model="m",
        )
        assert (
            converter.responses_to_anthropic_response(response, model="m")["content"] == anthropic_response["content"]
        )


class TestSharedHelperBranches:
    """Cover egress/shared helper branches now owned by this module."""

    def test_empty_anthropic_content_yields_empty_message(self) -> None:
        conv = _converter()
        request_body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        resp = conv.anthropic_to_responses(
            {"content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}, request_body, "m"
        )
        out = conv.responses_to_anthropic_response(resp, model="m")
        assert out["content"] == [{"type": "text", "text": ""}]
        assert out["stop_reason"] == "end_turn"

    def test_output_message_refusal_becomes_text_block(self) -> None:
        blocks = _converter()._output_message_to_anthropic_blocks({"content": [{"type": "refusal", "refusal": "no"}]})
        assert blocks == [{"type": "text", "text": "no"}]

    def test_output_message_unsupported_part_raises(self) -> None:
        import pytest

        with pytest.raises(NotImplementedError):
            _converter()._output_message_to_anthropic_blocks({"content": [{"type": "weird"}]})

    def test_egress_assistant_refusal_block(self) -> None:
        blocks = _converter()._content_to_anthropic_blocks([{"type": "refusal", "refusal": "no"}], "assistant")
        assert blocks == [{"type": "text", "text": "no"}]

    def test_egress_image_url_dict_form(self) -> None:
        block = _converter()._input_image_to_anthropic_block(
            {"type": "input_image", "image_url": {"url": PNG_DATA_URL}}
        )
        assert block["source"]["media_type"] == "image/png"
        assert block["source"]["data"] == "aGVsbG8="

    def test_egress_image_url_non_string_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            _converter()._input_image_to_anthropic_block({"type": "input_image", "image_url": 123})

    def test_parse_image_data_url_jpg_normalized_and_validations(self) -> None:
        import pytest

        conv = _converter()
        media_type, data = conv._parse_image_data_url("data:image/jpg;base64,aGVsbG8=")
        assert media_type == "image/jpeg" and data == "aGVsbG8="

        with pytest.raises(ValueError):  # no base64 data
            conv._parse_image_data_url("data:image/png;base64,")
        with pytest.raises(ValueError):  # not declared base64
            conv._parse_image_data_url("data:image/png,aGVsbG8=")
        with pytest.raises(ValueError):  # unsupported media type
            conv._parse_image_data_url("data:image/tiff;base64,aGVsbG8=")
        with pytest.raises(ValueError):  # invalid base64 payload
            conv._parse_image_data_url("data:image/png;base64,!!!notb64!!!")

    def test_content_to_text_list_and_unsupported(self) -> None:
        import pytest

        conv = _converter()
        assert conv._content_to_text([{"type": "input_text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
        with pytest.raises(NotImplementedError):
            conv._content_to_text([{"type": "input_image", "image_url": "x"}])

    def test_json_object_from_arguments_rejects_non_object(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            _converter()._json_object_from_arguments("[1, 2]")

    def test_copy_tool_choice_required_maps_to_any(self) -> None:
        conv = _converter()
        anthropic_body: dict = {}
        conv._copy_tool_choice({"tool_choice": "required"}, anthropic_body)
        assert anthropic_body["tool_choice"] == {"type": "any"}


class TestSchemaClassification:
    def test_anthropic_request_fields_match_pinned_sdk(self) -> None:
        classified = (
            MAPPED_ANTHROPIC_REQUEST_FIELDS | IGNORED_ANTHROPIC_REQUEST_FIELDS | REJECTED_ANTHROPIC_REQUEST_FIELDS
        )
        assert classified == set(get_type_hints(MessageCreateParamsBase))
        assert not MAPPED_ANTHROPIC_REQUEST_FIELDS & IGNORED_ANTHROPIC_REQUEST_FIELDS
        assert not MAPPED_ANTHROPIC_REQUEST_FIELDS & REJECTED_ANTHROPIC_REQUEST_FIELDS
        assert not IGNORED_ANTHROPIC_REQUEST_FIELDS & REJECTED_ANTHROPIC_REQUEST_FIELDS

    def test_anthropic_content_variants_match_pinned_sdk(self) -> None:
        tags = {get_args(get_type_hints(content_type)["type"])[0] for content_type in get_args(ContentBlockParam)}
        assert MAPPED_ANTHROPIC_CONTENT_BLOCK_TYPES | REJECTED_ANTHROPIC_CONTENT_BLOCK_TYPES == tags
        assert not MAPPED_ANTHROPIC_CONTENT_BLOCK_TYPES & REJECTED_ANTHROPIC_CONTENT_BLOCK_TYPES

    def test_anthropic_tool_variants_match_pinned_sdk(self) -> None:
        variants = {tool_type.__name__ for tool_type in get_args(ToolUnionParam)}
        assert MAPPED_ANTHROPIC_TOOL_VARIANTS | REJECTED_ANTHROPIC_TOOL_VARIANTS == variants
        assert not MAPPED_ANTHROPIC_TOOL_VARIANTS & REJECTED_ANTHROPIC_TOOL_VARIANTS

    def test_anthropic_stop_reasons_match_pinned_sdk(self) -> None:
        stop_reason_annotation = Message.model_fields["stop_reason"].annotation
        stop_reason_literal = next(arg for arg in get_args(stop_reason_annotation) if get_args(arg))
        assert MAPPED_ANTHROPIC_STOP_REASONS | REJECTED_ANTHROPIC_STOP_REASONS == set(get_args(stop_reason_literal))
        assert not MAPPED_ANTHROPIC_STOP_REASONS & REJECTED_ANTHROPIC_STOP_REASONS

    def test_responses_request_fields_match_pinned_models(self) -> None:
        assert MAPPED_RESPONSES_REQUEST_FIELDS | REJECTED_RESPONSES_REQUEST_FIELDS == set(
            NeMoGymResponseCreateParamsNonStreaming.model_fields
        )
        assert not MAPPED_RESPONSES_REQUEST_FIELDS & REJECTED_RESPONSES_REQUEST_FIELDS

    def test_responses_input_variants_match_pinned_models(self) -> None:
        union = get_args(NeMoGymResponseInputItem)[0]
        tags = {get_args(item_type.model_fields["type"].annotation)[0] for item_type in get_args(union)}
        assert MAPPED_RESPONSES_INPUT_ITEM_TYPES | REJECTED_RESPONSES_INPUT_ITEM_TYPES == tags
        assert not MAPPED_RESPONSES_INPUT_ITEM_TYPES & REJECTED_RESPONSES_INPUT_ITEM_TYPES
