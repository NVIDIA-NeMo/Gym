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

from anthropic.types import Message

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


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

    def test_system_role_message_passes_through(self) -> None:
        # Anthropic allows a "system" role inside messages (distinct from the top-level system
        # param); it is forwarded as a system input item rather than dropped or merged.
        params = _converter().anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [
                    {"role": "system", "content": "stay terse"},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        assert params.input[0].role == "system"
        assert params.input[0].content == "stay terse"
        assert params.input[1].role == "user"

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
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"}],
                    }
                ],
            }
        )
        out = params.input[0]
        assert out.type == "function_call_output"
        assert out.call_id == "toolu_1"
        assert out.output == "Sunny"

    def test_tool_result_block_list_content_is_flattened(self) -> None:
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
        assert params.input[0].output == "a\nb"

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

    def test_tools_and_tool_choice_variants(self) -> None:
        conv = _converter()
        params = conv.anthropic_request_to_responses(
            {
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
                "tool_choice": {"type": "any"},
            }
        )
        assert params.tools[0]["type"] == "function"
        assert params.tools[0]["name"] == "f"
        assert params.tools[0]["parameters"] == {"type": "object", "properties": {}}
        assert params.tool_choice == "required"

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
        import pytest

        with pytest.raises(NotImplementedError):
            _converter()._anthropic_image_to_input_part({"source": {"type": "url", "url": "http://x"}})

    def test_unsupported_image_media_type_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            _converter()._anthropic_image_to_input_part(
                {"source": {"type": "base64", "media_type": "image/tiff", "data": "x"}}
            )

    def test_unsupported_tool_result_block_raises(self) -> None:
        import pytest

        with pytest.raises(NotImplementedError):
            _converter()._anthropic_tool_result_content_to_text([{"type": "image", "source": {}}])


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
        assert out["usage"] == {"input_tokens": 5, "output_tokens": 7}

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
        assert out["usage"] == {"input_tokens": 0, "output_tokens": 0}

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
        assert delta_events[0]["delta"] == {"type": "thinking_delta", "thinking": "ponder"}

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
                }
            ],
            "tool_choice": {"type": "auto"},
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
            {"name": "lookup", "description": "Look up weather.", "input_schema": {"type": "object", "properties": {}}}
        ]
        assert rebuilt["tool_choice"] == {"type": "auto"}

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
