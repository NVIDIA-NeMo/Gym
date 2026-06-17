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
import pytest

from nemo_gym.anthropic_utils import NeMoGymAnthropicMessageCreateParamsNonStreaming
from nemo_gym.claude_converter import ClaudeConverter
from nemo_gym.openai_utils import NeMoGymChatCompletion


def _req(**kwargs) -> NeMoGymAnthropicMessageCreateParamsNonStreaming:
    kwargs.setdefault("max_tokens", 1024)
    kwargs.setdefault("messages", [])
    return NeMoGymAnthropicMessageCreateParamsNonStreaming.model_validate(kwargs)


def _roles(cc) -> list[str]:
    return [m["role"] for m in cc.messages]


class TestMessagesToChatCompletion:
    def setup_method(self) -> None:
        self.c = ClaudeConverter()

    def test_simple_text_message(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(_req(messages=[{"role": "user", "content": "hello"}]))
        assert cc.messages == [{"role": "user", "content": "hello"}]
        assert cc.max_tokens == 1024

    def test_system_string(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(system="be brief", messages=[{"role": "user", "content": "hi"}])
        )
        assert cc.messages[0] == {"role": "system", "content": "be brief"}

    def test_system_block_list_joined(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(
                system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        assert cc.messages[0] == {"role": "system", "content": "a b"}

    def test_system_block_list_without_text_is_dropped(self) -> None:
        # A system list carrying no text blocks yields no system message.
        cc = self.c.messages_to_chat_completion_create_params(
            _req(system=[], messages=[{"role": "user", "content": "hi"}])
        )
        assert "system" not in _roles(cc)

    def test_system_role_message(self) -> None:
        # Anthropic also allows "system" as a message role (not just the top-level param).
        cc = self.c.messages_to_chat_completion_create_params(
            _req(messages=[{"role": "system", "content": "stay terse"}])
        )
        assert cc.messages == [{"role": "system", "content": "stay terse"}]

    def test_assistant_text_via_block_list(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(messages=[{"role": "assistant", "content": [{"type": "text", "text": "ok"}]}])
        )
        assert cc.messages == [{"role": "assistant", "content": "ok"}]

    def test_assistant_tool_use_with_text(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "checking"},
                            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
                        ],
                    }
                ]
            )
        )
        msg = cc.messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "checking"
        assert msg["tool_calls"][0]["id"] == "toolu_1"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}

    def test_tool_use_only_has_null_content(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(
                messages=[
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}]}
                ]
            )
        )
        assert cc.messages[0]["content"] is None

    def test_tool_result_block_list_flattened(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [{"type": "text", "text": "72F"}, {"type": "text", "text": "sunny"}],
                            }
                        ],
                    }
                ]
            )
        )
        assert cc.messages == [{"role": "tool", "tool_call_id": "toolu_1", "content": "72F sunny"}]

    def test_tool_result_string_content(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(
                messages=[
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "done"}]}
                ]
            )
        )
        assert cc.messages[0] == {"role": "tool", "tool_call_id": "t", "content": "done"}

    def test_tool_result_non_text_block_json_encoded(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t",
                                "content": [
                                    {"type": "image", "source": {"type": "url", "url": "http://x/y.png"}},
                                ],
                            }
                        ],
                    }
                ]
            )
        )
        # Non-text blocks are JSON-encoded so their data survives the flatten.
        assert '"type": "image"' in cc.messages[0]["content"]

    def test_thinking_blocks_dropped(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "secret", "signature": "s"},
                            {"type": "redacted_thinking", "data": "xxx"},
                            {"type": "text", "text": "answer"},
                        ],
                    }
                ]
            )
        )
        assert cc.messages == [{"role": "assistant", "content": "answer"}]

    def test_empty_content_list_falls_back_to_empty_message(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(_req(messages=[{"role": "user", "content": []}]))
        assert cc.messages == [{"role": "user", "content": ""}]

    def test_unsupported_block_type_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            self.c.messages_to_chat_completion_create_params(
                _req(
                    messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "u"}}]}]
                )
            )

    def test_tools_converted(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(
                messages=[{"role": "user", "content": "hi"}],
                tools=[
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {"loc": {"type": "string"}}},
                    }
                ],
            )
        )
        assert cc.tools[0]["type"] == "function"
        assert cc.tools[0]["function"]["name"] == "get_weather"
        assert cc.tools[0]["function"]["description"] == "Get weather"
        assert cc.tools[0]["function"]["parameters"]["properties"]["loc"] == {"type": "string"}

    def test_tool_without_input_schema_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            self.c.messages_to_chat_completion_create_params(
                _req(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                )
            )

    @pytest.mark.parametrize(
        "anthropic_choice,openai_choice",
        [("auto", "auto"), ("any", "required"), ("none", "none")],
    )
    def test_tool_choice_string(self, anthropic_choice: str, openai_choice: str) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(messages=[{"role": "user", "content": "hi"}], tool_choice={"type": anthropic_choice})
        )
        assert cc.tool_choice == openai_choice

    def test_tool_choice_specific_tool(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(messages=[{"role": "user", "content": "hi"}], tool_choice={"type": "tool", "name": "get_weather"})
        )
        assert cc.tool_choice == {"type": "function", "function": {"name": "get_weather"}}

    def test_tool_choice_unknown_dict_defaults_auto(self) -> None:
        assert self.c._convert_tool_choice({"type": "mystery"}) == "auto"

    def test_tool_choice_bare_string(self) -> None:
        # Anthropic also accepts a bare string tool_choice.
        assert self.c._convert_tool_choice("any") == "required"

    def test_scalar_params_mapped(self) -> None:
        cc = self.c.messages_to_chat_completion_create_params(
            _req(
                model="claude-opus-4-8",
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                stop_sequences=["STOP"],
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        assert cc.model == "claude-opus-4-8"
        assert cc.temperature == 0.5
        assert cc.top_p == 0.9
        assert cc.stop == ["STOP"]
        # top_k has no Chat Completions equivalent and is dropped.
        assert not hasattr(cc, "top_k") or cc.model_dump().get("top_k") is None

    def test_simple_message_unsupported_role_raises(self) -> None:
        # Defensive branch: Anthropic message roles are constrained to
        # user/assistant/system by validation, so exercise it directly.
        with pytest.raises(NotImplementedError):
            self.c._simple_message("developer", "x")

    def test_flatten_tool_result_scalar_fallback(self) -> None:
        assert self.c._flatten_tool_result_content(123) == "123"


def _chat_completion(message: dict, finish_reason: str = "stop", usage: dict | None = None) -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "backend-model",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            "usage": usage,
        }
    )


class TestChatCompletionToMessage:
    def setup_method(self) -> None:
        self.c = ClaudeConverter()

    def test_text_response(self) -> None:
        msg = self.c.chat_completion_to_message(
            _chat_completion(
                {"role": "assistant", "content": "hello"},
                usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            )
        )
        assert msg.type == "message"
        assert msg.role == "assistant"
        assert msg.model == "backend-model"
        assert msg.stop_reason == "end_turn"
        assert msg.stop_sequence is None
        assert [b.type for b in msg.content] == ["text"]
        assert msg.content[0].text == "hello"
        assert (msg.usage.input_tokens, msg.usage.output_tokens) == (3, 4)
        assert msg.id.startswith("msg_")

    def test_model_override(self) -> None:
        msg = self.c.chat_completion_to_message(
            _chat_completion({"role": "assistant", "content": "x"}), model="claude-opus-4-8"
        )
        assert msg.model == "claude-opus-4-8"

    def test_tool_calls_response(self) -> None:
        msg = self.c.chat_completion_to_message(
            _chat_completion(
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
                finish_reason="tool_calls",
            )
        )
        assert [b.type for b in msg.content] == ["text", "tool_use"]
        tool_use = msg.content[1]
        assert tool_use.id == "call_1"
        assert tool_use.name == "get_weather"
        assert tool_use.input == {"city": "Paris"}
        assert msg.stop_reason == "tool_use"

    def test_tool_call_with_invalid_json_arguments(self) -> None:
        msg = self.c.chat_completion_to_message(
            _chat_completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c", "type": "function", "function": {"name": "f", "arguments": "not json"}}
                    ],
                },
                finish_reason="tool_calls",
            )
        )
        assert msg.content[0].type == "tool_use"
        assert msg.content[0].input == {"raw": "not json"}

    def test_empty_message_gets_empty_text_block(self) -> None:
        msg = self.c.chat_completion_to_message(_chat_completion({"role": "assistant", "content": None}))
        assert [b.type for b in msg.content] == ["text"]
        assert msg.content[0].text == ""

    def test_usage_absent_defaults_to_zero(self) -> None:
        msg = self.c.chat_completion_to_message(_chat_completion({"role": "assistant", "content": "x"}, usage=None))
        assert (msg.usage.input_tokens, msg.usage.output_tokens) == (0, 0)

    @pytest.mark.parametrize(
        "finish_reason,stop_reason",
        [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "end_turn"),
            ("function_call", "tool_use"),
        ],
    )
    def test_finish_reason_mapping(self, finish_reason: str, stop_reason: str) -> None:
        msg = self.c.chat_completion_to_message(
            _chat_completion({"role": "assistant", "content": "x"}, finish_reason=finish_reason)
        )
        assert msg.stop_reason == stop_reason


class TestRoundTrip:
    def test_request_then_response(self) -> None:
        c = ClaudeConverter()
        # Anthropic request -> Chat Completions request.
        cc_req = c.messages_to_chat_completion_create_params(
            _req(system="be helpful", messages=[{"role": "user", "content": "hi"}])
        )
        assert _roles(cc_req) == ["system", "user"]
        # A backend Chat Completions response -> Anthropic Messages response.
        msg = c.chat_completion_to_message(
            _chat_completion({"role": "assistant", "content": "hi there"}), model="claude-opus-4-8"
        )
        assert msg.content[0].text == "hi there"
        assert msg.role == "assistant"
