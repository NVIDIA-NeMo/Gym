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
import json

from responses_api_models.claude_messages_model.converter import (
    anthropic_messages_to_chat,
    anthropic_tools_to_chat,
    build_anthropic_message,
    chat_message_to_anthropic_blocks,
    sse_events_for_message,
)


class TestAnthropicToChat:
    def test_system_and_user_text(self):
        out = anthropic_messages_to_chat(
            system=[{"type": "text", "text": "be brief"}],
            messages=[{"role": "user", "content": "hi"}],
        )
        assert out == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_assistant_tool_use_then_tool_result_roundtrip(self):
        # an assistant tool_use turn, then the user turn carrying its tool_result
        out = anthropic_messages_to_chat(
            system=None,
            messages=[
                {"role": "user", "content": "run ls"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "sure"},
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file.txt"}]},
            ],
        )
        assert out[0] == {"role": "user", "content": "run ls"}
        assert out[1]["role"] == "assistant"
        assert out[1]["content"] == "sure"
        assert out[1]["tool_calls"][0]["id"] == "t1"
        assert out[1]["tool_calls"][0]["function"]["name"] == "Bash"
        assert json.loads(out[1]["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}
        # tool message must follow the assistant tool_call, keyed by the same id
        assert out[2] == {"role": "tool", "tool_call_id": "t1", "content": "file.txt"}

    def test_thinking_folded_inline(self):
        out = anthropic_messages_to_chat(
            system=None,
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ok"}],
                }
            ],
        )
        assert out[0]["content"] == "<think>hmm</think>ok"

    def test_tools_conversion(self):
        chat_tools = anthropic_tools_to_chat(
            [
                {
                    "name": "Read",
                    "description": "read",
                    "input_schema": {"type": "object", "properties": {"p": {"type": "string"}}},
                }
            ]
        )
        assert chat_tools[0]["type"] == "function"
        assert chat_tools[0]["function"]["name"] == "Read"
        assert chat_tools[0]["function"]["parameters"]["properties"]["p"]["type"] == "string"


class TestChatToAnthropic:
    def test_text_only_message_blocks(self):
        blocks = chat_message_to_anthropic_blocks({"role": "assistant", "content": "hello", "tool_calls": None})
        assert blocks == [{"type": "text", "text": "hello"}]

    def test_tool_call_blocks(self):
        blocks = chat_message_to_anthropic_blocks(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "Bash", "arguments": '{"command":"ls"}'}}],
            }
        )
        # empty content + tool call -> only the tool_use block (no empty text block)
        assert blocks == [{"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]

    def test_empty_message_gets_empty_text_block(self):
        blocks = chat_message_to_anthropic_blocks({"role": "assistant", "content": None, "tool_calls": None})
        assert blocks == [{"type": "text", "text": ""}]

    def test_sse_sequence_shape(self):
        msg = build_anthropic_message(
            message={"role": "assistant", "content": "hi", "tool_calls": None},
            finish_reason="stop",
            model="m",
            input_tokens=3,
            output_tokens=1,
            message_id="msg_1",
        )
        assert msg["stop_reason"] == "end_turn"
        events = [name for name, _ in sse_events_for_message(msg)]
        assert events == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

    def test_stop_reason_tool_use(self):
        msg = build_anthropic_message(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c", "function": {"name": "B", "arguments": "{}"}}],
            },
            finish_reason="tool_calls",
            model="m",
            input_tokens=1,
            output_tokens=1,
            message_id="msg_2",
        )
        assert msg["stop_reason"] == "tool_use"
