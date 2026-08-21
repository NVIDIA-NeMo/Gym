# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from responses_api_models.sglang_model.tool_parsers import (
    normalize_tool_call_arguments,
    parse_qwen3_coder_tool_calls,
)


TOOL_CALL = (
    "<tool_call>\n"
    "<function=editor>\n"
    "<parameter=line_number>\n42\n</parameter>\n"
    "<parameter=command>\n"
    "text spanning\nmultiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_parses_multiline_tool_call_and_keeps_content() -> None:
    tool_calls, content = parse_qwen3_coder_tool_calls(f"reasoning\n{TOOL_CALL}")

    assert content == "reasoning"
    assert tool_calls[0]["function"]["name"] == "editor"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "line_number": "42",
        "command": "text spanning\nmultiple lines",
    }


def test_uses_tool_schema_for_argument_types() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "editor",
                "parameters": {
                    "properties": {
                        "line_number": {"type": "integer"},
                        "command": {"type": "string"},
                    }
                },
            },
        }
    ]

    tool_calls, _ = parse_qwen3_coder_tool_calls(TOOL_CALL, tools)

    assert json.loads(tool_calls[0]["function"]["arguments"])["line_number"] == 42


def test_no_tool_call_is_passthrough() -> None:
    assert parse_qwen3_coder_tool_calls("plain text") == ([], "plain text")


def test_parses_multiple_tool_calls() -> None:
    second = TOOL_CALL.replace("editor", "shell")

    tool_calls, content = parse_qwen3_coder_tool_calls(f"{TOOL_CALL}\n{second}")

    assert [call["function"]["name"] for call in tool_calls] == ["editor", "shell"]
    assert content == ""


def test_invalid_typed_value_falls_back_to_string() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "editor",
                "parameters": {"properties": {"line_number": {"type": "integer"}}},
            },
        }
    ]
    text = TOOL_CALL.replace("42", "not-a-number")

    tool_calls, _ = parse_qwen3_coder_tool_calls(text, tools)

    assert json.loads(tool_calls[0]["function"]["arguments"])["line_number"] == "not-a-number"


def test_unknown_tool_parameters_remain_strings() -> None:
    tool_calls, _ = parse_qwen3_coder_tool_calls(TOOL_CALL, tools=[])

    assert json.loads(tool_calls[0]["function"]["arguments"])["line_number"] == "42"


def test_shell_text_with_angle_brackets_is_preserved() -> None:
    text = (
        "<tool_call>\n<function=shell>\n"
        "<parameter=command>\n"
        "grep -rn 'x < y && y > z' src/ | head -5\n"
        "</parameter>\n"
        "</function>\n</tool_call>"
    )

    tool_calls, content = parse_qwen3_coder_tool_calls(text)

    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "command": "grep -rn 'x < y && y > z' src/ | head -5"
    }
    assert content == ""


def test_normalizes_string_arguments_without_mutating_input() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "editor", "arguments": '{"command": "view"}'},
                }
            ],
        },
    ]

    normalized = normalize_tool_call_arguments(messages)

    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {"command": "view"}
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"command": "view"}'


def test_normalizes_parser_output() -> None:
    tool_calls, _ = parse_qwen3_coder_tool_calls(TOOL_CALL)

    normalized = normalize_tool_call_arguments([{"role": "assistant", "tool_calls": tool_calls}])

    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
        "line_number": "42",
        "command": "text spanning\nmultiple lines",
    }


def test_normalizer_leaves_non_object_arguments_unchanged() -> None:
    def message(arguments):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "editor", "arguments": arguments},
                }
            ],
        }

    mapping = message({"command": "view"})
    assert normalize_tool_call_arguments([mapping])[0]["tool_calls"][0]["function"]["arguments"] == {"command": "view"}
    for raw in ("not json", "[1, 2]"):
        assert normalize_tool_call_arguments([message(raw)])[0]["tool_calls"][0]["function"]["arguments"] == raw
