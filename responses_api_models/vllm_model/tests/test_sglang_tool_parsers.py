# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit tests for the qwen3_coder tool-call text parser (SGLang /generate path).

Fixtures mirror the canonical tool-call example embedded in Qwen3-Coder-style
chat templates, which instruct the model to emit::

    <tool_call>
    <function=example_function_name>
    <parameter=example_parameter_1>
    value_1
    </parameter>
    </function>
    </tool_call>
"""

import json

from responses_api_models.vllm_model.sglang_tool_parsers import (
    normalize_tool_call_arguments,
    parse_qwen3_coder_tool_calls,
)


# The template's own canonical example, verbatim shape.
TEMPLATE_EXAMPLE = (
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_template_canonical_example():
    tool_calls, content = parse_qwen3_coder_tool_calls(TEMPLATE_EXAMPLE)
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "example_function_name"
    args = json.loads(call["function"]["arguments"])
    assert args == {
        "example_parameter_1": "value_1",
        "example_parameter_2": ("This is the value for the second parameter\nthat can span\nmultiple lines"),
    }
    assert content == ""


def test_reasoning_prefix_kept_as_content():
    text = "Let me inspect the repository first.\n\n" + TEMPLATE_EXAMPLE
    tool_calls, content = parse_qwen3_coder_tool_calls(text)
    assert len(tool_calls) == 1
    assert content == "Let me inspect the repository first."


def test_no_tool_call_passthrough():
    text = "The bug is in utils.py line 42; here is my analysis."
    tool_calls, content = parse_qwen3_coder_tool_calls(text)
    assert tool_calls == []
    assert content == text


def test_multiple_tool_calls():
    text = TEMPLATE_EXAMPLE + "\n" + TEMPLATE_EXAMPLE.replace("example_function_name", "second_function")
    tool_calls, content = parse_qwen3_coder_tool_calls(text)
    assert [c["function"]["name"] for c in tool_calls] == [
        "example_function_name",
        "second_function",
    ]
    assert content == ""


def test_schema_aware_coercion():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "parameters": {
                    "properties": {
                        "line_number": {"type": "integer"},
                        "dry_run": {"type": "boolean"},
                        "tags": {"type": "array"},
                        "command": {"type": "string"},
                    }
                },
            },
        }
    ]
    text = (
        "<tool_call>\n"
        "<function=str_replace_editor>\n"
        "<parameter=line_number>\n42\n</parameter>\n"
        "<parameter=dry_run>\ntrue\n</parameter>\n"
        '<parameter=tags>\n["a", "b"]\n</parameter>\n'
        "<parameter=command>\nview\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    tool_calls, _ = parse_qwen3_coder_tool_calls(text, tools)
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {
        "line_number": 42,
        "dry_run": True,
        "tags": ["a", "b"],
        "command": "view",
    }


def test_coercion_falls_back_to_string_on_mismatch():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"properties": {"n": {"type": "integer"}}},
            },
        }
    ]
    text = "<tool_call>\n<function=f>\n<parameter=n>\nnot_a_number\n</parameter>\n</function>\n</tool_call>"
    tool_calls, _ = parse_qwen3_coder_tool_calls(text, tools)
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {"n": "not_a_number"}


def test_unknown_function_params_stay_strings():
    text = "<tool_call>\n<function=unlisted>\n<parameter=x>\n7\n</parameter>\n</function>\n</tool_call>"
    tool_calls, _ = parse_qwen3_coder_tool_calls(text, tools=[])
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {"x": "7"}


def test_bash_command_value_with_angle_brackets():
    # Shell text containing '<' / '>' must not confuse the param regex.
    text = (
        "<tool_call>\n<function=execute_bash>\n"
        "<parameter=command>\n"
        "grep -rn 'x < y && y > z' src/ | head -5\n"
        "</parameter>\n"
        "</function>\n</tool_call>"
    )
    tool_calls, content = parse_qwen3_coder_tool_calls(text)
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {"command": "grep -rn 'x < y && y > z' src/ | head -5"}
    assert content == ""


def test_normalize_decodes_string_arguments():
    # OpenAI-format history: arguments is a JSON string (as our parsers emit).
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "execute_bash",
                        "arguments": '{"command": "ls"}',
                    },
                }
            ],
        },
    ]
    out = normalize_tool_call_arguments(messages)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    # Inputs are not mutated (the session cache holds the originals).
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'
    assert out[0] is messages[0]


def test_normalize_roundtrips_parser_output():
    tool_calls, _ = parse_qwen3_coder_tool_calls(
        "<tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>"
    )
    out = normalize_tool_call_arguments([{"role": "assistant", "tool_calls": tool_calls}])
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"k": "v"}


def test_normalize_leaves_non_object_arguments():
    def msg(args):
        return {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": args}}],
        }

    # Already a mapping: untouched.
    out = normalize_tool_call_arguments([msg({"k": 1})])
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"k": 1}
    # Non-JSON string and JSON-but-not-object string: left as-is.
    for raw in ("not json", "[1, 2]"):
        out = normalize_tool_call_arguments([msg(raw)])
        assert out[0]["tool_calls"][0]["function"]["arguments"] == raw
    # Messages without tool_calls pass through by identity.
    plain = {"role": "tool", "content": "ok"}
    assert normalize_tool_call_arguments([plain])[0] is plain
