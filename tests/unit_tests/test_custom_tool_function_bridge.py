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
"""The function bridge for Responses ``custom`` (free-form) tools.

Fixture shapes mirror what the Codex CLI sends in code mode: one ``custom``
``exec`` tool with a lark grammar ``format`` carried inside an
``additional_tools`` developer item, ``custom_tool_call`` replay items with a
bare name, and ``custom_tool_call_output`` items whose output is a plain
string on success and a list of ``input_text`` parts on failure.
"""

import json

import pytest
from pydantic import TypeAdapter

from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseCustomToolCall,
    NeMoGymResponseCustomToolCallForTraining,
    NeMoGymResponseOutputItem,
    training_variant_of,
)
from nemo_gym.responses_converter import CUSTOM_TOOL_BRIDGE_PARAMETER, ResponsesConverter
from nemo_gym.responses_streaming import (
    flatten_namespace_tools,
    restore_namespace_tool_calls,
    sanitize_streaming_responses_body,
)
from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint
from nemo_gym.token_id_capture.terminal import _item_content_key, _served_item_id


EXEC_TOOL = {
    "type": "custom",
    "name": "exec",
    "description": "Run JavaScript code to orchestrate tool calls.",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: SOURCE\nSOURCE: /[\\s\\S]+/\n",
    },
}

EXEC_CODE = 'const out = await tools.shell({command: ["echo", "héllo"]});'


def _bridge_converter() -> ResponsesConverter:
    return ResponsesConverter(
        return_token_id_information=False, uses_reasoning_parser=False, custom_tool_strategy="function_bridge"
    )


def _passthrough_converter() -> ResponsesConverter:
    return ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=False)


def _params(input_items: list, tools: list) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": input_items, "tools": tools, "model": "policy"}
    )


def _chat_completion_with_tool_calls(tool_calls: list) -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "policy",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                }
            ],
        }
    )


def test_bridge_lowers_custom_tool_definition_to_one_string_parameter_function() -> None:
    params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
    chat_params = _bridge_converter().responses_to_chat_completion_create_params(params)
    (tool,) = chat_params.tools
    assert tool["type"] == "function"
    function = tool["function"]
    assert function["name"] == "exec"
    assert function["description"] == EXEC_TOOL["description"]
    assert function["parameters"]["required"] == [CUSTOM_TOOL_BRIDGE_PARAMETER]
    assert function["parameters"]["properties"][CUSTOM_TOOL_BRIDGE_PARAMETER]["type"] == "string"


def test_passthrough_keeps_the_chat_custom_tool_form() -> None:
    params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
    chat_params = _passthrough_converter().responses_to_chat_completion_create_params(params)
    (tool,) = chat_params.tools
    assert tool["type"] == "custom"
    assert tool["custom"]["name"] == "exec"


@pytest.mark.parametrize(
    "output",
    [
        "Script completed\nWall time 0.0 seconds\nOutput:\n",
        [
            {"type": "input_text", "text": "Script failed\nWall time 0.0 seconds\nOutput:\n"},
            {"type": "input_text", "text": "Script error:\nSyntaxError: Illegal return statement"},
        ],
    ],
)
def test_bridge_replays_custom_tool_history_as_the_lowered_function_turn(output) -> None:
    params = _params(
        [
            {"type": "message", "role": "user", "content": "go"},
            {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "exec", "input": EXEC_CODE},
            {"type": "custom_tool_call_output", "id": "ctco_1", "call_id": "call_1", "output": output},
        ],
        [EXEC_TOOL],
    )
    chat_params = _bridge_converter().responses_to_chat_completion_create_params(params)
    assistant = next(m for m in chat_params.messages if m["role"] == "assistant")
    (tool_call,) = assistant["tool_calls"]
    assert tool_call["function"]["name"] == "exec"
    assert json.loads(tool_call["function"]["arguments"]) == {CUSTOM_TOOL_BRIDGE_PARAMETER: EXEC_CODE}
    tool_message = next(m for m in chat_params.messages if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1"


def test_passthrough_still_refuses_replayed_custom_tool_history() -> None:
    params = _params(
        [{"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "exec", "input": EXEC_CODE}],
        [EXEC_TOOL],
    )
    with pytest.raises(NotImplementedError, match="custom_tool_call"):
        _passthrough_converter().responses_to_chat_completion_create_params(params)


def test_bridge_round_trips_the_free_form_payload_byte_for_byte() -> None:
    converter = _bridge_converter()
    for payload in (EXEC_CODE, 'x = {"a": 1}\n', "", "line1\nline2\té中"):
        encoded = converter.bridge_custom_tool_arguments(payload)
        assert converter.unbridge_custom_tool_arguments(encoded) == payload


def test_unbridge_returns_malformed_arguments_verbatim() -> None:
    converter = _bridge_converter()
    assert converter.unbridge_custom_tool_arguments("not json at all") == "not json at all"
    assert converter.unbridge_custom_tool_arguments('{"other": 1}') == '{"other": 1}'
    assert converter.unbridge_custom_tool_arguments(None) == ""


def test_lift_turns_a_bridged_function_call_back_into_a_custom_tool_call_item() -> None:
    converter = _bridge_converter()
    params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
    arguments = converter.bridge_custom_tool_arguments(EXEC_CODE)
    chat_completion = _chat_completion_with_tool_calls(
        [{"id": "call_7", "type": "function", "function": {"name": "exec", "arguments": arguments}}]
    )
    response = converter.chat_completion_to_response(params, chat_completion)
    (item,) = response.output
    assert item.type == "custom_tool_call"
    assert item.name == "exec"
    assert item.call_id == "call_7"
    assert item.input == EXEC_CODE


def test_lift_keeps_function_calls_that_are_not_declared_custom() -> None:
    converter = _bridge_converter()
    params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
    chat_completion = _chat_completion_with_tool_calls(
        [{"id": "call_8", "type": "function", "function": {"name": "wait", "arguments": '{"cell_id": "c1"}'}}]
    )
    (item,) = converter.chat_completion_to_response(params, chat_completion).output
    assert item.type == "function_call"
    assert item.name == "wait"


def test_lift_delivers_a_malformed_bridged_payload_verbatim() -> None:
    converter = _bridge_converter()
    params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
    chat_completion = _chat_completion_with_tool_calls(
        [{"id": "call_9", "type": "function", "function": {"name": "exec", "arguments": "while(true){}"}}]
    )
    (item,) = converter.chat_completion_to_response(params, chat_completion).output
    assert item.type == "custom_tool_call"
    assert item.input == "while(true){}"


def test_lift_handles_a_chat_native_custom_tool_call_under_any_strategy() -> None:
    for converter in (_bridge_converter(), _passthrough_converter()):
        params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
        chat_completion = _chat_completion_with_tool_calls(
            [{"id": "call_10", "type": "custom", "custom": {"name": "exec", "input": EXEC_CODE}}]
        )
        (item,) = converter.chat_completion_to_response(params, chat_completion).output
        assert item.type == "custom_tool_call"
        assert item.input == EXEC_CODE


def test_two_turn_replay_re_renders_the_arguments_the_engine_parsed() -> None:
    """The arguments string a replayed turn renders equals the one the lift consumed."""
    converter = _bridge_converter()
    params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
    arguments = converter.bridge_custom_tool_arguments(EXEC_CODE)
    chat_completion = _chat_completion_with_tool_calls(
        [{"id": "call_11", "type": "function", "function": {"name": "exec", "arguments": arguments}}]
    )
    (item,) = converter.chat_completion_to_response(params, chat_completion).output
    next_params = _params(
        [
            {"type": "message", "role": "user", "content": "go"},
            item.model_dump(),
            {"type": "custom_tool_call_output", "id": "ctco_2", "call_id": "call_11", "output": "done"},
        ],
        [EXEC_TOOL],
    )
    chat_params = converter.responses_to_chat_completion_create_params(next_params)
    assistant = next(m for m in chat_params.messages if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["arguments"] == arguments


def _codex_code_mode_body() -> dict:
    return {
        "model": "policy",
        "stream": True,
        "tools": [],
        "input": [
            {
                "type": "additional_tools",
                "id": "at_1",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "functions",
                        "description": "",
                        "tools": [
                            {k: v for k, v in EXEC_TOOL.items() if k != "type"} | {"type": "custom"},
                            {
                                "type": "function",
                                "name": "wait",
                                "description": "Waits on a yielded exec cell.",
                                "parameters": {"type": "object", "properties": {}},
                                "strict": False,
                            },
                        ],
                    }
                ],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        ],
    }


def test_sanitizer_hoists_carrier_custom_tools_only_when_carried() -> None:
    carried, _ = sanitize_streaming_responses_body(_codex_code_mode_body(), carry_custom_tools=True)
    names = {(tool.get("type"), tool.get("name")) for tool in carried["tools"]}
    assert names == {("custom", "functions__exec"), ("function", "functions__wait")}

    dropped, _ = sanitize_streaming_responses_body(_codex_code_mode_body(), carry_custom_tools=False)
    names = {(tool.get("type"), tool.get("name")) for tool in dropped["tools"]}
    assert names == {("function", "functions__wait")}


def test_sanitizer_folds_namespace_into_replayed_custom_tool_calls() -> None:
    body = _codex_code_mode_body()
    body["input"].append(
        {
            "type": "custom_tool_call",
            "id": "ctc_1",
            "call_id": "call_1",
            "namespace": "functions",
            "name": "exec",
            "input": EXEC_CODE,
        }
    )
    body["input"].append({"type": "custom_tool_call_output", "id": "ctco_1", "call_id": "call_1", "output": "done"})
    cleaned, ns_map = sanitize_streaming_responses_body(body, carry_custom_tools=True)
    replayed = next(item for item in cleaned["input"] if item.get("type") == "custom_tool_call")
    assert replayed["name"] == "functions__exec"
    assert "namespace" not in replayed
    assert ns_map["functions__exec"] == ("functions", "exec")


def test_restore_splits_flattened_custom_tool_call_names() -> None:
    _, ns_map = flatten_namespace_tools(_codex_code_mode_body()["input"][0]["tools"])
    (restored,) = restore_namespace_tool_calls(
        [{"type": "custom_tool_call", "name": "functions__exec", "call_id": "c", "input": "x"}], ns_map
    )
    assert restored["namespace"] == "functions"
    assert restored["name"] == "exec"


def test_flatten_preserves_each_sub_tools_own_type() -> None:
    flat, _ = flatten_namespace_tools(_codex_code_mode_body()["input"][0]["tools"])
    assert [(tool["type"], tool["name"]) for tool in flat] == [
        ("custom", "functions__exec"),
        ("function", "functions__wait"),
    ]


def test_custom_tool_call_has_a_training_variant_and_discriminates_with_token_metadata() -> None:
    assert training_variant_of(NeMoGymResponseCustomToolCall) is NeMoGymResponseCustomToolCallForTraining
    item = {
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_1",
        "name": "exec",
        "input": EXEC_CODE,
        "status": "completed",
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3],
        "generation_log_probs": [-0.5],
    }
    validated = TypeAdapter(NeMoGymResponseOutputItem).validate_python(item)
    assert isinstance(validated, NeMoGymResponseCustomToolCallForTraining)


def test_fingerprints_tell_custom_tool_payloads_apart() -> None:
    def transcript(code: str) -> list[dict]:
        return [
            {"role": "user", "content": "go"},
            {"type": "custom_tool_call", "call_id": "call_1", "name": "exec", "input": code},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "done"},
        ]

    one = assistant_fingerprint(transcript(EXEC_CODE))
    other = assistant_fingerprint(transcript("const y = 1;"))
    assert one != other
    assert assistant_fingerprint(transcript(EXEC_CODE)) == one


def test_terminal_keys_identify_custom_tool_calls_by_call_id_and_payload() -> None:
    item = {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "functions__exec", "input": "x"}
    assert _served_item_id(item) == "call_1"
    kind, name, payload = _item_content_key(item)
    assert (kind, name) == ("custom_tool_call", "exec")
    restored = {
        "type": "custom_tool_call",
        "call_id": "call_1",
        "namespace": "functions",
        "name": "exec",
        "input": "x",
    }
    assert _item_content_key(restored) == (kind, name, payload)
