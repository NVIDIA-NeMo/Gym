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
import logging

import pytest
from pydantic import TypeAdapter

from nemo_gym.core import NeMoGymChatCompletion, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseOutputItem
from nemo_gym.core.responses_api_types import (
    NeMoGymResponseCustomToolCall,
    NeMoGymResponseCustomToolCallForTraining,
    training_variant_of,
)
from nemo_gym.models import ResponsesConverter, flatten_namespace_tools, restore_namespace_tool_calls
from nemo_gym.models.chat_completions_converter import _CUSTOM_TOOL_BRIDGE_PARAMETER
from nemo_gym.models.responses_streaming import sanitize_streaming_responses_body
from nemo_gym.models.token_capture.fingerprint import assistant_fingerprint
from nemo_gym.models.token_capture.terminal import _item_content_key, _served_item_id


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

# EXEC_CODE as a vLLM tool parser serializes the bridged one-parameter object
# (default json.dumps separators, non-ASCII kept raw). A literal, so the replay
# expectation does not depend on the converter's own encoder.
ENGINE_ARGUMENTS = '{"input": "const out = await tools.shell({command: [\\"echo\\", \\"héllo\\"]});"}'


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
    assert function["parameters"]["required"] == [_CUSTOM_TOOL_BRIDGE_PARAMETER]
    assert function["parameters"]["properties"][_CUSTOM_TOOL_BRIDGE_PARAMETER]["type"] == "string"


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
    assert json.loads(tool_call["function"]["arguments"]) == {_CUSTOM_TOOL_BRIDGE_PARAMETER: EXEC_CODE}
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
        encoded = converter._encode_bridged_input(payload)
        assert converter._decode_bridged_input(encoded) == payload


def test_decode_returns_malformed_arguments_verbatim() -> None:
    converter = _bridge_converter()
    assert converter._decode_bridged_input("not json at all") == "not json at all"
    assert converter._decode_bridged_input('{"other": 1}') == '{"other": 1}'
    assert converter._decode_bridged_input(None) == ""


def test_decode_logs_each_malformed_shape_with_the_call_context(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="nemo_gym.models.chat_completions_converter"):
        assert ResponsesConverter._decode_bridged_input("not json", tool_name="exec", call_id="call_a") == "not json"
        assert (
            ResponsesConverter._decode_bridged_input('{"input": 5}', tool_name="exec", call_id="call_b")
            == '{"input": 5}'
        )
        assert (
            ResponsesConverter._decode_bridged_input(
                '{"input": "x", "timeout": 5}', tool_name="exec", call_id="call_c"
            )
            == "x"
        )
    assert "call_a" in caplog.text
    assert "call_b" in caplog.text
    assert "call_c" in caplog.text and "timeout" in caplog.text


def test_lift_turns_a_bridged_function_call_back_into_a_custom_tool_call_item() -> None:
    converter = _bridge_converter()
    params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
    chat_completion = _chat_completion_with_tool_calls(
        [{"id": "call_7", "type": "function", "function": {"name": "exec", "arguments": ENGINE_ARGUMENTS}}]
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


def _replay_arguments_after_lift(converter: ResponsesConverter, engine_arguments: str) -> str:
    """Lift a bridged call with the given engine arguments, replay it, return the rendered arguments."""
    params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
    chat_completion = _chat_completion_with_tool_calls(
        [{"id": "call_11", "type": "function", "function": {"name": "exec", "arguments": engine_arguments}}]
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
    return assistant["tool_calls"][0]["function"]["arguments"]


def test_two_turn_replay_re_renders_a_well_formed_call_to_the_engines_arguments() -> None:
    """A replayed turn renders the literal arguments string the engine produced."""
    assert _replay_arguments_after_lift(_bridge_converter(), ENGINE_ARGUMENTS) == ENGINE_ARGUMENTS


def test_two_turn_replay_preserves_the_parsed_object_across_encoding_differences() -> None:
    """Compact separators or ASCII escapes re-render byte-different but parse to the same object.

    vLLM parses a replayed call's arguments into a dict before rendering the chat
    template, so the template sees the parsed object, not the byte string.
    """
    compact = json.dumps({_CUSTOM_TOOL_BRIDGE_PARAMETER: EXEC_CODE}, separators=(",", ":"), ensure_ascii=True)
    replayed = _replay_arguments_after_lift(_bridge_converter(), compact)
    assert replayed != compact
    assert json.loads(replayed) == json.loads(compact)


def test_two_turn_replay_re_wraps_a_malformed_calls_raw_text_lift() -> None:
    """The replay limit: a malformed call's raw-text lift re-encodes as a different object.

    The engine produced ``{"input": 5}``; the lift delivers that raw text to the
    harness, and the replay wraps the raw text as the string payload, so the
    parsed object no longer matches what the model generated.
    """
    replayed = _replay_arguments_after_lift(_bridge_converter(), '{"input": 5}')
    assert json.loads(replayed) == {_CUSTOM_TOOL_BRIDGE_PARAMETER: '{"input": 5}'}


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
    carried, _ = sanitize_streaming_responses_body(_codex_code_mode_body(), hoist_carrier_custom_tools=True)
    names = {(tool.get("type"), tool.get("name")) for tool in carried["tools"]}
    assert names == {("custom", "functions__exec"), ("function", "functions__wait")}

    dropped, _ = sanitize_streaming_responses_body(_codex_code_mode_body(), hoist_carrier_custom_tools=False)
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
    cleaned, ns_map = sanitize_streaming_responses_body(body, hoist_carrier_custom_tools=True)
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


def test_bridge_lowers_a_custom_tool_without_a_description() -> None:
    params = _params([{"type": "message", "role": "user", "content": "go"}], [{"type": "custom", "name": "exec"}])
    chat_params = _bridge_converter().responses_to_chat_completion_create_params(params)
    (tool,) = chat_params.tools
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "exec"
    assert "description" not in tool["function"]


def test_grammar_drop_is_warned_once_per_tool_name(caplog) -> None:
    tool = dict(EXEC_TOOL, name="exec_grammar_once")
    with caplog.at_level(logging.WARNING, logger="nemo_gym.models.chat_completions_converter"):
        ResponsesConverter._custom_tool_to_bridged_function(tool)
        ResponsesConverter._custom_tool_to_bridged_function(tool)
    assert caplog.text.count("exec_grammar_once") == 1


def test_bridge_rewrites_a_custom_tool_choice_reference_to_the_function_form() -> None:
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [{"type": "message", "role": "user", "content": "go"}],
            "tools": [EXEC_TOOL],
            "model": "policy",
            "tool_choice": {"type": "custom", "name": "exec"},
        }
    )
    chat_params = _bridge_converter().responses_to_chat_completion_create_params(params)
    assert chat_params.tool_choice == {"type": "function", "function": {"name": "exec"}}


def test_bridge_rewrites_custom_references_inside_allowed_tools() -> None:
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [{"type": "message", "role": "user", "content": "go"}],
            "tools": [
                EXEC_TOOL,
                {
                    "type": "function",
                    "name": "wait",
                    "description": "Waits on a yielded exec cell.",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": False,
                },
            ],
            "model": "policy",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "custom", "name": "exec"}, {"type": "function", "name": "wait"}],
            },
        }
    )
    chat_params = _bridge_converter().responses_to_chat_completion_create_params(params)
    assert chat_params.tool_choice == {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "auto",
            "tools": [
                {"type": "function", "function": {"name": "exec"}},
                {"type": "function", "function": {"name": "wait"}},
            ],
        },
    }


def test_bridge_rejects_a_name_declared_both_as_function_and_as_custom() -> None:
    tools = [
        EXEC_TOOL,
        {
            "type": "function",
            "name": "exec",
            "description": "A JSON-arguments tool that shares the custom tool's name.",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        },
    ]
    params = _params([{"type": "message", "role": "user", "content": "go"}], tools)
    with pytest.raises(ValueError, match="both as a function tool and as a custom tool"):
        _bridge_converter().responses_to_chat_completion_create_params(params)
    # Passthrough keeps both declarations and the backend decides.
    chat_params = _passthrough_converter().responses_to_chat_completion_create_params(params)
    assert len(chat_params.tools) == 2


def test_a_namespaced_custom_sub_tool_in_top_level_tools_follows_the_top_level_rule() -> None:
    """Pins the passthrough-path change the bridge commit made to `flatten_namespace_tools`.

    A namespaced custom sub-tool keeps its declared type, so under passthrough it
    follows the same rule as a top-level custom tool: kept for the backend to
    decide, and removed by ``drop_custom_tools``. It used to be coerced to a
    parameterless function and silently dropped at validation.
    """

    def body() -> dict:
        return {
            "model": "policy",
            "stream": True,
            "tools": [_codex_code_mode_body()["input"][0]["tools"][0]],
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}],
        }

    kept, _ = sanitize_streaming_responses_body(body())
    assert ("custom", "functions__exec") in {(t.get("type"), t.get("name")) for t in kept["tools"]}

    dropped, _ = sanitize_streaming_responses_body(body(), drop_custom_tools=True)
    assert ("custom", "functions__exec") not in {(t.get("type"), t.get("name")) for t in dropped["tools"]}


def test_lift_wraps_a_bridged_call_in_the_training_variant_with_its_token_ids() -> None:
    converter = ResponsesConverter(
        return_token_id_information=True, uses_reasoning_parser=False, custom_tool_strategy="function_bridge"
    )
    params = _params([{"type": "message", "role": "user", "content": "go"}], [EXEC_TOOL])
    chat_completion = NeMoGymChatCompletion.model_validate(
        {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1,
            "model": "policy",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_15",
                                "type": "function",
                                "function": {"name": "exec", "arguments": ENGINE_ARGUMENTS},
                            }
                        ],
                        "prompt_token_ids": [1, 2, 3],
                        "generation_token_ids": [4, 5],
                        "generation_log_probs": [-0.1, -0.2],
                    },
                }
            ],
        }
    )
    (item,) = converter.chat_completion_to_response(params, chat_completion).output
    assert isinstance(item, NeMoGymResponseCustomToolCallForTraining)
    assert item.type == "custom_tool_call"
    assert item.input == EXEC_CODE
    assert item.prompt_token_ids == [1, 2, 3]
    assert item.generation_token_ids == [4, 5]


def test_chat_shaped_custom_tool_calls_lower_under_the_bridge_only() -> None:
    items = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_16", "type": "custom", "custom": {"name": "exec", "input": EXEC_CODE}}],
        },
        {"role": "tool", "tool_call_id": "call_16", "content": "done"},
    ]
    chat_params = _bridge_converter().responses_to_chat_completion_create_params(_params(items, [EXEC_TOOL]))
    assistant = next(m for m in chat_params.messages if m["role"] == "assistant")
    (tool_call,) = assistant["tool_calls"]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "exec"
    assert json.loads(tool_call["function"]["arguments"]) == {_CUSTOM_TOOL_BRIDGE_PARAMETER: EXEC_CODE}

    with pytest.raises(NotImplementedError, match="chat-shaped custom tool call"):
        _passthrough_converter().responses_to_chat_completion_create_params(_params(items, [EXEC_TOOL]))
