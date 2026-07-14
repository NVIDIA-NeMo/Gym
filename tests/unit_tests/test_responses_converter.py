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
"""Unit tests for the shared Responses API <-> Chat Completions converter."""

import pytest
from openai.types.completion_usage import CompletionUsage

from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionMessage,
    NeMoGymChatCompletionMessageToolCall,
    NeMoGymChoice,
    NeMoGymEasyInputMessage,
    NeMoGymFunction,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputText,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.responses_converter import (
    ResponsesConverter,
    ResponsesConverterState,
    VLLMConverter,
    VLLMConverterResponsesToChatCompletionsState,
    split_responses_input_output_items,
)


FIXED_UUID = "123"


class FakeUUID:
    hex = FIXED_UUID


@pytest.fixture
def converter() -> ResponsesConverter:
    return ResponsesConverter(return_token_id_information=False)


@pytest.fixture(autouse=True)
def _fixed_uuid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())


# ===========================================================================
# Backward-compatible aliases
# ===========================================================================


def test_backwards_compatible_aliases():
    assert VLLMConverter is ResponsesConverter
    assert VLLMConverterResponsesToChatCompletionsState is ResponsesConverterState


# ===========================================================================
# Reasoning helpers
# ===========================================================================


def test_wrap_reasoning_in_think_tags_filters_empty():
    assert ResponsesConverter._wrap_reasoning_in_think_tags(["a", "", "b"]) == "<think>a</think><think>b</think>"
    assert ResponsesConverter._wrap_reasoning_in_think_tags([]) == ""


def test_extract_reasoning_from_content(converter: ResponsesConverter):
    matches, cleaned = converter._extract_reasoning_from_content(
        "before<think>thought 1</think>middle<think>thought 2</think>after"
    )
    assert matches == ["thought 1", "thought 2"]
    assert cleaned == "beforemiddleafter"

    matches, cleaned = converter._extract_reasoning_from_content("no reasoning here")
    assert matches == []
    assert cleaned == "no reasoning here"


# ===========================================================================
# ResponsesConverterState.flush_assistant
# ===========================================================================


def test_flush_assistant_noop_on_empty_buffer():
    state = ResponsesConverterState(return_token_id_information=False)
    state.flush_assistant()
    assert state.messages == []


def test_flush_assistant_emits_plain_message():
    state = ResponsesConverterState(return_token_id_information=False)
    state.content_buffer = "hello"
    state.flush_assistant()
    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "assistant"
    assert state.messages[0]["content"] == "hello"
    # Buffers are reset after a flush.
    assert state.content_buffer == ""
    assert state.tool_calls_buffer == []


def test_flush_assistant_emits_training_message_when_token_info_present():
    state = ResponsesConverterState(return_token_id_information=True)
    state.content_buffer = "hello"
    from nemo_gym.openai_utils import TokenIDLogProbMixin

    state.token_information = TokenIDLogProbMixin(
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.1],
    )
    state.flush_assistant()
    assert state.messages[0]["prompt_token_ids"] == [1, 2]
    assert state.messages[0]["generation_token_ids"] == [3]


# ===========================================================================
# responses_to_chat_completion_create_params
# ===========================================================================


def test_responses_to_chat_completion_string_input(converter: ResponsesConverter):
    params = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(input="just a string")
    )
    assert params.messages == [{"role": "user", "content": [{"type": "text", "text": "just a string"}]}]


def test_responses_to_chat_completion_all_message_roles(converter: ResponsesConverter):
    params = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(role="system", content="sys", type="message"),
                NeMoGymEasyInputMessage(role="developer", content="dev", type="message"),
                NeMoGymEasyInputMessage(role="user", content="usr", type="message"),
                # type is inferred from the presence of a role.
                NeMoGymEasyInputMessage(role="user", content="no type given"),
                NeMoGymEasyInputMessage(
                    role="assistant",
                    content=[NeMoGymResponseInputText(text="assistant content", type="input_text")],
                    type="message",
                ),
            ]
        )
    )
    roles = [m["role"] for m in params.messages]
    assert roles == ["system", "developer", "user", "user", "assistant"]
    assert params.messages[-1]["content"] == "assistant content"


def test_responses_to_chat_completion_input_image_part(converter: ResponsesConverter):
    params = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {"type": "input_image", "image_url": "http://img", "detail": "high"},
                    ],
                }
            ]
        )
    )
    parts = params.messages[0]["content"]
    assert {"type": "text", "text": "what is this?"} in parts
    assert {"type": "image_url", "image_url": {"url": "http://img", "detail": "high"}} in parts


def test_responses_to_chat_completion_unsupported_part_raises(converter: ResponsesConverter):
    # Exercise the converter directly with an unsupported content part type. A raw
    # ResponseCreateParams would reject this at schema-validation time, so we call the
    # message formatter directly to cover the converter's own guard.
    with pytest.raises(NotImplementedError):
        converter._format_message(
            {"role": "user", "content": [{"type": "input_audio", "text": "x"}]},
            ResponsesConverterState(return_token_id_information=False),
        )


def test_responses_to_chat_completion_assistant_invalid_content_raises(converter: ResponsesConverter):
    with pytest.raises(NotImplementedError):
        converter._format_message(
            {"role": "assistant", "content": 42},
            ResponsesConverterState(return_token_id_information=False),
        )


def test_responses_to_chat_completion_function_call_and_output(converter: ResponsesConverter):
    params = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "user", "type": "message", "content": "call a tool"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "nyc"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            ]
        )
    )
    # user message, assistant message with tool call, tool result
    assert params.messages[0]["role"] == "user"
    assistant_msg = params.messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"
    tool_msg = params.messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "sunny"


def test_responses_to_chat_completion_reasoning_prepended(converter: ResponsesConverter):
    reasoning = NeMoGymResponseReasoningItem(
        id="rs_1",
        type="reasoning",
        status="completed",
        summary=[NeMoGymSummary(type="summary_text", text="thinking...")],
    )
    params = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                reasoning.model_dump(),
                {"role": "assistant", "type": "message", "content": "the answer"},
            ]
        )
    )
    assert params.messages[0]["content"] == "<think>thinking...</think>the answer"


def test_responses_to_chat_completion_reasoning_without_summary_is_noop(converter: ResponsesConverter):
    reasoning = NeMoGymResponseReasoningItem(id="rs_1", type="reasoning", status="completed", summary=[])
    params = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                reasoning.model_dump(),
                {"role": "assistant", "type": "message", "content": "the answer"},
            ]
        )
    )
    assert params.messages[0]["content"] == "the answer"


def test_responses_to_chat_completion_model_and_max_tokens_and_tools(converter: ResponsesConverter):
    params = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input="hi",
            model="my-model",
            max_output_tokens=128,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": True,
                }
            ],
        )
    )
    assert params.model == "my-model"
    assert params.max_tokens == 128
    assert params.tools[0]["type"] == "function"
    assert params.tools[0]["function"]["name"] == "get_weather"


def test_responses_to_chat_completion_token_id_information_path():
    converter = ResponsesConverter(return_token_id_information=True)
    params = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "assistant",
                    "type": "message",
                    "content": "trained answer",
                    "prompt_token_ids": [1, 2, 3],
                    "generation_token_ids": [4, 5],
                    "generation_log_probs": [-0.1, -0.2],
                }
            ]
        )
    )
    msg = params.messages[0]
    assert msg["prompt_token_ids"] == [1, 2, 3]
    assert msg["generation_token_ids"] == [4, 5]


# ===========================================================================
# postprocess_assistant_message_dict / postprocess_chat_response
# ===========================================================================


def test_postprocess_extracts_reasoning_when_enabled(converter: ResponsesConverter):
    output = converter.postprocess_assistant_message_dict(
        {"role": "assistant", "content": "<think>reasoning</think>the answer"}
    )
    assert isinstance(output[0], NeMoGymResponseReasoningItem)
    assert output[0].summary[0].text == "reasoning"
    assert isinstance(output[1], NeMoGymResponseOutputMessage)
    assert output[1].content[0].text == "the answer"


def test_postprocess_keeps_think_inline_when_disabled():
    converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=False)
    output = converter.postprocess_assistant_message_dict(
        {"role": "assistant", "content": "<think>reasoning</think>the answer"}
    )
    assert all(not isinstance(item, NeMoGymResponseReasoningItem) for item in output)
    assert output[0].content[0].text == "<think>reasoning</think>the answer"


def test_postprocess_empty_output_emits_empty_message(converter: ResponsesConverter):
    output = converter.postprocess_assistant_message_dict({"role": "assistant", "content": ""})
    assert len(output) == 1
    assert isinstance(output[0], NeMoGymResponseOutputMessage)
    assert output[0].content[0].text == ""


def test_postprocess_tool_calls(converter: ResponsesConverter):
    output = converter.postprocess_assistant_message_dict(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}},
            ],
        }
    )
    tool_calls = [item for item in output if isinstance(item, NeMoGymResponseFunctionToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].call_id == "call_1"


def test_postprocess_chat_response_via_choice(converter: ResponsesConverter):
    choice = NeMoGymChoice(
        index=0,
        finish_reason="stop",
        message=NeMoGymChatCompletionMessage(role="assistant", content="hello"),
    )
    output = converter.postprocess_chat_response(choice)
    assert output[0].content[0].text == "hello"


def test_postprocess_token_id_information_wraps_last_item():
    converter = ResponsesConverter(return_token_id_information=True)
    output = converter.postprocess_assistant_message_dict(
        {
            "role": "assistant",
            "content": "answer",
            "prompt_token_ids": [1, 2],
            "generation_token_ids": [3],
            "generation_log_probs": [-0.1],
        }
    )
    assert isinstance(output[-1], NeMoGymResponseOutputMessageForTraining)
    assert output[-1].prompt_token_ids == [1, 2]


# ===========================================================================
# chat_completions_messages_to_responses_items
# ===========================================================================


def test_chat_messages_to_responses_items_all_roles(converter: ResponsesConverter):
    items = converter.chat_completions_messages_to_responses_items(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": None},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
    )
    # system, user (None -> ""), assistant message, tool output
    assert items[1].content == ""
    assert any(isinstance(item, NeMoGymFunctionCallOutput) for item in items)


def test_chat_messages_to_responses_items_unrecognized_role_raises(converter: ResponsesConverter):
    with pytest.raises(NotImplementedError):
        converter.chat_completions_messages_to_responses_items([{"role": "alien", "content": "x"}])


# ===========================================================================
# chat_completion_to_response
# ===========================================================================


def test_chat_completion_to_response_sanity(converter: ResponsesConverter):
    actual_response = converter.chat_completion_to_response(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            model="",
            input=[
                dict(
                    role="user",
                    content="hello",
                ),
            ],
        ),
        chat_completion=NeMoGymChatCompletion(
            id="",
            created=0,
            model="",
            object="chat.completion",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="tool_calls",
                    message=NeMoGymChatCompletionMessage(
                        role="assistant",
                        content="hi",
                        tool_calls=[],
                    ),
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
            ),
        ),
    )

    expected_response = NeMoGymResponse(
        id="resp_123",
        created_at=0.0,
        model="",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_123",
                content=[
                    NeMoGymResponseOutputText(text="hi", type="output_text", annotations=[]),
                ],
                role="assistant",
            )
        ],
        parallel_tool_calls=True,
        usage=NeMoGymResponseUsage(
            input_tokens=1,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=2,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=3,
        ),
        tool_choice="auto",
        tools=[],
    )

    assert expected_response == actual_response


# ===========================================================================
# split_responses_input_output_items
# ===========================================================================


def test_split_empty_returns_empty():
    assert split_responses_input_output_items([]) == ([], [])


def test_split_on_assistant_message():
    user = NeMoGymEasyInputMessage(role="user", content="hi", type="message")
    assistant = NeMoGymResponseOutputMessage(
        id="msg_1",
        role="assistant",
        type="message",
        status="completed",
        content=[NeMoGymResponseOutputText(type="output_text", text="hi", annotations=[])],
    )
    inputs, outputs = split_responses_input_output_items([user, assistant])
    assert inputs == [user]
    assert outputs == [assistant]


def test_split_on_function_call():
    user = NeMoGymEasyInputMessage(role="user", content="hi", type="message")
    fc = NeMoGymResponseFunctionToolCall(
        id="call_1",
        call_id="call_1",
        name="get_weather",
        arguments="{}",
        type="function_call",
        status="completed",
    )
    inputs, outputs = split_responses_input_output_items([user, fc])
    assert inputs == [user]
    assert outputs == [fc]


def test_split_on_reasoning():
    user = NeMoGymEasyInputMessage(role="user", content="hi", type="message")
    reasoning = NeMoGymResponseReasoningItem(id="rs_1", type="reasoning", summary=[], status="completed")
    inputs, outputs = split_responses_input_output_items([user, reasoning])
    assert inputs == [user]
    assert outputs == [reasoning]


def test_round_trip_with_tool_calls(converter: ResponsesConverter):
    """A chat message with reasoning + tool calls survives a round trip back to chat params."""
    choice = NeMoGymChoice(
        index=0,
        finish_reason="tool_calls",
        message=NeMoGymChatCompletionMessage(
            role="assistant",
            content="<think>thinking</think>chatting",
            tool_calls=[
                NeMoGymChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=NeMoGymFunction(name="get_weather", arguments='{"city": "nyc"}'),
                )
            ],
        ),
    )
    output_items = converter.postprocess_chat_response(choice)
    params = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(input=[item.model_dump() for item in output_items])
    )
    assistant_msg = params.messages[0]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "<think>thinking</think>chatting"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"
