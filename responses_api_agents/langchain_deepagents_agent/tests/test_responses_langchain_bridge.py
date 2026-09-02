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

import asyncio
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from responses_api_agents.langchain_deepagents_agent.responses_langchain_bridge import (
    GymResponsesChatModel,
    _request_context,
    _text,
    to_gym_input,
    to_langchain,
    to_langchain_ai_message,
    to_responses,
)


# --- to_langchain / to_responses / _text -----------------------------------------------------------


def test_text_handles_str_and_list_and_other():
    assert _text("hello") == "hello"
    assert _text([{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]) == "ab"
    assert _text(123) == "123"


def test_to_langchain_round_trip_against_typed_items():
    items = [
        NeMoGymEasyInputMessage(role="user", content="hi"),
        NeMoGymEasyInputMessage(role="system", content="be nice"),
    ]
    messages = to_langchain(items)
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage) and messages[0].content == "hi"
    assert isinstance(messages[1], SystemMessage) and messages[1].content == "be nice"


def test_to_langchain_converts_prior_function_call_history_instead_of_dropping_it():
    items = [
        NeMoGymResponseFunctionToolCall(call_id="call_1", name="search", arguments='{"q": "x"}'),
        NeMoGymFunctionCallOutput(call_id="call_1", output="result text"),
    ]
    messages = to_langchain(items)
    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls == [{"name": "search", "args": {"q": "x"}, "id": "call_1", "type": "tool_call"}]
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].content == "result text"
    assert messages[1].tool_call_id == "call_1"


def test_to_langchain_function_call_malformed_args_falls_back_to_empty_dict():
    items = [NeMoGymResponseFunctionToolCall(call_id="call_1", name="search", arguments="not json")]
    (message,) = to_langchain(items)
    assert message.tool_calls[0]["args"] == {}


def test_to_responses_uses_last_ai_message_with_content():
    messages = [AIMessage(content=""), AIMessage(content="final answer")]
    result = to_responses(messages, "policy_model")
    assert result["output"][0]["content"][0]["text"] == "final answer"
    assert result["model"] == "policy_model"


def test_to_responses_preserves_full_tool_call_trace_not_just_final_text():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}]),
        ToolMessage(content="result text", tool_call_id="call_1"),
        AIMessage(content="final answer"),
    ]
    result = to_responses(messages, "policy_model")
    response = NeMoGymResponse.model_validate(result)

    call, call_output, message = response.output
    assert isinstance(call, NeMoGymResponseFunctionToolCall)
    assert call.call_id == "call_1"
    assert call.name == "search"
    assert call.arguments == '{"q": "x"}'
    assert isinstance(call_output, NeMoGymFunctionCallOutput)
    assert call_output.call_id == "call_1"
    assert call_output.output == "result text"
    assert isinstance(message, NeMoGymResponseOutputMessage)
    assert isinstance(message.content[0], NeMoGymResponseOutputText)
    assert message.content[0].text == "final answer"


# --- to_gym_input / to_langchain_ai_message ---------------------------------------------------------


def test_to_gym_input_covers_all_message_kinds():
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="thinking", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}]),
        ToolMessage(content="result text", tool_call_id="call_1"),
    ]
    items = to_gym_input(messages)
    assert items[0] == {"type": "message", "role": "system", "content": "sys"}
    assert items[1] == {"type": "message", "role": "user", "content": "hi"}
    assert items[2] == {"type": "message", "role": "assistant", "content": "thinking"}
    assert items[3] == {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": '{"q": "x"}'}
    assert items[4] == {"type": "function_call_output", "call_id": "call_1", "output": "result text"}


def test_to_langchain_ai_message_extracts_text_and_tool_calls():
    gym_response = NeMoGymResponse(
        id="resp_1",
        created_at=0,
        model="policy_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_1",
                content=[NeMoGymResponseOutputText(text="hello ", annotations=[])],
            ),
            NeMoGymResponseFunctionToolCall(call_id="call_1", name="search", arguments='{"q": "x"}'),
        ],
        parallel_tool_calls=False,
        tools=[],
        tool_choice="auto",
    )
    message = to_langchain_ai_message(gym_response)
    assert message.content == "hello "
    assert message.id == "resp_1"
    assert message.tool_calls == [{"name": "search", "args": {"q": "x"}, "id": "call_1", "type": "tool_call"}]


def test_bind_tools_unnests_chat_completions_shape_into_responses_shape():
    from langchain_core.tools import tool

    @tool
    def search(q: str) -> str:
        """Search the web."""
        return q

    model = GymResponsesChatModel(agent=MagicMock())
    bound = model.bind_tools([search], tool_choice="search")

    bound_kwargs = bound.kwargs
    assert bound_kwargs["tool_choice"] == "search"
    (tool_schema,) = bound_kwargs["tools"]
    assert tool_schema["type"] == "function"
    assert tool_schema["name"] == "search"
    assert "function" not in tool_schema  # un-nested, not Chat-Completions-shaped


# --- contextvar isolation under concurrency (the one unverified piece of the design) -----------------


@pytest.mark.asyncio
async def test_request_context_isolated_across_concurrent_requests():
    seen = []

    async def read_after_delay(value, delay):
        token = _request_context.set(value)
        try:
            await asyncio.sleep(delay)
            seen.append(_request_context.get())
        finally:
            _request_context.reset(token)

    await asyncio.gather(
        read_after_delay({"rollout_id": "r1", "cookies": {"a": "1"}}, 0.02),
        read_after_delay({"rollout_id": "r2", "cookies": {"a": "2"}}, 0.0),
    )

    ids = {entry["rollout_id"] for entry in seen}
    assert ids == {"r1", "r2"}
    for entry in seen:
        assert entry["cookies"]["a"] == entry["rollout_id"][-1]
