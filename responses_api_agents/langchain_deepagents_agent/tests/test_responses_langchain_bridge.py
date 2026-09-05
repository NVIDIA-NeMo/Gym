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
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.runnables.config import var_child_runnable_config

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
    _text,
    to_gym_input,
    to_langchain,
    to_langchain_ai_message,
    to_responses,
)


def _fake_model_response(text: str = "ok", cookies: dict | None = None):
    """A minimal stand-in for aiohttp's ClientResponse, shaped like server_client.post()'s return value —
    enough for raise_for_status()/get_response_json() to succeed against a real NeMoGymResponse body."""
    resp = MagicMock()
    resp.ok = True
    resp.cookies = cookies or {}
    resp.read = AsyncMock(return_value=orjson.dumps(to_responses([AIMessage(content=text)], "test-model")))
    return resp


def _fake_tool_call_response(name: str, args: dict, call_id: str = "call_1"):
    """Like `_fake_model_response()`, but the model "decides" to call a tool instead of answering —
    used to force the `task` tool so a real deepagents subagent invocation actually happens."""
    resp = MagicMock()
    resp.ok = True
    resp.cookies = {}
    message = AIMessage(content="", tool_calls=[tool_call(name=name, args=args, id=call_id)])
    resp.read = AsyncMock(return_value=orjson.dumps(to_responses([message], "test-model")))
    return resp


def _make_model(post: AsyncMock) -> GymResponsesChatModel:
    agent = MagicMock()
    agent.server_client.post = post
    agent.config.model_server.name = "policy_model"
    return GymResponsesChatModel(agent=agent)


@contextmanager
def _ambient_config(config: RunnableConfig):
    """Set the ambient `RunnableConfig` the way a real async call chain needs: `var_child_runnable_config`
    set directly, so it stays visible across `await` points within this task. Verified against a real
    `deepagents.create_deep_agent()` graph run that this is what actually reaches `_agenerate()` in
    practice — `langchain_core.runnables.config.set_config_context` is a different tool: it wraps a
    `contextvars.Context.run()` call, which only carries the value for a single *synchronous* callable, not
    an `await`ed block, so it doesn't apply here."""
    token = var_child_runnable_config.set(config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


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


# --- RunnableConfig propagation into _agenerate() ------------------------------------------------------


@pytest.mark.asyncio
async def test_agenerate_reads_model_url_path_and_cookies_from_ambient_config():
    post = AsyncMock(return_value=_fake_model_response())
    model = _make_model(post)
    config: RunnableConfig = {
        "configurable": {"model_url_path": "/v1/responses", "model_cookies": {"cookies": {"sid": "seed"}}}
    }

    with _ambient_config(config):
        await model._agenerate([HumanMessage(content="hi")])

    assert post.call_args.kwargs["url_path"] == "/v1/responses"
    assert post.call_args.kwargs["cookies"] == {"sid": "seed"}


@pytest.mark.asyncio
async def test_agenerate_raises_clear_error_without_ambient_config():
    model = _make_model(AsyncMock())
    with pytest.raises(RuntimeError, match="model_url_path"):
        await model._agenerate([HumanMessage(content="hi")])


@pytest.mark.asyncio
async def test_agenerate_chains_model_cookies_across_sequential_calls_without_mutating_shared_config():
    post = AsyncMock(
        side_effect=[_fake_model_response(cookies={"sid": "turn1"}), _fake_model_response(cookies={"sid": "turn2"})]
    )
    model = _make_model(post)
    cookie_holder = {"cookies": None}
    config: RunnableConfig = {"configurable": {"model_url_path": "/v1/responses", "model_cookies": cookie_holder}}

    with _ambient_config(config):
        # First call: nothing chained yet (matches DeepAgentsAgent.responses() seeding `cookies: None` —
        # never the resources-server's own session cookies).
        await model._agenerate([HumanMessage(content="hi")])
        assert post.call_args.kwargs["cookies"] is None

        # Second call in the same rollout: picks up the first response's cookies.
        await model._agenerate([HumanMessage(content="hi again")])
        assert post.call_args.kwargs["cookies"] == {"sid": "turn1"}

    # The mutation landed on the holder object referenced from `configurable`, not a replacement dict —
    # this is what makes it visible to the next call without re-`.set()`ing the whole config.
    assert cookie_holder["cookies"] == {"sid": "turn2"}


@pytest.mark.asyncio
async def test_agenerate_propagates_into_a_nested_runnable_call():
    """Proves the ambient-config mechanism actually reaches a call made from inside another Runnable —
    the shape of a deepagents subagent invoking its own nested model call — not just a direct call."""
    from langchain_core.runnables import RunnableLambda

    post = AsyncMock(return_value=_fake_model_response())
    model = _make_model(post)

    async def inner(_):
        return await model._agenerate([HumanMessage(content="nested")])

    nested = RunnableLambda(inner)
    config: RunnableConfig = {
        "configurable": {"model_url_path": "/nested/v1/responses", "model_cookies": {"cookies": None}}
    }

    await nested.ainvoke({}, config=config)

    assert post.call_args.kwargs["url_path"] == "/nested/v1/responses"


@pytest.mark.asyncio
async def test_task_tool_propagates_ambient_config_to_subagent_model_call():
    """Proves the ambient-config mechanism reaches a real deepagents subagent invocation, not just a bare
    `RunnableLambda` (see the test above) — deepagents' own `task`/`atask` tool
    (deepagents/middleware/subagents.py) is what production actually routes nested calls through, and it
    only stamps `{"configurable": {"ls_agent_type": "subagent"}}` explicitly onto the subagent's config,
    relying on LangGraph's ambient-config merge for `model_url_path`/`model_cookies` to reach it."""
    from deepagents import create_deep_agent

    post = AsyncMock(
        side_effect=[
            # Main agent delegates to the subagent instead of answering directly.
            _fake_tool_call_response("task", {"description": "look something up", "subagent_type": "web-searcher"}),
            # Subagent's own model call, made from inside atask() — no tool calls, so its loop ends here.
            _fake_model_response(text="found it"),
            # Main agent's follow-up call once the subagent's result comes back as a tool message.
            _fake_model_response(text="final answer"),
        ]
    )
    model = _make_model(post)
    agent = create_deep_agent(
        model=model,
        tools=[],
        subagents=[
            {
                "name": "web-searcher",
                "description": "Delegate any lookup to this subagent.",
                "system_prompt": "Look things up and report back.",
                "tools": [],
            }
        ],
    )
    config: RunnableConfig = {"configurable": {"model_url_path": "/v1/responses", "model_cookies": {"cookies": None}}}

    await agent.ainvoke({"messages": [HumanMessage(content="look something up")]}, config=config)

    # All three calls happened at all: if config propagation were broken, the subagent's model call
    # would hit GymResponsesChatModel._agenerate()'s RuntimeError guard instead of reaching `post`.
    assert post.call_count == 3
    for call in post.call_args_list:
        assert call.kwargs["url_path"] == "/v1/responses"


@pytest.mark.asyncio
async def test_agenerate_isolated_across_concurrent_top_level_configs():
    """Two staggered concurrent rollouts must never see each other's model_url_path/cookies — the
    property the old ContextVar-based test checked, now verified against RunnableConfig instead."""
    seen = []

    async def run_with_config(rollout_suffix: str, delay: float):
        post = AsyncMock(return_value=_fake_model_response())
        model = _make_model(post)
        config: RunnableConfig = {
            "configurable": {
                "model_url_path": f"/rollout-{rollout_suffix}/v1/responses",
                "model_cookies": {"cookies": {"a": rollout_suffix}},
            }
        }
        with _ambient_config(config):
            await asyncio.sleep(delay)
            await model._agenerate([HumanMessage(content="hi")])
        seen.append((rollout_suffix, post.call_args.kwargs["url_path"], post.call_args.kwargs["cookies"]))

    await asyncio.gather(run_with_config("r1", 0.02), run_with_config("r2", 0.0))

    for suffix, url_path, cookies in seen:
        assert url_path == f"/rollout-{suffix}/v1/responses"
        assert cookies == {"a": suffix}


def test_ensure_config_with_no_ambient_context_has_no_configurable():
    # Sanity check underpinning the RuntimeError test above: outside any `set_config_context`, there's no
    # ambient RunnableConfig to accidentally read stale values from.
    assert not ensure_config().get("configurable")
