# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A turn that exhausts max_output_tokens must stay a turn, not end the rollout."""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from responses_api_agents.apex_agent import stirrup_runtime


pytest.importorskip("stirrup")

from stirrup.clients.chat_completions_client import ChatCompletionsClient  # noqa: E402
from stirrup.core.models import SystemMessage, UserMessage  # noqa: E402


def _response(finish_reason: str = "stop", content: str = "partial answer", tool_calls=None, reasoning=None):
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.message.reasoning_content = reasoning
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 7
    response.usage.completion_tokens_details = MagicMock(reasoning_tokens=3)
    return response


def _tool_call(name: str = "finish"):
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function = MagicMock()
    tool_call.function.name = name
    tool_call.function.arguments = '{"final_answer": "x", "status": "completed"}'
    return tool_call


def _client(*, kwargs=None, **client_kwargs):
    client_class = stirrup_runtime.make_length_tolerant_client_class(ChatCompletionsClient)
    client = client_class(
        model="m",
        base_url="http://test",
        api_key="k",
        max_tokens=4096,
        kwargs={"temperature": 1.0, "top_p": 1.0} if kwargs is None else kwargs,
        **client_kwargs,
    )
    client._client = MagicMock()
    return client


MESSAGES = [SystemMessage(content="sys"), UserMessage(content="do the task")]


def _text(sent_message) -> str:
    """Stirrup serialises user content as text blocks; compare the joined text."""
    content = sent_message["content"]
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content)


@pytest.mark.asyncio
async def test_length_finish_returns_the_truncated_turn_instead_of_raising() -> None:
    client = _client()
    original_openai_client = client._client
    client._client.chat.completions.create = AsyncMock(return_value=_response("length", reasoning="thinking..."))

    message = await client.generate(MESSAGES, tools={})

    assert message.content == "partial answer"
    assert message.reasoning is not None and message.reasoning.content == "thinking..."
    assert message.tool_calls == []
    assert (message.token_usage.input, message.token_usage.reasoning, message.token_usage.answer) == (10, 3, 4)
    assert client.length_truncations == 1
    assert client._client is original_openai_client


@pytest.mark.asyncio
async def test_max_tokens_finish_reason_is_treated_like_length() -> None:
    client = _client()
    client._client.chat.completions.create = AsyncMock(return_value=_response("max_tokens"))

    message = await client.generate(MESSAGES, tools={})

    assert message.content == "partial answer"
    assert client.length_truncations == 1


@pytest.mark.asyncio
async def test_stop_finish_is_untouched_and_arms_nothing() -> None:
    client = _client()
    fake_create = AsyncMock(return_value=_response("stop"))
    client._client.chat.completions.create = fake_create

    await client.generate(MESSAGES, tools={})
    await client.generate(MESSAGES, tools={})

    assert client.length_truncations == 0
    assert client.recovery_turns == 0
    for call in fake_create.await_args_list:
        sent = call.kwargs
        assert _text(sent["messages"][-1]) == "do the task"
        assert "extra_body" not in sent
        assert sent["max_completion_tokens"] == 4096


@pytest.mark.asyncio
async def test_recovery_turn_adds_a_transient_notice_and_disables_thinking() -> None:
    client = _client()
    fake_create = AsyncMock(side_effect=[_response("length"), _response("stop"), _response("stop")])
    client._client.chat.completions.create = fake_create
    original_kwargs = dict(client._kwargs)
    messages = list(MESSAGES)

    await client.generate(messages, tools={})
    await client.generate(messages, tools={})
    await client.generate(messages, tools={})

    first, recovery, after = (call.kwargs for call in fake_create.await_args_list)
    assert "extra_body" not in first
    assert recovery["messages"][-1]["role"] == "user"
    assert _text(recovery["messages"][-1]) == stirrup_runtime.LENGTH_TRUNCATION_RECOVERY_NUDGE
    assert _text(recovery["messages"][-2]) == "do the task"
    assert recovery["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert recovery["temperature"] == 1.0
    # The notice is transient: it is neither kept in the caller's history nor repeated.
    assert len(messages) == len(MESSAGES)
    assert _text(after["messages"][-1]) == "do the task"
    assert "extra_body" not in after
    assert client._kwargs == original_kwargs
    assert client.recovery_turns == 1


@pytest.mark.asyncio
async def test_recovery_preserves_other_extra_body_settings() -> None:
    client = _client(kwargs={"extra_body": {"chat_template_kwargs": {"foo": 1}, "other": 2}})
    fake_create = AsyncMock(side_effect=[_response("length"), _response("stop")])
    client._client.chat.completions.create = fake_create

    await client.generate(MESSAGES, tools={})
    await client.generate(MESSAGES, tools={})

    recovery = fake_create.await_args_list[1].kwargs
    assert recovery["extra_body"] == {"chat_template_kwargs": {"foo": 1, "enable_thinking": False}, "other": 2}
    assert client._kwargs == {"extra_body": {"chat_template_kwargs": {"foo": 1}, "other": 2}}


@pytest.mark.asyncio
async def test_length_finish_with_a_tool_call_counts_but_does_not_arm_recovery() -> None:
    client = _client()
    fake_create = AsyncMock(side_effect=[_response("length", tool_calls=[_tool_call()]), _response("stop")])
    client._client.chat.completions.create = fake_create

    message = await client.generate(MESSAGES, tools={})
    await client.generate(MESSAGES, tools={})

    assert [tool_call.name for tool_call in message.tool_calls] == ["finish"]
    assert client.length_truncations == 1
    assert client.recovery_turns == 0
    second = fake_create.await_args_list[1].kwargs
    assert _text(second["messages"][-1]) == "do the task"
    assert "extra_body" not in second


@pytest.mark.asyncio
async def test_truncation_recovery_can_be_switched_off() -> None:
    client = _client(truncation_recovery=False)
    fake_create = AsyncMock(side_effect=[_response("length"), _response("stop")])
    client._client.chat.completions.create = fake_create

    await client.generate(MESSAGES, tools={})
    await client.generate(MESSAGES, tools={})

    assert client.length_truncations == 1
    assert client.recovery_turns == 0
    second = fake_create.await_args_list[1].kwargs
    assert _text(second["messages"][-1]) == "do the task"
    assert "extra_body" not in second


@pytest.mark.asyncio
async def test_client_and_kwargs_are_restored_when_the_request_raises() -> None:
    client = _client()
    original_openai_client = client._client
    original_kwargs = client._kwargs
    client._recover_from_truncation = True
    client._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await client.generate(MESSAGES, tools={})

    assert client._client is original_openai_client
    assert client._kwargs is original_kwargs
    assert client.length_truncations == 0


def test_default_client_construction_keeps_stock_behaviour_switches() -> None:
    client = _client()

    assert client.truncation_recovery is True
    assert client.length_truncations == 0
    assert client.recovery_turns == 0
    assert isinstance(client, ChatCompletionsClient)
    assert type(client).__name__ == "LengthTolerantChatCompletionsClient"


def test_run_stirrup_rollout_builds_the_length_tolerant_client() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(stirrup_runtime.run_stirrup_rollout)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]

    factory_calls = [call for call in calls if call.func.id == "make_length_tolerant_client_class"]
    assert len(factory_calls) == 1
    assert isinstance(factory_calls[0].args[0], ast.Name)
    assert factory_calls[0].args[0].id == "ChatCompletionsClient"

    # The stock class must not be instantiated directly any more.
    assert not [call for call in calls if call.func.id == "ChatCompletionsClient"]

    constructions = [call for call in calls if any(keyword.arg == "truncation_recovery" for keyword in call.keywords)]
    assert len(constructions) == 1
    assert any(keyword.arg == "max_tokens" for keyword in constructions[0].keywords)

    result_keys = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant)
    }
    assert {"n_length_truncations", "n_truncation_recovery_turns"} <= result_keys


def _finish_call(reason: str = "done"):
    tool_call = MagicMock()
    tool_call.id = "call_finish"
    tool_call.function = MagicMock()
    tool_call.function.name = "finish"
    tool_call.function.arguments = f'{{"reason": "{reason}", "paths": []}}'
    return tool_call


@pytest.mark.asyncio
async def test_stock_client_ends_the_rollout_on_a_length_finish() -> None:
    """Control: without the wrapper, the same exchange surfaces as a context overflow."""
    from stirrup import Agent
    from stirrup.core.exceptions import ContextOverflowError

    client = ChatCompletionsClient(model="m", base_url="http://test", api_key="k", max_tokens=4096)
    client._client = MagicMock()
    client._client.chat.completions.create = AsyncMock(
        side_effect=[_response("length", content="long deliberation"), _response("stop", tool_calls=[_finish_call()])]
    )
    agent = Agent(client=client, name="control", max_turns=4, system_prompt="sys")

    with pytest.raises(ContextOverflowError):
        async with agent.session() as session:
            await session.run("do the task")


@pytest.mark.asyncio
async def test_agent_loop_recovers_from_a_length_finish_and_finishes() -> None:
    from stirrup import Agent

    client = _client()
    fake_create = AsyncMock(
        side_effect=[_response("length", content="long deliberation"), _response("stop", tool_calls=[_finish_call()])]
    )
    client._client.chat.completions.create = fake_create
    agent = Agent(client=client, name="apex_test", max_turns=4, system_prompt="sys")

    async with agent.session() as session:
        finish_params, history, _metadata = await session.run("do the task")

    assert finish_params is not None and finish_params.reason == "done"
    assert client.length_truncations == 1
    assert client.recovery_turns == 1
    recovery = fake_create.await_args_list[1].kwargs
    assert _text(recovery["messages"][-1]) == stirrup_runtime.LENGTH_TRUNCATION_RECOVERY_NUDGE
    assert recovery["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    # The truncated turn is kept in the history; the recovery notice is not.
    flat = [message for turn in history for message in turn]
    assistant_contents = [
        getattr(message, "content", "") for message in flat if type(message).__name__ == "AssistantMessage"
    ]
    assert "long deliberation" in assistant_contents
    assert not any("SYSTEM NOTICE" in str(getattr(message, "content", "")) for message in flat)
