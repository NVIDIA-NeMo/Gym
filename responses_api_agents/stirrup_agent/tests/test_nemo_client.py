# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit tests for ``DynamicMaxTokensChatCompletionsClient``.

Covers per-call sampling kwargs (``temperature``, ``top_p``,
``enable_thinking``) and the configurable per-call completion cap
(``max_completion_tokens_cap``) — all of which used to be hardcoded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from stirrup.core.models import AssistantMessage, SystemMessage, TokenUsage, ToolCall, ToolMessage, UserMessage

from responses_api_agents.stirrup_agent.nemo_agent import NeMoUserMessage
from responses_api_agents.stirrup_agent.nemo_client import (
    DynamicMaxTokensChatCompletionsClient,
)
from responses_api_agents.stirrup_agent.stirrup_utils import (
    restore_tool_messages_for_model,
    to_provider_openai_messages,
)


def _make_response(content: str = "ok"):
    """Build a fake openai chat.completions response shape."""
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = []
    choice.message.reasoning_content = None
    choice.message.reasoning = None
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.completion_tokens_details = None
    return response


@pytest.mark.asyncio
async def test_generate_forwards_configured_sampling_kwargs() -> None:
    """``temperature``, ``top_p``, ``enable_thinking``, and the cap on
    ``max_completion_tokens`` should land on the wire request_kwargs."""
    client = DynamicMaxTokensChatCompletionsClient(
        model="m",
        max_tokens=10_000,
        base_url="http://test",
        api_key="k",
        temperature=0.42,
        top_p=0.7,
        enable_thinking=False,
        max_completion_tokens_cap=2048,
    )
    fake_create = AsyncMock(return_value=_make_response())
    client._client = MagicMock()
    client._client.chat.completions.create = fake_create

    messages = [SystemMessage(content="sys"), UserMessage(content="hi")]
    await client.generate(messages, tools={})

    fake_create.assert_awaited_once()
    sent = fake_create.await_args.kwargs

    assert sent["temperature"] == pytest.approx(0.42)
    assert sent["top_p"] == pytest.approx(0.7)
    assert sent["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert sent["max_completion_tokens"] <= 2048


@pytest.mark.asyncio
async def test_generate_restores_tool_result_messages_for_openai_payload() -> None:
    """Normal model calls must use provider-valid tool-call history."""
    client = DynamicMaxTokensChatCompletionsClient(
        model="m",
        max_tokens=10_000,
        base_url="http://test",
        api_key="k",
    )
    fake_create = AsyncMock(return_value=_make_response())
    client._client = MagicMock()
    client._client.chat.completions.create = fake_create

    messages = [
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(tool_call_id="call_1", name="code_exec", arguments='{"cmd":"true"}')],
            token_usage=TokenUsage(input=1, answer=1, reasoning=0),
        ),
        NeMoUserMessage(content="ok", name="code_exec", success=True, tool_call_id="call_1"),
    ]

    await client.generate(messages, tools={})

    sent_messages = fake_create.await_args.kwargs["messages"]
    assert sent_messages[0]["role"] == "assistant"
    assert sent_messages[0]["tool_calls"][0]["id"] == "call_1"
    assert sent_messages[1]["role"] == "tool"
    assert sent_messages[1]["tool_call_id"] == "call_1"


def test_provider_openai_messages_convert_nemo_user_messages() -> None:
    """Stirrup serialization helper owns provider-compatible history conversion."""
    messages = [
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(tool_call_id="call_1", name="code_exec", arguments='{"cmd":"true"}')],
            token_usage=TokenUsage(input=1, answer=1, reasoning=0),
        ),
        NeMoUserMessage(content="ok", name="code_exec", success=True, tool_call_id="call_1"),
    ]

    restored = restore_tool_messages_for_model(messages)
    serialized = to_provider_openai_messages(messages)

    assert restored[0] is messages[0]
    assert isinstance(restored[1], ToolMessage)
    assert restored[1].tool_call_id == "call_1"
    assert restored[1].content == "ok"
    assert serialized[0]["role"] == "assistant"
    assert serialized[1]["role"] == "tool"
    assert serialized[1]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_max_completion_tokens_cap_overrides_dynamic_size() -> None:
    """When the dynamic computation exceeds the cap, the cap should win."""
    client = DynamicMaxTokensChatCompletionsClient(
        model="m",
        max_tokens=1_000_000,  # huge context window → dynamic_max would exceed cap
        base_url="http://test",
        api_key="k",
        max_completion_tokens_cap=512,
    )
    fake_create = AsyncMock(return_value=_make_response())
    client._client = MagicMock()
    client._client.chat.completions.create = fake_create

    await client.generate([UserMessage(content="hi")], tools={})

    sent = fake_create.await_args.kwargs
    assert sent["max_completion_tokens"] == 512


def test_defaults_match_pre_lift_behaviour() -> None:
    """Sanity: omitting the new kwargs must keep the historical defaults
    (temperature=1.0, top_p=0.95, enable_thinking=True, cap=64000) so existing
    deployments that don't set the new config fields are unaffected."""
    client = DynamicMaxTokensChatCompletionsClient(
        model="m",
        max_tokens=10_000,
        base_url="http://test",
        api_key="k",
    )
    assert client._temperature == 1.0
    assert client._top_p == 0.95
    assert client._enable_thinking is True
    assert client._max_completion_tokens_cap == 64000
    assert client._truncation_recovery is True


# ---------------------------------------------------------------------------
# Thinking-overrun recovery
# ---------------------------------------------------------------------------


def _make_client(**kwargs):
    client = DynamicMaxTokensChatCompletionsClient(
        model="m",
        max_tokens=200_000,
        base_url="http://test",
        api_key="k",
        max_completion_tokens_cap=64_000,
        **kwargs,
    )
    client._client = MagicMock()
    return client


def _overrun_response():
    """A call that spent its whole budget thinking and never called a tool."""
    response = _make_response(content="<think>" + "reasoning " * 50)
    response.choices[0].finish_reason = "length"
    response.choices[0].message.tool_calls = []
    return response


def _tool_call_response(finish_reason: str = "stop"):
    response = _make_response(content="")
    response.choices[0].finish_reason = finish_reason
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "code_exec"
    tool_call.function.arguments = '{"cmd":"true"}'
    response.choices[0].message.tool_calls = [tool_call]
    return response


@pytest.mark.asyncio
async def test_budget_overrun_without_tool_call_arms_recovery() -> None:
    """The next call drops thinking and carries an explicit instruction."""
    client = _make_client()
    fake_create = AsyncMock(side_effect=[_overrun_response(), _make_response()])
    client._client.chat.completions.create = fake_create

    await client.generate([UserMessage(content="hi")], tools={})
    assert client._recover_from_truncation is True
    assert client._truncation_overruns == 1

    first_sent = fake_create.await_args_list[0].kwargs
    assert first_sent["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert len(first_sent["messages"]) == 1

    await client.generate([UserMessage(content="hi")], tools={})
    recovery_sent = fake_create.await_args_list[1].kwargs

    assert recovery_sent["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert len(recovery_sent["messages"]) == 2
    assert "hit the output token limit" in recovery_sent["messages"][-1]["content"]
    # The budget must NOT be clamped: recovery turns are usually large
    # single-shot deliverable writes (median 34.6k tokens on the measured run),
    # so shrinking it would convert a recovery into a truncated tool call.
    assert recovery_sent["max_completion_tokens"] == 64_000


@pytest.mark.asyncio
async def test_recovery_applies_to_one_turn_only() -> None:
    client = _make_client()
    fake_create = AsyncMock(side_effect=[_overrun_response(), _make_response(), _make_response()])
    client._client.chat.completions.create = fake_create

    for _ in range(3):
        await client.generate([UserMessage(content="hi")], tools={})

    assert fake_create.await_args_list[1].kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    third = fake_create.await_args_list[2].kwargs
    assert third["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert len(third["messages"]) == 1
    assert client._recover_from_truncation is False


@pytest.mark.asyncio
async def test_consecutive_overruns_each_arm_recovery() -> None:
    """The failure is sticky, so re-arming on every overrun is the point."""
    client = _make_client()
    fake_create = AsyncMock(side_effect=[_overrun_response(), _overrun_response(), _make_response()])
    client._client.chat.completions.create = fake_create

    for _ in range(3):
        await client.generate([UserMessage(content="hi")], tools={})

    assert client._truncation_overruns == 2
    for index in (1, 2):
        sent = fake_create.await_args_list[index].kwargs
        assert sent["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_truncated_call_that_still_made_a_tool_call_is_left_alone() -> None:
    """Truncation only matters when nothing was accomplished."""
    client = _make_client()
    fake_create = AsyncMock(side_effect=[_tool_call_response(finish_reason="length"), _make_response()])
    client._client.chat.completions.create = fake_create

    await client.generate([UserMessage(content="hi")], tools={})
    assert client._recover_from_truncation is False
    assert client._truncation_overruns == 0

    await client.generate([UserMessage(content="hi")], tools={})
    second = fake_create.await_args_list[1].kwargs
    assert second["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert len(second["messages"]) == 1


@pytest.mark.asyncio
async def test_text_only_response_that_finished_normally_is_left_alone() -> None:
    """A short text-only turn is a different problem; don't touch it."""
    client = _make_client()
    fake_create = AsyncMock(side_effect=[_make_response(), _make_response()])
    client._client.chat.completions.create = fake_create

    await client.generate([UserMessage(content="hi")], tools={})
    assert client._recover_from_truncation is False

    await client.generate([UserMessage(content="hi")], tools={})
    assert fake_create.await_args_list[1].kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_recovery_can_be_disabled_for_unsteered_rollouts() -> None:
    """RL rollouts that must not be steered opt out; detection still logs."""
    client = _make_client(truncation_recovery=False)
    fake_create = AsyncMock(side_effect=[_overrun_response(), _make_response()])
    client._client.chat.completions.create = fake_create

    await client.generate([UserMessage(content="hi")], tools={})
    assert client._truncation_overruns == 1

    await client.generate([UserMessage(content="hi")], tools={})
    second = fake_create.await_args_list[1].kwargs
    assert second["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert len(second["messages"]) == 1


@pytest.mark.asyncio
async def test_recovery_nudge_never_enters_the_trajectory() -> None:
    """The instruction is a wire-only steer; agent history stays clean."""
    client = _make_client()
    fake_create = AsyncMock(side_effect=[_overrun_response(), _make_response()])
    client._client.chat.completions.create = fake_create

    history = [UserMessage(content="hi")]
    await client.generate(history, tools={})
    await client.generate(history, tools={})

    assert len(history) == 1
    assert history[0].content == "hi"
