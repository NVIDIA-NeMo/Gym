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
