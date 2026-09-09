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
"""Compaction hardening for the Gym Stirrup adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (
    AssistantMessage,
    Reasoning,
    SummaryMessage,
    SystemMessage,
    TokenUsage,
    ToolMessage,
    UserMessage,
)
from stirrup.prompts import MESSAGE_SUMMARIZER

from responses_api_agents.stirrup_agent.nemo_agent import (
    NeMoAgent,
    NeMoSessionAgent,
    _is_meaningful_compaction_summary,
    _sanitize_compaction_summary,
)


def _words(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


def _assistant(content: str, *, message_id: str = "assistant-1") -> AssistantMessage:
    return AssistantMessage(
        id=message_id,
        content=content,
        token_usage=TokenUsage(input=1, answer=1, reasoning=0),
    )


class _FakeClient:
    model_slug = "fake-model"
    max_tokens = 262_144

    def __init__(self, *results: AssistantMessage | Exception) -> None:
        self.results = list(results)
        self.calls: list[tuple[list, dict]] = []

    async def generate(self, messages, tools):
        self.calls.append((messages, tools))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _RecoveryAwareFakeClient(_FakeClient):
    def __init__(
        self,
        *results: AssistantMessage | Exception,
        recovery: bool,
        overruns: int,
        mutate_during_generate: bool = False,
    ) -> None:
        super().__init__(*results)
        self._recover_from_truncation = recovery
        self._truncation_overruns = overruns
        self.mutate_during_generate = mutate_during_generate
        self.recovery_seen: list[bool] = []

    async def generate(self, messages, tools):
        self.recovery_seen.append(self._recover_from_truncation)
        if self.mutate_during_generate:
            self._recover_from_truncation = True
            self._truncation_overruns += 1
        return await super().generate(messages, tools)


def _agent(
    client: _FakeClient,
    logger: MagicMock | None = None,
    *,
    min_words: int = 1,
) -> NeMoAgent:
    return NeMoAgent(
        client=client,
        name="test_agent",
        tools=[],
        logger=logger or MagicMock(),
        min_compaction_summary_words=min_words,
    )


def _history() -> list:
    return [
        SystemMessage(content="system"),
        UserMessage(content="original task"),
        _assistant("worked", message_id="assistant-1"),
        ToolMessage(content="tool result", tool_call_id="call-1", name="code_exec", success=True),
    ]


def test_sanitize_compaction_summary_removes_only_leading_think_blocks() -> None:
    summary = "  <THINK>private reasoning</THINK>\n\n" + _words(50) + " literal <think> tag"

    sanitized = _sanitize_compaction_summary(summary)

    assert sanitized.startswith("word0 word1")
    assert "private reasoning" not in sanitized
    assert sanitized.endswith("literal <think> tag")


@pytest.mark.parametrize(
    "summary",
    [
        "",
        "   ",
        "<think>reasoning without a closing tag",
        "<think>reasoning</think>\n\n",
        "<think>first</think> <think>second but unclosed",
    ],
)
def test_sanitize_compaction_summary_rejects_empty_or_unclosed_think(summary: str) -> None:
    assert _sanitize_compaction_summary(summary) == ""


def test_meaningful_compaction_summary_uses_configured_minimum() -> None:
    assert not _is_meaningful_compaction_summary(_words(49), min_words=50)
    assert _is_meaningful_compaction_summary(_words(50), min_words=50)


@pytest.mark.parametrize("summary", ["Done.", "已完成并保留所有关键进展。"])
def test_generic_compaction_accepts_concise_and_cjk_summaries(summary: str) -> None:
    assert _is_meaningful_compaction_summary(summary)


def test_compaction_minimum_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _agent(_FakeClient(), min_words=0)


def test_session_agent_uses_the_hardened_compaction_override() -> None:
    assert NeMoSessionAgent.summarize_messages is NeMoAgent.summarize_messages


@pytest.mark.asyncio
async def test_summarization_retries_degenerate_outputs_and_persists_only_sanitized_text() -> None:
    good_summary = _words(60)
    client = _FakeClient(
        _assistant("<think>reasoning only"),
        _assistant("too short to preserve a long conversation"),
        AssistantMessage(
            content=f"<think>private chain of thought</think>\n{good_summary}",
            reasoning=Reasoning(content="separate private reasoning"),
        ),
    )
    logger = MagicMock()
    agent = _agent(client, logger, min_words=50)
    history = _history()

    archived, compacted = await agent.summarize_messages(history, {"assistant-1": {}})

    assert archived == history
    assert len(client.calls) == 3
    assert client.calls[0][0][-1].content == MESSAGE_SUMMARIZER
    assert client.calls[0][1]
    assert "text only" in client.calls[1][0][-1].content.lower()
    assert client.calls[1][1]
    assert client.calls[2][1] == {}

    summaries = [message for message in compacted if isinstance(message, SummaryMessage)]
    assert len(summaries) == 1
    persisted = summaries[0].content
    assert good_summary in persisted
    assert "<think>" not in persisted.lower()
    assert "private" not in persisted.lower()
    assert compacted[-1] == UserMessage(content="Got it, thanks!")

    logger.context_summarization_complete.assert_called_once()
    logged_summary, logged_bridge = logger.context_summarization_complete.call_args.args
    assert logged_summary == good_summary
    assert "<think>" not in logged_bridge.lower()


@pytest.mark.asyncio
async def test_summarization_fails_closed_after_three_degenerate_outputs() -> None:
    client = _FakeClient(
        _assistant("<think>one</think>"),
        _assistant(_words(10)),
        _assistant("<think>unclosed"),
    )
    logger = MagicMock()
    agent = _agent(client, logger, min_words=50)
    metadata = {"assistant-1": {"code_exec": []}}

    with pytest.raises(RuntimeError, match="usable text after 3 attempts"):
        await agent.summarize_messages(_history(), metadata)

    assert len(client.calls) == 3
    assert metadata == {"assistant-1": {"code_exec": []}}
    logger.context_summarization_complete.assert_not_called()


@pytest.mark.asyncio
async def test_summarization_preserves_context_overflow_unwind_semantics() -> None:
    history = [
        SystemMessage(content="system"),
        UserMessage(content="original task"),
        _assistant("first turn", message_id="assistant-1"),
        ToolMessage(content="first result", tool_call_id="call-1", name="code_exec", success=True),
        _assistant("second turn", message_id="assistant-2"),
        ToolMessage(content="second result", tool_call_id="call-2", name="code_exec", success=True),
    ]
    client = _FakeClient(ContextOverflowError("too large"), _assistant(_words(60)))
    agent = _agent(client, min_words=50)
    metadata = {"assistant-1": {"first": []}, "assistant-2": {"second": []}}

    archived, compacted = await agent.summarize_messages(history, metadata)

    assert archived == history[:4]
    assert "assistant-1" in metadata
    assert "assistant-2" not in metadata
    assert any(isinstance(message, SummaryMessage) for message in compacted)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_summary_calls_do_not_consume_an_armed_policy_recovery() -> None:
    client = _RecoveryAwareFakeClient(_assistant(_words(60)), recovery=True, overruns=7)
    agent = _agent(client, min_words=50)

    await agent.summarize_messages(_history(), {})

    assert client.recovery_seen == [False]
    assert client._recover_from_truncation is True
    assert client._truncation_overruns == 7


@pytest.mark.asyncio
async def test_summary_calls_do_not_arm_policy_recovery_or_increment_overruns() -> None:
    client = _RecoveryAwareFakeClient(
        _assistant("too short"),
        _assistant("still too short"),
        _assistant(_words(60)),
        recovery=False,
        overruns=2,
        mutate_during_generate=True,
    )
    agent = _agent(client, min_words=50)

    await agent.summarize_messages(_history(), {})

    assert client.recovery_seen == [False, False, False]
    assert client._recover_from_truncation is False
    assert client._truncation_overruns == 2


@pytest.mark.asyncio
async def test_session_run_replays_only_the_sanitized_summary_after_compaction() -> None:
    good_summary = _words(60)
    client = _FakeClient(
        AssistantMessage(
            content="first policy turn",
            token_usage=TokenUsage(input=190_000, answer=10, reasoning=0),
        ),
        _assistant(f"<think>private summary reasoning</think>\n{good_summary}"),
        _assistant("second policy turn"),
    )
    agent = NeMoAgent(
        client=client,
        name="test_agent",
        tools=[],
        logger=MagicMock(),
        max_turns=2,
        context_summarization_cutoff=0.7,
        min_compaction_summary_words=50,
    )

    async with agent.session(cache_on_interrupt=False) as session:
        assert isinstance(session, NeMoSessionAgent)
        _, history, _ = await session.run("original task")

    assert len(history) == 2
    assert any(isinstance(message, SummaryMessage) for message in history[1])
    second_policy_messages = client.calls[2][0]
    serialized_context = "\n".join(
        message.content for message in second_policy_messages if isinstance(message.content, str)
    )
    assert good_summary in serialized_context
    assert "private summary reasoning" not in serialized_context
    assert "<think>" not in serialized_context.lower()
