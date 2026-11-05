# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compaction hardening for the Gym Stirrup adapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI
from pydantic import BaseModel
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (
    AssistantMessage,
    Reasoning,
    SummaryMessage,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from stirrup.prompts import MESSAGE_SUMMARIZER

import responses_api_agents.stirrup_agent.nemo_agent as nemo_agent_module
from responses_api_agents.stirrup_agent.nemo_agent import (
    NeMoAgent,
    NeMoSessionAgent,
    _is_meaningful_compaction_summary,
    _sanitize_compaction_summary,
)
from responses_api_agents.stirrup_agent.nemo_client import DynamicMaxTokensChatCompletionsClient


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
@pytest.mark.parametrize("envelope", ["error", "detail"])
async def test_http_context_rejection_compacts_after_unwinding_only_the_latest_turn(envelope) -> None:
    """Exercise SDK decoding, the adapter exception, and the existing compaction recovery together."""
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={envelope: {"code": "context_length_exceeded", "message": "rendered prompt is full"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "summary-response",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Keep the first result and continue."},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
        )

    client = DynamicMaxTokensChatCompletionsClient(
        model="test-model",
        api_key="test",
        max_tokens=4096,
        min_completion_tokens=4096,
        max_completion_tokens_cap=4096,
        enable_thinking=True,
        truncation_recovery=False,
    )
    await client._client.close()

    class CodeExecParams(BaseModel):
        cmd: str

    code_exec = Tool(
        name="code_exec",
        description="Run a command.",
        parameters=CodeExecParams,
        executor=lambda _: None,
    )
    history = [
        SystemMessage(content="system"),
        UserMessage(content="original task"),
        AssistantMessage(
            id="first",
            content="first completed turn",
            tool_calls=[ToolCall(tool_call_id="call-1", name="code_exec", arguments='{"cmd":"true"}')],
            token_usage=TokenUsage(input=1, answer=1, reasoning=0),
        ),
        ToolMessage(content="first result", tool_call_id="call-1", name="code_exec", success=True),
        _assistant("latest partial work", message_id="latest"),
    ]
    metadata = {"first": {"code_exec": ["first-result"]}, "latest": {"discarded": []}}
    async with AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as sdk:
        client._client = sdk
        agent = NeMoAgent(client=client, name="test_agent", tools=[code_exec], logger=MagicMock())
        archived, compacted = await agent.summarize_messages(history, metadata)

    assert archived == history[:4]
    assert len(history) == 5
    assert metadata == {"first": {"code_exec": ["first-result"]}}
    assert len(requests) == 2
    assert requests[0]["messages"][-2]["content"] == [{"type": "text", "text": "latest partial work"}]
    assert requests[1]["messages"][-2]["content"] == [{"type": "text", "text": "first result"}]
    assert requests[1]["messages"][-3]["tool_calls"][0]["id"] == "call-1"
    assert requests[1]["messages"][-2]["tool_call_id"] == "call-1"
    assert any(tool["function"]["name"] == "code_exec" for tool in requests[1]["tools"])
    assert all(request["max_completion_tokens"] == 4096 for request in requests)
    assert all(request["chat_template_kwargs"]["enable_thinking"] for request in requests)
    assert compacted[:2] == history[:2]
    summary = next(message for message in compacted if isinstance(message, SummaryMessage))
    assert "Keep the first result and continue." in summary.content


@pytest.mark.asyncio
@pytest.mark.parametrize("summarized", [False, True])
async def test_http_context_recovery_cannot_discard_the_only_turn_after_a_progress_boundary(summarized) -> None:
    client = DynamicMaxTokensChatCompletionsClient(model="test-model", api_key="test", truncation_recovery=False)
    await client._client.close()
    error = APIStatusError(
        "rendered prompt is full",
        response=httpx.Response(400, request=httpx.Request("POST", "http://test/chat/completions")),
        body={"detail": {"code": "context_length_exceeded"}},
    )
    client._client = MagicMock()
    client._client.chat.completions.create = AsyncMock(side_effect=error)
    history = [SystemMessage(content="system"), UserMessage(content="original task")]
    if summarized:
        history.extend([SummaryMessage(content="earlier progress"), UserMessage(content="Got it, thanks!")])
    history.append(_assistant("only completed turn", message_id="only"))
    before = list(history)
    metadata = {"only": {"work": []}}
    agent = _agent(client)

    with pytest.raises(ContextOverflowError, match="summarized context" if summarized else "original prompt"):
        await agent.summarize_messages(history, metadata)

    assert history == before
    assert metadata == {"only": {"work": []}}
    assert client._client.chat.completions.create.await_count == 1


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


@pytest.mark.asyncio
async def test_summary_exhaustion_preserves_original_history_and_safe_attempt_diagnostics() -> None:
    """A terminal compaction must not lose the current turn or expose response text in diagnostics."""
    history = _history()
    history.extend([_assistant("UNWOUND_PRIVATE_PROGRESS", message_id="assistant-2"), UserMessage(content="continue")])
    secret = "DO_NOT_EXPORT_PRIVATE_REASONING"
    client = _FakeClient(
        ContextOverflowError("too large"),
        AssistantMessage(content="", reasoning=Reasoning(content=secret)),
        _assistant(f"<think>{secret}</think>"),
        _assistant("   "),
    )
    agent = _agent(client)
    metadata = {"assistant-1": {"first": []}, "assistant-2": {"discarded": []}}

    with pytest.raises(RuntimeError, match="usable text after 3 attempts") as caught:
        await agent.summarize_messages(history, metadata)

    error_type = getattr(nemo_agent_module, "CompactionSummaryError", None)
    assert error_type is not None and isinstance(caught.value, error_type)
    assert caught.value.messages == history
    assert caught.value.messages is not history
    assert len(client.calls) == 4
    assert "assistant-2" not in metadata
    assert [event["event"] for event in agent.compaction_events] == [
        "context_overflow",
        "overflow_unwind",
        "summary_rejected",
        "summary_rejected",
        "summary_rejected",
    ]
    assert agent.compaction_events[1]["dropped_message_count"] == 2
    attempts = caught.value.attempts
    assert [attempt["attempt"] for attempt in attempts] == [1, 2, 3]
    assert [attempt["rejection_reason"] for attempt in attempts] == ["empty_text", "reasoning_only", "empty_text"]
    assert attempts[0]["reasoning_present"] is True
    assert attempts[0]["reasoning_chars"] == len(secret)
    assert attempts[1]["content_chars"] == len(secret) + 15
    assert all(attempt["sanitized_chars"] == 0 for attempt in attempts)
    assert all(attempt["tool_calls_count"] == 0 for attempt in attempts)
    assert attempts[1]["usage"] == {"input": 1, "answer": 1, "reasoning": 0}
    exported = json.dumps({"attempts": attempts, "events": agent.compaction_events})
    assert secret not in exported
    assert "UNWOUND_PRIVATE_PROGRESS" not in exported
    assert "<think>" not in exported


@pytest.mark.asyncio
async def test_summary_acceptance_records_sanitized_metadata_without_changing_replay() -> None:
    client = _FakeClient(_assistant("<think>PRIVATE_THINKING</think> kept result"))
    agent = _agent(client)

    _, compacted = await agent.summarize_messages(_history(), {})

    event = agent.compaction_events[-1]
    assert event["event"] == "summary_accepted"
    assert event["attempt"] == 1
    assert event["sanitized_chars"] == 11
    assert event["sanitized_words"] == 2
    assert event["rejection_reason"] is None
    assert "PRIVATE_THINKING" not in json.dumps(event)
    assert "kept result" not in json.dumps(event)
    assert "kept result" in next(message.content for message in compacted if isinstance(message, SummaryMessage))


@pytest.mark.asyncio
async def test_context_recovery_boundary_is_recorded_without_dropping_protected_history() -> None:
    history = _history()
    agent = _agent(_FakeClient(ContextOverflowError("PRIVATE_PROVIDER_DIAGNOSTIC")))

    with pytest.raises(ContextOverflowError):
        await agent.summarize_messages(history, {})

    assert [event["event"] for event in agent.compaction_events] == ["context_overflow", "overflow_unwind_blocked"]
    assert agent.compaction_events[-1]["history_message_count"] == 4
    assert "PRIVATE_PROVIDER_DIAGNOSTIC" not in json.dumps(agent.compaction_events)


@pytest.mark.asyncio
async def test_session_compaction_events_remain_available_after_run() -> None:
    client = _FakeClient(
        AssistantMessage(content="first policy turn", token_usage=TokenUsage(input=190_000, answer=10, reasoning=0)),
        _assistant("working summary"),
        _assistant("second policy turn"),
        AssistantMessage(content="next session turn", token_usage=TokenUsage(input=190_000, answer=10, reasoning=0)),
        _assistant("next session summary"),
        _assistant("last policy turn"),
    )
    agent = NeMoAgent(
        client=client, name="test_agent", tools=[], logger=MagicMock(), max_turns=2, context_summarization_cutoff=0.7
    )

    for task in ("original task", "another task"):
        async with agent.session(cache_on_interrupt=False) as session:
            await session.run(task)
            assert [event["event"] for event in session.compaction_events] == ["summary_accepted"]
