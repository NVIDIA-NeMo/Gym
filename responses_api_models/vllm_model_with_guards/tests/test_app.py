# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import SESSION_ID_KEY, NeMoGymAsyncOpenAI, ServerClient
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
)
from responses_api_models.vllm_model_with_guards.app import VLLMModelWithGuards, VLLMModelWithGuardsConfig


def _make_model(**config_overrides: Any) -> VLLMModelWithGuards:
    config = VLLMModelWithGuardsConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="policy_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
        **config_overrides,
    )
    return VLLMModelWithGuards(
        config=config,
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )


def _messages(*commands: str) -> list[dict]:
    """A conversation in which each assistant turn made one bash call with the given command."""
    messages: list[dict] = [{"role": "user", "content": "go"}]
    for i, command in enumerate(commands):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{i}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call-{i}", "content": "ok"})
    return messages


def _custom_messages(*calls: tuple[str, str]) -> list[dict]:
    """A conversation in which each assistant turn made one custom tool call, given as (name, input)."""
    messages: list[dict] = [{"role": "user", "content": "go"}]
    for i, (name, tool_input) in enumerate(calls):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"call-{i}", "type": "custom", "custom": {"name": name, "input": tool_input}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call-{i}", "content": "ok"})
    return messages


async def _chat(messages: list[dict], **guards: Any) -> tuple[NeMoGymChatCompletion, AsyncMock]:
    server = _make_model(**guards)
    upstream = NeMoGymChatCompletion(
        id="chtcmpl-up",
        object="chat.completion",
        created=1691418000,
        model="dummy_model",
        choices=[
            NeMoGymChoice(
                index=0,
                finish_reason="stop",
                message=NeMoGymChatCompletionMessage(role="assistant", content="from the engine", tool_calls=[]),
            )
        ],
    )
    client = MagicMock(spec=NeMoGymAsyncOpenAI)
    client.create_chat_completion = AsyncMock(return_value=upstream.model_dump())
    server._clients = [client]
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "session-1"}
    request.headers = {}
    result = await server.chat_completions(request, NeMoGymChatCompletionCreateParamsNonStreaming(messages=messages))
    return result, client.create_chat_completion


def test_rejects_responses_native() -> None:
    with pytest.raises(ValueError, match="does not support is_responses_native"):
        _make_model(is_responses_native=True)


@pytest.mark.parametrize(
    ("commands", "expect_none"), [(("a", "b"), False), (("a", "b", "c"), True), (("a", "b", "c", "d"), True)]
)
async def test_text_only_guard_forces_tool_choice_none_from_the_limit_on(
    commands: tuple[str, ...], expect_none: bool
) -> None:
    result, upstream = await _chat(_messages(*commands), guard_text_only_after_assistant_turns=3)

    assert result.choices[0].message.content == "from the engine"
    assert (upstream.await_args.kwargs.get("tool_choice") == "none") is expect_none


async def test_turn_limit_ends_the_conversation_without_calling_the_engine() -> None:
    result, upstream = await _chat(_messages("a", "b", "c"), guard_max_assistant_turns=3)

    assert result.choices[0].finish_reason == "stop"
    assert not result.choices[0].message.tool_calls
    assert "3 assistant turns" in result.choices[0].message.content
    upstream.assert_not_awaited()


async def test_identical_tool_calls_end_the_conversation_without_calling_the_engine() -> None:
    result, upstream = await _chat(_messages("ls", "true", "true", "true"), guard_identical_tool_calls=3)

    assert result.choices[0].finish_reason == "stop"
    assert not result.choices[0].message.tool_calls
    assert "same tool call" in result.choices[0].message.content
    upstream.assert_not_awaited()


@pytest.mark.parametrize(
    ("commands", "guards"),
    [
        (("a", "b"), {"guard_max_assistant_turns": 3}),
        (("true", "true", "ls"), {"guard_identical_tool_calls": 3}),
        (("true", "ls", "true"), {"guard_identical_tool_calls": 3}),
        (("true", "true", "true"), {}),
    ],
    ids=["below the turn limit", "a run broken by the last call", "not consecutive", "guards off"],
)
async def test_other_conversations_pass_through(commands: tuple[str, ...], guards: dict) -> None:
    result, upstream = await _chat(_messages(*commands), **guards)

    assert result.choices[0].message.content == "from the engine"
    upstream.assert_awaited_once()
    assert upstream.await_args.kwargs.get("tool_choice") != "none"


async def test_differing_custom_tool_calls_reach_the_engine() -> None:
    """Custom calls with other names or inputs are progress, not a loop."""
    for calls in (
        (("apply_patch", "a"), ("apply_patch", "b"), ("apply_patch", "c")),  # same tool, different inputs
        (("apply_patch", "a"), ("run_tests", "a"), ("apply_patch", "a")),  # different tools, same input
    ):
        result, upstream = await _chat(_custom_messages(*calls), guard_identical_tool_calls=3)

        assert result.choices[0].message.content == "from the engine", calls
        upstream.assert_awaited_once()


async def test_identical_custom_tool_calls_end_the_conversation() -> None:
    result, upstream = await _chat(
        _custom_messages(("apply_patch", "a"), ("apply_patch", "x"), ("apply_patch", "x"), ("apply_patch", "x")),
        guard_identical_tool_calls=3,
    )

    assert "same tool call" in result.choices[0].message.content
    upstream.assert_not_awaited()
