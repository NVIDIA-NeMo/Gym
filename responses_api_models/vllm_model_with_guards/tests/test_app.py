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
from nemo_gym.models.token_capture.harness_notes.opencode import OPENCODE_MAX_STEPS_PROMPT
from responses_api_models.vllm_model.app import VLLMModel
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


async def _chat(
    messages: list[dict], headers: dict | None = None, **guards: Any
) -> tuple[NeMoGymChatCompletion, AsyncMock]:
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
    request.headers = headers or {}
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


async def test_custom_tool_inputs_are_compared_byte_for_byte() -> None:
    """Free-form custom input is not JSON: "1" and " 1 " are different inputs, so the guard must not stop on them."""
    result, upstream = await _chat(
        _custom_messages(("apply_patch", "1"), ("apply_patch", " 1 ")), guard_identical_tool_calls=2
    )

    assert result.choices[0].message.content == "from the engine"
    upstream.assert_awaited_once()


async def test_identical_raw_custom_inputs_still_end_the_conversation() -> None:
    result, upstream = await _chat(
        _custom_messages(("apply_patch", " 1 "), ("apply_patch", " 1 ")), guard_identical_tool_calls=2
    )

    assert "same tool call" in result.choices[0].message.content
    upstream.assert_not_awaited()


async def test_function_arguments_are_still_compared_as_json() -> None:
    """Function arguments keep their JSON normalization: key order and whitespace do not make a call different."""
    messages = [{"role": "user", "content": "go"}]
    for i, arguments in enumerate(('{"a": 1, "b": 2}', '{ "b": 2,  "a": 1 }')):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": arguments}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})

    result, upstream = await _chat(messages, guard_identical_tool_calls=2)

    assert "same tool call" in result.choices[0].message.content
    upstream.assert_not_awaited()


async def test_parallel_calls_mixing_function_and_custom_keep_their_order() -> None:
    def turn(i: int, calls: list[dict]) -> list[dict]:
        return [
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": f"t{i}", "content": "ok"},
        ]

    function_call = {"id": "f", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
    custom_call = {"id": "c", "type": "custom", "custom": {"name": "apply_patch", "input": "x"}}
    # the same two calls in a different order are a different signature, so the conversation is progressing
    messages = [
        {"role": "user", "content": "go"},
        *turn(0, [function_call, custom_call]),
        *turn(1, [custom_call, function_call]),
    ]
    result, upstream = await _chat(messages, guard_identical_tool_calls=2)
    assert result.choices[0].message.content == "from the engine"
    upstream.assert_awaited_once()

    # the same two calls in the same order are a loop
    messages = [
        {"role": "user", "content": "go"},
        *turn(0, [function_call, custom_call]),
        *turn(1, [function_call, custom_call]),
    ]
    result, upstream = await _chat(messages, guard_identical_tool_calls=2)
    assert "same tool call" in result.choices[0].message.content
    upstream.assert_not_awaited()


@pytest.mark.parametrize("as_parts", [False, True], ids=["string notice", "text-part notice"])
async def test_the_harness_step_limit_notice_is_not_a_model_turn(as_parts: bool) -> None:
    """OpenCode appends its step-limit notice as a trailing assistant message; it must not hide a loop or add a turn."""
    notice: Any = [{"type": "text", "text": OPENCODE_MAX_STEPS_PROMPT}] if as_parts else OPENCODE_MAX_STEPS_PROMPT
    with_notice = [*_messages("true", "true"), {"role": "assistant", "content": notice}]

    result, upstream = await _chat(with_notice, guard_identical_tool_calls=2)
    assert "same tool call" in result.choices[0].message.content
    upstream.assert_not_awaited()

    # Two real turns plus the notice stay below a limit of three.
    result, upstream = await _chat(with_notice, guard_max_assistant_turns=3)
    assert result.choices[0].message.content == "from the engine"
    # ... and the body sent upstream still carries the notice.
    assert upstream.await_args.kwargs["messages"][-1]["content"] == notice


@pytest.mark.parametrize(
    "last",
    [
        {"role": "assistant", "content": OPENCODE_MAX_STEPS_PROMPT + " (edited)"},
        {"role": "assistant", "content": "I will keep going."},
        {
            "role": "assistant",
            "content": OPENCODE_MAX_STEPS_PROMPT,
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "bash", "arguments": '{"command": "x"}'}}
            ],
        },
    ],
    ids=["near match", "ordinary text", "notice text with a tool call"],
)
async def test_other_trailing_assistant_messages_still_count_and_break_the_streak(last: dict) -> None:
    messages = [*_messages("true", "true"), last]

    result, upstream = await _chat(messages, guard_identical_tool_calls=2)
    assert result.choices[0].message.content == "from the engine"  # the streak is broken by the last message
    upstream.assert_awaited_once()

    result, upstream = await _chat(messages, guard_max_assistant_turns=3)
    assert "3 assistant turns" in result.choices[0].message.content
    upstream.assert_not_awaited()


@pytest.mark.parametrize(
    "pin",
    [{"tool_choice": "required"}, {"tool_choice": "auto"}, {"tool_choice": None}, {"tool_choice": {"type": "x"}}],
)
def test_text_only_guard_rejects_a_tool_choice_pin_that_would_undo_it(pin: dict) -> None:
    with pytest.raises(ValueError, match="no tool_choice sampling pin"):
        _make_model(guard_text_only_after_assistant_turns=1, sampling_overrides=pin)


@pytest.mark.parametrize("pins", [None, {}, {"tool_choice": "none"}, {"temperature": 0.2, "max_tokens": 100}])
async def test_text_only_guard_accepts_no_pin_none_or_unrelated_pins(pins: dict | None) -> None:
    result, upstream = await _chat(_messages("a"), guard_text_only_after_assistant_turns=1, sampling_overrides=pins)

    assert result.choices[0].message.content == "from the engine"
    assert upstream.await_args.kwargs.get("tool_choice") == "none"


def test_a_tool_choice_pin_is_fine_without_the_text_only_guard() -> None:
    _make_model(guard_max_assistant_turns=3, sampling_overrides={"tool_choice": "required"})
    _make_model(sampling_overrides={"tool_choice": "required"})


def test_text_only_guard_rejects_the_completions_backend() -> None:
    with pytest.raises(ValueError, match="requires use_completions_api=false"):
        _make_model(guard_text_only_after_assistant_turns=1, use_completions_api=True)


def test_other_guards_keep_the_completions_backend_available() -> None:
    _make_model(guard_max_assistant_turns=3, guard_identical_tool_calls=2, use_completions_api=True)


async def test_a_compaction_summary_request_is_not_stopped_by_the_task_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent produces and checks the summary; a local stop here would answer OpenCode's compaction with no state."""
    delegated = AsyncMock(return_value="parent summary")
    monkeypatch.setattr(VLLMModel, "chat_completions", delegated)
    headers = {"X-NeMo-Gym-OpenCode-Agent": "compaction"}
    server = _make_model(guard_max_assistant_turns=1, compaction_summary_max_tokens=512)
    request = MagicMock()
    request.headers = headers

    result = await server.chat_completions(
        request, NeMoGymChatCompletionCreateParamsNonStreaming(messages=_messages("a", "b"))
    )
    assert result == "parent summary"
    delegated.assert_awaited_once()

    # Without the header, or without the cap, the same history is still stopped locally.
    for hdrs, cap in (({}, 512), ({"X-NeMo-Gym-OpenCode-Agent": "build"}, 512), (headers, None)):
        delegated.reset_mock()
        server = _make_model(guard_max_assistant_turns=1, compaction_summary_max_tokens=cap)
        request.headers = hdrs
        request.session = {SESSION_ID_KEY: "s"}
        result = await server.chat_completions(
            request, NeMoGymChatCompletionCreateParamsNonStreaming(messages=_messages("a", "b"))
        )
        assert "1 assistant turns" in result.choices[0].message.content, (hdrs, cap)
        delegated.assert_not_awaited()
