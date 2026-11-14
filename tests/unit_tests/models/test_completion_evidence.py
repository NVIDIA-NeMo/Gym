# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture describes the served dialect, not a second interpretation of it."""

from copy import deepcopy

import pytest
from pydantic import TypeAdapter, ValidationError

from nemo_gym.models.anthropic_converter import AnthropicConverter
from nemo_gym.models.token_capture.completion import (
    completion_items,
    completion_metadata,
    completion_replay_items,
    output_item_fingerprint,
    output_validation_item,
)
from nemo_gym.models.token_capture.records import response_to_output_items, strip_token_fields
from nemo_gym.models.token_capture.replay import replay_context, summarize_replay
from nemo_gym.models.token_capture.staging.records import CallRecord


@pytest.mark.parametrize("text", ["<think>work</think>answer", "literal <think> tag", "ordinary answer"])
def test_chat_preserves_served_text_and_tools(text):
    message = {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": f"call-{i}", "type": "function", "function": {"name": "check", "arguments": "{}"}} for i in range(2)
        ],
    }
    response = {"choices": [{"message": message}]}
    original = deepcopy(response)
    items = completion_items(response)
    assert [item["type"] for item in items] == ["message", "function_call", "function_call"]
    assert items[0]["content"][0]["text"] == text
    assert [item["call_id"] for item in items[1:]] == ["call-0", "call-1"]
    assert "id" not in items[0]
    assert response == original


def test_native_responses_reasoning_stays_reasoning_and_is_not_ignored():
    reasoning = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "work"}]}
    answer = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}
    assert completion_items({"output": [reasoning, answer]}) == [reasoning, answer]
    edited = deepcopy(reasoning)
    edited["summary"][0]["text"] = "changed"
    assert output_item_fingerprint(reasoning) != output_item_fingerprint(edited)


def test_chat_reasoning_removal_changes_evidence():
    def evidence(text):
        return output_item_fingerprint(
            completion_items({"choices": [{"message": {"role": "assistant", "content": text}}]})[0]
        )

    assert evidence("<think>work</think>answer") != evidence("answer")
    assert evidence("<think>work</think>answer") != evidence("<think>edited</think>answer")


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({}, ("unknown", None)),
        ({"status": "cancelled"}, ("failed", None)),
        ({"choices": [{"finish_reason": "abort"}]}, ("failed", "abort")),
        ({"status": "queued", "choices": [{"finish_reason": "stop"}]}, ("unknown", "stop")),
        ({"status": "in_progress"}, ("unknown", None)),
        ({"status": "future_status"}, ("unknown", None)),
        ({"choices": [{"finish_reason": "future_reason"}]}, ("unknown", "future_reason")),
        ({"status": "completed", "error": {}}, ("failed", None)),
        ({"status": "completed"}, ("completed", None)),
        ({"choices": [{"finish_reason": "tool_calls"}]}, ("completed", "tool_calls")),
        ({"choices": [{"finish_reason": "length"}]}, ("incomplete", "length")),
        ({"stop_reason": "max_tokens"}, ("incomplete", "max_tokens")),
    ],
)
def test_completion_status_is_bounded_and_does_not_invent_success(payload, expected):
    assert completion_metadata(payload) == expected


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed", "unknown", None])
def test_manifest_completion_status_vocabulary(status):
    adapter = TypeAdapter(CallRecord.model_fields["response_status"].annotation)
    assert adapter.validate_python(status) == status
    with pytest.raises(ValidationError):
        adapter.validate_python("cancelled")


def test_validation_evidence_preserves_detector_text_and_rejects_wrong_shapes():
    adapter = TypeAdapter(CallRecord.model_fields["last_output_item"].annotation)
    item = {
        "type": "message",
        "content": [{"type": "output_text", "text": "<think>original", "tokens": [1]}, {}],
        "summary": [{"type": "summary_text", "text": "reasoning"}],
        "generation_token_ids": [2],
    }
    evidence = output_validation_item(item)
    assert adapter.validate_python(evidence) == {
        "type": "message",
        "content": [{"text": "<think>original"}, {}],
        "summary": [{"text": "reasoning"}],
    }
    for invalid in ({"content": "wrong"}, {"content": [{"text": 123}]}, {"summary": ["wrong"]}):
        with pytest.raises(ValidationError):
            adapter.validate_python(invalid)


@pytest.mark.parametrize("dialect", ["responses", "chat"])
def test_completion_replay_strips_tokens_before_joining_history(dialect):
    tokens = {"prompt_token_ids": [1, 2], "generation_token_ids": [3], "generation_log_probs": [-0.5]}
    output = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
        {"type": "function_call", "call_id": "call-1", "name": "check", "arguments": "{}", **tokens},
    ]
    served = (
        {"output": output}
        if dialect == "responses"
        else {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning": "think",
                        "tool_calls": [
                            {"id": "call-1", "type": "function", "function": {"name": "check", "arguments": "{}"}}
                        ],
                        **tokens,
                    }
                }
            ]
        }
    )
    original = deepcopy(served)
    # Historical token metadata and newly served metadata must not be counted together.
    history = [{"role": "assistant", "content": "previous", **tokens}]
    expected_items = strip_token_fields(response_to_output_items(served))[0]
    actual_items = completion_replay_items(served)
    assert actual_items == expected_items
    expected = summarize_replay(replay_context(history + expected_items, render_digest="0" * 64))
    actual = summarize_replay(replay_context(history + actual_items, render_digest="0" * 64))
    assert actual == expected
    assert served == original


def test_messages_replay_matches_echoed_prefix():
    served = {
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [
            {"type": "text", "text": "delegating"},
            {"type": "tool_use", "id": "call-1", "name": "Agent", "input": {"prompt": "solve"}},
        ],
    }
    echoed = AnthropicConverter().anthropic_request_to_responses(
        {
            "model": "test-model",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": served["content"]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "ok"}]},
            ],
        }
    )
    items = [item.model_dump(mode="json") for item in echoed.input]
    completion = summarize_replay(replay_context(items[:1] + completion_replay_items(served)))
    prefix = summarize_replay(replay_context(items), item_count=completion.item_count)
    generic = summarize_replay(replay_context(items[:1] + strip_token_fields(response_to_output_items(served))[0]))
    assert completion.source_digest == prefix.source_digest
    assert generic.source_digest != prefix.source_digest
