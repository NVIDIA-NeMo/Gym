# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture describes the served dialect, not a second interpretation of it."""

from copy import deepcopy

import pytest

from nemo_gym.token_id_capture.completion import completion_items, output_item_fingerprint


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
