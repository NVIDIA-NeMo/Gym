# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from resources_servers.partner_if.native_evidence import captured_artifact, visible_events


def test_visible_events_preserve_calls_and_observations_but_not_hidden_reasoning():
    raw = [
        {"type": "reasoning", "summary": [{"text": "private reasoning"}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Plan first"}],
            "reasoning": "hidden",
            "model": "actor-identity",
        },
        {"type": "function_call", "name": "shell", "call_id": "a", "arguments": '{"command":"ls"}'},
        {"type": "function_call_output", "call_id": "a", "output": "files"},
    ]
    events = visible_events(raw)
    assert [event.source_index for event in events] == [1, 2, 3]
    assert [event.kind for event in events] == ["assistant", "tool_call", "tool_result"]
    assert events[1].arguments == {"command": "ls"}
    serialized = json.dumps([event.model_dump() for event in events])
    assert "private reasoning" not in serialized and "actor-identity" not in serialized
    assert "Plan first" in serialized and "files" in serialized


@pytest.mark.parametrize(
    "item",
    [
        {"type": "computer_call"},
        {"type": "message", "role": "assistant", "content": [{"type": "image", "url": "private"}]},
    ],
)
def test_unsupported_visible_evidence_is_not_silently_omitted(item):
    with pytest.raises(ValueError, match="Unsupported"):
        visible_events([item])


def test_tool_argument_string_is_retained_even_when_it_is_malformed_json():
    event = visible_events([{"type": "function_call", "name": "shell", "call_id": "a", "arguments": "{"}])[0]
    assert event.arguments == "{"  # An agent's malformed call is evidence, not a capture failure.


def test_capture_requires_explicit_receipt_and_preserves_absence_vs_error():
    assert captured_artifact({"exists": True, "text": "hello"}).text == "hello"
    assert captured_artifact({"exists": False, "text": ""}).status == "absent"
    assert captured_artifact({"error": "copy failed"}).status == "error"
    assert captured_artifact({}).status == "error"
    assert captured_artifact({"exists": True}).status == "error"


def test_partial_capture_error_does_not_become_confirmed_absence():
    result = captured_artifact({"exists": False, "error": "permission denied"})
    assert result.status == "error"
