# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode export must retain the actual model-facing interface on rejected calls."""

import json
from copy import deepcopy

import pytest

from resources_servers.agentic_if.episode.opencode_export import export_output


def test_rejected_call_preserves_actor_arguments_and_error() -> None:
    # Discarding the hook's receipt must not train the actor on backend argument names.
    state = {
        "status": "error",
        "input": {
            "filePath": "/app/example.py",
            "__nemo_gym_profile": {
                "configured_args": {"path": "/app/example.py", "unexpected": True},
                "names": {"filePath": "path"},
            },
        },
        "error": "Invalid filePath argument",
    }
    trace = {
        "messages": [
            {
                "info": {"role": "assistant"},
                "parts": [{"type": "tool", "tool": "read", "callID": "call-1", "state": state}],
            }
        ]
    }
    before = deepcopy(trace)

    call, result = export_output(trace)

    assert json.loads(call.arguments) == {"path": "/app/example.py", "unexpected": True}
    assert result.output == "Invalid path argument"
    assert call.call_id == result.call_id == "call-1"
    assert trace == before


def test_text_and_reasoning_are_kept_in_order_and_step_markers_are_dropped() -> None:
    trace = {
        "messages": [
            {
                "info": {"role": "user"},
                "parts": [{"type": "text", "text": "Fix it."}, {"type": "file", "url": "file:///app/a.py"}],
            },
            {
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "step-start"},
                    {"type": "reasoning", "text": "Look first."},
                    {"type": "text", "text": "Done."},
                    {"type": "patch", "hash": "abc", "files": ["/app/a.py"]},
                    {"type": "step-finish", "reason": "stop"},
                ],
            },
        ]
    }
    request, reasoning, reply = export_output(trace)
    # a user message keeps its text parts only
    assert request.role == "user" and [part["text"] for part in request.content] == ["Fix it."]
    assert reasoning.type == "reasoning" and [part.text for part in reasoning.summary] == ["Look first."]
    assert reply.role == "assistant" and [part.text for part in reply.content] == ["Done."]


@pytest.mark.parametrize(
    "message",
    [
        {"info": {"role": "assistant"}, "parts": [{"type": "snapshot", "snapshot": "abc"}]},
        {"info": {"role": "system"}, "parts": []},
    ],
    ids=["part_type", "role"],
)
def test_an_unknown_part_or_role_fails_rather_than_being_dropped(message) -> None:
    with pytest.raises(NotImplementedError):
        export_output({"messages": [message]})
