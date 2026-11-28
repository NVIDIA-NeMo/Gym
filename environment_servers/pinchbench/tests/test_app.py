# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts the upstream verifier relies on across harnesses."""

import pytest
from pydantic import ValidationError

from environment_servers.pinchbench.app import Plan, conversations, transcript
from environment_servers.pinchbench.task_input import TaskSelector
from nemo_gym.agents.session_client import TurnRecord
from nemo_gym.core import NeMoGymResponse


def test_fresh_conversation_keeps_the_first_two_prompts_together() -> None:
    plan = Plan(
        prompts=["store this", {"prompt": "update it"}, {"prompt": "recall", "new_session": True}],
        timeout=60,
        grading_type="automated",
        requires_fws=False,
    )
    assert conversations(plan) == [["store this", "update it"], ["recall"]]


def test_task_selector_cannot_select_another_file_in_the_verifier_image() -> None:
    with pytest.raises(ValidationError):
        TaskSelector(task_id="../scripts/worker")


def test_transcript_preserves_tool_evidence_and_final_text() -> None:
    response = NeMoGymResponse.model_validate(
        {
            "id": "r",
            "object": "response",
            "created_at": 1,
            "model": "test",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "c",
                    "type": "function_call",
                    "name": "read",
                    "arguments": '{"path":"in.txt"}',
                    "call_id": "call1",
                },
                {"type": "function_call_output", "call_id": "call1", "output": "file contents"},
                {
                    "type": "message",
                    "id": "m",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                },
            ],
        }
    )
    turn = TurnRecord(turn=1, input="read a file", response=response, duration_seconds=1)
    messages = [event["message"] for event in transcript([turn], agent="test", session_id="s")]
    assert messages[0] == {"role": "user", "content": [{"type": "text", "text": "read a file"}]}
    assert {"type": "toolCall", "name": "read", "id": "call1", "arguments": {"path": "in.txt"}} in messages[1][
        "content"
    ]
    assert messages[2] == {
        "role": "toolResult",
        "toolCallId": "call1",
        "toolName": "read",
        "content": [{"type": "text", "text": "file contents"}],
    }
    assert messages[-1] == {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}
