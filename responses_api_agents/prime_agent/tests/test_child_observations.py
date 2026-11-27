# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.config import ModelServerRef
from responses_api_agents.prime_agent.child_observations import child_observations


def observe(events, termination="completed"):
    return child_observations(
        {"events": events, "termination": termination},
        system_prompt="Native SDK system prompt",
        model_ref=ModelServerRef(type="responses_api_models", name="policy"),
        invocation_id="child",
    )


def assistant(response_id="upstream-response"):
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "responseId": response_id,
            "stopReason": "toolUse",
            "content": [{"type": "toolCall", "id": "call", "name": "ipython", "arguments": {"code": "1/0"}}],
        },
    }


def test_response_identity_and_executed_conversation_are_exported():
    bundle = observe(
        [
            {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "Inspect"}]}},
            assistant(),
            {
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call",
                    "content": [{"type": "text", "text": "division by zero"}],
                },
            },
        ]
    )
    invocation = bundle.records[0]
    assert invocation.model_calls[0].response_id == "upstream-response"
    assert invocation.model_calls[0].model_ref.name == "policy"
    assert invocation.conversation[0].content == "Native SDK system prompt"
    assert invocation.conversation[1].content == "Inspect"
    assert invocation.conversation[-2].call_id == invocation.conversation[-1].call_id == "call"
    assert invocation.conversation[-1].output == "division by zero"
    assert not bundle.gaps


@pytest.mark.parametrize(
    ("kernel_status", "expected"), [("ok", "completed"), ("error", "failed"), ("timeout", "timeout")]
)
def test_kernel_outcome_is_preserved_even_when_sdk_iserror_is_false(kernel_status, expected):
    bundle = observe(
        [
            assistant(),
            {"type": "tool_execution_start", "toolCallId": "call", "toolName": "ipython", "gymObservedAt": 10.0},
            {
                "type": "tool_execution_end",
                "toolCallId": "call",
                "gymObservedAt": 10.1,
                "isError": False,
                "result": {"isError": False, "details": {"status": kernel_status, "durationMs": 90}},
            },
        ]
    )
    tool = bundle.records[1]
    assert tool.status == expected
    assert tool.started_at == 10
    assert tool.completed_at == 10.1
    assert tool.duration_ms == 90


def test_missing_response_id_is_a_gap_and_is_never_fabricated():
    bundle = observe([assistant(None)])
    assert not bundle.records[0].model_calls
    assert {g.code for g in bundle.gaps} == {"model_response_id_unavailable", "model_call_ownership_unavailable"}


def test_turn_limit_preserves_last_answer_and_excludes_abort_control_event():
    bundle = observe(
        [assistant(), {"type": "message_end", "message": {"role": "assistant", "stopReason": "aborted"}}], "turn_limit"
    )
    invocation = bundle.records[0]
    assert invocation.status == "incomplete"
    assert invocation.stop_reason == "max_iterations"
    assert len(invocation.model_calls) == 1
    assert not bundle.gaps


def test_missing_tool_end_remains_incomplete():
    bundle = observe([assistant(), {"type": "tool_execution_start", "toolCallId": "call", "toolName": "ipython"}])
    assert bundle.records[1].status == "incomplete"
    assert bundle.records[1].completed_at is None
