# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An interrupted checkpoint must not imply that all captured policy calls were observed."""

from copy import deepcopy

from nemo_gym.rollout_health import evaluate_record


def test_recovered_checkpoint_with_unpersisted_reply_keeps_capture_scope_unobserved():
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "response": {
            "output": [{"type": "message", "role": "assistant", "content": "saved decision"}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
        "ng_trajectory": {
            "task_id": "0",
            "rollout_id": "0-0",
            "turns": [
                {
                    "invocation_id": "root",
                    "task_id": "0",
                    "rollout_id": "0-0",
                    "turn_no": 1,
                    "timestamp": 1.0,
                    "answer": "saved decision",
                    "step_count": 1,
                    "model_calls": [{"model_call_id": "saved-call"}],
                }
            ],
            "model_calls": [],
        },
    }
    for call_id, response_id, timestamp in (
        ("saved-call", "saved-reply", 1.0),
        ("unsaved-call", "unsaved-reply", 2.0),
    ):
        record["ng_trajectory"]["model_calls"].append(
            {
                "model_call_id": call_id,
                "started_at": timestamp,
                "request": {"input": "actual model request"},
                "response": {"output_text": "actual model reply"},
                "response_metadata": {
                    "response_id": response_id,
                    "status_code": 200,
                    "response_status": "completed",
                    "finish_reason": "stop",
                },
                "token_stats": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )

    # Every persisted turn has a valid capture and matching token usage, while
    # another successful reply never reached the interrupted producer checkpoint.
    # Without the producer's scope gap, the bound-call checks alone look healthy.
    control = evaluate_record(record, frozenset())
    assert control.verdict == "healthy"
    assert control.policy_calls_observed is True
    assert control.model_calls == 2
    assert control.unobserved == []
    assert control.findings == []

    recovered = deepcopy(record)
    recovered["ng_trajectory"]["gaps"] = [
        {
            "code": "turn_model_call_scope_incomplete",
            "invocation_id": "root",
            "detail": "The agent was evicted before its latest reply was persisted.",
        }
    ]
    health = evaluate_record(recovered, frozenset())
    assert health.verdict == "unobserved"
    assert health.policy_calls_observed is False
    assert health.model_calls == 2
    assert health.unobserved == ["rollout_token_count_mismatch"]
    assert health.findings == []
