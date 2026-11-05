# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.rollouts.rollout_perf import _build_ng_perf
from nemo_gym.trajectory.captured_calls import build_captured_calls
from nemo_gym.trajectory.trajectory_record import NG_TRAJECTORY_KEY


def test_captured_calls_outside_agent_turns_are_counted_without_changing_agent_totals() -> None:
    result = _three_call_trajectory()

    assert build_captured_calls(result) == {
        "calls": 3,
        "calls_without_usage": 0,
        "attributed_call_coverage": 1 / 3,
        "by_role": {
            "agent": {"calls": 1, "calls_without_usage": 0, "prompt_tokens": 100, "call_latency_ms": 40.0},
            "unattributed": {"calls": 2, "calls_without_usage": 0, "prompt_tokens": 500, "call_latency_ms": 50.0},
        },
    }
    ng_perf = _build_ng_perf(result, rollout_latency_ms=None)
    assert ng_perf["token_observability_coverage"] == 1.0 and ng_perf["prompt_tokens"] == 100


def test_verifier_owned_calls_get_their_own_role() -> None:
    result = _three_call_trajectory(
        verifier_invocations=[
            {
                "invocation_id": "verify-1",
                "model_calls": [{"model_call_id": "call-2"}, {"model_call_id": "call-1"}],
            }
        ]
    )

    captured = build_captured_calls(result)

    assert captured["attributed_call_coverage"] == 2 / 3
    assert {role: totals["calls"] for role, totals in captured["by_role"].items()} == {
        "agent": 1,
        "verifier": 1,
        "unattributed": 1,
    }
    assert captured["by_role"]["verifier"]["prompt_tokens"] == 200


def test_captured_calls_count_shared_references_once_and_keep_missing_usage_out_of_totals() -> None:
    result = {
        NG_TRAJECTORY_KEY: {
            "task_id": "t",
            "rollout_id": "t-0",
            "invocations": [
                {"invocation_id": "root", "model_calls": [{"model_call_id": "call-1"}]},
                {"invocation_id": "sub", "model_calls": [{"model_call_id": "call-1"}]},
            ],
            "model_calls": [
                {"model_call_id": "call-1", "token_stats": {"prompt_tokens": 100, "completion_tokens": 7}},
                {"model_call_id": "call-2"},
            ],
        }
    }

    captured = build_captured_calls(result)

    assert captured["calls"] == 2 and captured["calls_without_usage"] == 1
    assert captured["by_role"]["agent"] == {
        "calls": 1,
        "calls_without_usage": 0,
        "prompt_tokens": 100,
        "completion_tokens": 7,
    }
    assert captured["by_role"]["unattributed"] == {"calls": 1, "calls_without_usage": 1}


def test_captured_calls_distinguish_no_calls_from_no_capture() -> None:
    assert build_captured_calls({}) is None
    empty = {NG_TRAJECTORY_KEY: {"task_id": "t", "rollout_id": "t-0"}}
    assert build_captured_calls(empty) == {
        "calls": 0,
        "calls_without_usage": 0,
        "attributed_call_coverage": None,
        "by_role": {},
    }


def _three_call_trajectory(**extra: object) -> dict:
    """NVIDIA-NeMo/Gym#4722: three captured calls, only the first in the single agent turn."""
    return {
        NG_TRAJECTORY_KEY: {
            "task_id": "t",
            "rollout_id": "t-0",
            "invocations": [{"invocation_id": "root", "model_calls": [{"model_call_id": "call-1"}]}],
            "model_calls": [
                {"model_call_id": "call-1", "duration_ms": 40.0, "token_stats": {"prompt_tokens": 100}},
                {"model_call_id": "call-2", "duration_ms": 30.0, "token_stats": {"prompt_tokens": 200}},
                {"model_call_id": "call-3", "duration_ms": 20.0, "token_stats": {"prompt_tokens": 300}},
            ],
            **extra,
        }
    }
