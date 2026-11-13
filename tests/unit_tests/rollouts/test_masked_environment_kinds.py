# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery of masked results that an environment saved under its own namespaced kind (agentic_turn_simulation)."""

import pytest

from nemo_gym.rollouts.rollout_recovery import masked_infrastructure_kind, masked_kept_answer_kind


_ANSWER = {
    "response": {
        "object": "response",
        "id": "r1",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
    }
}


def _masked(kind: str, **fields) -> dict:
    return {"reward": 0.0, "mask_sample": True, "failure_kind": kind, "failure_reason": "cut off"} | fields


@pytest.mark.parametrize("kind", ["agentic_turn_simulation:time_cap", "agentic_turn_simulation:simulator_failed"])
def test_a_saved_namespaced_environment_mask_is_planned_as_an_environment_failure(kind):
    """Rows saved before the registered kind: the wall-clock cap and a failed simulation. Without a saved answer the
    rollout is retried within its attempt cap; with one it is held for an approved rerun, as any masked kept answer."""
    assert masked_infrastructure_kind(_masked(kind)) == "environment_episode_failed"
    assert masked_infrastructure_kind(_masked(kind) | _ANSWER) is None
    assert masked_kept_answer_kind(_masked(kind) | _ANSWER) == "environment_episode_failed"


def test_a_row_saved_with_the_registered_kind_is_planned_the_same_way():
    row = _masked("environment_episode_failed", failure_detail="agentic_turn_simulation:time_cap")
    assert masked_infrastructure_kind(row) == "environment_episode_failed"
    assert masked_kept_answer_kind(row | _ANSWER) == "environment_episode_failed"


@pytest.mark.parametrize(
    "kind", ["agentic_turn_simulation:context_overflow", "agentic_turn_simulation:recording_cut_off"]
)
def test_a_masked_row_property_or_other_namespaced_kind_stays_final(kind):
    assert masked_infrastructure_kind(_masked(kind)) is None
    assert masked_kept_answer_kind(_masked(kind) | _ANSWER) is None


def test_a_scored_model_outcome_is_no_masked_environment_failure():
    unfinished = {"reward": 0.0, "code_gate": "unfinished_turn", "stop_reason": "context_overflow", "steps": 3}
    assert masked_infrastructure_kind(unfinished) is None
    assert masked_kept_answer_kind(unfinished | _ANSWER) is None
