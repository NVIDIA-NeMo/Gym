# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.sol_execbench.metrics import aggregate_results


def row(task="a", *, index=0, outcome="PASSED", infrastructure_error=False, score=0.5):
    return dict(
        task_id=task,
        rollout_index=index,
        protocol_sha256="protocol",
        outcome=outcome,
        infrastructure_error=infrastructure_error,
        mask_sample=infrastructure_error or score is None,
        solved=outcome == "PASSED",
        sol_score=score,
    )


def aggregate(rows, *, samples_per_task=1):
    return aggregate_results(rows, task_ids=["a", "b"], samples_per_task=samples_per_task, protocol_sha256="protocol")


def test_complete_correctness_and_sol_score():
    metrics = aggregate([row(score=1), row("b", outcome="INVALID_SOLUTION", score=0)])
    assert metrics.agent_metrics["correctness_complete"]
    assert metrics.agent_metrics["score_complete"]
    assert metrics.agent_metrics["expected_samples"] == metrics.agent_metrics["observed_samples"] == 2
    assert metrics.agent_metrics["missing_samples"] == metrics.agent_metrics["unresolved"] == 0
    assert metrics.agent_metrics["passed"] == metrics.agent_metrics["outcome/PASSED"] == 1
    assert metrics.key_metrics == {
        "correctness_at_1": 0.5,
        "pass_at_1": 0.5,
        "sol_score": 0.5,
        "sol_score_best_of_1": 0.5,
    }


def test_repeat_means_and_best_scores_include_partial_candidate_failures():
    rows = [
        row("b", index=1, score=0.4),
        row(index=1, outcome="CANDIDATE_FAILED", score=0.8),
        row("b", outcome="CANDIDATE_FAILED", score=0),
        row(score=0.2),
    ]
    rows[0]["_ng_rollout_index"] = rows[0].pop("rollout_index")
    metrics = aggregate(rows, samples_per_task=2)
    assert metrics.key_metrics == pytest.approx(
        {"correctness_at_1": 0.5, "pass_at_2": 1.0, "sol_score": 0.35, "sol_score_best_of_2": 0.6}
    )


def test_unavailable_score_preserves_complete_correctness():
    metrics = aggregate([row(score=None), row("b", outcome="CANDIDATE_FAILED", score=0)])
    assert metrics.agent_metrics["correctness_complete"]
    assert not metrics.agent_metrics["score_complete"]
    assert metrics.key_metrics == {
        "correctness_at_1": 0.5,
        "pass_at_1": 0.5,
        "sol_score": None,
        "sol_score_best_of_1": None,
    }


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [row()],
        [row(), row("b", outcome="NATIVE_UNRESOLVED", infrastructure_error=True, score=None)],
        [row(), row("b", infrastructure_error=True)],
    ],
)
def test_missing_or_infrastructure_failure_suppresses_both_metrics(rows):
    metrics = aggregate(rows)
    assert not metrics.agent_metrics["correctness_complete"]
    assert not metrics.agent_metrics["score_complete"]
    assert all(value is None for value in metrics.key_metrics.values())
    assert metrics.agent_metrics["missing_samples"] == 2 - len(rows)


@pytest.mark.parametrize("score", [True, False, "0.5", float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_fail_closed(score):
    with pytest.raises(ValueError, match="sol_score"):
        aggregate([row(score=score)])


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"task_id": "other"}, "task/repeat"),
        ({"rollout_index": True}, "task/repeat"),
        ({"rollout_index": -1}, "task/repeat"),
        ({"rollout_index": 1}, "task/repeat"),
        ({"protocol_sha256": "other"}, "Protocol"),
        ({"solved": 1}, "boolean"),
        ({"infrastructure_error": 0}, "boolean"),
        ({"outcome": "TIMEOUT"}, "outcome"),
        ({"outcome": "TIMEOUT", "solved": False}, "Unmeasured"),
        ({"mask_sample": True}, "mask_sample"),
        ({"mask_sample": 0}, "mask_sample"),
        ({"sol_score": None}, "mask_sample"),
        ({"infrastructure_error": True}, "mask_sample"),
    ],
)
def test_slot_provenance_and_result_invariants_fail_closed(updates, message):
    result = row()
    result.update(updates)
    with pytest.raises(ValueError, match=message):
        aggregate([result])


def test_duplicate_slot_and_missing_mask_fail_closed():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([row(), row()])
    result = row()
    del result["mask_sample"]
    with pytest.raises(ValueError, match="mask_sample"):
        aggregate([result])


@pytest.mark.parametrize("task_ids,samples", [([], 1), (["a", "a"], 1), ([""], 1), (["a"], 0), (["a"], True)])
def test_invalid_denominators_fail_closed(task_ids, samples):
    with pytest.raises(ValueError):
        aggregate_results([], task_ids=task_ids, samples_per_task=samples, protocol_sha256="protocol")
