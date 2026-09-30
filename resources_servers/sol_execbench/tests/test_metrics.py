# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.sol_execbench.metrics import aggregate_results


def row(task="a", *, outcome="PASSED", masked=False):
    return dict(
        task_id=task,
        rollout_index=0,
        protocol_sha256="protocol",
        outcome=outcome,
        infrastructure_error=masked,
        mask_sample=masked,
        solved=outcome == "PASSED",
    )


def aggregate(rows):
    return aggregate_results(rows, task_ids=["a", "b"], samples_per_task=1, protocol_sha256="protocol")


def test_complete_correctness_without_invented_sol():
    metrics = aggregate([row(), row("b", outcome="CANDIDATE_FAILED")])
    assert metrics.agent_metrics["complete"]
    assert metrics.key_metrics == {
        "official/correctness_at_1": 0.5,
        "official/pass_at_1": 0.5,
        "official/sol_score": None,
    }


@pytest.mark.parametrize("rows", [[row()], [row(), row("b", outcome="NATIVE_UNRESOLVED", masked=True)]])
def test_missing_and_unresolved_suppress_official_metrics(rows):
    assert all(value is None for value in aggregate(rows).key_metrics.values())


def test_duplicate_protocol_and_outcome_mismatch_fail_closed():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([row(), row()])
    mismatch = row()
    mismatch["protocol_sha256"] = "other"
    with pytest.raises(ValueError, match="Protocol"):
        aggregate([mismatch])
    mismatch = row()
    mismatch["outcome"] = "TIMEOUT"
    with pytest.raises(ValueError, match="outcome"):
        aggregate([mismatch])
