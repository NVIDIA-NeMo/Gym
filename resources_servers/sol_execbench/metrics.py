# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-denominator native correctness metrics, with unresolved runs withheld."""

from collections import Counter

from nemo_gym.config_types import AggregateMetrics


def aggregate_results(
    rows: list[dict], *, task_ids: list[str], samples_per_task: int, protocol_sha256: str
) -> AggregateMetrics:
    """Require each declared task/repeat slot before publishing correctness metrics."""
    if (
        not task_ids
        or len(task_ids) != len(set(task_ids))
        or any(not isinstance(task, str) or not task for task in task_ids)
    ):
        raise ValueError("task_ids must be nonempty and unique")
    if type(samples_per_task) is not int or samples_per_task <= 0:
        raise ValueError("samples_per_task must be a positive integer")
    slots = set()
    counts = Counter()
    solved_tasks = set()
    for row in rows:
        task_id = row.get("task_id")
        index = row.get("_ng_rollout_index", row.get("rollout_index", 0))
        if task_id not in task_ids or type(index) is not int or not 0 <= index < samples_per_task:
            raise ValueError("Unexpected task/repeat slot")
        if (task_id, index) in slots:
            raise ValueError("Duplicate task/repeat slot")
        if row.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Protocol mismatch")
        if type(row.get("infrastructure_error")) is not bool or type(row.get("solved")) is not bool:
            raise ValueError("Every result requires boolean infrastructure_error and solved")
        if row["solved"] != (row.get("outcome") == "PASSED"):
            raise ValueError("solved must match PASSED outcome")
        if not row["infrastructure_error"] and row.get("outcome") not in {
            "PASSED",
            "CANDIDATE_FAILED",
            "INVALID_SOLUTION",
        }:
            raise ValueError("Unmeasured outcome must remain masked")
        if row["infrastructure_error"] and row["solved"]:
            raise ValueError("Unresolved results cannot be solved")
        if row.get("mask_sample", row["infrastructure_error"]) != row["infrastructure_error"]:
            raise ValueError("mask_sample must equal infrastructure_error")
        slots.add((task_id, index))
        counts["unresolved"] += row["infrastructure_error"]
        counts["passed"] += row["solved"]
        counts[f"outcome/{row['outcome']}"] += 1
        if row["solved"]:
            solved_tasks.add(task_id)
    expected = len(task_ids) * samples_per_task
    missing = expected - len(slots)
    complete = missing == 0 and counts["unresolved"] == 0
    official = {
        "official/correctness_at_1": counts["passed"] / expected if complete else None,
        f"official/pass_at_{samples_per_task}": len(solved_tasks) / len(task_ids) if complete else None,
        "official/sol_score": None,
    }
    return AggregateMetrics(
        agent_metrics={
            "complete": complete,
            "expected_samples": expected,
            "observed_samples": len(slots),
            "missing_samples": missing,
            "protocol_sha256": protocol_sha256,
            **counts,
            **official,
        },
        key_metrics=official,
    )
