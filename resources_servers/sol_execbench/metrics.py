# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-denominator correctness and SOLscore metrics with separate validity gates."""

import math
from collections import Counter
from statistics import mean

from nemo_gym.config_types import AggregateMetrics


def aggregate_results(
    rows: list[dict], *, task_ids: list[str], samples_per_task: int, protocol_sha256: str
) -> AggregateMetrics:
    """Require every task/repeat slot before publishing correctness or SOLscore."""
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
    scores = {task_id: [] for task_id in task_ids}
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
        score = row.get("sol_score")
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("sol_score must be null or a finite number in [0, 1]")
        mask = row.get("mask_sample")
        if type(mask) is not bool or mask != (row["infrastructure_error"] or score is None):
            raise ValueError("mask_sample must equal infrastructure_error or missing sol_score")
        slots.add((task_id, index))
        counts["unresolved"] += row["infrastructure_error"]
        counts["passed"] += row["solved"]
        counts[f"outcome/{row['outcome']}"] += 1
        if row["solved"]:
            solved_tasks.add(task_id)
        if score is not None:
            scores[task_id].append(score)
    expected = len(task_ids) * samples_per_task
    missing = expected - len(slots)
    correctness_complete = missing == 0 and counts["unresolved"] == 0
    score_complete = correctness_complete and all(len(values) == samples_per_task for values in scores.values())
    metrics = {
        "correctness_at_1": counts["passed"] / expected if correctness_complete else None,
        f"pass_at_{samples_per_task}": len(solved_tasks) / len(task_ids) if correctness_complete else None,
        "sol_score": mean(mean(values) for values in scores.values()) if score_complete else None,
        f"sol_score_best_of_{samples_per_task}": mean(max(values) for values in scores.values())
        if score_complete
        else None,
    }
    return AggregateMetrics(
        agent_metrics={
            "correctness_complete": correctness_complete,
            "score_complete": score_complete,
            "expected_samples": expected,
            "observed_samples": len(slots),
            "missing_samples": missing,
            "protocol_sha256": protocol_sha256,
            **counts,
            **metrics,
        },
        key_metrics=metrics,
    )
