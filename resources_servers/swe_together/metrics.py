# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure-aware SWE-Together summaries, keeping missing measurements separate."""

from collections import defaultdict
from statistics import mean


def aggregate(records: list[dict], *, planned: int, repeats: int = 2) -> dict:
    by_task = defaultdict(list)
    graded = []
    for record in records:
        if record.get("mask_sample") or record.get("judge_score") is None:
            continue
        score = float(record["judge_score"])
        graded.append(score)
        task = record["task_id"]
        by_task[task if isinstance(task, str) else task["task_id"]].append(score)
    complete = [scores for scores in by_task.values() if len(scores) == repeats]
    corrections = [r["user_correction"] for r in records if r.get("user_correction") is not None]
    return {
        "planned": planned,
        "returned": len(records),
        "graded": len(graded),
        "masked": sum(bool(r.get("mask_sample")) for r in records),
        "missing": max(0, planned - len(records)),
        "MeanJudge": mean(graded) if graded else None,
        "pass@1": mean(mean(s >= 0.85 for s in scores) for scores in by_task.values()) if by_task else None,
        "stable_solve_rate": mean(mean(scores) >= 0.85 for scores in complete) if complete else None,
        "pass2": mean(all(s >= 0.85 for s in scores) for scores in complete) if repeats == 2 and complete else None,
        "complete_tasks": len(complete),
        "user_correction": mean(corrections) if corrections else None,
        "tagged_episodes": len(corrections),
        "upstream_compatibility_mean_zero_filled": sum(graded) / planned if planned else None,
    }
