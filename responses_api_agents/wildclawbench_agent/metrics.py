# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish a full-suite score only for one valid, source-bound attempt of every task."""

import json
from math import isfinite

from responses_api_agents.wildclawbench_agent.data import source_pin


def benchmark_metrics(rows: list[dict]) -> dict[str, float | int]:
    pin = source_pin()
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate tasks: merge one attempt per task, not shard means or repeats")
    protocols = {json.dumps(row["evaluation_protocol"], sort_keys=True) for row in rows}
    if len(protocols) > 1:
        raise ValueError("Mixed evaluated models or evaluation protocol (context, generation, judge or image)")
    for row in rows:
        if row["task_id"] not in pin["tasks"] or row["source_revision"] != pin["revision"]:
            raise ValueError("Mixed benchmark sources or unknown task")
        expected = pin["files"][pin["tasks"][row["task_id"]]]
        if row["task_sha256"] != expected:
            raise ValueError("Task source hash differs from pinned grader")
        reward = row["reward"]
        if isinstance(reward, bool) or not isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Task reward must be finite and in [0, 1]")
        if not row.get("mask_sample") and row.get("native_reward") != reward:
            raise ValueError("Gym reward differs from original grader reward")
    valid = [row for row in rows if not row.get("mask_sample")]
    metrics = {
        "wildclawbench/num_collected": len(rows),
        "wildclawbench/num_scored": len(valid),
        "wildclawbench/num_masked": len(rows) - len(valid),
        "wildclawbench/coverage": len(valid) / len(pin["tasks"]),
    }
    if valid:
        metrics["wildclawbench/partial_mean_score"] = sum(row["reward"] for row in valid) / len(valid)
    if set(ids) == pin["tasks"].keys() and len(valid) == len(ids):
        metrics["wildclawbench/overall_score"] = metrics["wildclawbench/partial_mean_score"]
    return metrics
