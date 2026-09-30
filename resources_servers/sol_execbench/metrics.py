# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregate SOL results against an explicit task and repeat manifest."""

import math
from collections import Counter

from nemo_gym.config_types import AggregateMetrics


MEASURED_OUTCOMES = frozenset(
    {
        "PASSED",
        "CANDIDATE_FAILED",
        "CANDIDATE_SYNTAX_ERROR",
        "COMPILE_ERROR",
        "NO_SOLUTION",
        "INVALID_SOLUTION",
        "CANDIDATE_IMPORT_ERROR",
        "MISSING_ENTRYPOINT",
    }
)


def aggregate_sol_results(
    rows: list[dict],
    *,
    task_ids: list[str],
    samples_per_task: int,
    protocol_sha256: str,
    timeout_zero_sensitivity: bool = False,
) -> AggregateMetrics:
    """Compute fixed-denominator metrics without treating unknown results as failures.

    Every observed row must identify one unique expected task/repeat slot and the
    same protocol. Missing rows and infrastructure errors suppress all official
    headline metrics. Optional timeout-zero metrics are a sensitivity analysis,
    available only when every slot exists and every unknown is a recorded timeout.
    Raw passing scores are retained. Headline best-of-K includes zero as an
    alternative for each task, matching the evaluator's aggregation convention.
    """
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
    ):
        raise ValueError("task_ids must be a nonempty list of nonempty strings")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")
    if type(samples_per_task) is not int or samples_per_task <= 0:
        raise ValueError("samples_per_task must be a positive integer")
    if not isinstance(protocol_sha256, str) or not protocol_sha256:
        raise ValueError("protocol_sha256 must be a nonempty string")
    if type(timeout_zero_sensitivity) is not bool:
        raise ValueError("timeout_zero_sensitivity must be a boolean")
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")

    expected_tasks = set(task_ids)
    by_task: dict[str, list[dict]] = {task_id: [] for task_id in task_ids}
    seen: set[tuple[str, int]] = set()
    outcomes: Counter[str] = Counter()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {position} must be a dictionary")
        task_id, rollout_index = row.get("task_id"), row.get("_ng_rollout_index")
        if not isinstance(task_id, str) or task_id not in expected_tasks:
            raise ValueError(f"Row {position} has an unexpected task_id")
        if type(rollout_index) is not int or not 0 <= rollout_index < samples_per_task:
            raise ValueError(f"Row {position} has an invalid _ng_rollout_index")
        slot = (task_id, rollout_index)
        if slot in seen:
            raise ValueError(f"Duplicate task/repeat slot: {slot}")
        if row.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"Row {position} has a protocol mismatch")
        if type(row.get("infrastructure_error")) is not bool or type(row.get("solved")) is not bool:
            raise ValueError(f"Row {position} must have boolean infrastructure_error and solved fields")
        infrastructure_error, solved = row["infrastructure_error"], row["solved"]
        if "mask_sample" in row and (
            type(row["mask_sample"]) is not bool or row["mask_sample"] != infrastructure_error
        ):
            raise ValueError(f"Row {position} mask_sample must equal infrastructure_error")
        outcome = row.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError(f"Row {position} must have a nonempty outcome")
        if solved != (outcome == "PASSED"):
            raise ValueError(f"Row {position} solved must be true exactly when outcome is PASSED")
        if not infrastructure_error and outcome not in MEASURED_OUTCOMES:
            raise ValueError(f"Row {position} outcome {outcome!r} must remain an infrastructure error")
        if "sol_score" not in row:
            raise ValueError(f"Row {position} is missing sol_score")
        score = row["sol_score"]
        if infrastructure_error:
            if solved or score is not None:
                raise ValueError(f"Row {position} infrastructure errors must be unsolved with a null score")
        else:
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError(f"Row {position} must have a finite numeric sol_score (not boolean)")
            if not solved and score != 0:
                raise ValueError(f"Row {position} unsolved candidates must have zero sol_score")
        seen.add(slot)
        outcomes[outcome] += 1
        by_task[task_id].append(row)

    groups = []
    for task_id, task_rows in by_task.items():
        passing = [row for row in task_rows if row["solved"]]
        missing = samples_per_task - len(task_rows)
        infrastructure_errors = sum(row["infrastructure_error"] for row in task_rows)
        groups.append(
            {
                "task_id": task_id,
                "expected_samples": samples_per_task,
                "observed_samples": len(task_rows),
                "missing_samples": missing,
                "infrastructure_errors": infrastructure_errors,
                "unresolved_samples": missing + infrastructure_errors,
                "observed_passes": len(passing),
                "candidate_failures": len(task_rows) - infrastructure_errors - len(passing),
                "observed_best_passing_sol": max((row["sol_score"] for row in passing), default=0.0),
                "outcome_counts": dict(Counter(row["outcome"] for row in task_rows)),
            }
        )

    expected = len(task_ids) * samples_per_task
    missing = expected - len(rows)
    infrastructure_errors = sum(group["infrastructure_errors"] for group in groups)
    unresolved = missing + infrastructure_errors
    observed_passes = sum(group["observed_passes"] for group in groups)
    solved_tasks = sum(group["observed_passes"] > 0 for group in groups)
    eligible = missing == 0 and all(
        not row["infrastructure_error"] or row["outcome"] == "EVALUATION_TIMEOUT" for row in rows
    )
    scores = {
        f"sol_best{samples_per_task}": math.fsum(max(0.0, group["observed_best_passing_sol"]) for group in groups)
        / len(task_ids),
        "correctness_at_1": observed_passes / expected,
        f"pass_at_{samples_per_task}": solved_tasks / len(task_ids),
    }
    official = {f"official/{name}": value if unresolved == 0 else None for name, value in scores.items()}
    agent_metrics = {
        "protocol_sha256": protocol_sha256,
        "complete": unresolved == 0,
        "counts/task_count": len(task_ids),
        "counts/samples_per_task": samples_per_task,
        "counts/expected_samples": expected,
        "counts/observed_samples": len(rows),
        "counts/missing_samples": missing,
        "counts/infrastructure_errors": infrastructure_errors,
        "counts/unresolved_samples": unresolved,
        "counts/observed_passes": observed_passes,
        "counts/observed_solved_tasks": solved_tasks,
        "counts/candidate_failures": len(rows) - infrastructure_errors - observed_passes,
        **{f"outcomes/{name}": count for name, count in sorted(outcomes.items())},
        **official,
    }
    key_metrics = dict(official)
    if timeout_zero_sensitivity:
        sensitivity = {f"timeout_zero/{name}": value if eligible else None for name, value in scores.items()}
        agent_metrics.update(sensitivity)
        agent_metrics["timeout_zero/eligible"] = eligible
        agent_metrics["timeout_zero/note"] = (
            "Sensitivity analysis only: recorded EVALUATION_TIMEOUT infrastructure errors are assigned zero. "
            "Missing records or other infrastructure errors suppress these metrics. "
            "Official metrics remain null with any unresolved slots."
        )
        key_metrics.update(sensitivity)
    return AggregateMetrics(group_level_metrics=groups, agent_metrics=agent_metrics, key_metrics=key_metrics)
