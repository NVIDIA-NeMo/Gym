# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit real collector results without treating missing measurements as zero."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def _finite_number(value: object, *, maximum: float | None = None) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        and (maximum is None or value <= maximum)
    )


def healthy_record(row: dict) -> list[str]:
    if not isinstance(row, dict):
        return ["invalid_record"]
    problems = []
    if row.get("mask_sample") or row.get("failure_kind") or row.get("_ng_failure_class") or row.get("failure"):
        problems.append("failed_or_masked")
    score = row.get("judge_score")
    if not _finite_number(score, maximum=1):
        problems.append("missing_or_invalid_grade")
    elif not _finite_number(row.get("reward"), maximum=1) or row["reward"] != float(score >= 0.85):
        problems.append("reward_disagrees_with_grade")
    verdict = row.get("verdict")
    verdict = verdict if isinstance(verdict, dict) else {}
    goals = verdict.get("goal_results")
    if (
        not isinstance(goals, list)
        or not goals
        or any(
            not isinstance(goal, dict)
            or not isinstance(goal.get("id"), str)
            or not goal["id"]
            or not isinstance(goal.get("met"), bool)
            for goal in goals
        )
        or len({goal["id"] for goal in goals}) != len(goals)
    ):
        problems.append("missing_frozen_goal_evidence")
    if not _finite_number(verdict.get("judge_score"), maximum=1) or verdict.get("judge_score") != score:
        problems.append("verdict_disagrees_with_grade")
    if str(verdict.get("judge_runtime", "")).split()[:1] != ["2.1.108"]:
        problems.append("unqualified_judge_runtime")
    activations = row.get("ng_activations")
    valid_activations = (
        isinstance(activations, list)
        and bool(activations)
        and all(
            isinstance(activation, dict)
            and type(activation.get("activation_id")) is int
            and activation["activation_id"] == index
            for index, activation in enumerate(activations)
        )
    )
    if not valid_activations:
        problems.append("missing_or_unordered_activations")
    else:
        dispatched = False
        native_session = None
        for index, activation in enumerate(activations):
            response = activation.get("response")
            metadata = response.get("metadata") if isinstance(response, dict) else None
            if not isinstance(metadata, dict):
                problems.append("unqualified_candidate_runtime")
                break
            if metadata.get("native_input_dispatched") == "false":
                observation = activation.get("observation")
                observation = observation if isinstance(observation, dict) else {}
                steps = row.get("ng_steps")
                step = steps[index] if isinstance(steps, list) and index < len(steps) else {}
                if (
                    not dispatched
                    or index != len(activations) - 1
                    or not native_session
                    or metadata.get("opencode_session_id") != native_session
                    or response.get("status") != "incomplete"
                    or response.get("output") != []
                    or activation.get("turn_complete") is not False
                    or activation.get("stop_reason") != "session_budget_exhausted"
                    or observation.get("harness_steps") != 0
                    or observation.get("events") != []
                    or not isinstance(step, dict)
                    or step.get("continue_episode") is not False
                    or step.get("stop_reason") != "session_budget_exhausted"
                ):
                    problems.append("invalid_undispatched_budget_checkpoint")
                    break
            elif metadata.get("opencode_version") != "1.15.13":
                problems.append("unqualified_candidate_runtime")
                break
            else:
                dispatched = True
                native_session = metadata.get("opencode_session_id")
    close = row.get("ng_agent_close")
    if (
        not isinstance(close, dict)
        or close.get("cleanup_confirmed") is not True
        or close.get("activations") != activations
    ):
        problems.append("unconfirmed_candidate_close")
    steps = row.get("ng_steps")
    if not valid_activations or not isinstance(steps, list) or len(steps) != len(activations):
        problems.append("missing_resource_steps")
    elif any(
        not isinstance(step, dict)
        or type(step.get("activation_id")) is not int
        or step["activation_id"] != index
        or step.get("continue_episode") is not (index < len(steps) - 1)
        or (step.get("responses_create_params") is not None) != step["continue_episode"]
        for index, step in enumerate(steps)
    ):
        problems.append("invalid_resource_steps")
    if row.get("tagger_error") or not _finite_number(row.get("user_correction")):
        problems.append("missing_user_correction")
    coverage = row.get("intent_coverage")
    if (
        row.get("coverage_error")
        or not isinstance(coverage, dict)
        or not all(
            _finite_number(coverage.get(metric), maximum=1)
            for metric in ("coverage_rate", "weighted_coverage", "scope_precision", "overall_score")
        )
    ):
        problems.append("missing_intent_coverage")
    return problems


def audit(paths: list[Path], expected: set[str]) -> dict:
    attempts = []
    healthy = set()
    returned = set()
    reasons = Counter()
    for path in paths:
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = row.get("_ng_task_id", row.get("task_id")) if isinstance(row, dict) else None
                task_id = identity.get("task_id") if isinstance(identity, dict) else identity
                if isinstance(row, dict) and isinstance(row.get("result"), dict):
                    row = row["result"]
                problems = healthy_record(row)
                if isinstance(row, dict) and row.get("task_id") is not None:
                    result_identity = row["task_id"]
                    result_task = (
                        result_identity.get("task_id") if isinstance(result_identity, dict) else result_identity
                    )
                    if result_task != task_id:
                        problems.append("task_identity_mismatch")
                if task_id not in expected:
                    problems.append("unexpected_task")
                returned.add(task_id)
                reasons.update(problems)
                if not problems:
                    healthy.add(task_id)
                attempts.append({"file": str(path), "line": line_number, "task_id": task_id, "problems": problems})
    missing = sorted(expected - healthy)
    return {
        "planned_tasks": len(expected),
        "returned_tasks": len(returned & expected),
        "attempts": len(attempts),
        "healthy_graded_tasks": len(healthy),
        "healthy_attempts": sum(not attempt["problems"] for attempt in attempts),
        "tasks_without_healthy_grade": missing,
        "problem_counts": dict(reasons),
        "attempt_records": attempts,
        "passed": bool(expected) and not missing and not (returned - expected),
        "scope": (
            "At least one healthy graded attempt per planned task, including continuation, scoring, metrics, "
            "and agent-close evidence. Separately audit provider cleanup and reference repeat coverage."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="Materialized input JSONL; use all 109 for release.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {json.loads(line)["task_id"]["task_id"] for line in args.tasks.read_text().splitlines() if line.strip()}
    report = audit(args.results, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "attempt_records"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
