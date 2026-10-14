# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supplement Gym trajectory health with SWE-Together lifecycle and scoring evidence.

A healthy smoke attempt may solve none of the frozen goals. This audit checks that
candidate continuation, cleanup, grading, and interaction measurements succeeded.
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from environment_servers.swe_together.coverage import compute_scores
from environment_servers.swe_together.evaluation import PASS_THRESHOLD, derive_score
from environment_servers.swe_together.task import TaskData, load_task


def _number(value: object, *, maximum: float | None = None) -> bool:
    return (
        type(value) in (int, float)
        and (type(value) is int or math.isfinite(value))
        and value >= 0
        and (maximum is None or value <= maximum)
    )


def _counter(value: object) -> bool:
    return type(value) is int and value >= 0


def _grade_problems(row: dict, goals: dict[str, object]) -> list[str]:
    problems = []
    verdict = row.get("verdict")
    if not isinstance(verdict, dict):
        return ["missing_frozen_goal_evidence", "unconfirmed_judge_close"]
    rows = verdict.get("goal_results")
    if not isinstance(rows, list) or not all(isinstance(goal, dict) for goal in rows):
        problems.append("invalid_frozen_goal_evidence")
    else:
        try:
            derived = derive_score(goals, verdict)
        except (KeyError, TypeError, ValueError):
            problems.append("invalid_frozen_goal_evidence")
        else:
            score = derived["judge_score"]
            if not _number(row.get("judge_score"), maximum=1) or row["judge_score"] != score:
                problems.append("grade_disagrees_with_frozen_goals")
            if not _number(verdict.get("judge_score"), maximum=1) or verdict["judge_score"] != score:
                problems.append("verdict_disagrees_with_frozen_goals")
            if verdict.get("verdict") != derived["verdict"]:
                problems.append("verdict_label_disagrees_with_frozen_goals")
            if not _number(row.get("reward"), maximum=1) or row["reward"] != float(score >= PASS_THRESHOLD):
                problems.append("reward_disagrees_with_grade")
    close = verdict.get("judge_close")
    if (
        not isinstance(close, dict)
        or close.get("cleanup_confirmed") is not True
        or not isinstance(close.get("agent_session_id"), str)
        or not close["agent_session_id"]
        or ("judge_close" in row and row["judge_close"] != close)
    ):
        problems.append("unconfirmed_judge_close")
    return problems


def _lifecycle_problems(row: dict) -> list[str]:
    problems = []
    activations = row.get("ng_activations")
    steps = row.get("ng_steps")
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
    valid_steps = (
        valid_activations
        and isinstance(steps, list)
        and len(steps) == len(activations)
        and all(
            isinstance(step, dict)
            and type(step.get("activation_id")) is int
            and step["activation_id"] == index
            and step.get("kind") == ("continue" if index < len(steps) - 1 else "stop")
            and (step.get("responses_create_params") is not None) == (step["kind"] == "continue")
            for index, step in enumerate(steps)
        )
    )
    if not valid_steps:
        problems.append("invalid_next_user_turns")
    conversation_id = None
    if valid_activations:
        for index, activation in enumerate(activations):
            response = activation.get("response")
            observation = activation.get("observation")
            if not isinstance(response, dict) or not isinstance(observation, dict):
                problems.append("missing_activation_evidence")
                break
            metadata = response.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            native_id = metadata.get("native_conversation_id")
            if (
                not isinstance(native_id, str)
                or not native_id
                or (conversation_id is not None and native_id != conversation_id)
            ):
                problems.append("missing_or_changed_native_conversation")
                break
            launched = metadata.get("launch_attempted")
            accepted = metadata.get("native_input_accepted")
            if launched not in {"true", "false"} or accepted not in {"true", "false", "unknown"}:
                problems.append("missing_candidate_input_evidence")
                break
            if (launched == "false" and accepted != "false") or (launched == "true" and accepted == "false"):
                problems.append("inconsistent_candidate_input_evidence")
                break
            if accepted != "true":
                step = steps[index] if valid_steps else {}
                if (
                    conversation_id is None
                    or index != len(activations) - 1
                    or response.get("status") != "incomplete"
                    or response.get("output") != []
                    or response.get("usage") is not None
                    or activation.get("turn_complete") is not False
                    or activation.get("stop_reason") != "session_budget_exhausted"
                    or observation.get("events") != []
                    or observation.get("harness_steps") != 0
                    or step.get("kind") != "stop"
                    or step.get("stop_reason") != "session_budget_exhausted"
                ):
                    problems.append("invalid_unaccepted_budget_checkpoint")
                    break
            else:
                if response.get("status") not in ("completed", "incomplete") or not isinstance(
                    response.get("output"), list
                ):
                    problems.append("failed_candidate_activation")
                    break
                if activation.get("turn_complete") is not (response["status"] == "completed"):
                    problems.append("invalid_activation_completion")
                conversation_id = native_id
            events = observation.get("events")
            if (
                not isinstance(events, list)
                or not all(isinstance(event, dict) and _counter(event.get("sequence")) for event in events)
                or [event["sequence"] for event in events] != sorted({event["sequence"] for event in events})
                or not _number(observation.get("duration_seconds"))
                or not _number(observation.get("elapsed_seconds"))
                or (observation.get("harness_steps") is not None and not _counter(observation["harness_steps"]))
            ):
                problems.append("invalid_activation_observations")
                break
    close = row.get("ng_agent_close")
    if (
        not isinstance(close, dict)
        or close.get("cleanup_confirmed") is not True
        or not isinstance(close.get("agent_session_id"), str)
        or not close["agent_session_id"]
        or not valid_activations
        or close.get("activations") != activations
    ):
        problems.append("unconfirmed_candidate_close")
    cleanup = row.get("cleanup_receipts")
    for role in ("candidate_sandbox", "judge_sandbox"):
        receipt = cleanup.get(role) if isinstance(cleanup, dict) else None
        if not isinstance(receipt, dict) or receipt.get("cleanup_confirmed") is not True:
            problems.append("unconfirmed_" + role + "_close")
    submission = row.get("submission")
    if (
        not isinstance(submission, dict)
        or not isinstance(submission.get("archive_sha256"), str)
        or re.fullmatch(r"[a-f0-9]{64}", submission["archive_sha256"]) is None
        or not _counter(submission.get("members"))
        or not isinstance(submission.get("restored_roots"), list)
        or not submission["restored_roots"]
        or not all(
            isinstance(root, str) and root.startswith("/") and root != "/" for root in submission["restored_roots"]
        )
    ):
        problems.append("missing_frozen_submission_receipt")
    return problems


def _metric_problems(row: dict, n_intents: int | None) -> list[str]:
    problems = []
    count = row.get("n_simulator_messages")
    correction = row.get("user_correction")
    tags = row.get("message_tags")
    valid_correction = row.get("tagger_error") is None and _counter(count) and _number(correction)
    if valid_correction and count == 0:
        valid_correction = correction == 0
    elif valid_correction:
        valid_correction = (
            isinstance(tags, list)
            and len(tags) == count
            and all(
                isinstance(tag, dict)
                and _counter(tag.get("trial_idx"))
                and isinstance(tag.get("tags"), list)
                and all(isinstance(value, str) for value in tag["tags"])
                for tag in tags
            )
        )
        if valid_correction:
            valid_correction = {tag["trial_idx"] for tag in tags} == set(range(1, count + 1)) and correction == round(
                sum(("correction" in tag["tags"]) + 0.2 * ("nudge" in tag["tags"]) for tag in tags), 4
            )
    if not valid_correction:
        problems.append("missing_or_invalid_user_correction")
    coverage = row.get("intent_coverage")
    metrics = ("coverage_rate", "weighted_coverage", "scope_precision", "overall_score")
    if (
        row.get("coverage_error") is not None
        or not isinstance(coverage, dict)
        or coverage.get("schema_warnings")
        or not all(_number(coverage.get(metric), maximum=1) for metric in metrics)
    ):
        problems.append("missing_or_invalid_intent_coverage")
    elif "match_table" not in coverage:
        if n_intents not in (None, 0) or (n_intents == 0 and any(coverage[metric] != 1 for metric in metrics)):
            problems.append("missing_intent_coverage_evidence")
    else:
        table = coverage["match_table"]
        valid_table = (
            isinstance(table, dict) and _counter(table.get("n_intents")) and _counter(table.get("n_trial_msgs"))
        )
        if valid_table:
            valid_table = (n_intents is None or table["n_intents"] == n_intents) and (
                _counter(count) and table["n_trial_msgs"] == count + 1
            )
        if valid_table:
            entries = table.get("per_intent")
            valid_table = (
                isinstance(entries, list)
                and len(entries) == table["n_intents"]
                and all(
                    isinstance(entry, dict)
                    and _counter(entry.get("intent_id"))
                    and _number(entry.get("match_confidence"), maximum=1)
                    and (
                        entry.get("matched_trial_idx") is None
                        or (
                            _counter(entry["matched_trial_idx"]) and entry["matched_trial_idx"] < table["n_trial_msgs"]
                        )
                    )
                    for entry in entries
                )
            )
            if valid_table:
                valid_table = {entry["intent_id"] for entry in entries} == set(range(table["n_intents"]))
        derived = compute_scores(table, table["n_intents"], table["n_trial_msgs"]) if valid_table else None
        if derived is None or any(coverage[metric] != derived[metric] for metric in metrics):
            problems.append("intent_coverage_disagrees_with_evidence")
    return problems


def healthy_record(row: object, *, goals: dict[str, object], n_intents: int | None = None) -> list[str]:
    """Return reasons an attempt fails smoke health, recomputing its frozen-goal score."""
    if not isinstance(row, dict):
        return ["invalid_record"]
    problems = []
    if row.get("mask_sample") or any(
        row.get(key) is not None for key in ("failure_kind", "_ng_failure_class", "failure")
    ):
        problems.append("failed_or_masked")
    return problems + _grade_problems(row, goals) + _lifecycle_problems(row) + _metric_problems(row, n_intents)


def audit(
    paths: Sequence[Path],
    expected_task_data: Mapping[str, Mapping[str, object]],
    *,
    task_source: Path | None = None,
) -> dict[str, object]:
    """Select the first healthy attempt per task in supplied file order, retaining every failure."""
    task_source = task_source or Path(
        os.environ.get(
            "SWE_TOGETHER_SOURCE", Path(os.environ.get("SWE_TOGETHER_WORK_DIR", "outputs/swe_together")) / "source"
        )
    )
    goals = {}
    n_intents = {}
    for task_id, data in expected_task_data.items():
        task = TaskData.model_validate(data)
        if task.task_id != task_id:
            raise ValueError(f"Expected task identity disagrees with task data: {task_id}")
        directory = task_source / "tasks" / task_id
        load_task(directory, task)
        goals[task_id] = json.loads((directory / "canonical_goals.json").read_text())
        n_intents[task_id] = len(json.loads((directory / "oracle_intents.json").read_text())["intents"])
    attempts = []
    selected = {}
    returned = set()
    reasons = Counter()
    for path in paths:
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    outer = json.loads(line)
                except json.JSONDecodeError:
                    outer = None
                row = outer.get("result", outer) if isinstance(outer, dict) else outer
                identity = outer.get("_ng_task_id", outer.get("task_id")) if isinstance(outer, dict) else None
                if identity is None and isinstance(row, dict):
                    identity = row.get("task_id")
                task_id = identity.get("task_id") if isinstance(identity, dict) else identity
                if not isinstance(task_id, str) or task_id not in expected_task_data:
                    problems = ["unexpected_task"]
                else:
                    problems = healthy_record(row, goals=goals[task_id], n_intents=n_intents[task_id])
                    if isinstance(row, dict) and row.get("task_id") != task_id:
                        problems.append("task_identity_mismatch")
                    if (
                        isinstance(outer, dict)
                        and outer.get("failure") is not None
                        and "failed_or_masked" not in problems
                    ):
                        problems.append("failed_or_masked")
                    returned.add(task_id)
                location = {"file": str(path), "line": line_number, "task_id": task_id, "problems": problems}
                attempts.append(location)
                reasons.update(problems)
                if not problems and task_id not in selected:
                    selected[task_id] = location
    missing = sorted(set(expected_task_data) - selected.keys())
    return {
        "passed": bool(expected_task_data) and not missing and "unexpected_task" not in reasons,
        "planned_tasks": len(expected_task_data),
        "returned_tasks": len(returned),
        "attempts": len(attempts),
        "healthy_attempts": sum(not record["problems"] for record in attempts),
        "healthy_graded_tasks": len(selected),
        "tasks_without_healthy_grade": missing,
        "selected_attempts": selected,
        "problem_counts": dict(reasons),
        "attempt_records": attempts,
        "scope": "Smoke health requires one healthy graded attempt per task, independent of solve reward. Gym trajectory health runs separately.",
    }


def main() -> None:
    """Audit collector output against the materialized taskset and its pinned task assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="Materialized input taskset JSONL.")
    parser.add_argument(
        "--source",
        "--task-source",
        dest="task_source",
        type=Path,
        help="Pinned upstream checkout; defaults to SWE_TOGETHER_SOURCE.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {}
    for line in args.tasks.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            task_id = row["task_id"]["task_id"]
            if task_id in expected:
                raise ValueError(f"Duplicate planned task: {task_id}")
            expected[task_id] = row["task_input"]["task_data"]
    report = audit(args.results, expected, task_source=args.task_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"attempt_records", "selected_attempts"}},
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
