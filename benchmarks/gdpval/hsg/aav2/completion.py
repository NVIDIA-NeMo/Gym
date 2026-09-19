#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native Gym outputs without changing journals, caches, or scores."""

import argparse
import json
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected nonempty JSONL objects: {path}")
    return rows


def task_ids(dataset: Path) -> set[str]:
    rows = read_rows(dataset)
    ids = [row.get("task_id") for row in rows]
    if any(not isinstance(value, str) or not value or value in (".", "..") or "/" in value for value in ids):
        raise ValueError("dataset requires path-safe task_id strings")
    if len(set(ids)) != len(ids):
        raise ValueError("dataset has duplicate task IDs; this launcher supports one repeat per task")
    return set(ids)


def rollout_complete(dataset: Path, deliverables: Path, missing_task_ids: set[str] | None = None) -> int:
    expected = task_ids(dataset)
    missing_task_ids = missing_task_ids or set()
    if not missing_task_ids <= expected:
        raise ValueError("missing-task allowance contains tasks outside the dataset")
    for task_id in sorted(expected):
        marker = deliverables / f"task_{task_id}" / "repeat_0" / "finish_params.json"
        if task_id in missing_task_ids:
            if marker.exists() or marker.is_symlink():
                raise ValueError(f"previously missing finish marker appeared; use a fresh import: {task_id}")
            continue
        if not marker.is_file() or marker.is_symlink():
            raise ValueError(f"task has no finish marker: {task_id}")
        value = json.loads(marker.read_text())
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"invalid finish marker: {marker}")
    return len(expected)


def judge_complete(dataset: Path, output: Path, mode: str = "full") -> int:
    expected = task_ids(dataset)
    trials = {"smoke": 1, "pilot": 2, "full": 4}[mode]
    expected_count = min(len(expected), {"smoke": 4, "pilot": 12, "full": len(expected)}[mode])
    rows = read_rows(output)
    stages = {row.get("expected_final_stage_index") for row in rows}
    if len(stages) != 1 or any(type(stage) is not int or stage < 0 for stage in stages):
        raise ValueError("judgments do not declare a consistent final stage")
    final_stage = stages.pop()
    if final_stage != (0 if mode == "smoke" else 1):
        raise ValueError("judgments do not match the requested stage protocol")
    for row in rows:
        judge = row.get("judge_response") or {}
        if judge.get("manual_imputation") and row.get("stage_index") != 1:
            raise ValueError("calibration must contain actual judgments, never imputed losses")
        judged, invalid = judge.get("total_judged"), judge.get("total_invalid")
        valid_votes = judged == trials and invalid == 0
        if mode == "full":
            valid_votes = (
                type(judged) is int
                and type(invalid) is int
                and 0 < judged <= trials
                and 0 <= invalid <= trials - judged
            )
        if (
            row.get("error")
            or row.get("_ng_failure_class")
            or row.get("invalid_judge_response")
            or judge.get("error")
            or judge.get("scoring_error")
            or judge.get("ref_errors")
            or not valid_votes
        ):
            raise ValueError(f"task lacks valid votes: {row.get('task_id')}")
    final_rows = [row for row in rows if row.get("stage_index") == final_stage]
    final_ids = {row.get("task_id") for row in final_rows}
    if len(final_rows) != len(final_ids) or not final_ids <= expected:
        raise ValueError("final stage has duplicate, missing, or unexpected tasks")
    if mode == "full":
        journal = read_rows(output.with_stem(output.stem + "_multistage_state").with_suffix(".jsonl"))
        plans = {row["stage_index"]: row for row in journal if row.get("status") == "planned"}
        calibration = [row for row in rows if row.get("stage_index") == 0]
        calibration_ids = {row.get("task_id") for row in calibration}
        planned_calibration = plans[0]["task_ids"]
        if (
            len(calibration) != min(45, len(expected))
            or len(calibration_ids) != len(calibration)
            or calibration_ids != set(planned_calibration)
            or len(planned_calibration) != len(calibration)
            or not calibration_ids <= expected
        ):
            raise ValueError("calibration requires every planned task to have valid votes")
        plan = plans[final_stage]
        if set(plan["task_ids"]) != expected or len(plan["task_ids"]) != expected_count:
            raise ValueError("final stage plan does not match the dataset")
        missing = expected - final_ids
        if missing:
            failures = output.with_stem(output.stem + "_failures").with_suffix(".jsonl")
            failed_ids = set()
            if failures.exists():
                with failures.open() as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        failure = json.loads(line)
                        task_id = failure.get("task_id")
                        if (
                            failure.get("stage_index") == final_stage
                            and failure.get("_ng_failure_class")
                            and not failure.get("_ng_no_persist")
                            and task_id in missing
                            and failure.get("reference_ids") == [plan["task_reference_ids"][task_id]]
                        ):
                            failed_ids.add(task_id)
            if failed_ids != missing:
                raise ValueError("final stage has missing tasks without recorded failed attempts")
    elif len(final_rows) != expected_count:
        raise ValueError("final stage has duplicate, missing, or unexpected tasks")
    metrics_path = output.with_stem(output.stem + "_aggregate_metrics").with_suffix(".json")
    entries = json.loads(metrics_path.read_text())
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("expected one agent's native aggregate metrics")
    metrics = entries[0]["agent_metrics"]
    partial = len(final_rows) < expected_count
    required = {"final_stage_present": 1, "final_stage_complete": int(not partial)}
    if mode == "full":
        required.update(final_stage_fit=1, final_stage_degraded=int(partial))
        elo = metrics.get(f"comparison/stage_{final_stage}/eval_elo")
        if not isinstance(elo, (int, float)) or not math.isfinite(elo):
            raise ValueError("native aggregate has no finite final-stage fit")
    if any(metrics.get(f"comparison/{key}") != value for key, value in required.items()):
        raise ValueError("native aggregate reports an incomplete or unfit final stage")
    return len(final_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    rollout = sub.add_parser("rollout")
    rollout.add_argument("--dataset", type=Path, required=True)
    rollout.add_argument("--deliverables", type=Path, required=True)
    judge = sub.add_parser("judge")
    judge.add_argument("--dataset", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--mode", choices=("smoke", "pilot", "full"), default="full")
    args = parser.parse_args()
    try:
        count = (
            rollout_complete(args.dataset, args.deliverables)
            if args.phase == "rollout"
            else judge_complete(args.dataset, args.output, args.mode)
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"INCOMPLETE: {error}") from error
    if args.phase == "judge":
        expected = len(task_ids(args.dataset))
        expected = min(expected, {"smoke": 4, "pilot": 12, "full": expected}[args.mode])
        imputed = sum(
            bool((row.get("judge_response") or {}).get("manual_imputation"))
            for row in read_rows(args.output)
            if row.get("stage_index") == 1
        )
        status = "PARTIAL" if count < expected else "COMPLETE"
        print(
            f"{status}: judge, {count - imputed}/{expected} actually judged tasks, "
            f"{imputed} imputed-loss tasks, {expected - count} failed tasks excluded"
        )
    else:
        print(f"COMPLETE: {args.phase}, {count} tasks")


if __name__ == "__main__":
    main()
