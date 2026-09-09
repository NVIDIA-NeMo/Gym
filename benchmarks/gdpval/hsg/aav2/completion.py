#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native Gym outputs without changing journals, caches, or scores."""

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
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


def rollout_complete(dataset: Path, deliverables: Path) -> int:
    expected = task_ids(dataset)
    for task_id in sorted(expected):
        marker = deliverables / f"task_{task_id}" / "repeat_0" / "finish_params.json"
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
        if (
            row.get("error")
            or row.get("_ng_failure_class")
            or row.get("invalid_judge_response")
            or judge.get("error")
            or judge.get("scoring_error")
            or judge.get("ref_errors")
            or judge.get("total_judged") != trials
            or judge.get("total_invalid") != 0
        ):
            raise ValueError(f"task lacks {trials} valid votes: {row.get('task_id')}")
    final_rows = [row for row in rows if row.get("stage_index") == final_stage]
    final_ids = {row.get("task_id") for row in final_rows}
    if len(final_rows) != expected_count or len(final_ids) != expected_count or not final_ids <= expected:
        raise ValueError("final stage has duplicate, missing, or unexpected tasks")
    metrics_path = output.with_stem(output.stem + "_aggregate_metrics").with_suffix(".json")
    entries = json.loads(metrics_path.read_text())
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("expected one agent's native aggregate metrics")
    metrics = entries[0]["agent_metrics"]
    required = {"final_stage_present": 1, "final_stage_complete": 1}
    if mode == "full":
        required.update(final_stage_fit=1, final_stage_degraded=0)
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
    print(f"COMPLETE: {args.phase}, {count} tasks")


if __name__ == "__main__":
    main()
