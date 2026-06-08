#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize swe_agents golden-patch validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def is_pass(row: dict[str, Any]) -> bool:
    if row.get("resolved") is True:
        return True
    try:
        return float(row.get("reward", 0.0)) == 1.0
    except (TypeError, ValueError):
        return False


def instance_id(row: dict[str, Any]) -> str:
    cfg = row.get("instance_config") or {}
    problem_info = cfg.get("problem_info") or {}
    if isinstance(problem_info, str):
        try:
            problem_info = json.loads(problem_info)
        except json.JSONDecodeError:
            problem_info = {}
    return (
        problem_info.get("instance_id")
        or cfg.get("instance_id")
        or row.get("instance_id")
        or row.get("id")
        or "<unknown>"
    )


def eval_root(row: dict[str, Any]) -> Path | None:
    cfg = row.get("instance_config") or {}
    output_for_eval = cfg.get("output_for_eval_path")
    if not output_for_eval:
        return None
    path = Path(output_for_eval)
    if len(path.parents) < 3:
        return None
    return path.parents[2]


def first_instance_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        cfg = row.get("instance_config")
        if isinstance(cfg, dict):
            return cfg
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", required=True, help="Result JSONL from ng_collect_rollouts")
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--show-settings", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.output_jsonl)
    rows = load_rows(path)
    passed = [row for row in rows if is_pass(row)]
    failed = [row for row in rows if not is_pass(row)]

    summary = {
        "output": str(path),
        "rows": len(rows),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": (len(passed) / len(rows) if rows else None),
    }
    if args.expected_count is not None:
        summary["expected_count"] = args.expected_count
        summary["remaining"] = max(args.expected_count - len(rows), 0)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"output: {path}")
        print(f"rows: {summary['rows']}")
        print(f"passed: {summary['passed']}")
        print(f"failed: {summary['failed']}")
        if summary["pass_rate"] is None:
            print("pass_rate: n/a")
        else:
            print(f"pass_rate: {summary['pass_rate'] * 100:.2f}%")
        if args.expected_count is not None:
            print(f"remaining: {summary['remaining']}")

    if args.show_settings and rows:
        cfg = first_instance_config(rows)
        print("settings:")
        for key in (
            "dataset_path",
            "concurrency",
            "apptainer_memory_limit_mb",
            "swebench_tests_timeout",
            "verify_golden_patch",
        ):
            print(f"  {key}: {cfg.get(key)}")

    if args.show_failures and failed:
        print("failures:")
        for row in failed:
            root = eval_root(row)
            print(instance_id(row))
            print(f"  eval_root: {root}")
            if root is not None:
                print(f"  report: {root / 'eval_results' / 'report.json'}")
                print(f"  test_output: {root / 'eval_results' / 'test_output.log'}")
                print(f"  eval_script: {root / 'container_scripts' / 'eval_script.sh'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
