#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate one rollout shard against a shared GDPVal deliverables root."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence


class ShardCoverageError(ValueError):
    """Raised when shard inputs or completion markers violate the contract."""


def _regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not path.is_symlink()


def _read_task_ids(dataset: Path) -> list[str]:
    try:
        resolved = dataset.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ShardCoverageError(f"dataset is unavailable: {dataset}: {exc}") from exc
    if not _regular_file(resolved):
        raise ShardCoverageError(f"dataset is not a regular non-symlink file: {resolved}")

    task_ids: list[str] = []
    with resolved.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                raise ShardCoverageError(f"dataset has a blank line at {resolved}:{line_number}")
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ShardCoverageError(f"invalid dataset JSON at {resolved}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ShardCoverageError(f"dataset row {line_number} is not an object")
            task_id = row.get("task_id")
            if (
                not isinstance(task_id, str)
                or not task_id
                or task_id in (".", "..")
                or "/" in task_id
                or "\x00" in task_id
                or (os.altsep is not None and os.altsep in task_id)
            ):
                raise ShardCoverageError(f"dataset row {line_number} has an invalid path-safe task_id")
            task_ids.append(task_id)

    if not task_ids:
        raise ShardCoverageError(f"dataset has no rows: {resolved}")
    if len(set(task_ids)) != len(task_ids):
        raise ShardCoverageError("dataset task_id values are not unique")
    return task_ids


def _validate_marker(marker: Path) -> str | None:
    if not _regular_file(marker):
        return "missing_or_nonregular"
    try:
        value: Any = json.loads(marker.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "invalid_json"
    if value is not None and not isinstance(value, dict):
        return "not_an_object_or_null"
    return None


def shard_coverage(dataset: Path, deliverables: Path) -> dict[str, Any]:
    task_ids = _read_task_ids(dataset)
    try:
        root = deliverables.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ShardCoverageError(f"deliverables root is unavailable: {deliverables}: {exc}") from exc
    if not root.is_dir():
        raise ShardCoverageError(f"deliverables root is not a directory: {root}")

    completed: list[str] = []
    invalid: dict[str, str] = {}
    for task_id in task_ids:
        marker = root / f"task_{task_id}" / "repeat_0" / "finish_params.json"
        error = _validate_marker(marker)
        if error is None:
            completed.append(task_id)
        else:
            invalid[task_id] = error

    # This number is informational only. Sibling shards intentionally publish
    # into the same root, so only membership of this shard's task IDs gates PASS.
    shared_markers = sum(1 for marker in root.glob("task_*/repeat_0/finish_params.json") if _regular_file(marker))
    missing = [task_id for task_id in task_ids if task_id not in completed]
    return {
        "status": "PASS" if not missing else "INCOMPLETE",
        "dataset": str(dataset.expanduser().resolve()),
        "deliverables": str(root),
        "expected": len(task_ids),
        "completed": len(completed),
        "missing": missing,
        "invalid": invalid,
        "shared_markers": shared_markers,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--deliverables", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = shard_coverage(args.dataset, args.deliverables)
    except (OSError, ShardCoverageError) as exc:
        print(f"SHARD_COVERAGE_ERROR: {exc}", file=sys.stderr)
        return 64
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "SHARD_COVERAGE "
            f"status={report['status']} completed={report['completed']}/{report['expected']} "
            f"shared_markers={report['shared_markers']} missing={len(report['missing'])} "
            f"invalid={len(report['invalid'])}"
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
