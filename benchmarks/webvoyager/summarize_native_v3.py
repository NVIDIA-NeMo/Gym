# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Account for every native WebVoyager task without hiding failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_TASKS = 552


def task_id_from_row(row: dict[str, Any]) -> str:
    task_id = row.get("task_id") or row.get("id")
    if task_id:
        return str(task_id)
    metadata = (row.get("responses_create_params") or {}).get("metadata") or {}
    return str(metadata.get("task_id") or "")


def summarize(
    rows: list[dict[str, Any]],
    *,
    expected_task_ids: set[str] | None = None,
    superseded_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    for row in rows:
        task_id = task_id_from_row(row)
        if not task_id:
            continue
        duplicates[task_id] += 1
        by_task[task_id] = row

    expected = expected_task_ids or set(by_task)
    expected_count = len(expected_task_ids) if expected_task_ids is not None else EXPECTED_TASKS
    completed_ids = set(by_task) & expected
    missing_ids = sorted(expected - set(by_task))
    unexpected_ids = sorted(set(by_task) - expected) if expected_task_ids is not None else []
    expected_rows = [by_task[task_id] for task_id in completed_ids]
    success = sum(bool(row.get("task_success")) for row in expected_rows)
    invalid_ids = sorted(task_id for task_id in completed_ids if bool(by_task[task_id].get("mask_sample")))
    invalid = len(invalid_ids)
    retry_ids = sorted(set(missing_ids) | set(invalid_ids))
    failures = Counter(
        str(row.get("failure_kind") or "policy_failure") for row in expected_rows if not row.get("task_success")
    )
    superseded = superseded_task_ids or set()
    duplicate_ids = sorted(task_id for task_id, count in duplicates.items() if count > 1 and task_id not in superseded)
    superseded_ids = sorted(task_id for task_id, count in duplicates.items() if count > 1 and task_id in superseded)
    return {
        "expected": expected_count,
        "completed_unique": len(completed_ids),
        "missing": max(0, expected_count - len(completed_ids)),
        "missing_task_ids": missing_ids,
        "invalid_task_ids": invalid_ids,
        "retry_task_ids": retry_ids,
        "unexpected_task_ids": unexpected_ids,
        "success": success,
        "strict_sr": success / expected_count if expected_count else 0.0,
        "invalid_or_infrastructure": invalid,
        "duplicate_task_ids": duplicate_ids,
        "superseded_task_ids": superseded_ids,
        "failure_kinds": dict(sorted(failures.items())),
        "comparable": (
            len(completed_ids) == expected_count and invalid == 0 and not duplicate_ids and not unexpected_ids
        ),
    }


def _jsonl_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("rollouts.jsonl") if candidate.is_file())
    return [path]


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    """Discard trajectory payloads that are irrelevant to score reconciliation.

    Native WebVoyager rollout rows can contain many screenshots and reach tens
    of megabytes each.  Keeping those rows alive while reading an entire wave
    makes a fixed-size score summary consume memory proportional to the full
    trajectory archive.  Reconciliation only needs the task identity and four
    scalar result fields, so compact each row before retaining it.
    """

    return {
        "task_id": task_id_from_row(row),
        "task_success": row.get("task_success"),
        "mask_sample": row.get("mask_sample"),
        "failure_kind": row.get("failure_kind"),
    }


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for jsonl in _jsonl_files(path):
            with jsonl.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        rows.append(_summary_row(json.loads(line)))
    return rows


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    task_ids = {task_id_from_row(row) for row in rows}
    task_ids.discard("")
    if len(rows) != len(task_ids):
        raise ValueError(f"dataset task IDs are missing or duplicated: rows={len(rows)} unique={len(task_ids)}")
    return rows, task_ids


def write_missing_rows(dataset_rows: list[dict[str, Any]], missing_ids: set[str], output: Path) -> None:
    selected = [row for row in dataset_rows if task_id_from_row(row) in missing_ids]
    if len(selected) != len(missing_ids):
        raise ValueError(f"missing-row closure mismatch: expected={len(missing_ids)} selected={len(selected)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    partial.replace(output)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--missing-output", type=Path)
    parser.add_argument(
        "--superseded-ids-jsonl",
        type=Path,
        help="Task IDs intentionally rerun; the last loaded result wins without counting as an accidental duplicate.",
    )
    args = parser.parse_args()
    dataset_rows: list[dict[str, Any]] | None = None
    expected_task_ids: set[str] | None = None
    if args.dataset:
        dataset_rows, expected_task_ids = load_dataset(args.dataset)
    superseded_task_ids: set[str] | None = None
    if args.superseded_ids_jsonl:
        _, superseded_task_ids = load_dataset(args.superseded_ids_jsonl)
    report = summarize(
        load_rows(args.rollouts),
        expected_task_ids=expected_task_ids,
        superseded_task_ids=superseded_task_ids,
    )
    if args.missing_output:
        if dataset_rows is None:
            parser.error("--missing-output requires --dataset")
        write_missing_rows(dataset_rows, set(report["retry_task_ids"]), args.missing_output)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        write_report(report, args.output)
    print(text)
