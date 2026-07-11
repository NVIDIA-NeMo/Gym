#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail fast when two OSWorld inputs materialize different task definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FIELDS = ("instruction", "config", "evaluator")


def _records(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value
        return

    value = json.loads(text)
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict) and isinstance(value.get("tasks"), list):
        items = value["tasks"]
    elif isinstance(value, dict):
        items = list(value.values()) if value and all(isinstance(item, dict) for item in value.values()) else [value]
    else:
        raise ValueError(f"{path}: expected an object, task map, task list, or JSONL")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: expected every task to be a JSON object")
        yield item


def _task(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("verifier_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("osworld_task"), dict):
        return metadata["osworld_task"]
    if isinstance(record.get("osworld_task"), dict):
        return record["osworld_task"]
    return record


def _task_id(record: dict[str, Any]) -> str:
    task = _task(record)
    metadata = record.get("verifier_metadata")
    candidates = [
        task.get("id"),
        task.get("task_id"),
        metadata.get("task_id") if isinstance(metadata, dict) else None,
        record.get("id"),
        record.get("task_id"),
    ]
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return str(candidate)
    raise ValueError("OSWorld task record has no id/task_id")


def _canonical(task: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: task.get(field) for field in fields}


def _digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_tasks(path: Path, fields: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for record in _records(path):
        task_id = _task_id(record)
        if task_id in tasks:
            raise ValueError(f"{path}: duplicate task id {task_id}")
        tasks[task_id] = _canonical(_task(record), fields)
    return tasks


def compare_inputs(
    left_path: Path,
    right_path: Path,
    *,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    selected_ids: set[str] | None = None,
) -> dict[str, Any]:
    left = load_tasks(left_path, fields)
    right = load_tasks(right_path, fields)
    ids = selected_ids if selected_ids is not None else set(left) | set(right)
    missing_left = sorted(task_id for task_id in ids if task_id not in left)
    missing_right = sorted(task_id for task_id in ids if task_id not in right)
    mismatches = []
    matched = 0
    for task_id in sorted(ids - set(missing_left) - set(missing_right)):
        left_task = left[task_id]
        right_task = right[task_id]
        if left_task == right_task:
            matched += 1
            continue
        changed_fields = [field for field in fields if left_task.get(field) != right_task.get(field)]
        mismatches.append(
            {
                "task_id": task_id,
                "changed_fields": changed_fields,
                "left_sha256": _digest(left_task),
                "right_sha256": _digest(right_task),
                "field_differences": {
                    field: {"left": left_task.get(field), "right": right_task.get(field)} for field in changed_fields
                },
            }
        )
    return {
        "schema_version": 1,
        "left": str(left_path),
        "right": str(right_path),
        "fields": list(fields),
        "selected_tasks": len(ids),
        "matched": matched,
        "mismatched": len(mismatches),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "parity": not mismatches and not missing_left and not missing_right,
        "mismatches": mismatches,
    }


def _selected_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="Internal/canonical JSON or JSONL input")
    parser.add_argument("right", type=Path, help="Gym/candidate JSON or JSONL input")
    parser.add_argument("--task-ids", type=Path, help="Optional newline-delimited task-ID allowlist")
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    parser.add_argument("--output", type=Path, help="Optional path for the same JSON report")
    args = parser.parse_args()

    try:
        report = compare_inputs(
            args.left,
            args.right,
            fields=tuple(args.fields),
            selected_ids=_selected_ids(args.task_ids),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        rendered = json.dumps({"parity": False, "error": str(exc)}, ensure_ascii=False, indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["parity"] else 1


if __name__ == "__main__":
    sys.exit(main())
