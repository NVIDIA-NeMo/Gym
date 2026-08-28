#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move retryable Apex infrastructure failures out of the resume-success JSONL."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


RETRYABLE_FAILURE_CLASSES = frozenset({"judge_failed", "timeout_exceeded"})
_TIMEOUT_PATTERN = re.compile(r"\b(?:timed?\s*out|timeout|deadline exceeded)\b", re.IGNORECASE)


def failure_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_failures.jsonl")


def classify_retryable_row(row: dict[str, Any]) -> str | None:
    failure_class = row.get("_ng_failure_class")
    if failure_class in RETRYABLE_FAILURE_CLASSES:
        return str(failure_class)
    if row.get("invalid_judge_response") is True or row.get("verifier_error"):
        return "judge_failed"
    apex_error = row.get("apex_error")
    if isinstance(apex_error, str) and _TIMEOUT_PATTERN.search(apex_error):
        return "timeout_exceeded"
    return None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise SystemExit(f"expected a JSON object in {path}:{line_number}")
            rows.append(row)
    return rows


def _replace_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _attempt_identity(row: dict[str, Any]) -> tuple[Any, int, int, Any]:
    return (
        row.get("task_id"),
        int(row.get("_ng_rollout_index", 0)),
        int(row.get("_ng_attempt_index", 0)),
        row.get("_ng_failure_class"),
    )


def prepare_retry_rows(output: Path) -> tuple[int, int]:
    """Quarantine retryable main rows and make matching sidecar rows nonterminal.

    Returns ``(quarantined_main_rows, reopened_terminal_rows)``. The sidecar is
    replaced before the main file; attempt identities make that ordering safe
    and idempotent if interrupted between replacements.
    """

    main_rows = _read_rows(output)
    failure_path = failure_path_for(output)
    failure_rows = _read_rows(failure_path)

    reopened = 0
    for row in failure_rows:
        if row.get("_ng_failure_class") in RETRYABLE_FAILURE_CLASSES and row.pop("_ng_failure_terminal", None):
            reopened += 1

    existing_attempts = {_attempt_identity(row) for row in failure_rows}
    kept_rows = []
    quarantined = 0
    for row in main_rows:
        failure_class = classify_retryable_row(row)
        if failure_class is None:
            kept_rows.append(row)
            continue

        quarantined += 1
        failure = dict(row)
        failure["_ng_failure_class"] = failure_class
        failure.pop("_ng_failure_terminal", None)
        failure["_ng_retry_quarantined_from_main"] = True
        identity = _attempt_identity(failure)
        if identity not in existing_attempts:
            failure_rows.append(failure)
            existing_attempts.add(identity)

    if quarantined or reopened:
        _replace_rows(failure_path, failure_rows)
    if quarantined:
        _replace_rows(output, kept_rows)
    return quarantined, reopened


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quarantined, reopened = prepare_retry_rows(args.output)
    print(f"quarantined={quarantined} reopened_terminal={reopened}")


if __name__ == "__main__":
    main()
