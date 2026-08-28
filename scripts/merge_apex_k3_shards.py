#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge resume-safe Apex rollout shards into source order."""

import argparse
import json
import os
from collections import Counter
from collections.abc import Collection
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--print-retryable", action="store_true")
    return parser.parse_args()


def count_retryable(
    task_ids: Collection[str],
    num_repeats: int,
    succeeded: Collection[tuple[str, int]],
    failure_attempts: Counter[tuple[str, int]],
    terminal_failures: Collection[tuple[str, int]],
    max_attempts: int,
) -> int:
    expected = {(task_id, rollout_index) for task_id in task_ids for rollout_index in range(num_repeats)}
    maxed_out = {key for key, attempts in failure_attempts.items() if attempts >= max_attempts}
    return len(expected - set(succeeded) - set(terminal_failures) - maxed_out)


def exhausted_failure_rows(
    failures: dict[tuple[str, int], list[dict]],
    succeeded: Collection[tuple[str, int]],
    terminal_failures: Collection[tuple[str, int]],
    max_attempts: int,
) -> dict[tuple[str, int], dict]:
    """Return a scoreable zero row for failures that must no longer retry."""

    exhausted = set(terminal_failures) | {key for key, attempts in failures.items() if len(attempts) >= max_attempts}
    exhausted -= set(succeeded)
    rows = {}
    for key in exhausted:
        row = dict(failures[key][-1])
        failure_class = row.pop("_ng_failure_class", "unknown")
        row.pop("_ng_failure_terminal", None)
        row["_ng_exhausted_failure_class"] = failure_class
        row["reward"] = 0.0
        rows[key] = row
    return rows


def main() -> None:
    args = parse_args()
    if args.num_repeats < 1:
        raise SystemExit("--num-repeats must be positive")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")

    source_index: dict[str, int] = {}
    with args.source.open() as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            task_id = json.loads(line)["task_id"]
            if task_id in source_index:
                raise SystemExit(f"duplicate source task_id: {task_id}")
            source_index[task_id] = index

    rows: dict[tuple[str, int], dict] = {}
    for shard_index in range(args.shard_count):
        shard = args.output_dir / f"shard-{shard_index}.jsonl"
        if not shard.exists():
            continue
        with shard.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"invalid JSON in {shard}:{line_number}: {error}") from error
                task_id = row.get("task_id")
                if task_id not in source_index:
                    raise SystemExit(f"unknown task_id {task_id!r} in {shard}:{line_number}")
                rollout_index = int(row.get("_ng_rollout_index", 0))
                key = (task_id, rollout_index)
                normalized = dict(row)
                normalized["_ng_task_index"] = source_index[task_id]
                if key in rows and rows[key] != normalized:
                    raise SystemExit(f"conflicting duplicate task_id={task_id}, rollout={rollout_index}")
                rows[key] = normalized

    args.combined.parent.mkdir(parents=True, exist_ok=True)
    combined_failures = args.combined.with_name(args.combined.stem + "_failures.jsonl")
    temporary_failures = combined_failures.with_suffix(combined_failures.suffix + ".tmp")
    failure_attempts: Counter[tuple[str, int]] = Counter()
    failures_by_key: dict[tuple[str, int], list[dict]] = {}
    terminal_failures: set[tuple[str, int]] = set()
    with temporary_failures.open("w") as destination:
        for shard_index in range(args.shard_count):
            failure_path = args.output_dir / f"shard-{shard_index}_failures.jsonl"
            if not failure_path.exists():
                continue
            with failure_path.open() as source_stream:
                for line_number, line in enumerate(source_stream, 1):
                    if line.strip():
                        failure = json.loads(line)
                        task_id = failure.get("task_id")
                        if task_id not in source_index:
                            raise SystemExit(f"unknown task_id {task_id!r} in {failure_path}:{line_number}")
                        rollout_index = int(failure.get("_ng_rollout_index", 0))
                        key = (task_id, rollout_index)
                        failure_attempts[key] += 1
                        failures_by_key.setdefault(key, []).append(failure)
                        if failure.get("_ng_failure_terminal"):
                            terminal_failures.add(key)
                        destination.write(line if line.endswith("\n") else line + "\n")
    os.replace(temporary_failures, combined_failures)

    rows.update(
        exhausted_failure_rows(
            failures_by_key,
            rows,
            terminal_failures,
            args.max_attempts,
        )
    )
    for row in rows.values():
        row["_ng_task_index"] = source_index[row["task_id"]]
    ordered = sorted(rows.values(), key=lambda row: (source_index[row["task_id"]], row.get("_ng_rollout_index", 0)))
    temporary = args.combined.with_suffix(args.combined.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, args.combined)

    if args.print_retryable:
        succeeded = {(row["task_id"], int(row.get("_ng_rollout_index", 0))) for row in ordered}
        print(
            count_retryable(
                source_index,
                args.num_repeats,
                succeeded,
                failure_attempts,
                terminal_failures,
                args.max_attempts,
            )
        )
    else:
        print(len(ordered))


if __name__ == "__main__":
    main()
