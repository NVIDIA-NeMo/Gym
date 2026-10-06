# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare native NOOA rows from the existing GDP dataset preparation."""

import json
from pathlib import Path

from benchmarks.gdpval.prepare import OUTPUT_FPATH, prepare
from resources_servers.gdpval.sandbox_tasks import prepare_row


def prepare_native(*, source: Path = OUTPUT_FPATH, output: Path | None = None) -> Path:
    """Retain every task and verifier field while adding the NOOA file-task prompt."""
    if not source.exists():
        if source != OUTPUT_FPATH:
            raise FileNotFoundError(source)
        prepare()
    output = output or source.with_name("gdpval_nooa.jsonl")
    if source.resolve() == output.resolve():
        raise ValueError("NOOA preparation must not overwrite the source dataset")
    with source.open("rb") as stream:
        rows = [prepare_row(json.loads(line)) for line in stream if line.strip()]
    task_ids = [row["task_id"]["task_id"] for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Duplicate GDP task IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return output


if __name__ == "__main__":
    prepare_native()
