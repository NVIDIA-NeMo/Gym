# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add stable IDs to prepared single-answer JSONL without rebuilding model input."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from resources_servers.mm_coding_training.task_data import TaskData


def with_task_id(row: dict) -> dict:
    """Keep task/model data intact and refuse conflicting transport identities."""
    if "task_input" in row or "task_id" in row or "instance_id" in row:
        raise ValueError("Expected a single-answer flat row without task_id or instance_id")
    agent = row.get("agent_ref")
    if not isinstance(agent, dict) or agent.get("name") != "mm_coding_training_simple_agent":
        raise ValueError("Expected the mm_coding_training_simple_agent route")
    if not isinstance(row.get("responses_create_params"), dict):
        raise ValueError("Expected responses_create_params in the prepared row")
    task = TaskData.model_validate({"task": row.get("task"), "problem_id": row.get("problem_id")}).task
    return row | {"problem_id": f"{task.benchmark}:{task.task_id}"}


def prepare_task_ids(source: Path, output: Path) -> int:
    """Publish a complete distinct file, leaving source bytes and existing outputs untouched."""
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with source.open() as rows:
                for line in rows:
                    if not line.strip():
                        continue
                    row = with_task_id(json.loads(line))
                    if row["problem_id"] in seen:
                        raise ValueError(f"Duplicate coding task: {row['problem_id']}")
                    seen.add(row["problem_id"])
                    stream.write(json.dumps(row) + "\n")
            if not seen:
                raise ValueError("Input task collection is empty")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = prepare_task_ids(args.input, args.output)
    print(f"Prepared {count} single-answer rows at {args.output}")


if __name__ == "__main__":
    main()
