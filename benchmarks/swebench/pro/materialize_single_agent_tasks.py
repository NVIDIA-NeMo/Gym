# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert prepared SWE-bench Pro rows into EnvironmentServer tasks.

Each output row contains a task identity, model input, and benchmark-owned task
data. Its taskset selects the EnvironmentServer configured in hermes.yaml.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _task_id(row: dict[str, Any]) -> str:
    for key in ("task_id", "instance_id", "problem_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def materialize_row(
    row: dict[str, Any],
    *,
    taskset: str,
) -> dict[str, Any]:
    """Separate task identity and model input from the Resources Server's task data."""
    responses_create_params = row.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        raise ValueError("SWE Pro rows require responses_create_params")
    task_data = {
        key: value
        for key, value in row.items()
        if key not in {"agent_ref", "responses_create_params", "task_source"} and not key.startswith("_ng_")
    }
    return {
        "task_id": {
            "taskset": taskset,
            "task_id": _task_id(row),
        },
        "task_input": {
            "responses_create_params": responses_create_params,
            "task_data": task_data,
        },
    }


def main() -> None:
    """Write one EnvironmentServer task for each prepared input row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prepared SWE-bench Pro JSONL from prepare.py")
    parser.add_argument("output", type=Path, help="Task JSONL to pass to gym eval run -i")
    parser.add_argument(
        "--taskset",
        default="swebench_pro",
        help="Taskset key in environment_server_routes (default: swebench_pro)",
    )
    args = parser.parse_args()

    with args.input.open() as source, args.output.open("w") as target:
        for line in source:
            row = json.loads(line)
            target.write(
                json.dumps(
                    materialize_row(
                        row,
                        taskset=args.taskset,
                    )
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
