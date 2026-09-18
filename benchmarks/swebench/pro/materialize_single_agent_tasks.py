# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize flat SWE Pro rows for the taskset-routing smoke test."""

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
    revision: str,
) -> dict[str, Any]:
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
            "revision": revision,
        },
        "task_input": {
            "responses_create_params": responses_create_params,
            "task_data": task_data,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--taskset", default="swebench_pro_smoke")
    parser.add_argument("--revision", default="smoke-v1")
    args = parser.parse_args()

    with args.input.open() as source, args.output.open("w") as target:
        for line in source:
            row = json.loads(line)
            target.write(
                json.dumps(
                    materialize_row(
                        row,
                        taskset=args.taskset,
                        revision=args.revision,
                    )
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
