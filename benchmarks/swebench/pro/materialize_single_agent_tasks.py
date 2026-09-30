# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert prepared SWE-bench Pro rows into EnvironmentServer tasks.

Each output row contains a task identity, model input, and benchmark-owned task
data. Its taskset selects the EnvironmentServer configured in hermes.yaml.
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from nemo_gym.single_agent_task import materialize_single_agent_task


def materialize_row(
    row: Mapping[str, JsonValue],
    *,
    taskset: str,
    task_index: int | None = None,
) -> dict[str, JsonValue]:
    """Separate task identity and model input from the Resources Server's task data."""
    return materialize_single_agent_task(row, taskset=taskset, task_index=task_index).model_dump(
        mode="json", exclude_unset=True
    )


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
        for task_index, line in enumerate(source):
            row = json.loads(line)
            target.write(
                json.dumps(
                    materialize_row(
                        row,
                        taskset=args.taskset,
                        task_index=task_index,
                    )
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
