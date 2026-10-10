# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe the tasks of Harbor datasets; the half of Harbor provisioning that needs Harbor itself.

prepare_utils runs this in an isolated environment with only Harbor installed (`uv run --isolated --with harbor`),
because Harbor's dependencies conflict with Gym's. It reads a JSON request and writes one JSON object per task:

    python -m benchmarks.harbor.prepare_utils.harbor_side REQUEST.json OUTPUT.jsonl
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from harbor.models.job.config import DatasetConfig
from harbor.models.task.task import Task

from resources_servers.harbor_tasks.tasks import (
    builds_image,
    download_dataset_tasks,
    image_reference,
    unsupported_features,
)


async def describe(request: dict[str, Any]) -> list[dict[str, Any]]:
    """One record per task: what a row needs, which image it starts from, and why it is unsupported."""
    records = []
    for alias, dataset in request["datasets"].items():
        for path in await download_dataset_tasks(DatasetConfig.model_validate(dataset)):
            task = Task(path)
            records.append(
                {
                    "harbor_dataset": alias,
                    "task_name": task.name,
                    "instruction": task.instruction,
                    "agent_timeout_sec": task.config.agent.timeout_sec,
                    "agent_user": task.config.agent.user,
                    "image": image_reference(task, request.get("image_template")),
                    "builds_image": builds_image(task),
                    "environment_dir": str(task.paths.environment_dir),
                    "unsupported": unsupported_features(
                        task,
                        image_template=request.get("image_template"),
                        allow_unenforced_network_policy=request.get("allow_unenforced_network_policy", False),
                    ),
                }
            )
    return records


def main() -> None:
    request_path, output_path = (Path(arg) for arg in sys.argv[1:3])
    records = asyncio.run(describe(json.loads(request_path.read_text())))
    output_path.write_text("".join(json.dumps(record) + "\n" for record in records))


if __name__ == "__main__":
    main()
