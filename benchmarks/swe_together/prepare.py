# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize canonical tasks from a pinned upstream checkout and image inventory.

Run: python -m benchmarks.swe_together.prepare --source CHECKOUT --images images.json --output tasks.jsonl
The image map must contain image, image_digest and workdir for all selected IDs.
Assets stay in the source checkout, configured independently on Resources.
"""

import argparse
import json
import os
from pathlib import Path

from resources_servers.swe_together.task import MANIFEST, TaskData, load_task


def prepare(
    source: Path | str | None = None,
    images: Path | str | None = None,
    output: Path | str = "benchmarks/swe_together/data/full109.jsonl",
    tasks: list[str] | None = None,
) -> Path:
    source = source or os.environ.get("SWE_TOGETHER_SOURCE")
    images = images or os.environ.get("SWE_TOGETHER_IMAGES")
    if not source or not images:
        raise ValueError("Provide source/images or SWE_TOGETHER_SOURCE/SWE_TOGETHER_IMAGES")
    source, images, output = Path(source), Path(images), Path(output)
    manifest = json.loads(MANIFEST.read_text())
    image_map = json.loads(images.read_text())
    selected = tasks or [item["task_id"] for item in manifest["tasks"]]
    rows = []
    for task_id in selected:
        task = TaskData(
            task_id=task_id,
            **{key: value for key, value in image_map[task_id].items() if key in TaskData.model_fields},
        )
        load_task(source / "tasks" / task_id, task)
        rows.append(
            {
                # The catalog's BaseRunRequest validator requires this envelope.
                # Interactive Resources supplies the actual prompt at seed time.
                "responses_create_params": {"input": []},
                "task_id": {"taskset": "swe_together:full109", "task_id": task_id},
                "task_input": {"task_data": task.model_dump()},
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/swe_together/data/full109.jsonl"))
    parser.add_argument("--task", action="append")
    args = parser.parse_args()
    print(f"Prepared {prepare(args.source, args.images, args.output, args.task)}")
