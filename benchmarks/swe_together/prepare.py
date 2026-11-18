# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize canonical tasks from a pinned upstream checkout and image inventory.

Run: python -m benchmarks.swe_together.prepare --source CHECKOUT --images images.json --output tasks.jsonl
The image map must contain image, image_digest and workdir for all selected IDs.
Assets stay in the source checkout, configured on the SWE-Together environment server.
"""

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from environment_servers.swe_together.task import MANIFEST, SOURCE_REVISION, TaskData, load_task


def ensure_source(destination: Path) -> Path:
    """Cache the official source; task loading checks each pinned asset's SHA-256."""
    destination = destination.resolve()
    if (destination / "tasks").is_dir():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/Togetherbench/SWE-Together/tar.gz/{SOURCE_REVISION}"
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".source-") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        with urllib.request.urlopen(url, timeout=90) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(Path(temporary) / "unpacked", filter="data")
        checkout = Path(temporary) / "unpacked" / f"SWE-Together-{SOURCE_REVISION}"
        if not (checkout / "tasks").is_dir():
            raise ValueError("Pinned upstream archive has no task directory")
        checkout.rename(destination)
    return destination


def prepare(
    source: Path | str | None = None,
    images: Path | str | None = None,
    output: Path | str = "benchmarks/swe_together/data/full109.jsonl",
    tasks: list[str] | None = None,
) -> Path:
    """Materialize the canonical taskset from verified assets and immutable image pins."""
    work_dir = Path(os.environ.get("SWE_TOGETHER_WORK_DIR", "outputs/swe_together")).resolve()
    source = Path(source or os.environ.get("SWE_TOGETHER_SOURCE", work_dir / "source"))
    source = ensure_source(source)
    images = Path(images or os.environ.get("SWE_TOGETHER_IMAGES", Path(__file__).with_name("image-pins.json")))
    output = Path(output)
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
