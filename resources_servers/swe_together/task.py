# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified, host-only task assets and independent materialized task data."""

import hashlib
import json
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


SOURCE_REVISION = "891d19eb4b3a64a47c3d49bbd066a311e0133254"
MANIFEST = Path(__file__).parents[2] / "benchmarks/swe_together/manifest.json"


class TaskData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    image: str
    image_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    workdir: str = Field(default="/workspace", pattern=r"^/")
    source_revision: str = SOURCE_REVISION


def load_task(directory: Path, task: TaskData) -> dict:
    """Fail on absent or changed canonical assets before starting a sandbox."""
    if task.source_revision != SOURCE_REVISION:
        raise ValueError("Unsupported SWE-Together source revision")
    manifest = json.loads(MANIFEST.read_text())
    record = next((row for row in manifest["tasks"] if row["task_id"] == task.task_id), None)
    if record is None:
        raise ValueError(f"Task is not in the canonical 109: {task.task_id}")
    for relative, digest in record["files"].items():
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Task asset is missing or changed: {task.task_id}/{relative}")
    toml = tomllib.loads((directory / "task.toml").read_text())
    if task.image.split("@")[0] != toml["environment"]["docker_image"]:
        raise ValueError("Task image does not match the pinned official task definition")
    # The upstream regex scans the entire TOML, including [environment].
    match = re.search(r"build_timeout_sec\s*=\s*([0-9.]+)", (directory / "task.toml").read_text())
    return {
        "record": record,
        "environment": toml["environment"],
        "judge_timeout": 1200 if match and float(match[1]) >= 600 else 600,
    }
