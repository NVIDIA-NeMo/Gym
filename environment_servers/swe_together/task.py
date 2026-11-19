# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified, host-only task assets and independent materialized task data."""

import hashlib
import json
import re
import tomllib
from pathlib import Path

from environment_servers.swe_together.task_input import SOURCE_REVISION, TaskData


MANIFEST = Path(__file__).with_name("task-manifest.json")


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
        "agent_kwargs": toml.get("agent", {}).get("kwargs", {}),
        "judge_timeout": 1200 if match and float(match[1]) >= 600 else 600,
    }
