# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare task identities; Harbor retrieves the immutable official task packages."""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"


def prepare(task_names: list[str] | None = None, category: str | None = None) -> Path:
    manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text())
    tasks = manifest["tasks"]
    if len(tasks) != 66 or len({task["name"] for task in tasks}) != 66:
        raise ValueError("Terminal-Bench 4.0 must contain 66 unique tasks")
    if task_names is not None:
        unknown = set(task_names) - {task["name"] for task in tasks}
        if unknown:
            raise ValueError(f"Unknown TB4 task names: {sorted(unknown)}")
        tasks = [task for task in tasks if task["name"] in task_names]
    if category is not None:
        if category not in {"cpu", "compose", "gpu"}:
            raise ValueError(f"Unknown TB4 category: {category}")
        tasks = [task for task in tasks if task["category"] == category]
    if not tasks:
        raise ValueError("The TB4 task selection is empty")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as output:
        for task in tasks:
            output.write(json.dumps({"task_name": task["name"], "responses_create_params": {"input": []}}) + "\n")
    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
