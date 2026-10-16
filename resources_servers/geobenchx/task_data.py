# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load and materialize the official GeoBenchX task set for NemoGym."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from resources_servers.geobenchx.catalog import response_tools


ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = (ROOT / "upstream" / "COMMIT").read_text().strip()
# Not committed; downloaded from the pinned upstream commit (see DATA-ATTESTATION.md).
UPSTREAM_TASKS = ROOT / "upstream" / "tasks_and_reference_solutions.json"
UPSTREAM_TASKS_URL = (
    f"https://raw.githubusercontent.com/Solirinai/GeoBenchX/{UPSTREAM_COMMIT}"
    "/benchmark_set/tasks_and_reference_solutions.json"
)
UPSTREAM_TASKS_SHA256 = "ec21a16df85279093e3a599b9b4ecb6906510c549972955eca9b92f2dac64ae3"
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "benchmark.jsonl"
VALIDATION_FRACTION = 0.26
# Upstream prompts use the run date. A fixed date keeps prompts identical across
# baseline, training, and post-training runs.
BENCHMARK_DATE = date(2026, 10, 1)

SYSTEM_PROMPT = """You are a geographer who answers questions with maps whenever possible.
Use the supplied GIS tools and snapshot catalogs. Do not provide Python code. Provide the
dataset names used, the resulting map or calculation, and concise explanations.

Rules:
- If the task is not geospatial and needs no supplied data, answer directly without tools.
- If a geospatial task cannot be solved with the available datasets and tools, call reject_task.
- Reject when a dataset covers only part of the requested geography.
- If no date is specified, use the latest period with non-empty data for over 70% of objects.
- Use the same year for both variables in a bivariate map.
- Do not ask for clarification. Infer reasonable values and complete the task.
- You may make at most 25 tool calls.
"""


def fetch_tasks(path: Path = UPSTREAM_TASKS) -> Path:
    with urllib.request.urlopen(UPSTREAM_TASKS_URL, timeout=60) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != UPSTREAM_TASKS_SHA256:
        raise RuntimeError(f"{UPSTREAM_TASKS_URL} has sha256 {digest}, expected {UPSTREAM_TASKS_SHA256}")
    partial = path.with_suffix(".partial")
    partial.write_bytes(content)
    partial.replace(path)
    return path


def load_tasks() -> dict[str, dict[str, Any]]:
    if not UPSTREAM_TASKS.exists():
        fetch_tasks()
    records = json.loads(UPSTREAM_TASKS.read_text())["tasks"]
    return {record["task_ID"]: record for record in records}


TASKS = load_tasks()


def task_kind(task: dict[str, Any]) -> str:
    kinds = set()
    for solution in task["reference_solutions"]:
        names = [step["function_name"] for step in solution.get("steps") or []]
        kinds.add("control" if not names else "reject" if names == ["reject_task"] else "solve")
    return "+".join(sorted(kinds))


def primary_group(task: dict[str, Any]) -> str:
    return next(
        (
            label
            for label in task["task_labels"]
            if label
            in {
                "Merge, Visualize",
                "Process, Merge, Visualize",
                "Spatial operations",
                "Heatmaps, Contour Lines",
                "Control question",
            }
        ),
        task["task_labels"][0],
    )


def dataset_row(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_ID"],
        "activity": "geobenchx",
        "group": primary_group(task),
        "task_kind": task_kind(task),
        "task_labels": task["task_labels"],
        "reference_solution_count": len(task["reference_solutions"]),
        "responses_create_params": {
            "input": [
                {
                    "role": "system",
                    "content": f"{SYSTEM_PROMPT}\nBenchmark date: {BENCHMARK_DATE.isoformat()}.",
                },
                {"role": "user", "content": task["task_text"]},
            ],
            "tools": response_tools(),
            "parallel_tool_calls": False,
            "temperature": 0,
        },
    }


def split_of(tasks: dict[str, dict[str, Any]] = TASKS) -> dict[str, str]:
    """Fixed train/validation assignment, stratified by group and task kind.

    Within each stratum, tasks are ordered by a hash of the task ID, so the split
    does not depend on file order and is stable across rebuilds.
    """
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task_id, task in tasks.items():
        strata[(primary_group(task), task_kind(task))].append(task_id)
    split = {}
    for ids in strata.values():
        ids.sort(key=lambda task_id: hashlib.sha256(task_id.encode()).hexdigest())
        n_validation = round(len(ids) * VALIDATION_FRACTION)
        for index, task_id in enumerate(ids):
            split[task_id] = "validation" if index < n_validation else "train"
    return split


def _write(path: Path, tasks: list[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for task in tasks:
            stream.write(json.dumps(dataset_row(task), ensure_ascii=False, separators=(",", ":")) + "\n")


def materialize(data_dir: Path = DATA_DIR) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    split = split_of()
    train = [task for task_id, task in TASKS.items() if split[task_id] == "train"]
    validation = [task for task_id, task in TASKS.items() if split[task_id] == "validation"]
    outputs = {
        "benchmark.jsonl": list(TASKS.values()),
        "train.jsonl": train,
        "validation.jsonl": validation,
        "example.jsonl": train[:5],
    }
    for name, tasks in outputs.items():
        _write(data_dir / name, tasks)
    return [data_dir / name for name in outputs]


if __name__ == "__main__":
    for path in materialize():
        print(path)
