# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and materialize the pinned GeoBenchX records at runtime."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from collections import defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from resources_servers.geobenchx.catalog import response_tools


ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = (ROOT / "upstream" / "COMMIT").read_text().strip()
UPSTREAM_TASKS = ROOT / "upstream" / "tasks_and_reference_solutions.json"
UPSTREAM_TASKS_URL = (
    f"https://raw.githubusercontent.com/Solirinai/GeoBenchX/{UPSTREAM_COMMIT}"
    "/benchmark_set/tasks_and_reference_solutions.json"
)
UPSTREAM_TASKS_SHA256 = "ec21a16df85279093e3a599b9b4ecb6906510c549972955eca9b92f2dac64ae3"
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "benchmark.jsonl"
BENCHMARK_DATE = date(2026, 10, 1)


def fetch_tasks(path: Path = UPSTREAM_TASKS) -> Path:
    with urllib.request.urlopen(UPSTREAM_TASKS_URL, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != UPSTREAM_TASKS_SHA256:
        raise RuntimeError("downloaded GeoBenchX records do not match the pinned checksum")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        partial = Path(stream.name)
        stream.write(content)
    partial.replace(path)
    return path


def load_tasks() -> dict[str, dict[str, Any]]:
    if not UPSTREAM_TASKS.exists():
        fetch_tasks()
    return {task["task_ID"]: task for task in json.loads(UPSTREAM_TASKS.read_text())["tasks"]}


TASKS = load_tasks()


def task_kind(task: dict[str, Any]) -> str:
    names = [[s["function_name"] for s in x.get("steps") or []] for x in task["reference_solutions"]]
    return "+".join(sorted({"control" if not x else "reject" if x == ["reject_task"] else "solve" for x in names}))


def primary_group(task: dict[str, Any]) -> str:
    groups = {"Merge, Visualize", "Process, Merge, Visualize", "Spatial operations", "Heatmaps, Contour Lines", "Control question"}
    return next(label for label in task["task_labels"] if label in groups)


def task_families(tasks: dict[str, dict[str, Any]] = TASKS) -> list[list[str]]:
    parent = {key: key for key in tasks}
    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def text(task: dict[str, Any]) -> str:
        return " ".join(task["task_text"].lower().split())
    ids = sorted(tasks)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            left_steps = tasks[left]["reference_solutions"][0].get("steps") or []
            right_steps = tasks[right]["reference_solutions"][0].get("steps") or []
            same_tools = [x["function_name"] for x in left_steps] == [x["function_name"] for x in right_steps]
            threshold = 0.65 if ("reject" in task_kind(tasks[left])) != ("reject" in task_kind(tasks[right])) else 0.8
            if same_tools or SequenceMatcher(None, text(tasks[left]), text(tasks[right])).ratio() >= threshold:
                parent[root(left)] = root(right)
    families: dict[str, list[str]] = defaultdict(list)
    for key in ids:
        families[root(key)].append(key)
    return list(families.values())


def split_of(tasks: dict[str, dict[str, Any]] = TASKS) -> dict[str, str]:
    split = dict.fromkeys(tasks, "train")
    target = round(len(tasks) * 0.26)
    taken = 0
    for family in sorted(task_families(tasks), key=lambda x: hashlib.sha256(min(x).encode()).hexdigest()):
        if taken >= target:
            break
        split.update(dict.fromkeys(family, "validation"))
        taken += len(family)
    return split


def dataset_row(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_ID"], "activity": "geobenchx", "group": primary_group(task),
        "task_kind": task_kind(task), "task_labels": task["task_labels"],
        "reference_solution_count": len(task["reference_solutions"]),
        "responses_create_params": {
            "input": [{"role": "system", "content": f"You are a geographer. Use supplied GIS tools. Benchmark date: {BENCHMARK_DATE.isoformat()}."}, {"role": "user", "content": task["task_text"]}],
            "tools": response_tools(), "parallel_tool_calls": False, "temperature": 0,
        },
    }


def materialize(data_dir: Path = DATA_DIR) -> list[Path]:
    tasks = load_tasks()
    split = split_of(tasks)
    outputs = {
        "benchmark.jsonl": list(tasks.values()),
        "train.jsonl": [task for key, task in tasks.items() if split[key] == "train"],
        "validation.jsonl": [task for key, task in tasks.items() if split[key] == "validation"],
        "example.jsonl": [task for key, task in tasks.items() if split[key] == "train"][:5],
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        (data_dir / name).write_text("".join(json.dumps(dataset_row(row), ensure_ascii=False) + "\n" for row in rows))
    return [data_dir / name for name in outputs]


def prepare() -> Path:
    materialize()
    return DATASET_PATH


if __name__ == "__main__":
    print(prepare())
