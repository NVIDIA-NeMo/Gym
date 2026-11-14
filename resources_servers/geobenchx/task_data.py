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

import functools
import hashlib
import itertools
import json
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from difflib import SequenceMatcher
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
# Near-duplicate thresholds for task_families (difflib ratio on normalized prompts).
TEXT_RATIO = 0.8
EDIT_RATIO = 0.65
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


def _normalized_text(task: dict[str, Any]) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", task["task_text"].lower()))


def _solution_signature(task: dict[str, Any]) -> tuple[frozenset[str], tuple[str, ...]] | None:
    datasets, tools = set(), []
    for step in task["reference_solutions"][0].get("steps") or []:
        tools.append(step["function_name"])
        for key in ("dataset", "geodataset", "rasterdataset", "raster_path"):
            value = (step.get("arguments") or {}).get(key)
            if isinstance(value, str):
                datasets.add(value)
    return (frozenset(datasets), tuple(tools)) if datasets else None


def task_families(tasks: dict[str, dict[str, Any]] = TASKS) -> list[list[str]]:
    """Group tasks into near-duplicate families (connected components).

    Two tasks are linked when their reference solutions load the same datasets with
    the same tool sequence, or their prompts are close (difflib ratio on normalized
    text). Unsolvable tasks are often edits of a solvable one and have no datasets
    to match on, so solvable/unsolvable pairs link at a looser ratio.
    """
    ids = sorted(tasks)
    parent = {task_id: task_id for task_id in ids}

    def find(task_id: str) -> str:
        while parent[task_id] != task_id:
            parent[task_id] = parent[parent[task_id]]
            task_id = parent[task_id]
        return task_id

    text = {i: _normalized_text(tasks[i]) for i in ids}
    signature = {i: _solution_signature(tasks[i]) for i in ids}
    rejects = {i: "reject" in task_kind(tasks[i]) for i in ids}
    for a, b in itertools.combinations(ids, 2):
        if signature[a] is not None and signature[a] == signature[b]:
            parent[find(a)] = find(b)
            continue
        threshold = EDIT_RATIO if rejects[a] != rejects[b] else TEXT_RATIO
        if SequenceMatcher(None, text[a], text[b]).ratio() >= threshold:
            parent[find(a)] = find(b)
    families: dict[str, list[str]] = defaultdict(list)
    for task_id in ids:
        families[find(task_id)].append(task_id)
    return list(families.values())


@functools.cache
def _family_split() -> dict[str, str]:
    return _split_families(TASKS)


def split_of(tasks: dict[str, dict[str, Any]] = TASKS) -> dict[str, str]:
    """Fixed train/validation assignment that keeps near-duplicates on one side.

    Whole families (task_families) go to validation, visited in a fixed hash order,
    while they fit the per-stratum (group, task kind) quota. A first pass gives every
    group and task kind at least one validation family.
    """
    return dict(_family_split()) if tasks is TASKS else _split_families(tasks)


def _split_families(tasks: dict[str, dict[str, Any]]) -> dict[str, str]:
    def stratum(task_id: str) -> tuple[str, str]:
        return primary_group(tasks[task_id]), task_kind(tasks[task_id])

    families = sorted(task_families(tasks), key=lambda family: hashlib.sha256(min(family).encode()).hexdigest())
    quota = {s: round(n * VALIDATION_FRACTION) for s, n in Counter(map(stratum, tasks)).items()}
    target = round(len(tasks) * VALIDATION_FRACTION)
    taken: Counter[tuple[str, str]] = Counter()
    split = dict.fromkeys(tasks, "train")

    def take(family: list[str]) -> bool:
        need = Counter(map(stratum, family))
        if split[family[0]] == "validation" or any(taken[s] + n > quota[s] + 1 for s, n in need.items()):
            return False
        taken.update(need)
        split.update(dict.fromkeys(family, "validation"))
        return True

    for label in [*sorted({group for group, _ in quota}), *sorted({kind for _, kind in quota})]:
        if any(label in s for s in taken.elements()):
            continue
        for family in families:
            if any(label in stratum(task_id) for task_id in family) and take(family):
                break
    for family in families:
        if sum(taken.values()) >= target:
            break
        take(family)
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
