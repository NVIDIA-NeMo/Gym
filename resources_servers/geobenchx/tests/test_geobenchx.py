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

from __future__ import annotations

import json
from collections import Counter

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.geobenchx.catalog import TOOL_DEFINITIONS, response_tools
from resources_servers.geobenchx.scoring import score_task
from resources_servers.geobenchx.task_data import (
    TASKS,
    dataset_row,
    materialize,
    primary_group,
    split_of,
    task_families,
    task_kind,
)


def _calls(solution: dict) -> list[tuple[str, dict]]:
    return [(step["function_name"], dict(step.get("arguments") or {})) for step in solution.get("steps") or []]


def test_official_task_set_counts_and_groups() -> None:
    assert len(TASKS) == 202
    # The paper calls these 23 GIS tools plus the additional reject_task.
    assert len(TOOL_DEFINITIONS) == 24
    assert Counter(primary_group(task) for task in TASKS.values()) == {
        "Process, Merge, Visualize": 56,
        "Heatmaps, Contour Lines": 54,
        "Spatial operations": 53,
        "Merge, Visualize": 36,
        "Control question": 3,
    }
    kinds = Counter(task_kind(task) for task in TASKS.values())
    assert kinds["solve"] == 112
    assert kinds["reject"] == 79
    assert kinds["control"] == 3


def test_rows_are_valid_responses_api_tasks() -> None:
    task = next(iter(TASKS.values()))
    row = dataset_row(task)
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    assert len(params.tools) == 24
    assert params.temperature == 0
    assert {tool["name"] for tool in response_tools()} == set(TOOL_DEFINITIONS)


def _read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_materialize_writes_all_official_rows(tmp_path) -> None:
    materialize(tmp_path)
    rows = _read(tmp_path / "benchmark.jsonl")
    assert len(rows) == 202
    assert len({row["task_id"] for row in rows}) == 202


def test_train_and_validation_partition_the_benchmark(tmp_path) -> None:
    materialize(tmp_path)
    train = {row["task_id"] for row in _read(tmp_path / "train.jsonl")}
    validation = {row["task_id"] for row in _read(tmp_path / "validation.jsonl")}
    assert not train & validation
    assert train | validation == set(TASKS)
    assert len(validation) == 53
    assert {row["task_id"] for row in _read(tmp_path / "example.jsonl")} <= train


def test_split_keeps_each_near_duplicate_family_on_one_side() -> None:
    split = split_of()
    families = task_families()
    assert sorted(task_id for family in families for task_id in family) == sorted(TASKS)
    assert all(len({split[task_id] for task_id in family}) == 1 for family in families)


def test_split_covers_every_group_and_kind_in_validation() -> None:
    split = split_of()
    validation = [TASKS[task_id] for task_id, name in split.items() if name == "validation"]
    assert {primary_group(task) for task in validation} == {primary_group(task) for task in TASKS.values()}
    assert {task_kind(task) for task in validation} >= {"solve", "reject", "control"}
    assert split == split_of()


def test_reference_solutions_score_full_match() -> None:
    for task in TASKS.values():
        solution = task["reference_solutions"][0]
        assert score_task(_calls(solution), task)["geobenchx_match"] == 2


def test_output_handle_renames_do_not_change_match() -> None:
    task = next(
        task
        for task in TASKS.values()
        if any(len(solution.get("steps") or []) >= 4 for solution in task["reference_solutions"])
    )
    solution = next(solution for solution in task["reference_solutions"] if len(solution.get("steps") or []) >= 4)
    calls = _calls(solution)
    renamed = []
    for name, arguments in calls:
        renamed.append(
            (
                name,
                {key: ("renamed_handle" if key.startswith("output_") else value) for key, value in arguments.items()},
            )
        )
    assert score_task(renamed, task)["geobenchx_match"] == 2


def test_reject_and_control_tasks_require_correct_behavior() -> None:
    reject = next(task for task in TASKS.values() if task_kind(task) == "reject")
    control = next(task for task in TASKS.values() if task_kind(task) == "control")
    assert score_task([("reject_task", {})], reject)["geobenchx_match"] == 2
    inspected = [("load_geodata", {"geodataset": "Countries"}), ("reject_task", {})]
    assert score_task(inspected, reject)["geobenchx_match"] == 2
    assert score_task(list(reversed(inspected)), reject)["geobenchx_match"] == 0
    assert score_task([], reject)["geobenchx_match"] == 0
    assert score_task([], control)["geobenchx_match"] == 2
    assert score_task([("reject_task", {})], control)["geobenchx_match"] == 0


def test_colormap_difference_is_partial() -> None:
    task = next(
        task
        for task in TASKS.values()
        if any(
            step["function_name"] == "make_choropleth_map"
            for solution in task["reference_solutions"]
            for step in solution.get("steps") or []
        )
    )
    solution = next(
        solution
        for solution in task["reference_solutions"]
        if any(step["function_name"] == "make_choropleth_map" for step in solution["steps"])
    )
    calls = _calls(solution)
    changed = [
        (name, {**arguments, "colormap": "Hazards"}) if name == "make_choropleth_map" else (name, arguments)
        for name, arguments in calls
    ]
    assert score_task(changed, task)["geobenchx_match"] == 1
