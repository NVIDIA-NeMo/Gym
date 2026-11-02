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
import time
from collections import Counter

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.geobenchx.catalog import TOOL_DEFINITIONS, response_tools
from resources_servers.geobenchx.prepare_data import (
    TASKS,
    dataset_row,
    materialize,
    primary_group,
    split_of,
    task_families,
    task_kind,
)
from resources_servers.geobenchx.scoring import (
    HANDLE_KEYS,
    MAX_TOOL_CALLS,
    OUTPUT_KEYS,
    is_well_formed,
    parse_calls,
    score_task,
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
    malformed = 0
    for task in TASKS.values():
        for solution in task["reference_solutions"]:
            score = score_task(_calls(solution), task)
            if is_well_formed(solution["steps"]):
                assert score["geobenchx_match"] == 2
            else:
                # Upstream references that consume a handle no earlier step produced.
                malformed += 1
                assert score["geobenchx_match"] == 1
                assert score["match_detail"]["reference_well_formed"] is False
    assert malformed == 5


def test_output_handle_renames_do_not_change_match() -> None:
    task = next(
        task
        for task in TASKS.values()
        if any(len(solution.get("steps") or []) >= 4 for solution in task["reference_solutions"])
    )
    solution = next(solution for solution in task["reference_solutions"] if len(solution.get("steps") or []) >= 4)

    def rename(value):
        if isinstance(value, list):
            return [rename(item) for item in value]
        return f"renamed_{value}" if isinstance(value, str) else value

    renamed = [
        (name, {key: rename(value) if key in OUTPUT_KEYS | HANDLE_KEYS else value for key, value in arguments.items()})
        for name, arguments in _calls(solution)
    ]
    assert score_task(renamed, task)["geobenchx_match"] == 2
    # Renaming outputs without updating their consumers breaks the dataflow.
    unlinked = [
        (name, {key: rename(value) if key in OUTPUT_KEYS else value for key, value in arguments.items()})
        for name, arguments in _calls(solution)
    ]
    assert score_task(unlinked, task)["geobenchx_match"] < 2


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


def _load(dataset: str, handle: str) -> tuple[str, dict]:
    return "load_geodata", {"geodataset": dataset, "output_geodataframe_name": handle}


def _show(*handles: str) -> tuple[str, dict]:
    return "visualize_geographies", {"geodataframe_names": list(handles)}


def _task(*steps: tuple[str, dict]) -> dict:
    return {"reference_solutions": [{"steps": [{"function_name": name, "arguments": args} for name, args in steps]}]}


TWO_LAYERS = _task(_load("Countries", "a"), _load("Rivers", "b"), _show("a", "b"))


def test_scoring_follows_dataflow_not_handle_names() -> None:
    assert (
        score_task([_load("Countries", "x"), _load("Rivers", "y"), _show("x", "y")], TWO_LAYERS)["geobenchx_match"]
        == 2
    )
    # Independent producers commute.
    assert (
        score_task([_load("Rivers", "y"), _load("Countries", "x"), _show("x", "y")], TWO_LAYERS)["geobenchx_match"]
        == 2
    )
    # A consumer before its producers, a swapped binding, or a handle nothing produced cannot fully match.
    for calls in (
        [_show("x", "y"), _load("Countries", "x"), _load("Rivers", "y")],
        [_load("Countries", "x"), _load("Rivers", "y"), _show("y", "x")],
        [_load("Countries", "x"), _load("Rivers", "y"), _show("x", "missing")],
    ):
        assert score_task(calls, TWO_LAYERS)["geobenchx_match"] < 2


def test_scoring_compares_catalog_names_exactly() -> None:
    calls = [_load("countries", "x"), _load("Rivers", "y"), _show("x", "y")]
    assert score_task(calls, TWO_LAYERS)["geobenchx_match"] < 2


def test_scoring_skips_raster_paths_but_checks_the_raster_dataset() -> None:
    task = _task(
        ("get_raster_path", {"rasterdataset": "Snow 2024"}),
        ("get_raster_description", {"raster_path": "G:/My Drive/upstream/snow.tif"}),
    )
    calls = [
        ("get_raster_path", {"rasterdataset": "Snow 2024"}),
        ("get_raster_description", {"raster_path": "snapshot://rasters/x.tif"}),
    ]
    assert score_task(calls, task)["geobenchx_match"] == 2
    calls[0] = ("get_raster_path", {"rasterdataset": "Snow 2023"})
    assert score_task(calls, task)["geobenchx_match"] < 2


def test_scoring_prefers_a_late_exact_correction() -> None:
    task = _task(("make_choropleth_map", {"dataframe_name": "a", "colormap": "Reds"}))
    task["reference_solutions"][0]["steps"].insert(
        0, {"function_name": "load_data", "arguments": {"dataset": "GDP", "output_dataframe_name": "a"}}
    )
    load = ("load_data", {"dataset": "GDP", "output_dataframe_name": "a"})
    wrong = ("make_choropleth_map", {"dataframe_name": "a", "colormap": "Blues"})
    right = ("make_choropleth_map", {"dataframe_name": "a", "colormap": "Reds"})
    assert score_task([load, wrong], task)["geobenchx_match"] == 1
    assert score_task([load, wrong, right], task)["geobenchx_match"] == 2


def test_malformed_reference_caps_at_partial_but_alternatives_still_score() -> None:
    malformed = _task(_load("Countries", "a"), _show("never_produced"))["reference_solutions"][0]
    well_formed = TWO_LAYERS["reference_solutions"][0]
    calls = [_load("Countries", "a"), _show("never_produced")]
    assert score_task(calls, {"reference_solutions": [malformed]})["geobenchx_match"] == 1
    good = [_load("Countries", "x"), _load("Rivers", "y"), _show("x", "y")]
    assert score_task(good, {"reference_solutions": [malformed, well_formed]})["geobenchx_match"] == 2


def test_scoring_rejects_malformed_and_over_limit_calls() -> None:
    assert (
        score_task([_load("Countries", "x"), _load("Rivers", "y"), ("visualize_geographies", None)], TWO_LAYERS)[
            "geobenchx_match"
        ]
        < 2
    )
    good = [_load("Countries", "x"), _load("Rivers", "y"), _show("x", "y")]
    padding = [("get_unique_values", {"dataframe_name": "x", "column": "NAME_EN"})] * (MAX_TOOL_CALLS - len(good) + 1)
    assert score_task(good + padding, TWO_LAYERS)["geobenchx_match"] == 0
    assert score_task(good + padding[1:], TWO_LAYERS)["geobenchx_match"] == 2
    assert parse_calls([{"type": "function_call", "name": "reject_task", "arguments": "{not json"}]) == [
        ("reject_task", None)
    ]
    reject = _task(("reject_task", {}))
    assert score_task([("reject_task", None)], reject)["geobenchx_match"] == 0
    assert score_task([("reject_task", {"reason": "no data for Mars"})], reject)["geobenchx_match"] == 2


def test_scoring_time_is_bounded_for_interchangeable_calls() -> None:
    task = _task(*[_load("Countries", f"g{i}") for i in range(8)], _show(*[f"g{i}" for i in range(8)]))
    calls = [_load("Countries", f"c{i}") for i in range(MAX_TOOL_CALLS - 1)] + [_show(*["c0"] * 8)]
    started = time.monotonic()
    result = score_task(calls, task)
    assert time.monotonic() - started < 5
    assert result["geobenchx_match"] == 1
