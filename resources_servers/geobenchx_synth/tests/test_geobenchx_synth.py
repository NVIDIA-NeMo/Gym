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

from resources_servers.geobenchx_synth import templates as T
from resources_servers.geobenchx_synth.catalog import dataset_row, offered_entries
from resources_servers.geobenchx_synth.scoring import parse_answer, score


def _call(name):
    return {"type": "function_call", "name": name, "arguments": "{}", "call_id": name}


def _message(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


SOLVE = {"task_kind": "solve", "answers": [412.0, 415.0], "rel_tol": 0.0, "abs_tol": 0.0}


def test_parse_answer_takes_last_answer_line():
    assert parse_answer("Answer: 3\nOn reflection...\nAnswer: 1,204.5") == 1204.5
    assert parse_answer("**Answer:** 17") == 17.0
    assert parse_answer("There are 17 schools.") is None


def test_solve_accepts_any_replayed_answer():
    assert score(SOLVE, [_call("load_geodata"), _message("Answer: 415")])["reward"] == 1.0
    assert score(SOLVE, [_call("load_geodata"), _message("Answer: 413")])["reward"] == 0.0


def test_solve_with_reject_scores_zero():
    out = [_call("load_geodata"), _call("reject_task"), _message("Answer: 412")]
    assert score(SOLVE, out)["reward"] == 0.0


def test_reject_requires_reject_last():
    task = {"task_kind": "reject", "answers": [], "rel_tol": 0.0, "abs_tol": 0.0}
    assert score(task, [_call("load_geodata"), _call("reject_task"), _message("Rejected.")])["reward"] == 1.0
    assert score(task, [_call("reject_task"), _call("load_geodata"), _message("Answer: 3")])["reward"] == 0.0
    assert score(task, [_message("Answer: 3")])["reward"] == 0.0


def test_control_needs_no_tools():
    task = {"task_kind": "control", "answers": [640.0], "rel_tol": 0.005, "abs_tol": 0.0}
    assert score(task, [_message("Answer: 640")])["reward"] == 1.0
    assert score(task, [_call("load_geodata"), _message("Answer: 640")])["reward"] == 0.0


ENTRIES = [
    {"name": "Counties, Iowa study counties", "kind": "counties", "state": "Iowa"},
    {"name": "OpenStreetMap points of interest, Iowa study counties", "kind": "osm_pois", "state": "Iowa"},
    {
        "name": "Microsoft building footprints, Story County, Iowa",
        "kind": "buildings",
        "state": "Iowa",
        "county": "Story County",
    },
    {
        "name": "USDA Cropland Data Layer 2022, Story County, Iowa, 30 m",
        "kind": "cdl",
        "state": "Iowa",
        "county": "Story County",
        "year": 2022,
    },
    {"name": "Counties, Kansas study counties", "kind": "counties", "state": "Kansas"},
    {"name": "Counties, Georgia study counties", "kind": "counties", "state": "Georgia"},
    {"name": "Cropland Data Layer acreage, Iowa", "kind": "table", "state": "Iowa"},
    {"name": "Cropland Data Layer acreage, Georgia", "kind": "table", "state": "Georgia"},
]


def test_offered_entries_stay_in_split():
    offered = offered_entries(ENTRIES, "Iowa", ["Iowa", "Kansas"])
    assert {e["state"] for e in offered} == {"Iowa", "Kansas"}


def test_dataset_row_restricts_catalog_enums():
    cat = T.Catalog(tuple(ENTRIES))
    task = T.buildings_on_crop(cat, "Iowa", "Story County", 2022, "corn")
    record = {**task.__dict__, "task_id": task.task_id, "split": "train", "answers": [5.0]}
    row = dataset_row(record, offered_entries(ENTRIES, "Iowa", ["Iowa", "Kansas"]))
    tools = {t["name"]: t for t in row["responses_create_params"]["tools"]}
    assert tools["load_data"]["parameters"]["properties"]["dataset"]["enum"] == ["Cropland Data Layer acreage, Iowa"]
    assert (
        "Cropland Data Layer acreage, Iowa"
        not in tools["load_geodata"]["parameters"]["properties"]["geodataset"]["enum"]
    )
    assert tools["get_raster_path"]["parameters"]["properties"]["rasterdataset"]["enum"] == [
        "USDA Cropland Data Layer 2022, Story County, Iowa, 30 m"
    ]
    assert (
        "Counties, Georgia study counties"
        not in tools["load_geodata"]["parameters"]["properties"]["geodataset"]["enum"]
    )
    assert task.plans[0][-1]["arguments"]["raster_path"] == {
        "raster": "USDA Cropland Data Layer 2022, Story County, Iowa, 30 m"
    }


def test_evaluate_combines_sub_results():
    from resources_servers.geobenchx_synth.synth_worker import _evaluate

    store = {"a": [1, 2, 3], "b": [1], "length": {"total_length_km": 4.0}, "y2020": [1, 2], "y2021": [1]}
    rows = lambda n: {"type": "rows", "name": n}  # noqa: E731
    assert _evaluate(store, {"type": "sub", "a": rows("b"), "b": rows("a")}) == -2
    assert _evaluate(store, {"type": "div", "a": rows("a"), "b": {"type": "length_km", "name": "length"}}) == 0.75
    assert _evaluate(store, {"type": "pct", "a": rows("b"), "b": rows("a")}) == 100 / 3
    assert _evaluate(store, {"type": "max", "items": [rows("a"), rows("missing")]}) == 3
    assert _evaluate(store, {"type": "argmax", "options": {"2020": rows("y2020"), "2021": rows("y2021")}}) == 2020.0
    assert _evaluate(store, {"type": "argmax", "options": {"2020": rows("b"), "2021": rows("y2021")}}) is None
    assert _evaluate(store, {"type": "div", "a": rows("a"), "b": rows("missing")}) is None


def test_evaluate_reads_table_columns():
    import pandas as pd

    from resources_servers.geobenchx_synth.synth_worker import _evaluate

    store = {"one": pd.DataFrame({"Corn acres": [120]}), "many": pd.DataFrame({"Corn acres": [100, 300, None]})}
    column = lambda n, agg: {"type": "column", "name": n, "column": "Corn acres", "agg": agg}  # noqa: E731
    assert _evaluate(store, column("one", "single")) == 120
    assert _evaluate(store, column("many", "single")) is None
    assert _evaluate(store, column("many", "sum")) == 400
    assert _evaluate(store, column("many", "mean")) == 200
    assert _evaluate(store, column("many", "max")) == 300
    assert _evaluate({"one": pd.DataFrame({"County": ["Story"]})}, column("one", "single")) is None


def test_inferred_prompts_name_no_dataset_or_class():
    cat = T.Catalog(tuple(ENTRIES))
    task = T.inferred_buildings_on_crop(cat, "Iowa", "Story County", 2022, "corn")
    assert "Cropland Data Layer" not in task.prompt and "code" not in task.prompt
    assert task.task_id != T.buildings_on_crop(cat, "Iowa", "Story County", 2022, "corn").task_id


def test_reject_prompts_name_counties_outside_catalog():
    for task in T.reject_tasks("Iowa", ["Story County"]):
        assert task.task_kind == "reject"
        assert task.plans == [[{"tool": "reject_task", "arguments": {}}]]


def _full_catalog(state, counties, years=(2020, 2021, 2022, 2023)):
    entries = [
        {"name": f"{kind} {state}", "kind": kind, "state": state}
        for kind in ("counties", "osm_roads", "osm_pois", "table")
    ]
    for county in counties:
        entries.append({"name": f"buildings {county}", "kind": "buildings", "state": state, "county": county})
        for year in years:
            entries.append(
                {"name": f"cdl {year} {county}", "kind": "cdl", "state": state, "county": county, "year": year}
            )
    return entries


def test_subtle_rejects_come_with_solvable_twins():
    cat = T.Catalog(
        tuple(_full_catalog("Iowa", ["Story County", "Polk County"]) + _full_catalog("Nebraska", ["Hall County"]))
    )
    tasks = T.subtle_rejects(
        cat,
        "Iowa",
        "Story County",
        "Polk County",
        "corn",
        ("Kansas", "Riley County"),
        ("Nebraska", "Hall County"),
        "school",
    )
    for reject, twin in zip(tasks[::2], tasks[1::2]):
        assert reject.task_kind == "reject" and twin.task_kind == "solve"
        assert twin.template == f"{reject.template}_twin"
    offered = [t for t in tasks if t.template.startswith("subtle_not_offered")]
    assert "Riley County, Kansas" in offered[0].prompt and "Hall County, Nebraska" in offered[1].prompt


def test_table_largest_change_takes_the_max_difference():
    import pandas as pd

    from resources_servers.geobenchx_synth.synth_worker import _evaluate

    cat = T.Catalog(tuple(_full_catalog("Iowa", ["Story County", "Polk County"])))
    task = T.table_largest_change(cat, "Iowa", ["Story County", "Polk County"], 2020, 2023, "corn")
    assert task.plans[0][1]["arguments"]["filters"] == {"County": ["Story"], "Year": [2020]}
    frame = lambda v: pd.DataFrame({"Corn acres": [v]})  # noqa: E731
    store = {"f0": frame(100), "l0": frame(90), "f1": frame(50), "l1": frame(80)}
    assert _evaluate(store, task.extract) == 30
