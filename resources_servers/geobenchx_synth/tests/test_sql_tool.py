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

import json
import os

import pytest

from resources_servers.geobenchx_synth.catalog import dataset_row, offered_entries, offered_states, split_states
from resources_servers.geobenchx_synth.sql_tool import SqlSandbox, StateDatabases, sql_tool_definition, table_name


duckdb = pytest.importorskip("duckdb")
EXTENSION_DIR = os.environ.get("GEOBENCHX_SQL_EXTENSION_DIR")

# Iowa and Nebraska are partners in the train split; Kansas is a same-split outsider.
STATES = ("Iowa", "Nebraska", "Kansas")


def _entries():
    entries = []
    for state in STATES:
        slug = state.lower()
        entries += [
            {
                "name": f"OpenStreetMap points of interest, {state}",
                "file": f"{slug}_osm_pois.gpkg",
                "kind": "osm_pois",
            },
            {"name": f"Acreage by county, {state}", "file": f"{slug}_cdl_acreage.csv", "kind": "table"},
            {"name": f"Cropland Data Layer 2021, {state}", "file": f"cdl_2021_{slug}.tif", "kind": "cdl"},
        ]
        for entry in entries[-3:]:
            entry.update(state=state, split="train")
    return entries


@pytest.fixture(scope="module")
def sql_data(tmp_path_factory):
    """Tables as build_sql.py writes them: WKB geom (EPSG:4326) and geom_3857, plain Parquet tables."""
    root = tmp_path_factory.mktemp("sqldata")
    config = {"extension_directory": EXTENSION_DIR} if EXTENSION_DIR else {}
    connection = duckdb.connect(config=config)
    try:
        connection.execute("INSTALL spatial")
        connection.execute("LOAD spatial")
    except duckdb.Error as error:
        pytest.skip(f"DuckDB spatial extension unavailable: {error}")
    connection.execute("SET geometry_always_xy = true")
    manifest = []
    for i, state in enumerate(STATES):
        slug = state.lower()
        (root / slug).mkdir()
        # Three schools 0, 100 and 1000 m east of a reference point, in EPSG:3857 meters.
        connection.execute(
            f"""COPY (
                SELECT x AS osm_id, 'school' AS fclass, 'Story County' AS county,
                       ST_AsWKB(ST_Transform(p, 'EPSG:3857', 'EPSG:4326')) AS geom, ST_AsWKB(p) AS geom_3857
                FROM (SELECT x, ST_Point({i * 1e5} + x, 5e6) AS p FROM (VALUES (0), (100), (1000)) t(x))
            ) TO '{root}/{slug}/{slug}_osm_pois.parquet' (FORMAT PARQUET)"""
        )
        connection.execute(
            f"""COPY (SELECT 1 AS geoid, 'Story' AS county, 2021 AS year, {100 + i} AS corn_acres)
                TO '{root}/{slug}/{slug}_cdl_acreage.parquet' (FORMAT PARQUET)"""
        )
    connection.close()
    for entry in _entries():
        if entry["kind"] != "cdl":
            table = table_name(entry)
            manifest.append({**entry, "table": table, "parquet": f"{entry['state'].lower()}/{table}.parquet"})
    (root / "catalog_sql.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture(scope="module")
def databases(sql_data, tmp_path_factory):
    return StateDatabases(str(sql_data), str(tmp_path_factory.mktemp("sql")), EXTENSION_DIR)


@pytest.fixture
def sandbox(databases):
    sandbox = SqlSandbox(databases, timeout_s=2.0, max_rows=2)
    sandbox.reset("s", offered_states("Iowa", "train", _entries()))
    yield sandbox
    sandbox.close()


def _value(result):
    return result.splitlines()[1]


NEAR = """SELECT count(*) FROM iowa_osm_pois p JOIN iowa_osm_pois q ON q.osm_id = 0
          AND ST_DWithin(ST_GeomFromWKB(p.geom_3857), ST_GeomFromWKB(q.geom_3857), 150)"""


def test_offered_states_match_the_generated_catalog():
    entries = _entries()
    offered = offered_entries(entries, "Iowa", split_states(entries)["train"])
    assert offered_states("Iowa", "train", entries) == sorted({e["state"] for e in offered}, key=STATES.index)
    assert offered_states("Iowa", "train", entries) == ["Iowa", "Nebraska"]


def test_portable_queries_measure_in_the_documented_crs(sandbox):
    assert _value(sandbox.run("s", NEAR)) == "2"
    assert _value(sandbox.run("s", "SELECT corn_acres FROM nebraska_cdl_acreage")) == "101"


def test_wkb_columns_are_shown_as_text_and_rows_are_truncated(sandbox):
    result = sandbox.run("s", "SELECT osm_id, geom, ST_Buffer(ST_GeomFromWKB(geom_3857), 1) AS b FROM iowa_osm_pois")
    lines = result.splitlines()
    assert lines[0] == "osm_id | geom | b"
    assert lines[1].split(" | ")[1].startswith("POINT") and lines[1].split(" | ")[2].startswith("POLYGON")
    assert len(lines) == 4 and lines[-1].startswith("(first 2 rows shown")


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM kansas_osm_pois",
        "SELECT * FROM kansas.kansas_osm_pois",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_parquet('/etc/passwd')",
        "COPY (SELECT 1) TO 'out.csv'",
        "ATTACH 'other.duckdb' AS other",
        "CREATE TEMP TABLE t AS SELECT 1",
        "SET enable_external_access = true",
        "INSTALL httpfs",
        "SELECT 1; SELECT 2",
    ],
)
def test_sandbox_refuses_everything_but_offered_reads(sandbox, query):
    assert sandbox.run("s", query).startswith("Error:")


def test_long_queries_are_interrupted(sandbox):
    result = sandbox.run("s", "SELECT count(*) FROM range(100000000000) a, range(100) b")
    assert result == "Error: query exceeded the 2 s time limit"


def test_sessions_without_state_get_no_database(databases):
    sandbox = SqlSandbox(databases)
    sandbox.reset("s", [])
    assert sandbox.run("s", "SELECT 1").startswith("Error: no spatial database")


def test_tool_sets_and_sql_tables_follow_the_offered_catalog():
    entries = _entries()
    offered = offered_entries(entries, "Iowa", split_states(entries)["train"])
    task = {
        "task_id": "t", "template": "poi_count", "task_kind": "solve", "state": "Iowa", "county": "Story County",
        "split": "train", "answers": [1.0], "rel_tol": 0.0, "abs_tol": 0.0, "prompt": "How many schools?",
    }  # fmt: skip
    names = {mode: [t["name"] for t in dataset_row(task, offered, mode)["responses_create_params"]["tools"]] for mode in ("gis", "both", "sql")}  # fmt: skip
    assert "run_sql" not in names["gis"]
    assert names["both"] == names["gis"] + ["run_sql"]
    assert names["sql"] == ["run_sql", "reject_task"]
    description = sql_tool_definition(offered)["description"]
    assert "iowa_osm_pois" in description and "nebraska_cdl_acreage" in description
    assert "kansas" not in description and "cdl_2021_iowa" not in description
