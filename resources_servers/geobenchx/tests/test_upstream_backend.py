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

import sys
import textwrap
from pathlib import Path

import pytest

from resources_servers.geobenchx.upstream_backend import UpstreamToolPool


FAKE_TOOLS = """
import os
import time


def load_data(dataset, output_dataframe_name, state):
    state["data_store"][output_dataframe_name] = {"dataset": dataset, "root": os.environ["STATDATAPATH"]}
    state["image_store"].append({"base64": "x"})
    return f"Loaded {dataset} into {output_dataframe_name}"


def list_handles(state):
    return ",".join(sorted(state["data_store"]))


def explode(state):
    raise ValueError("bad column")


def sleep_forever(state):
    time.sleep(60)


def write_output(output_filename, state, output_folder=None):
    return os.path.join(output_folder, output_filename)


class Frame:
    def __init__(self, columns):
        self.columns = columns

    def drop(self, columns):
        return Frame([c for c in self.columns if c not in columns])


def make_joined(name, state):
    state["data_store"][name] = Frame(["geometry", "index_right"])
    return "ok"


def select_features_by_spatial_relationship(
    features_geodataframe_name, reference_geodataframe_name, state
):
    frames = [state["data_store"][features_geodataframe_name], state["data_store"][reference_geodataframe_name]]
    return ",".join(sorted({c for f in frames for c in f.columns}))
"""


@pytest.fixture
def pool(tmp_path: Path):
    src = tmp_path / "upstream"
    (src / "geobenchx").mkdir(parents=True)
    (src / "geobenchx" / "__init__.py").write_text("")
    (src / "geobenchx" / "tools.py").write_text(textwrap.dedent(FAKE_TOOLS))
    (tmp_path / "stat").mkdir()
    (tmp_path / "geo").mkdir()
    return UpstreamToolPool(
        python=sys.executable,
        upstream_src=str(src),
        stat_data_path=str(tmp_path / "stat"),
        geo_data_path=str(tmp_path / "geo"),
        scratch_path=str(tmp_path / "scratch"),
        num_workers=2,
        tool_timeout_s=10,
        max_sessions_per_worker=8,
    )


async def test_state_persists_within_a_session_and_is_isolated(pool: UpstreamToolPool, tmp_path: Path) -> None:
    try:
        out = await pool.call("a", "load_data", {"dataset": "GDP", "output_dataframe_name": "gdp"})
        assert out == {"ok": True, "result": "Loaded GDP into gdp", "images": 1}
        assert (await pool.call("a", "list_handles", {}))["result"] == "gdp"
        assert (await pool.call("b", "list_handles", {}))["result"] == ""
        await pool.reset("a")
        assert (await pool.call("a", "list_handles", {}))["result"] == ""
    finally:
        await pool.close()


async def test_unknown_arguments_are_dropped(pool: UpstreamToolPool) -> None:
    try:
        out = await pool.call("a", "load_data", {"dataset": "GDP", "output_dataframe_name": "g", "title": "x"})
        assert out["ok"]
    finally:
        await pool.close()


async def test_tool_errors_are_returned_to_the_model(pool: UpstreamToolPool) -> None:
    try:
        out = await pool.call("a", "explode", {})
        assert not out["ok"]
        assert out["result"].startswith("Error: ValueError('bad column')")
        unknown = await pool.call("a", "no_such_tool", {})
        assert unknown["result"] == "Error: unknown tool 'no_such_tool'"
    finally:
        await pool.close()


async def test_timeout_restarts_the_worker(pool: UpstreamToolPool) -> None:
    pool._timeout = 1
    try:
        out = await pool.call("a", "sleep_forever", {})
        assert "timed out" in out["result"]
        assert (await pool.call("a", "list_handles", {}))["ok"]
    finally:
        await pool.close()


async def test_file_outputs_stay_in_the_worker_scratch_dir(pool: UpstreamToolPool, tmp_path: Path) -> None:
    try:
        out = await pool.call("a", "write_output", {"output_filename": "../../etc/contours", "output_folder": "/etc"})
        path = Path(out["result"])
        assert path.name == "contours"
        assert path.parent.parent == tmp_path / "scratch"
        assert path.parent.name.startswith("worker-")
    finally:
        await pool.close()


async def test_spatial_selection_drops_previous_join_index(pool: UpstreamToolPool) -> None:
    try:
        await pool.call("a", "make_joined", {"name": "first"})
        await pool.call("a", "make_joined", {"name": "second"})
        out = await pool.call(
            "a",
            "select_features_by_spatial_relationship",
            {"features_geodataframe_name": "first", "reference_geodataframe_name": "second"},
        )
        assert out["result"] == "geometry"
    finally:
        await pool.close()


def test_missing_data_paths_fail_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="stat_data_path"):
        UpstreamToolPool(
            python=sys.executable,
            upstream_src=str(tmp_path),
            stat_data_path=str(tmp_path / "missing"),
            geo_data_path=str(tmp_path),
            scratch_path=str(tmp_path / "scratch"),
            num_workers=1,
            tool_timeout_s=1,
            max_sessions_per_worker=1,
        )
