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
# The promoted types run through the server: their own oracle scores 1.0, an empty answer 0, and the instances
# regenerate byte for byte against the hashes upstream recorded in each type's checks.json.
import importlib
import json
from pathlib import Path

import pytest

from resources_servers.swarmbench.app import CAPABILITY_ONLY, GRADES_WORLD, TRACKS, TYPE_DIRS, row_tools
from resources_servers.swarmbench.tests.test_app import _meta, _request, _seed, _server, _verify_body


TYPES_DIR = Path(__file__).parents[1] / "task_types"


def _smallest_n(name: str) -> int:
    return min(t[0] for t in TRACKS[name].TIERS.values())


def _oracle(name: str, world) -> str:
    # Family B keeps its oracle in baselines.py; it edits the repo and the reply is not graded.
    path = TYPE_DIRS[name].replace("/", ".")
    for module in ("solutions", "baselines"):
        if (TYPES_DIR / TYPE_DIRS[name] / f"{module}.py").exists():
            out = importlib.import_module(f"resources_servers.swarmbench.task_types.{path}.{module}").oracle(
                world.task
            )
            return out if isinstance(out, str) else json.dumps(out)
    raise AssertionError(f"{name} has no oracle")


@pytest.mark.parametrize("name", sorted(TYPE_DIRS))
@pytest.mark.asyncio
async def test_type_oracle_scores_one_and_empty_scores_zero(name: str, tmp_path: Path) -> None:
    server = _server(tmp_path)
    meta = _meta(name, n=_smallest_n(name))
    world = await _seed(server, "oracle", meta)
    assert world.task["prompt"].startswith("BENCHMARK DATA SHOULD NEVER APPEAR")
    result = await server.verify(_request("oracle"), _verify_body(meta, _oracle(name, world)))
    assert result.reward == result.score_exact == 1.0, result.score_detail
    assert result.capability_only == (name in CAPABILITY_ONLY)

    await _seed(server, "empty", meta)
    result = await server.verify(_request("empty"), _verify_body(meta, ""))
    assert result.reward == 0.0, result.score_detail
    assert ("empty_answer" in result.score_detail) == (name not in GRADES_WORLD)


@pytest.mark.parametrize("name", sorted(n for n, p in TYPE_DIRS.items() if (TYPES_DIR / p / "checks.json").exists()))
@pytest.mark.asyncio
async def test_type_instances_match_upstream_hashes(name: str, tmp_path: Path) -> None:
    rows = json.loads((TYPES_DIR / TYPE_DIRS[name] / "checks.json").read_text())["rows"]
    row = next(r for r in rows if r["seed"] == 0)
    family = importlib.import_module(f"resources_servers.swarmbench.task_types.{TYPE_DIRS[name].split('/')[0]}.common")
    world = await _seed(_server(tmp_path), "a", _meta(name, n=row["n"], seed=row["seed"]))
    assert family.instance_hash(world.task) == row["hash"]


@pytest.mark.parametrize("name", sorted(TYPE_DIRS))
def test_type_tools_are_routed_by_name(name: str) -> None:
    tools = row_tools(name, 1)
    assert tools and all(t["name"].startswith(f"{name}_") for t in tools)
    if name == "f60_dag":
        assert "_agent" in {t["name"]: t for t in tools}["f60_dag_post_value"]["parameters"]["properties"]
