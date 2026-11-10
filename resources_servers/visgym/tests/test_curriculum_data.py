# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest
from yaml import safe_load

from resources_servers.visgym.schemas import VisGymTaskRow
from resources_servers.visgym.tests.conftest import requires_visgym


VISGYM_ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = VISGYM_ROOT.parents[1]
PREPARE = GYM_ROOT / "environments" / "visgym" / "prepare.py"
# The generator's default budget; the index carries it so two variants cannot
# overwrite each other's index.
INDEX_NAME = "maze_2d_easy_curriculum_5x5_7x7_9x9_11x11_manifest_index_t1024.json"


def _generate_curriculum(output_dir: Path) -> Path:
    """Run the canonical prepare entry point and return its manifest index.

    The full curriculum is 5120 rows across five files (~8 MB), so it is
    generated on demand instead of being committed. The generator is pure and
    deterministic, which is what makes that safe — see
    test_curriculum_generator_is_deterministic.
    """
    subprocess.run(
        [sys.executable, str(PREPARE), "maze", "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_dir / INDEX_NAME


@pytest.fixture(scope="module")
def curriculum_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("maze_curriculum")
    _generate_curriculum(output_dir)
    return output_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _without_task_idx(rows: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "task_idx"} for row in rows]


def test_curriculum_is_ordered_and_schema_valid(curriculum_dir: Path) -> None:
    index = json.loads((curriculum_dir / INDEX_NAME).read_text())
    combined_path = curriculum_dir / index["combined"]["path"]
    combined_rows = _read_jsonl(combined_path)

    assert index["curriculum_name"] == "maze_size_5_7_9_11"
    assert index["curriculum_order"] == "ascending_maze_size"
    assert index["shuffle_required"] is False
    assert index["samples_per_stage"] == 1280
    assert index["total_rows"] == 5120
    assert len(combined_rows) == 5120

    expected_sizes = ["5x5"] * 1280 + ["7x7"] * 1280 + ["9x9"] * 1280 + ["11x11"] * 1280
    expected_horizons = [10] * 1280 + [20] * 1280 + [34] * 1280 + [52] * 1280
    assert [row["task_metadata"]["maze_size"] for row in combined_rows] == expected_sizes
    assert [row["horizon_cap"] for row in combined_rows] == expected_horizons
    assert [row["task_idx"] for row in combined_rows] == list(range(5120))
    assert len({row["task_id"] for row in combined_rows}) == 5120
    assert len({row["seed"] for row in combined_rows}) == 5120

    for row in combined_rows:
        size = int(row["task_metadata"]["maze_size"].split("x", 1)[0])
        assert row["env_kwargs"] == {
            "maze_width": size,
            "maze_height": size,
        }
        VisGymTaskRow.model_validate(row)

    offset = 0
    for stage in index["stages"]:
        stage_rows = _read_jsonl(curriculum_dir / stage["path"])
        assert len(stage_rows) == stage["rows"] == 1280
        expected_slice = combined_rows[offset : offset + len(stage_rows)]
        assert _without_task_idx(stage_rows) == _without_task_idx(expected_slice)
        offset += len(stage_rows)
    assert offset == len(combined_rows)


def test_curriculum_generator_is_deterministic(curriculum_dir: Path, tmp_path: Path) -> None:
    """Two independent runs must be byte-identical.

    Determinism is the contract that lets the launcher regenerate the manifest
    on any node instead of shipping it: a curriculum that differs run to run
    would silently change which seeds a resumed job trains on.
    """
    _generate_curriculum(tmp_path)

    index = json.loads((curriculum_dir / INDEX_NAME).read_text())
    expected_paths = [curriculum_dir / INDEX_NAME, curriculum_dir / index["combined"]["path"]]
    expected_paths.extend(curriculum_dir / stage["path"] for stage in index["stages"])

    for expected_path in expected_paths:
        generated_path = tmp_path / expected_path.name
        assert generated_path.read_bytes() == expected_path.read_bytes()


@pytest.mark.parametrize(
    "config_path",
    [
        GYM_ROOT / "environments" / "visgym" / "config.yaml",
        VISGYM_ROOT / "configs" / "visgym_maze2d_thinking_agent.yaml",
    ],
)
def test_maze_agent_backstops_cover_generated_horizons(
    curriculum_dir: Path,
    config_path: Path,
) -> None:
    index = json.loads((curriculum_dir / INDEX_NAME).read_text())
    max_horizon = max(stage["horizon_cap"] for stage in index["stages"])
    config = safe_load(config_path.read_text())
    agent_max_steps = config["visgym_agent"]["responses_api_agents"]["visgym_agent"]["max_steps"]
    assert agent_max_steps >= max_horizon


def _optimal_turns_for_current_maze(env) -> int:
    """Return shortest-path moves plus the terminal stop action."""
    start = tuple(int(value) for value in env._agent_location)
    target = tuple(int(value) for value in env._target_location)
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (row, column), moves = queue.popleft()
        if (row, column) == target:
            return moves + 1
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (row + row_delta, column + column_delta)
            if (
                0 <= candidate[0] < env.maze_height
                and 0 <= candidate[1] < env.maze_width
                and env.maze_map[candidate] == 0
                and candidate not in visited
            ):
                visited.add(candidate)
                queue.append((candidate, moves + 1))
    raise AssertionError("VisGym generated a maze with no route to the target")


@requires_visgym
def test_every_default_curriculum_maze_has_recovery_slack(curriculum_dir: Path) -> None:
    """No generated seed may time out even under an optimal policy.

    Keep this tied to the real pinned VisGym runtime: checking only constants in
    the manifest previously missed 129 impossible episodes.
    """
    import gymnasium as gym

    index = json.loads((curriculum_dir / INDEX_NAME).read_text())
    rows = _read_jsonl(curriculum_dir / index["combined"]["path"])

    for size in (5, 7, 9, 11):
        env = gym.make("maze_2d/easy", maze_width=size, maze_height=size).unwrapped
        # Rendering is irrelevant to graph reachability and dominates this
        # exhaustive 5,120-seed contract test.
        env._get_obs = lambda: None
        try:
            for row in (candidate for candidate in rows if candidate["env_kwargs"]["maze_width"] == size):
                env.reset(seed=row["seed"])
                optimal_turns = _optimal_turns_for_current_maze(env)
                assert row["horizon_cap"] >= optimal_turns + 3, row["task_id"]
        finally:
            env.close()
