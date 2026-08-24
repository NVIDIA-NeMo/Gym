# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from resources_servers.visgym.schemas import VisGymTaskRow


VISGYM_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = VISGYM_ROOT / "scripts" / "create_maze_curriculum.py"
# The generator's default budget; the index carries it so two variants cannot
# overwrite each other's index.
INDEX_NAME = "maze_2d_easy_curriculum_5x5_7x7_9x9_11x11_manifest_index_t1024.json"


def _generate_curriculum(output_dir: Path) -> Path:
    """Run the committed generator and return the manifest index it wrote.

    The full curriculum is 5120 rows across five files (~8 MB), so it is
    generated on demand instead of being committed. The generator is pure and
    deterministic, which is what makes that safe — see
    test_curriculum_generator_is_deterministic.
    """
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output-dir", str(output_dir)],
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
    expected_horizons = [8] * 1280 + [12] * 1280 + [25] * 1280 + [35] * 1280
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
