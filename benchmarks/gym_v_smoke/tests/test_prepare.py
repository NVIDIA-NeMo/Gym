# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from benchmarks.gym_v_smoke import prepare as preparation


def test_reproducible_unique_bounded_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "BENCHMARK_DIR", tmp_path)
    output = preparation.prepare()
    first = output.read_bytes()
    assert preparation.prepare().read_bytes() == first
    rows = [json.loads(line) for line in first.splitlines()]
    assert len(rows) == 25
    assert len({row["task_id"] for row in rows}) == 25
    assert len({row["seed"] for row in rows}) == 25
    assert all(row["horizon_cap"] > 0 for row in rows)
    assert all(row["task_metadata"]["split"] == "benchmark" for row in rows)
    expected_budget = {
        "Games/FrozenLake-v0": 1024,
        "Spatial/DoorKey-v0": 1024,
        "Algorithmic/GameOfLife-v0": 2048,
        "Graphs/GridComponent-v0": 2048,
        "Graphs/TreeEvenPartitioning-v0": 2048,
    }
    assert all(row["responses_create_params"]["max_output_tokens"] == expected_budget[row["env_id"]] for row in rows)
    source = Path(preparation.ROOT) / "resources_servers/gym_v/data/example.jsonl"
    source_seeds = {json.loads(line)["seed"] for line in source.read_text().splitlines()}
    assert source_seeds.isdisjoint(row["seed"] for row in rows)
