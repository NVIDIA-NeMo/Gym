# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from harbor.models.job.config import DatasetConfig
from harbor.models.task.id import PackageTaskId

from benchmarks.terminal_bench_4 import prepare as preparation


def test_prepared_names_match_harbor_package_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "OUTPUT_PATH", tmp_path / "benchmark.jsonl")
    output = preparation.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads((preparation.BENCHMARK_DIR / "manifest.json").read_text())
    ids = [PackageTaskId(org="terminal-bench", name=task["name"], ref=task["ref"]) for task in manifest["tasks"]]
    assert len(rows) == len(ids) == 66
    for row in rows:
        config = DatasetConfig(name=manifest["dataset"], ref=manifest["ref"], task_names=[row["task_name"]])
        assert len(config._filter_task_ids(ids)) == 1


@pytest.mark.parametrize("category,count", [("cpu", 52), ("compose", 11), ("gpu", 3)])
def test_category_selections(tmp_path, monkeypatch, category, count):
    monkeypatch.setattr(preparation, "OUTPUT_PATH", tmp_path / "benchmark.jsonl")
    assert len(preparation.prepare(category=category).read_text().splitlines()) == count


def test_explicit_names_and_invalid_selections(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "OUTPUT_PATH", tmp_path / "benchmark.jsonl")
    row = json.loads(preparation.prepare(task_names=["ks-solver-cpp"]).read_text())
    assert row["task_name"] == "terminal-bench/ks-solver-cpp"
    with pytest.raises(ValueError, match="Unknown TB4 task"):
        preparation.prepare(task_names=["missing"])
    with pytest.raises(ValueError, match="Unknown TB4 category"):
        preparation.prepare(category="missing")
    with pytest.raises(ValueError, match="empty"):
        preparation.prepare(task_names=[])
