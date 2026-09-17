# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.terminal_bench_4 import prepare as preparation
from responses_api_agents.miniswe_sandboxed_agent.harness import MiniSWEConfig


def test_prepared_names_match_pinned_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "OUTPUT_PATH", tmp_path / "benchmark.jsonl")
    output = preparation.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads((preparation.BENCHMARK_DIR / "manifest.json").read_text())
    tasks = {"terminal-bench/" + task["name"]: task["ref"] for task in manifest["tasks"]}
    assert len(rows) == len(tasks) == 66
    assert {row["task_name"] for row in rows} == tasks.keys()
    for row in rows:
        assert row["task_ref"] == tasks[row["task_name"]]
        assert row["dataset_ref"] == manifest["ref"]
        assert "path" not in row


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


@pytest.mark.parametrize(
    "overrides,steps,timeout", [({}, 500, 30), ({"tb4_max_steps": 0, "tb4_step_timeout_sec": 45}, 0, 45)]
)
def test_benchmark_limits_resolve_defaults_and_client_overrides(overrides, steps, timeout):
    root = Path(preparation.__file__).resolve().parents[2]
    config = OmegaConf.merge(OmegaConf.load(root / "benchmarks/terminal_bench_4/miniswe.yaml"), overrides)
    agent = config.terminal_bench_4_miniswe.responses_api_agents.miniswe_sandboxed_agent
    harness = config.terminal_bench_4.resources_servers.terminal_bench_4.harness
    assert harness.step_limit == steps
    assert harness.step_timeout_sec == timeout
    assert agent.datasets[0].num_repeats == 1
    assert MiniSWEConfig.model_fields["step_timeout_sec"].default == 600
