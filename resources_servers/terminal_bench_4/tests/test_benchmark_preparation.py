# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.terminal_bench_4 import prepare as preparation
from nemo_gym.task_data import TaskDataValidator, load_task_data_schema
from responses_api_agents.miniswe_sandboxed_agent.harness import MiniSWEConfig


SERVER_DIR = Path(__file__).resolve().parents[1]


def task_validator() -> TaskDataValidator:
    return TaskDataValidator("terminal_bench_4", load_task_data_schema(SERVER_DIR), "test.jsonl")


def test_prepared_names_match_pinned_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "OUTPUT_PATH", tmp_path / "benchmark.jsonl")
    output = preparation.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads((preparation.BENCHMARK_DIR / "manifest.json").read_text())
    tasks = {"terminal-bench/" + task["name"]: task["ref"] for task in manifest["tasks"]}
    assert len(rows) == len(tasks) == 66
    assert {row["task_name"] for row in rows} == tasks.keys()
    validator = task_validator()
    for index, row in enumerate(rows):
        validator.validate_row(index, row)
        assert row["task_id"] == row["task_name"]
        assert row["task_ref"] == tasks[row["task_name"]]
        assert row["dataset_ref"] == manifest["ref"]
        assert "path" not in row
    assert validator.report.clean, validator.report.summary()


@pytest.mark.parametrize("missing", ["task_name", "task_ref", "dataset_ref"])
def test_schema_rejects_missing_task_identity(missing: str) -> None:
    row = {"task_name": "terminal-bench/ks-solver-cpp", "task_ref": "sha256:task", "dataset_ref": "sha256:dataset"}
    del row[missing]
    validator = task_validator()
    validator.validate_row(0, row)
    assert validator.report.error_rows == 1
    assert missing in validator.report.summary()


def test_example_rollouts_match_pinned_tasks() -> None:
    examples = [json.loads(line) for line in (SERVER_DIR / "data/example.jsonl").read_text().splitlines()]
    rollouts = [json.loads(line) for line in (SERVER_DIR / "data/example_rollouts.jsonl").read_text().splitlines()]
    manifest = json.loads((preparation.BENCHMARK_DIR / "manifest.json").read_text())
    tasks = {"terminal-bench/" + task["name"]: task for task in manifest["tasks"]}
    assert len(examples) == len(rollouts) == 5
    assert len({row["task_name"] for row in examples}) == 5
    assert {tasks[row["task_name"]]["category"] for row in examples} == {"cpu", "compose", "gpu"}
    validator = task_validator()
    for index, (example, rollout) in enumerate(zip(examples, rollouts, strict=True)):
        validator.validate_row(index, example)
        assert example["task_ref"] == tasks[example["task_name"]]["ref"]
        assert example["dataset_ref"] == manifest["ref"]
        for key in ("task_name", "task_ref", "dataset_ref"):
            assert rollout[key] == example[key]
        assert rollout["evaluation_completed"] is True
        assert rollout["infrastructure_error"] is None
        assert rollout["response"]["output"]
        assert rollout["harness_version"] == "2.4.6"
    assert validator.report.clean, validator.report.summary()


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
