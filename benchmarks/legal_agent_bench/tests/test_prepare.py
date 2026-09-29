# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Legal Agent Bench benchmark wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.legal_agent_bench import prepare as benchmark_prepare
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from resources_servers.legal_agent_bench.prepare import EXPECTED_TASK_COUNT, INDEX_FILENAME


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
CONFIG_FPATH = BENCHMARK_DIR / "config.yaml"


def _write_task_index(parent: Path, count: int) -> tuple[Path, list[str]]:
    tasks_dir = parent / "tasks"
    tasks_dir.mkdir()
    task_names = [f"practice-area__task-{index:04d}" for index in range(count)]
    rows = []
    for task_name in task_names:
        rows.append(
            {
                "agent_ref": {
                    "name": "legal_agent_bench_harbor_agent",
                    "type": "responses_api_agents",
                },
                "instance_id": f"legal_agent_bench::{task_name}",
                "responses_create_params": {
                    "input": [],
                    "temperature": 1.0,
                    "top_p": 0.95,
                },
            }
        )
    (tasks_dir / INDEX_FILENAME).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return tasks_dir, task_names


def _mock_asset_preparation(monkeypatch: pytest.MonkeyPatch, tasks_dir: Path) -> list[tuple[str, bool]]:
    calls: list[tuple[str, bool]] = []

    def fake_prepare_assets(asset: str, *, force: bool = False):
        calls.append((asset, force))
        return {"tasks": tasks_dir, "skills": tasks_dir.parent / "skills"}

    monkeypatch.setattr(benchmark_prepare, "prepare_assets", fake_prepare_assets)
    return calls


def test_prepare_writes_deterministic_complete_benchmark_index(monkeypatch, tmp_path) -> None:
    tasks_dir, task_names = _write_task_index(tmp_path, EXPECTED_TASK_COUNT)
    output_path = tmp_path / "output" / "legal_agent_bench_benchmark.jsonl"
    calls = _mock_asset_preparation(monkeypatch, tasks_dir)
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_FPATH", output_path)

    assert benchmark_prepare.prepare() == output_path
    first_content = output_path.read_bytes()
    assert benchmark_prepare.prepare(force=True) == output_path
    assert output_path.read_bytes() == first_content
    assert calls == [("all", False), ("all", True)]

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == EXPECTED_TASK_COUNT
    assert [row["instance_id"].split("::", 1)[1] for row in rows] == task_names
    assert all(
        row["agent_ref"]
        == {
            "name": benchmark_prepare.BENCHMARK_AGENT_NAME,
            "type": "responses_api_agents",
        }
        for row in rows
    )


def test_wrong_row_count_does_not_replace_existing_output(monkeypatch, tmp_path) -> None:
    tasks_dir, _ = _write_task_index(tmp_path, 1)
    output_path = tmp_path / "benchmark.jsonl"
    output_path.write_text("existing output\n", encoding="utf-8")
    _mock_asset_preparation(monkeypatch, tasks_dir)
    monkeypatch.setattr(benchmark_prepare, "EXPECTED_TASK_COUNT", 2)
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_FPATH", output_path)

    with pytest.raises(ValueError, match="Expected 2 LAB benchmark rows, found 1"):
        benchmark_prepare.prepare()
    assert output_path.read_text(encoding="utf-8") == "existing output\n"


def test_asset_preparation_failure_does_not_replace_existing_output(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "benchmark.jsonl"
    output_path.write_text("existing output\n", encoding="utf-8")
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_FPATH", output_path)

    def fail_preparation(*args, **kwargs):
        raise RuntimeError("asset preparation failed")

    monkeypatch.setattr(benchmark_prepare, "prepare_assets", fail_preparation)

    with pytest.raises(RuntimeError, match="asset preparation failed"):
        benchmark_prepare.prepare()
    assert output_path.read_text(encoding="utf-8") == "existing output\n"


def test_malformed_source_index_does_not_replace_existing_output(monkeypatch, tmp_path) -> None:
    tasks_dir, _ = _write_task_index(tmp_path, 1)
    (tasks_dir / INDEX_FILENAME).write_text("not JSON\n", encoding="utf-8")
    output_path = tmp_path / "benchmark.jsonl"
    output_path.write_text("existing output\n", encoding="utf-8")
    _mock_asset_preparation(monkeypatch, tasks_dir)
    monkeypatch.setattr(benchmark_prepare, "EXPECTED_TASK_COUNT", 1)
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_FPATH", output_path)

    with pytest.raises(ValueError, match="Invalid LAB task index JSON on line 1"):
        benchmark_prepare.prepare()
    assert output_path.read_text(encoding="utf-8") == "existing output\n"


def test_benchmark_config_is_isolated_and_resolves_shared_cache_paths() -> None:
    benchmark = BenchmarkConfig.from_config_path(CONFIG_FPATH, strict=False)
    assert benchmark is not None
    assert benchmark.name == "legal_agent_bench"
    assert benchmark.agent_name == "legal_agent_bench_benchmark_harbor_agent"
    assert benchmark.num_repeats == 1
    assert benchmark.dataset.prompt_config is None
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/legal_agent_bench/data/legal_agent_bench_benchmark.jsonl")
    assert benchmark.dataset.prepare_script == Path("benchmarks/legal_agent_bench/prepare.py")

    initial_config = OmegaConf.merge(
        OmegaConf.load(CONFIG_FPATH),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
    )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial_config)
    assert "legal_agent_bench" not in resolved
    assert "legal_agent_bench_harbor_agent" not in resolved

    resource = resolved.legal_agent_bench_benchmark_resources_server.resources_servers.legal_agent_bench
    agent = resolved.legal_agent_bench_benchmark_harbor_agent.responses_api_agents.harbor_agent
    assert agent.harbor_datasets.legal_agent_bench.local_dataset_path == resource.harbor_tasks_dir
    assert agent.harbor_agent_kwargs.skills_dir == resource.harness_skills_dir
    assert agent.harbor_agent_kwargs.max_turns == 60
    assert len(agent.datasets) == 1
    assert agent.datasets[0].type == "benchmark"
