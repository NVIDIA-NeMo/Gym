# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Legal Agent Bench benchmark wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.legal_agent_bench import prepare as benchmark_prepare
from nemo_gym.cli.eval import _multiprocess_benchmark_prepare_fn
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.environment import BenchmarkConfig
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
    assert all("agent_ref" not in row for row in rows)


def test_prepare_subset_forwards_selection_and_environment_cache_paths(monkeypatch, tmp_path) -> None:
    tasks_dir, task_names = _write_task_index(tmp_path, 2)
    selection = tmp_path / "selection.jsonl"
    selection.write_bytes((tasks_dir / INDEX_FILENAME).read_bytes())
    original_selection = selection.read_bytes()
    output = tmp_path / "benchmark.jsonl"
    output.write_text("existing full index\n")
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_FPATH", output)
    monkeypatch.setenv("LEGAL_AGENT_BENCH_TASK_CACHE_DIR", str(tasks_dir))
    monkeypatch.setenv("LEGAL_AGENT_BENCH_SKILLS_DIR", str(tmp_path / "skills"))
    calls = []

    def prepare_assets(asset, **kwargs):
        calls.append((asset, kwargs))
        return {"tasks": tasks_dir}

    monkeypatch.setattr(benchmark_prepare, "prepare_assets", prepare_assets)
    assert benchmark_prepare.prepare(input=selection) == selection
    assert selection.read_bytes() == original_selection
    assert output.read_text() == "existing full index\n"
    assert calls == [
        (
            "all",
            {
                "force": False,
                "tasks_dir": str(tasks_dir),
                "skills_dir": str(tmp_path / "skills"),
                "task_ids": tuple(task_names),
            },
        )
    ]


@pytest.mark.parametrize(
    "filename",
    ["config.yaml", "config_native.yaml"],
)
def test_subset_config_and_cli_prepare_use_input_without_publishing_full_index(
    monkeypatch, tmp_path, filename
) -> None:
    tasks, _ = _write_task_index(tmp_path, 1)
    selection = tmp_path / "selection.jsonl"
    # These caller-provided settings must survive preparation unchanged.
    selection.write_text(
        json.dumps(
            {
                "instance_id": "legal_agent_bench::practice-area__task-0000",
                "responses_create_params": {"input": [], "temperature": 0.2, "top_p": None},
            }
        )
        + "\n"
    )
    original = selection.read_bytes()
    full_index = tmp_path / "full.jsonl"
    full_index.write_text("full dataset\n")
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_FPATH", full_index)
    monkeypatch.setattr(benchmark_prepare, "prepare_assets", lambda *args, **kwargs: {"tasks": tasks})
    config = OmegaConf.merge(
        OmegaConf.load(BENCHMARK_DIR / filename),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"prepare_script_args": {"input": str(selection)}},
    )
    benchmark = BenchmarkConfig.from_initial_config_dict(
        path=BENCHMARK_DIR / filename, initial_config_dict=config, strict=False
    )
    assert benchmark.dataset.jsonl_fpath == selection
    _multiprocess_benchmark_prepare_fn(
        (benchmark, "benchmarks.legal_agent_bench.prepare", {"input": str(selection)}, None)
    )
    assert selection.read_bytes() == original
    assert full_index.read_text() == "full dataset\n"


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
    agent = resolved.legal_agent_bench_benchmark_harbor_agent.responses_api_agents.harbor_agent_general
    assert agent.job.datasets[0].path == resource.harbor_tasks_dir
    assert agent.job.agents[0].kwargs.skills_dir == resource.harness_skills_dir
    assert agent.job.agents[0].kwargs.max_turns == 60
    assert len(agent.datasets) == 1
    assert agent.datasets[0].type == "benchmark"


def test_native_archive_limits_override_defaults_through_config() -> None:
    from resources_servers.legal_agent_bench.session_app import LabSessionResourcesConfig

    initial = OmegaConf.merge(
        OmegaConf.load(BENCHMARK_DIR / "config_native.yaml"),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "lab_judges": [{"model_name": "gpt-5.5"}],
            "legal_agent_bench_benchmark_resources_server": {
                "resources_servers": {
                    "legal_agent_bench": {
                        "archive_limits": {"max_archive_bytes": 8 * 1024**3, "timeout_seconds": 1800}
                    }
                }
            },
        },
    )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)
    settings = OmegaConf.to_container(
        resolved.legal_agent_bench_benchmark_resources_server.resources_servers.legal_agent_bench, resolve=True
    )
    config = LabSessionResourcesConfig.model_validate({**settings, "name": "lab", "host": "127.0.0.1", "port": 0})
    assert config.archive_limits.max_archive_bytes == 8 * 1024**3
    assert config.archive_limits.timeout_seconds == 1800
    assert config.archive_limits.max_expanded_bytes == 16 * 1024**3


@pytest.mark.parametrize("concurrency", [1, 3])
def test_native_preset_uses_collection_concurrency(concurrency: int) -> None:
    initial = OmegaConf.merge(
        OmegaConf.load(BENCHMARK_DIR / "config_native.yaml"),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"num_samples_in_parallel": concurrency},
    )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)
    assert (
        resolved.legal_agent_bench_benchmark_resources_server.resources_servers.legal_agent_bench.concurrency
        == concurrency
    )
