# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Legal Agent Bench benchmark wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from benchmarks.legal_agent_bench import prepare as benchmark_prepare
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.config_types import ResponsesAPIAgentServerInstanceConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.train_data_utils import TrainDataProcessor
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
    assert benchmark.agent_name == "legal_agent_bench_benchmark_native_agent"
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
    assert "legal_agent_bench_native_agent" not in resolved

    resource = resolved.legal_agent_bench_benchmark_resources_server.resources_servers.legal_agent_bench
    agent = resolved.legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent
    assert agent.agent_server_module == "responses_api_agents.legal_agent_bench_native_agent.app"
    assert agent.runtime_tasks_dir == resource.harbor_tasks_dir
    assert agent.skills_dir == resource.harness_skills_dir
    assert agent.runtime_builder_provider_options == {}
    assert agent.agent_sandbox_provider_options == {}
    assert agent.verifier_sandbox_provider_options == {}
    assert agent.agent_kwargs.max_turns == 60
    assert len(agent.datasets) == 1
    assert agent.datasets[0].type == "benchmark"


def test_harbor_compatibility_variant_resolves() -> None:
    config_path = BENCHMARK_DIR / "config_harbor.yaml"
    benchmark = BenchmarkConfig.from_config_path(config_path, strict=False)
    assert benchmark is not None
    assert benchmark.name == "legal_agent_bench"
    assert benchmark.agent_name == "legal_agent_bench_benchmark_harbor_agent"

    initial_config = OmegaConf.merge(
        OmegaConf.load(config_path),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
    )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial_config)
    resource = resolved.legal_agent_bench_benchmark_harbor_resources_server.resources_servers.legal_agent_bench
    agent = resolved.legal_agent_bench_benchmark_harbor_agent.responses_api_agents.harbor_agent
    assert agent.harbor_datasets.legal_agent_bench.local_dataset_path == resource.harbor_tasks_dir
    assert agent.harbor_agent_kwargs.skills_dir == resource.harness_skills_dir
    assert agent.harbor_agent_kwargs.max_turns == 60
    assert agent.datasets[0].type == "benchmark"


@pytest.mark.parametrize(
    ("filename", "expected_agent", "expected_module"),
    [
        ("config_hermes.yaml", "legal_agent_bench_benchmark_hermes_agent", "responses_api_agents.hermes_agent.app"),
        (
            "config_claude_code.yaml",
            "legal_agent_bench_benchmark_claude_code_agent",
            "responses_api_agents.claude_code_agent.app",
        ),
        ("config_codex.yaml", "legal_agent_bench_benchmark_codex_agent", "responses_api_agents.codex_agent.app"),
    ],
)
def test_configurable_benchmark_variants_resolve(filename, expected_agent, expected_module) -> None:
    config_path = BENCHMARK_DIR / filename
    benchmark = BenchmarkConfig.from_config_path(config_path, strict=False)
    assert benchmark is not None
    assert benchmark.name == "legal_agent_bench"
    assert benchmark.agent_name == expected_agent
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/legal_agent_bench/data/legal_agent_bench_benchmark.jsonl")

    initial_config = OmegaConf.merge(
        OmegaConf.load(config_path),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
    )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial_config)
    agent_names = [
        name for name, value in resolved.items() if isinstance(value, DictConfig) and "responses_api_agents" in value
    ]
    resource_names = [
        name for name, value in resolved.items() if isinstance(value, DictConfig) and "resources_servers" in value
    ]
    assert agent_names == [expected_agent]
    assert resource_names == [f"{expected_agent.removesuffix('_agent')}_resources_server"]
    agent = resolved[expected_agent].responses_api_agents.legal_agent_bench_agent
    assert agent.agent_server_module == expected_module
    if expected_module == "responses_api_agents.claude_code_agent.app":
        assert agent.agent_kwargs.claude_code_version == "2.1.211"
    elif expected_module == "responses_api_agents.codex_agent.app":
        assert agent.agent_kwargs.codex_version == "0.144.4"
        assert agent.agent_kwargs.cwd == "/sandbox/nemo-gym-legal-agent-bench/workspace/output"
    assert agent.datasets[0].type == "benchmark"
    assert agent.runtime_tasks_dir.endswith("data/runtime/harbor_tasks/legal_agent_bench")
    assert agent.runtime_builder_provider_options == {}
    assert agent.agent_sandbox_provider_options == {}
    assert agent.verifier_sandbox_provider_options == {}


@pytest.mark.parametrize(
    ("filename", "expected_agent"),
    [
        ("config.yaml", "legal_agent_bench_benchmark_native_agent"),
        ("config_hermes.yaml", "legal_agent_bench_benchmark_hermes_agent"),
        ("config_claude_code.yaml", "legal_agent_bench_benchmark_claude_code_agent"),
        ("config_codex.yaml", "legal_agent_bench_benchmark_codex_agent"),
    ],
)
def test_configurable_variants_decode_phase_provider_options_from_environment(
    monkeypatch, filename, expected_agent
) -> None:
    monkeypatch.setenv(
        "NEMO_GYM_LAB_RUNTIME_BUILDER_PROVIDER_OPTIONS",
        "{policy: /tmp/lab-builder-policy.yaml}",
    )
    monkeypatch.setenv(
        "NEMO_GYM_LAB_AGENT_SANDBOX_PROVIDER_OPTIONS",
        "{policy: /tmp/lab-agent-policy.yaml}",
    )
    monkeypatch.setenv(
        "NEMO_GYM_LAB_VERIFIER_SANDBOX_PROVIDER_OPTIONS",
        "{policy: /tmp/lab-verifier-policy.yaml}",
    )
    initial_config = OmegaConf.merge(
        OmegaConf.load(BENCHMARK_DIR / filename),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
    )

    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial_config)
    agent = resolved[expected_agent].responses_api_agents.legal_agent_bench_agent

    assert agent.runtime_builder_provider_options == {"policy": "/tmp/lab-builder-policy.yaml"}
    assert agent.agent_sandbox_provider_options == {"policy": "/tmp/lab-agent-policy.yaml"}
    assert agent.verifier_sandbox_provider_options == {"policy": "/tmp/lab-verifier-policy.yaml"}


@pytest.mark.parametrize(
    "agent_name",
    [
        "legal_agent_bench_benchmark_native_agent",
        "legal_agent_bench_benchmark_harbor_agent",
        "legal_agent_bench_benchmark_hermes_agent",
        "legal_agent_bench_benchmark_claude_code_agent",
        "legal_agent_bench_benchmark_codex_agent",
    ],
)
def test_benchmark_collation_stamps_selected_agent_without_changing_source(tmp_path, agent_name) -> None:
    source = tmp_path / "legal_agent_bench.jsonl"
    neutral_row = {
        "instance_id": "legal_agent_bench::corporate__task",
        "responses_create_params": {"input": []},
    }
    source.write_text(json.dumps(neutral_row) + "\n", encoding="utf-8")
    agent_config = {
        "responses_api_agents": {
            "agent": {
                "host": "127.0.0.1",
                "port": 12345,
                "entrypoint": "app.py",
                "resources_server": {"type": "resources_servers", "name": "legal_agent_bench"},
                "model_server": {"type": "responses_api_models", "name": "policy_model"},
                "datasets": [
                    {
                        "name": "legal_agent_bench",
                        "type": "benchmark",
                        "jsonl_fpath": str(source),
                        "prepare_script": "benchmarks/legal_agent_bench/prepare.py",
                        "num_repeats": 1,
                    }
                ],
            }
        }
    }
    instance = ResponsesAPIAgentServerInstanceConfig(
        name=agent_name,
        server_type_config_dict=DictConfig(agent_config),
        responses_api_agents=agent_config["responses_api_agents"],
    )

    prepared = TrainDataProcessor()._collate_samples_single_type("benchmark", [instance])[0]
    collated_row = json.loads(prepared.read_text(encoding="utf-8"))

    assert json.loads(source.read_text(encoding="utf-8")) == neutral_row
    assert collated_row["agent_ref"] == {"type": "responses_api_agents", "name": agent_name}
