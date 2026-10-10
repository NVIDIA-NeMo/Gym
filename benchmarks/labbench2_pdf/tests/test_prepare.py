# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the direct-PDF LabBench2 benchmark wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.labbench2_pdf import prepare as benchmark_prepare
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
CONFIG_FPATH = BENCHMARK_DIR / "config.yaml"


def test_prepare_materializes_tasks_and_writes_benchmark_index(monkeypatch, tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    source_index = tasks_dir / "rollout_input.jsonl"
    output_path = tmp_path / "benchmark" / "labbench2_pdf_benchmark.jsonl"
    row = {
        "task_name": "figqa2-0001-example",
        "responses_create_params": {"input": []},
        "agent_ref": {"name": benchmark_prepare.BENCHMARK_AGENT_NAME},
    }
    calls = []

    def fake_prepare_data(options):
        calls.append(options)
        tasks_dir.mkdir(parents=True)
        source_index.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {"materialization": {"output_dir": str(tasks_dir)}}

    monkeypatch.setattr(benchmark_prepare, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(benchmark_prepare, "OUTPUT_FPATH", output_path)
    monkeypatch.setattr(benchmark_prepare, "TASKS_DIR", tasks_dir)

    result = benchmark_prepare.prepare(
        questions_dir="/tmp/questions",
        papers_dir=Path("/tmp/papers"),
        benchmarks=["figqa2"],
        limit_per_benchmark=1,
        build_image=False,
    )

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == json.dumps(row) + "\n"
    assert len(calls) == 1
    options = calls[0]
    assert options.questions_dir == Path("/tmp/questions")
    assert options.papers_dir == Path("/tmp/papers")
    assert options.output_dir == tasks_dir
    assert options.benchmarks == ("figqa2",)
    assert options.limit_per_benchmark == 1
    assert options.build_image is False
    assert options.agent_name == benchmark_prepare.BENCHMARK_AGENT_NAME
    assert options.overwrite_tasks is True


def test_prepare_rejects_unknown_script_arguments() -> None:
    with pytest.raises(TypeError, match="unsupported LabBench2 PDF preparation arguments: typo"):
        benchmark_prepare.prepare(typo=True)


def test_benchmark_config_is_discoverable_and_uses_generated_tasks() -> None:
    benchmark = BenchmarkConfig.from_config_path(CONFIG_FPATH, strict=False)
    assert benchmark is not None
    assert benchmark.name == "labbench2_pdf"
    assert benchmark.agent_name == benchmark_prepare.BENCHMARK_AGENT_NAME
    assert benchmark.num_repeats == 1
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/labbench2_pdf/data/labbench2_pdf_benchmark.jsonl")
    assert benchmark.dataset.prepare_script == Path("benchmarks/labbench2_pdf/prepare.py")

    initial_config = OmegaConf.merge(
        OmegaConf.load(CONFIG_FPATH),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
    )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial_config)
    assert "harbor_agent_general" not in resolved

    agent = resolved.labbench2_pdf_benchmark_harbor_agent.responses_api_agents.harbor_agent_general
    assert agent.harbor_dataset.path == "environments/labbench2_pdf/data/tasks_docker"
    assert len(agent.datasets) == 1
    assert agent.datasets[0].type == "benchmark"
