# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the paired Litmus Tier 1/2 example benchmark."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig


prepare_module = importlib.import_module("benchmarks.litmus-bench.prepare_tier12")
BENCHMARK_DIR = Path(__file__).resolve().parents[1]
CONFIG_FPATH = BENCHMARK_DIR / "config_tier12.yaml"


def _paired_rows(tier: int = 1) -> list[dict]:
    base = {
        "responses_create_params": {"input": [{"role": "user", "content": "Calculate the value."}]},
        "expected_answer": 3.0,
        "answer_type": "float",
        "property": "ExactMolWt",
        "uuid": "answer-contract",
        "question_uuid": "question-1",
        "pair_id": "question-1",
        "tier": tier,
    }
    direct = {**deepcopy(base), "method": "direct", "tool_use": False}
    tool = {**deepcopy(base), "method": "mcp-python", "tool_use": True}
    tool["responses_create_params"]["tools"] = [
        {"type": "function", "name": prepare_module.TOOL_NAME, "parameters": {"type": "object"}}
    ]
    return [direct, tool]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_bundled_example_is_balanced_and_paired() -> None:
    content = prepare_module._render_rows(prepare_module.SOURCE_FPATH)
    rows = [json.loads(line) for line in content.splitlines()]

    assert len(rows) == 20
    assert {row["tier"] for row in rows} == {1, 2}
    assert sum(row["tier"] == 1 for row in rows) == 10
    assert sum(row["tier"] == 2 for row in rows) == 10
    assert sum(row["method"] == "direct" for row in rows) == 10
    assert sum(row["method"] == "mcp-python" for row in rows) == 10


def test_render_rejects_pair_with_different_prompt(tmp_path) -> None:
    rows: list[dict] = []
    for tier in (1, 2):
        for index in range(prepare_module.EXPECTED_QUESTIONS_PER_TIER):
            pair = _paired_rows(tier)
            pair_id = f"tier-{tier}-question-{index}"
            for row in pair:
                row["question_uuid"] = pair_id
                row["pair_id"] = pair_id
            rows.extend(pair)
    rows[1]["responses_create_params"]["input"][0]["content"] = "Different prompt."
    source_path = tmp_path / "source.jsonl"
    _write_rows(source_path, rows)

    with pytest.raises(ValueError, match="does not use an identical prompt"):
        prepare_module._render_rows(source_path)


def test_prepare_writes_validated_content(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "prepared.jsonl"
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)

    assert prepare_module.prepare() == output_path
    assert output_path.read_text(encoding="utf-8") == prepare_module._render_rows(prepare_module.SOURCE_FPATH)


def test_config_wires_paired_benchmark_and_rdkit_sandbox() -> None:
    config = OmegaConf.load(CONFIG_FPATH)
    benchmark = BenchmarkConfig.from_config_path(CONFIG_FPATH, strict=False)

    assert benchmark is not None
    assert benchmark.name == "litmus-tier12-paired"
    assert benchmark.agent_name == "litmus_tier12_agent"
    assert benchmark.dataset.prepare_script == Path("benchmarks/litmus-bench/prepare_tier12.py")
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/litmus-bench/data/litmus-tier12-paired_benchmark.jsonl")
    assert benchmark.num_repeats == 1
    resource = config.litmus_tier12_resources_server.resources_servers.litmus_agent
    assert resource.sandbox_spec.image == "docker.io/mcs07/rdkit:latest"
    assert resource.sandbox_spec.workdir == "/tmp"
