# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Litmus-Bench v0.1 benchmark wrapper."""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


prepare_module = importlib.import_module("benchmarks.litmus-bench.prepare")
BENCHMARK_DIR = Path(__file__).resolve().parents[1]
CONFIG_FPATH = BENCHMARK_DIR / "config.yaml"


def _source_row(*, uuid: str = "litmus-1", use_box_format: bool = True) -> dict:
    return {
        "license": "CC BY 4.0",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": "How many heavy atoms are in CCO? Return an integer in the requested format.",
                }
            ]
        },
        "expected_answer": 3,
        "property_type": "count",
        "property": "HeavyAtomCount",
        "chembl_id": "CHEMBL545",
        "smiles": "CCO",
        "method": "direct",
        "use_box_format": use_box_format,
        "messages": [
            {"role": "user", "content": "placeholder"},
            {"role": "assistant", "content": "placeholder"},
        ],
        "metadata": {},
        "agent_ref": {"type": "responses_api_agents", "name": "rdkit_chemistry_agent"},
        "uuid": uuid,
        "used_in": ["ultra_v3"],
    }


def _patch_dataset(monkeypatch: pytest.MonkeyPatch, rows: list[dict], output_path: Path) -> list[tuple]:
    calls: list[tuple] = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return deepcopy(rows)

    monkeypatch.setattr(prepare_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(prepare_module, "EXPECTED_TEST_ROWS", len(rows))
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)
    return calls


def test_prepare_loads_pinned_test_split_and_preserves_source_fields(monkeypatch, tmp_path) -> None:
    rows = [_source_row(uuid="boxed", use_box_format=True), _source_row(uuid="parentheses", use_box_format=False)]
    rows[0]["expected_answer"] = 3.0  # `datasets` promotes the pinned test integers to floats.
    rows[1]["responses_create_params"]["tools"] = []
    output_path = tmp_path / "data" / "litmus-bench_benchmark.jsonl"
    calls = _patch_dataset(monkeypatch, rows, output_path)

    assert prepare_module.prepare() == output_path
    first_content = output_path.read_bytes()
    assert prepare_module.prepare() == output_path
    assert output_path.read_bytes() == first_content
    assert calls == [
        (
            (prepare_module.HF_DATASET,),
            {"split": "test", "revision": prepare_module.HF_REVISION},
        ),
        (
            (prepare_module.HF_DATASET,),
            {"split": "test", "revision": prepare_module.HF_REVISION},
        ),
    ]

    prepared_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["use_box_format"] for row in prepared_rows] == [True, False]
    assert type(prepared_rows[0]["expected_answer"]) is int
    assert "tools" not in prepared_rows[0]["responses_create_params"]
    assert prepared_rows[1]["responses_create_params"]["tools"] == []
    assert all("agent_ref" not in row for row in prepared_rows)
    for source, prepared in zip(rows, prepared_rows):
        expected = deepcopy(source)
        expected.pop("agent_ref")
        assert prepared == expected


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda row: row.pop("smiles"), "missing required fields"),
        (lambda row: row.update(method="mcp-python"), "method must be 'direct'"),
        (
            lambda row: row["responses_create_params"].update(tools=[{"type": "function", "name": "python"}]),
            "must not expose tools",
        ),
        (lambda row: row.update(expected_answer=3.5), "expected_answer must be integral"),
        (lambda row: row.update(property_type="float"), "unsupported property_type"),
        (lambda row: row.update(use_box_format="true"), "use_box_format must be a boolean"),
    ],
)
def test_prepare_rejects_release_schema_drift(monkeypatch, tmp_path, mutate, error) -> None:
    row = _source_row()
    mutate(row)
    output_path = tmp_path / "litmus-bench_benchmark.jsonl"
    _patch_dataset(monkeypatch, [row], output_path)

    with pytest.raises(ValueError, match=error):
        prepare_module.prepare()
    assert not output_path.exists()


def test_invalid_row_count_does_not_replace_existing_output(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "litmus-bench_benchmark.jsonl"
    output_path.write_text("existing output\n", encoding="utf-8")
    monkeypatch.setattr(prepare_module, "load_dataset", lambda *args, **kwargs: [_source_row()])
    monkeypatch.setattr(prepare_module, "EXPECTED_TEST_ROWS", 2)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)

    with pytest.raises(ValueError, match="has 1 rows; expected 2"):
        prepare_module.prepare()
    assert output_path.read_text(encoding="utf-8") == "existing output\n"


def test_benchmark_config_is_discoverable_and_isolated() -> None:
    benchmark = BenchmarkConfig.from_config_path(CONFIG_FPATH, strict=False)
    assert benchmark is not None
    assert benchmark.name == "litmus-bench"
    assert benchmark.agent_name == "litmus_bench_benchmark_agent"
    assert benchmark.num_repeats == 5
    assert benchmark.dataset.prompt_config is None
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/litmus-bench/data/litmus-bench_benchmark.jsonl")
    assert benchmark.dataset.prepare_script == Path("benchmarks/litmus-bench/prepare.py")

    initial_config = OmegaConf.merge(
        OmegaConf.load(CONFIG_FPATH),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
    )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial_config)
    assert "litmus_agent" not in resolved
    assert "litmus_agent_agent" not in resolved
    assert resolved.responses_create_params.max_output_tokens == 131072

    resource = resolved.litmus_bench_benchmark_resources_server.resources_servers.litmus_agent
    agent = resolved.litmus_bench_benchmark_agent.responses_api_agents.simple_agent
    assert resource.verified is False
    assert agent.resources_server.name == "litmus_bench_benchmark_resources_server"
    assert len(agent.datasets) == 1
    assert agent.datasets[0].name == "litmus-bench"
    assert agent.datasets[0].type == "benchmark"
    assert agent.datasets[0].prompt_config is None
    assert agent.datasets[0].num_repeats == 5
    assert agent.datasets[0].license == "CC-BY-4.0"
