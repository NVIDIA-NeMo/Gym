# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local ADME Tier-5 Litmus-Bench companion."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig


prepare_module = importlib.import_module("benchmarks.litmus-bench.prepare_adme_tier5")
BENCHMARK_DIR = Path(__file__).resolve().parents[1]


def _row(*, uuid: str, answer_type: str = "float", match: dict | None = None) -> dict:
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "Predict the property."}]},
        "expected_answer": 1.5 if answer_type == "float" else 1,
        "answer_type": answer_type,
        "property": "LogD",
        "method": "direct",
        "output_regex": r"Answer: ([-+]?\d*\.?\d+)",
        "uuid": uuid,
        "agent_ref": {"type": "responses_api_agents", "name": "stale_agent"},
    }
    if match is not None:
        row["match"] = match
    return row


def _write_source(source_dir: Path, subset: str, rows: list[dict]) -> None:
    source_path = source_dir / prepare_module.SUBSETS[subset] / "nemo_gym_data" / "validation.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_returns_one_path_and_drops_unscorable_float_rows(monkeypatch, tmp_path) -> None:
    source_dir = tmp_path / "source"
    data_dir = tmp_path / "prepared"
    _write_source(
        source_dir,
        "direct",
        [
            _row(uuid="kept", match={"rule": "abs_window", "abs_tol": 0.3}),
            _row(uuid="dropped"),
        ],
    )
    monkeypatch.setattr(prepare_module, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(prepare_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(prepare_module, "SPLIT", "validation")

    output_path = prepare_module.prepare("direct")

    assert output_path == data_dir / "adme-tier5-direct_benchmark.jsonl"
    prepared = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["uuid"] for row in prepared] == ["kept"]
    assert "agent_ref" not in prepared[0]


def test_prepare_preserves_scorable_boolean_rows(monkeypatch, tmp_path) -> None:
    source_dir = tmp_path / "source"
    data_dir = tmp_path / "prepared"
    _write_source(source_dir, "comparison", [_row(uuid="bool", answer_type="bool")])
    monkeypatch.setattr(prepare_module, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(prepare_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(prepare_module, "SPLIT", "validation")

    output_path = prepare_module.prepare("comparison")

    prepared = json.loads(output_path.read_text(encoding="utf-8"))
    assert prepared["uuid"] == "bool"
    assert prepared["answer_type"] == "bool"


def test_prepare_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="Unknown ADME Tier-5 subset"):
        prepare_module.prepare("missing")


@pytest.mark.parametrize("subset", ["direct", "analogue", "comparison"])
def test_bundled_example_has_ten_scorable_rows(monkeypatch, subset: str) -> None:
    monkeypatch.setattr(prepare_module, "SPLIT", "example")
    monkeypatch.delenv(prepare_module.SOURCE_DIR_ENV, raising=False)

    content, kept, total, dropped = prepare_module._render_subset(subset, prepare_module.SOURCE_DIR)

    assert len(content.splitlines()) == kept == total == 10
    assert dropped == {}


@pytest.mark.parametrize("subset", ["direct", "analogue", "comparison"])
def test_adme_config_wires_prepare_argument_and_dataset(subset: str) -> None:
    config_path = BENCHMARK_DIR / f"config_adme_{subset}.yaml"
    config = OmegaConf.load(config_path)
    benchmark = BenchmarkConfig.from_config_path(config_path, strict=False)

    assert config.prepare_script_args.subset == subset
    assert benchmark is not None
    assert benchmark.name == f"adme-tier5-{subset}"
    assert benchmark.agent_name == f"adme_tier5_{subset}_agent"
    assert benchmark.dataset.prepare_script == Path("benchmarks/litmus-bench/prepare_adme_tier5.py")
    assert benchmark.dataset.jsonl_fpath == Path(f"benchmarks/litmus-bench/data/adme-tier5-{subset}_benchmark.jsonl")
    assert benchmark.num_repeats == 5
