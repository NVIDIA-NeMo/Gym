# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ether0 benchmark wrapper: discovery, oracle wiring and preparation."""

import json
from pathlib import Path

import pytest

from benchmarks.ether0 import prepare
from nemo_gym.environment import BenchmarkConfig


GYM_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("benchmarks") / "ether0" / "config.yaml"


def _source_row(index: int) -> dict:
    return {
        "id": f"task-{index}",
        "problem": f"Question {index}",
        "problem_type": "molecule-name",
        "solution": "product_eval!:!CCO!:!molecule-name",
        "ideal": "CCO",
    }


def test_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(GYM_ROOT)
    config = BenchmarkConfig.from_config_path(CONFIG_PATH)
    assert config.name == "ether0"
    assert config.agent_name == "ether0_simple_agent"
    assert config.dataset.jsonl_fpath.resolve() == prepare.OUTPUT_FPATH
    assert config.dataset.prepare_script == Path("benchmarks") / "ether0" / "prepare.py"
    assert config.num_repeats == 1


def test_prepare_writes_one_formatted_row_per_task(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    requested = {}

    def load_dataset(name: str, **kwargs):
        requested.update(name=name, **kwargs)
        return [_source_row(0), _source_row(1)]

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    monkeypatch.setattr(prepare, "NUM_TASKS", 2)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "data" / "test.jsonl")

    output = prepare.prepare()

    assert requested == {"name": "futurehouse/ether0-benchmark", "split": "test", "revision": prepare.DATA_REVISION}
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["verifier_metadata"]["id"] for row in rows] == ["task-0", "task-1"]
    assert rows[0]["responses_create_params"]["input"][-1] == {"role": "user", "content": "Question 0"}
    assert rows[0]["agent_ref"]["name"] == "ether0_simple_agent"
    assert list(output.parent.iterdir()) == [output]


def test_prepare_rejects_an_unexpected_task_set(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: [_source_row(0)])
    monkeypatch.setattr(prepare, "NUM_TASKS", 2)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "data" / "test.jsonl")
    with pytest.raises(ValueError, match="Expected 2 distinct ether0 tasks"):
        prepare.prepare()
    assert not (tmp_path / "data").exists()
