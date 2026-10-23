# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDPval harnesses share source metrics, while rendered outputs stay distinct."""

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from benchmarks.gdpval import prepare
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.train_data_utils import TrainDataProcessor


def _config(harness: str, source: Path, output: Path) -> DictConfig:
    parser = GlobalConfigDictParser()
    filename = "config.yaml" if harness == "stirrup" else "nooa.yaml"
    _, parts = parser.load_extra_config_paths([f"benchmarks/gdpval/{filename}"])
    config = OmegaConf.merge(*parts)
    parser._recursively_swap_keys(config)
    if harness == "stirrup":
        dataset = config.gdpval_stirrup_agent.responses_api_agents.stirrup_agent.datasets[0]
    else:
        dataset = config.gdpval_resources_server.resources_servers.gdpval.datasets[0]
    dataset.jsonl_fpath = str(source)
    config.update({"mode": "train_preparation", "output_dirpath": str(output), "should_download": False})
    return config


@pytest.mark.parametrize("order", [("stirrup", "nooa"), ("nooa", "stirrup")])
def test_harness_reuse_preserves_source_and_output_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: tuple[str, str]
) -> None:
    row = {
        "task_id": "fixture",
        "sector": "Finance",
        "occupation": "Auditor",
        "prompt": "Audit {this}.",
        "reference_files": ["reference_files/ledger.xlsx"],
        "reference_file_urls": ["https://example.invalid/private-download-url"],
        "rubric_json": [{"criterion": "private grading criterion", "score": 1}],
        "rubric_pretty": "private grading rubric",
    }
    monkeypatch.setenv("TAVILY_API_KEY", "unused")
    monkeypatch.setenv("GDPVAL_SANDBOX_IMAGE", "fixture:1")
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: [row])
    monkeypatch.setattr("nemo_gym.global_config._find_open_port_using_range", lambda *args, **kwargs: 12345)
    monkeypatch.setattr(prepare, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "data" / "gdpval_benchmark.jsonl")
    source = prepare.prepare()
    original = source.read_bytes()
    source_row = json.loads(original)
    processor = TrainDataProcessor()
    sidecar = source.with_name("gdpval_benchmark_metrics.json")
    sidecar_bytes = None

    for harness in order:
        output = tmp_path / harness
        processor.run(_config(harness, source, output))
        rendered = json.loads((output / "benchmark.jsonl").read_text())
        prompt = json.dumps(rendered["responses_create_params"]["input"])
        assert ("code_exec" in prompt) == (harness == "stirrup")
        assert rendered["responses_create_params"]["tools"] == source_row["responses_create_params"]["tools"]
        assert row["reference_file_urls"][0] not in prompt
        assert row["rubric_pretty"] not in prompt
        assert row["rubric_json"][0]["criterion"] not in prompt
        assert source.read_bytes() == original
        if sidecar_bytes is None:
            sidecar_bytes = sidecar.read_bytes()
        assert sidecar.read_bytes() == sidecar_bytes
        assert json.loads(sidecar_bytes)["Number of examples"] == 1
        metrics = json.loads((output / "benchmark_metrics.json").read_text())
        assert metrics["Number of examples"] == 1
        expected_template = "default-stirrup.yaml" if harness == "stirrup" else "harness-neutral.yaml"
        assert metrics["prompt_config"].endswith(expected_template)

    # The same source is reusable, but an existing rendered output must not be
    # silently relabeled with the other harness's template.
    with pytest.raises(ValueError, match="conflicting aggregate metrics"):
        processor.run(_config(order[1], source, tmp_path / order[0]))
    assert (tmp_path / order[0] / "benchmark_metrics_conflict.json").exists()

    # Changing actual source statistics remains a conflict even with a new output.
    source.write_bytes(original + original)
    with pytest.raises(ValueError, match="conflicting aggregate metrics"):
        processor.run(_config(order[1], source, tmp_path / "changed-source"))
    assert source.with_name("gdpval_benchmark_metrics_conflict.json").exists()

    # Source counting must not bypass the existing rendering compatibility guard.
    incompatible_source = tmp_path / "already-rendered.jsonl"
    source_row["responses_create_params"]["input"] = "An existing prompt"
    incompatible_source.write_text(json.dumps(source_row) + "\n")
    with pytest.raises(ValueError, match="mutually exclusive"):
        processor.run(_config(order[1], incompatible_source, tmp_path / "invalid-rendering"))
