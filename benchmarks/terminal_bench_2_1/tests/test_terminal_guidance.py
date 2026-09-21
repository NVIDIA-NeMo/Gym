# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from benchmarks.terminal_bench_2_1 import prepare_terminal_guidance
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


@pytest.fixture
def source_dataset(tmp_path: Path) -> Path:
    rows = [
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Recover the lost file.\nKeep its café heading.\n"}],
                "max_output_tokens": 4096,
            },
            "task_name": "terminal-bench/example-git",
            "docker_image": "example/task:git",
            "task_folder": "benchmarks/terminal_bench_2_1/tasks/example-git",
            "verifier_metadata": {"expected_file": "recovered.txt"},
        },
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Write the requested module.\n\nDo not change the input data."}]
            },
            "task_name": "terminal-bench/example-module",
            "docker_image": "example/task:module",
            "task_folder": "benchmarks/terminal_bench_2_1/tasks/example-module",
        },
    ]
    source_path = tmp_path / "benchmark.jsonl"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return source_path


def test_guidance_preserves_source_messages_metadata_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_dataset: Path
) -> None:
    original_bytes = source_dataset.read_bytes()
    original = [json.loads(line) for line in original_bytes.splitlines()]
    output_path = tmp_path / "guided" / "benchmark_terminal_guidance.jsonl"
    monkeypatch.setattr(prepare_terminal_guidance, "prepare_original", Mock(return_value=source_dataset))
    monkeypatch.setattr(prepare_terminal_guidance, "OUTPUT_PATH", output_path)

    assert prepare_terminal_guidance.prepare() == output_path

    prepared = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(prepared) == len(original)
    for actual, expected in zip(prepared, original, strict=True):
        assert actual["responses_create_params"]["input"].pop() == {
            "role": "user",
            "content": prepare_terminal_guidance.TERMINAL_INTERACTION_GUIDANCE,
        }
        assert actual == expected
    assert source_dataset.read_bytes() == original_bytes


def test_repeated_preparation_uses_fresh_source_without_accumulating_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_dataset: Path
) -> None:
    output_path = tmp_path / "guided.jsonl"
    prepare_original = Mock(return_value=source_dataset)
    monkeypatch.setattr(prepare_terminal_guidance, "prepare_original", prepare_original)
    monkeypatch.setattr(prepare_terminal_guidance, "OUTPUT_PATH", output_path)

    prepare_terminal_guidance.prepare()
    first_output = output_path.read_bytes()
    prepare_terminal_guidance.prepare()
    assert output_path.read_bytes() == first_output

    rows = [json.loads(line) for line in source_dataset.read_text(encoding="utf-8").splitlines()]
    rows[0]["responses_create_params"]["input"][0]["content"] = "Updated task instruction."
    source_dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prepare_terminal_guidance.prepare()

    latest = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert latest["responses_create_params"]["input"] == [
        {"role": "user", "content": "Updated task instruction."},
        {"role": "user", "content": prepare_terminal_guidance.TERMINAL_INTERACTION_GUIDANCE},
    ]
    assert prepare_original.call_count == 3


def test_guidance_config_changes_only_dataset_selection() -> None:
    configurations = []
    for config_name in ("terminus_2.yaml", "terminus_2_terminal_guidance.yaml"):
        config = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                skip_load_from_cli=True,
                offline=True,
                initial_global_config_dict=OmegaConf.create(
                    {
                        "config_paths": [
                            f"benchmarks/terminal_bench_2_1/{config_name}",
                            "responses_api_models/vllm_model/configs/vllm_model.yaml",
                        ],
                        "policy_base_url": "http://127.0.0.1:1/v1",
                        "policy_api_key": "unused",
                        "policy_model_name": "offline-model",
                    }
                ),
            )
        )
        configurations.append(OmegaConf.to_container(config, resolve=True))
    original, guided = configurations
    original_dataset = original["terminal_bench_2_1_terminus_2_sandboxed_agent"]["responses_api_agents"][
        "terminus_2_sandboxed_agent"
    ].pop("datasets")
    guided_dataset = guided["terminal_bench_2_1_terminus_2_sandboxed_agent"]["responses_api_agents"][
        "terminus_2_sandboxed_agent"
    ].pop("datasets")

    assert len(original_dataset) == len(guided_dataset) == 1
    assert guided_dataset[0]["jsonl_fpath"] == str(prepare_terminal_guidance.OUTPUT_PATH.relative_to(Path.cwd()))
    assert guided_dataset[0]["prepare_script"] == "benchmarks/terminal_bench_2_1/prepare_terminal_guidance.py"
    assert guided_dataset[0]["type"] == original_dataset[0]["type"] == "benchmark"
    assert guided_dataset[0]["num_repeats"] == original_dataset[0]["num_repeats"] == 8
    # The provenance lists differ because the guided config includes the original.
    assert "benchmarks/terminal_bench_2_1/terminus_2_terminal_guidance.yaml" in guided.pop("config_paths")
    assert "benchmarks/terminal_bench_2_1/terminus_2.yaml" in original.pop("config_paths")
    assert guided == original
