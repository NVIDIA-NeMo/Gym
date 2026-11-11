# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline download and resolved-configuration coverage for frozen Arena subsets."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from benchmarks.lmarena_v3 import prepare as preparation
from nemo_gym.cli.eval import _datasets_to_prepare
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig


@pytest.mark.parametrize("prompt_set", ["full", "super35", "nano35"])
def test_prepare_selects_pinned_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_set: str):
    monkeypatch.setattr(preparation, "DATA_DIR", tmp_path)
    monkeypatch.setattr(preparation, "OUTPUT_FPATH", tmp_path / "lmarena_v3_validation.jsonl")
    download = Mock()
    monkeypatch.setattr(preparation, "download_jsonl_dataset", download)
    result = preparation.prepare() if prompt_set == "full" else preparation.prepare(prompt_set=prompt_set)
    dataset = "lmarena_v3" if prompt_set == "full" else f"lmarena_v3_{prompt_set}"
    config = download.call_args.args[0]
    assert config.dataset_name == dataset
    assert config.version == "0.0.1"
    assert config.artifact_fpath == f"{dataset}_validation.jsonl"
    assert result == tmp_path / config.artifact_fpath
    assert config.output_fpath == str(result)


def test_rejects_unknown_prompt_set(monkeypatch: pytest.MonkeyPatch):
    download = Mock()
    monkeypatch.setattr(preparation, "download_jsonl_dataset", download)
    with pytest.raises(ValueError, match="prompt_set"):
        preparation.prepare(prompt_set="unknown")
    download.assert_not_called()


@pytest.mark.parametrize("prompt_set", ["super35", "nano35"])
@pytest.mark.parametrize("glm", [False, True])
def test_subset_config_resolves(prompt_set: str, glm: bool):
    suffix = "_glm53_judge" if glm else "_gemini38flash_judge"
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [f"benchmarks/lmarena_v3/config_{prompt_set}{suffix}.yaml"],
                    "policy_base_url": "https://example.com/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "policy",
                }
            ),
        )
    )
    judge = resolved.judge_model.responses_api_models.vllm_model
    assert judge.model == ("nvidia/zai-org/glm-5.3" if glm else "gcp/google/gemini-3.8-flash")
    assert judge.uses_reasoning_parser is True
    assert resolved.prepare_script_args.prompt_set == prompt_set
    datasets = _datasets_to_prepare(resolved)
    assert len(datasets) == 1
    dataset = datasets[0].dataset
    assert dataset.name == f"lmarena_v3_{prompt_set}"
    assert str(dataset.jsonl_fpath) == f"benchmarks/lmarena_v3/data/lmarena_v3_{prompt_set}_validation.jsonl"
    arena = resolved.lmarena_v3_benchmark_resources_server.resources_servers.arena
    assert arena.style_control_method == "reference_length"
    assert arena.judge_concurrency == 32
    assert arena.judge_responses_create_params.temperature == 0.0
    assert arena.judge_responses_create_params.max_output_tokens == 32768
    assert resolved.responses_create_params.temperature == 1.0
    assert resolved.responses_create_params.top_p == 0.95
    assert resolved.responses_create_params.max_output_tokens == 32768
