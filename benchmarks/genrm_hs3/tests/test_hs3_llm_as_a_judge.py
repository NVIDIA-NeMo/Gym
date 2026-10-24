# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from benchmarks.genrm_hs3 import prepare_llm_as_a_judge as module
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


@pytest.mark.parametrize("cached", [True, False])
def test_prepare_preserves_candidates_and_labels(tmp_path, monkeypatch, cached):
    source = tmp_path / "genrm.jsonl"
    row = {
        "id": "test-1",
        "responses_create_params": {
            "input": [{"role": "user", "content": "What is 1+1?"}],
            "metadata": {"response_1": "2", "response_2": "3"},
        },
        "verifier_metadata": {"ground_truth_label": 2, "data_type": "test"},
    }
    original = json.dumps(row) + "\n"
    source.write_text(original)
    monkeypatch.setattr(module, "SOURCE", source if cached else tmp_path / "missing.jsonl")
    download = Mock(return_value=source)
    monkeypatch.setattr(module, "prepare_genrm", download)
    output = module.prepare(tmp_path / "llm.jsonl")
    result = json.loads(output.read_text())
    assert result["verifier_metadata"] == row["verifier_metadata"]
    assert result["id"] == row["id"]
    params = result["responses_create_params"]
    assert "metadata" not in params
    prompt = params["input"][0]["content"]
    assert "User: What is 1+1?" in prompt
    assert "[The Start of Response 1]\n2" in prompt
    assert "[The Start of Response 2]\n3" in prompt
    assert '"ranking": <1-6>' in prompt
    assert '"rubric_evaluations"' in prompt
    assert "Extremely Helpful - Completely aligned with what the user was asking for" in prompt
    assert "You must provide your answer before judging any answers." in prompt
    assert source.read_text() == original
    assert download.call_count == (0 if cached else 1)
    if cached:
        with pytest.raises(ValueError, match="must not overwrite"):
            module.prepare(source)


def test_config_selects_regular_model_and_numeric_scoring():
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": ["benchmarks/genrm_hs3/llm_as_a_judge.yaml"],
                    "policy_base_url": "https://example.com/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "test",
                }
            ),
        )
    )
    assert set(config.policy_model.responses_api_models) == {"vllm_model"}
    assert config.genrm_hs3_resources_server.resources_servers.genrm_eval.scoring_module.endswith("preferences")
    assert config.genrm_hs3_resources_server.resources_servers.genrm_eval.get("verdict_format", "genrm") == "genrm"
    dataset = config.genrm_hs3_benchmark_agent.responses_api_agents.simple_agent.datasets[0]
    assert str(dataset.jsonl_fpath).endswith("hs3_llm_as_a_judge_validation.jsonl")
