# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from importlib import import_module
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from resources_servers.genrm_eval.llm_as_a_judge import prepare_llm_input


@pytest.mark.parametrize("benchmark", ["genrm_hs4", "genrm_judgebench", "genrm_rmbench"])
@pytest.mark.parametrize("cached", [False, True])
def test_prepare_and_config(tmp_path, monkeypatch, benchmark, cached):
    module = import_module(f"benchmarks.{benchmark}.prepare_llm_as_a_judge")
    source = tmp_path / "genrm.jsonl"
    rows = [
        {
            "id": f"pair-{i}",
            "responses_create_params": {
                "input": [{"role": "user", "content": "Question?"}],
                "metadata": {"response_1": a, "response_2": b},
            },
            "verifier_metadata": {"label": label, "style": i},
        }
        for i, (a, b, label) in enumerate([("first", "second", "A>B"), ("second", "first", "B>A")])
    ]
    original = "".join(json.dumps(row) + "\n" for row in rows)
    source.write_text(original)
    monkeypatch.setattr(module, "SOURCE", source if cached else tmp_path / "missing.jsonl")
    download = Mock(return_value=source)
    monkeypatch.setattr(module, "prepare_genrm", download)
    output = module.prepare(tmp_path / "llm.jsonl")
    result = [json.loads(line) for line in output.read_text().splitlines()]
    for before, after in zip(rows, result, strict=True):
        assert before["id"] == after["id"]
        assert before["verifier_metadata"] == after["verifier_metadata"]
        params = after["responses_create_params"]
        assert "metadata" not in params
        text = params["input"][0]["content"]
        assert '"rubric_evaluations"' in text
        for number in (1, 2):
            assert (
                f"[The Start of Response {number}]\n{before['responses_create_params']['metadata'][f'response_{number}']}"
                in text
            )
    assert source.read_text() == original
    assert download.call_count == (0 if cached else 1)

    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [f"benchmarks/{benchmark}/llm_as_a_judge.yaml"],
                    "policy_base_url": "https://example.com/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "test",
                }
            ),
        )
    )
    assert set(config.policy_model.responses_api_models) == {"vllm_model"}
    settings = config[f"{benchmark}_resources_server"].resources_servers.genrm_eval
    expected = (
        "resources_servers.genrm_eval.preferences" if benchmark == "genrm_hs4" else f"benchmarks.{benchmark}.metrics"
    )
    assert settings.scoring_module == expected
    dataset = config[f"{benchmark}_benchmark_agent"].responses_api_agents.simple_agent.datasets[0]
    assert str(dataset.jsonl_fpath).endswith("_llm_as_a_judge_validation.jsonl")
    assert config.responses_create_params.temperature == 1.0
    assert config.responses_create_params.top_p == 0.95
    assert config.responses_create_params.max_output_tokens == 16384


def test_reject_overwrite_and_bad_input(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("original")
    with pytest.raises(ValueError, match="must not overwrite"):
        prepare_llm_input(source=source, output=source)
    assert source.read_text() == "original"
    source.write_text(json.dumps({"responses_create_params": {"input": []}}))
    output = tmp_path / "output.jsonl"
    with pytest.raises(ValueError, match="text conversation"):
        prepare_llm_input(source=source, output=output)
    assert not output.exists()
