# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from resources_servers.genrm_eval.llm_as_a_judge import prepare_llm_input


@pytest.mark.parametrize("benchmark", ["genrm_hs4", "genrm_judgebench", "genrm_rmbench", "genrm_rewardbench2"])
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
    assert config.responses_create_params.max_output_tokens == 32768


def test_text_blocks_match_string_conversations(tmp_path: Path) -> None:
    strings = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer.\nMore detail."},
        {"role": "user", "content": "Follow-up?"},
    ]
    blocks = [
        {"role": "system", "content": [{"type": "input_text", "text": "Be helpful."}]},
        {"role": "user", "content": [{"type": "input_text", "text": "First question?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "First answer.\n"},
                {"type": "output_text", "text": "More detail."},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Follow-"}, {"type": "input_text", "text": "up?"}],
        },
    ]
    prompts = []
    for name, turns in (("strings", strings), ("blocks", blocks)):
        source = tmp_path / f"{name}.jsonl"
        row = {
            "id": "pair-1",
            "responses_create_params": {
                "input": turns,
                "metadata": {"response_1": "Answer A", "response_2": "Answer B"},
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 1234,
            },
            "verifier_metadata": {"pair_id": "pair-1", "label": "B>A", "flipped": True},
        }
        original = json.dumps(row) + "\n"
        source.write_text(original)
        output = prepare_llm_input(source=source, output=tmp_path / f"{name}-llm.jsonl")
        result = json.loads(output.read_text())
        assert result["id"] == row["id"]
        assert result["verifier_metadata"] == row["verifier_metadata"]
        messages = result["responses_create_params"].pop("input")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert result["responses_create_params"] == {"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 1234}
        prompts.append(messages[0]["content"])
        assert source.read_text() == original
    assert prompts[0] == prompts[1]
    assert (
        "System: Be helpful.\nUser: First question?\nAssistant: First answer.\nMore detail.\nUser: Follow-up?"
        in prompts[1]
    )
    assert "[The Start of Response 1]\nAnswer A\n[The End of Response 1]" in prompts[1]
    assert "[The Start of Response 2]\nAnswer B\n[The End of Response 2]" in prompts[1]


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "input_image", "image_url": "https://example.com/image.png"}],
        [{"type": "input_text", "text": "Question?"}, {"type": "input_image", "image_url": "example.png"}],
        [{"type": "input_text", "text": 123}],
        [{"type": "input_audio"}],
    ],
)
def test_reject_non_text_preserves_destination(tmp_path: Path, content: object) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "llm.jsonl"
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Valid first row"}],
            "metadata": {"response_1": "A", "response_2": "B"},
        },
    }
    original = json.dumps(row) + "\n"
    row["responses_create_params"]["input"][0]["content"] = content
    original += json.dumps(row) + "\n"
    source.write_text(original)
    output.write_text("existing dataset")
    with pytest.raises(ValueError, match="LLM preparation requires text conversation messages"):
        prepare_llm_input(source=source, output=output)
    assert output.read_text() == "existing dataset"
    assert source.read_text() == original


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
