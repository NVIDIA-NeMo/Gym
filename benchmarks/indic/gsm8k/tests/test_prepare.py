# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.indic.gsm8k import prepare as prepare_module


def _rows(language_code: str) -> list[dict]:
    return [
        {
            "question": f"  {language_code} question {index}?  ",
            "answer": f"work {index} #### {index}",
            "judge_pass_stage": "synthetic",
        }
        for index in range(8)
    ]


def test_render_prompt_matches_reference_sampling_and_format() -> None:
    rows = _rows("hi")
    rng = random.Random(prepare_module.FEWSHOT_SEED)
    expected_rng = random.Random(prepare_module.FEWSHOT_SEED)
    sampled_indices = expected_rng.sample(range(len(rows)), prepare_module.NUM_FEWSHOT + 1)
    demo_indices = [index for index in sampled_indices if index != 0][: prepare_module.NUM_FEWSHOT]

    prompt = prepare_module.render_prompt(rows, 0, rng)
    expected = "".join(
        f"Question: hi question {index}?\nAnswer: work {index} #### {index}\n\n" for index in demo_indices
    )
    expected += "Question: hi question 0?\nAnswer:"

    assert prompt == expected
    assert "\\n" not in prompt


def test_prepare_pins_source_and_resets_rng_per_language(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_load_dataset(source_id, language_code, *, split, revision, token):
        calls.append((source_id, language_code, split, revision, token))
        return _rows(language_code)

    monkeypatch.setattr(prepare_module, "load_dataset", fake_load_dataset)
    output = tmp_path / "prepared.jsonl"
    prepare_module.prepare(["hi", "ta"], output)

    prepared = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(prepared) == 16
    assert calls == [
        (
            prepare_module.SOURCE_ID,
            "hi",
            prepare_module.SOURCE_SPLIT,
            prepare_module.SOURCE_REVISION,
            True,
        ),
        (
            prepare_module.SOURCE_ID,
            "ta",
            prepare_module.SOURCE_SPLIT,
            prepare_module.SOURCE_REVISION,
            True,
        ),
    ]
    assert prepared[0]["prompt"].replace("hi", "ta") == prepared[8]["prompt"]
    assert prepared[0]["responses_create_params"] == {
        "max_output_tokens": 4096,
        "temperature": 0.0,
    }
    assert prepared[0]["subset_for_metrics"] == "hi"
    assert prepared[8]["subset_for_metrics"] == "ta"


def test_config_disables_chat_template_and_pins_generation() -> None:
    repo_root = Path(__file__).parents[4]
    benchmark_config = OmegaConf.load(repo_root / "benchmarks/indic/gsm8k/config.yaml")
    resolved = OmegaConf.create()
    for config_path in benchmark_config.pop("config_paths"):
        resolved = OmegaConf.merge(resolved, OmegaConf.load(repo_root / config_path))
    resolved = OmegaConf.merge(resolved, benchmark_config)

    model = resolved.policy_model.responses_api_models.vllm_model
    assert model.use_completions_api is True
    assert model.get("render_chat_template", False) is False
    assert model.sampling_overrides.temperature == 0.0
    assert list(model.sampling_overrides.stop) == ["Question:", "</s>", "<|im_end|>"]
