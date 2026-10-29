# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from omegaconf import OmegaConf

from benchmarks.indic.gsm8k.prepare_posttrain import OUTPUT, build_rows


def source_row():
    return {
        "id": "one",
        "question": "English question",
        "answer": "Original work\n#### 18",
        "question_Hindi_translation": "हिन्दी प्रश्न",
        "answer_Hindi_translation": "bad translation #### 99",
        "question_Tamil_translation": "தமிழ் கேள்வி",
    }


def test_posttrain_uses_translations_and_original_answer_without_demos():
    rows = build_rows([source_row()], languages=["hi", "ta"], source_sha256="abc")
    assert [r["question"] for r in rows] == ["हिन्दी प्रश्न", "தமிழ் கேள்வி"]
    assert all(r["expected_answer"] == "Original work\n#### 18" for r in rows)
    assert all(r["responses_create_params"] == {"max_output_tokens": 120000} for r in rows)
    assert rows[0]["uuid"] != rows[1]["uuid"]
    assert "prompt" not in rows[0]


@pytest.mark.parametrize("change", [{"question_Hindi_translation": ""}, {"answer": "no final"}, {"id": None}])
def test_posttrain_rejects_invalid_data(change):
    with pytest.raises(ValueError):
        build_rows([{**source_row(), **change}], languages=["hi"], source_sha256="abc")


def test_posttrain_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        build_rows([source_row(), source_row()], languages=["hi"], source_sha256="abc")


def test_posttrain_chat_config_has_no_base_model_overrides():
    config = OmegaConf.load(OUTPUT.parent.parent / "config_posttrain.yaml")
    model = config.policy_model.responses_api_models.vllm_model
    assert model.use_completions_api is False
    assert model.chat_template_kwargs.enable_thinking is True
    assert not model.get("sampling_overrides")
