# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.ruler.prepare_utils import TEXT_COMPLETION_OUTPUT_TOKENS, _to_gym_sample


REPO_ROOT = Path(__file__).parents[2]


@pytest.fixture
def source_sample() -> dict:
    return {
        "input": "Question?",
        "answer_prefix": "  Answer:",
        "outputs": ["expected"],
        "length": 123,
    }


def test_chat_sample_is_unchanged(source_sample: dict) -> None:
    sample = _to_gym_sample(source_sample, "qa_1", "chat")

    assert sample["responses_create_params"] == {
        "input": [{"role": "user", "content": "Question?"}],
    }
    assert sample["outputs"] == ["expected"]


@pytest.mark.parametrize(
    ("data_format", "expected_content"),
    [("default", "Question?Answer:"), ("base", "Question?\nAnswer:")],
)
def test_text_completion_appends_answer_prefix(source_sample: dict, data_format: str, expected_content: str) -> None:
    sample = _to_gym_sample(source_sample, "qa_1", data_format)

    assert sample["responses_create_params"] == {
        "input": [{"role": "user", "content": expected_content}],
        "max_output_tokens": 32,
    }


@pytest.mark.parametrize(
    ("subset", "expected_tokens"),
    [
        ("niah_single_1", 128),
        ("vt", 30),
        ("cwe", 120),
        ("fwe", 50),
        ("qa_1", 32),
    ],
)
def test_text_completion_uses_ruler_task_limits(source_sample: dict, subset: str, expected_tokens: int) -> None:
    sample = _to_gym_sample(source_sample, subset, "default")

    assert sample["responses_create_params"]["max_output_tokens"] == expected_tokens
    assert TEXT_COMPLETION_OUTPUT_TOKENS[subset.split("_", maxsplit=1)[0]] == expected_tokens


def test_pretrain_profile_uses_raw_completions() -> None:
    config = OmegaConf.load(REPO_ROOT / "responses_api_models/local_vllm_model/configs/pretrain_text_completion.yaml")
    model_config = config.policy_model.responses_api_models.local_vllm_model

    assert model_config.use_completions_api is True
    assert model_config.render_chat_template is False
    assert model_config.chat_template_kwargs is None
    assert "chat_template" not in model_config.vllm_serve_kwargs
