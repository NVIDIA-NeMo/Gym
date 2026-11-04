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


@pytest.mark.parametrize("already_prepared", [True, False])
def test_preparation_reuses_locked_worker_or_shared_wheel_setup(tmp_path, monkeypatch, already_prepared):
    import json
    import os
    import subprocess
    import sys
    import venv
    from unittest.mock import Mock

    from benchmarks.ruler import prepare_utils
    from nemo_gym.cli import _locked_env

    benchmark = tmp_path / "site packages/benchmarks/ruler"
    benchmark.mkdir(parents=True)
    data = tmp_path / "data"
    isolated = tmp_path / "writable/preparation/.venv" if already_prepared else benchmark / ".venv"
    venv.EnvBuilder(with_pip=False).create(isolated)
    monkeypatch.setattr(prepare_utils, "BENCHMARK_DIR", benchmark)
    monkeypatch.setattr(prepare_utils, "DATA_DIR", data)
    monkeypatch.setattr(prepare_utils, "get_hf_token", lambda: None)
    if already_prepared:
        monkeypatch.setenv("NEMO_GYM_PREPARE_VENV", str(isolated))
    else:
        monkeypatch.delenv("NEMO_GYM_PREPARE_VENV", raising=False)
    sync = Mock()
    monkeypatch.setattr(_locked_env, "sync_environment", sync)
    generation_calls = []

    def generate(command, *, check, cwd, env):
        generation_calls.append(command)
        assert command[0] == str(isolated / "bin/python")
        assert check is True
        assert cwd == benchmark
        child = subprocess.check_output(["python", "-c", "import sys; print(sys.prefix)"], env=env, text=True)
        assert child.strip() == str(isolated)
        assert env["VIRTUAL_ENV"] == str(isolated)
        temporary = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--tmp_data_dir=")))
        subset = temporary / "ruler_data/qa_1"
        subset.mkdir(parents=True)
        (subset / "test.jsonl").write_text(
            json.dumps({"input": "question", "answer_prefix": "Answer:", "outputs": ["answer"], "length": 10}) + "\n"
        )

    monkeypatch.setattr(prepare_utils, "run", generate)
    parent_path = os.environ["PATH"]
    output = prepare_utils.prepare_helper("ruler.jsonl", "local/model", 1024)
    assert len(generation_calls) == 1
    if already_prepared:
        sync.assert_not_called()
    else:
        sync.assert_called_once_with(benchmark, isolated, sys.executable, [], [])
    row = json.loads(output.read_text())
    assert row["outputs"] == ["answer"]
    assert row["responses_create_params"]["input"][-1]["content"] == "Answer:"
    assert os.environ["PATH"] == parent_path
