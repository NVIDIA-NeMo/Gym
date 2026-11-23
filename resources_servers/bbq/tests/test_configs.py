# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_main_config_is_prototype_only() -> None:
    config = yaml.safe_load((ROOT / "configs/bbq.yaml").read_text(encoding="utf-8"))
    server = config["bbq"]["resources_servers"]["bbq"]
    assert server["verified"] is False
    assert server["judge_model_server"]["name"] == "bbq_judge_model"
    assert "grpo" not in str(config).lower()
    assert "training" not in config


def test_judge_config_is_environment_backed_and_contains_no_secret() -> None:
    text = (ROOT / "configs/bbq_judge_model.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    judge = config["bbq_judge_model"]["responses_api_models"]["vllm_model"]
    assert "BBQ_JUDGE_BASE_URL" in judge["base_url"]
    assert "BBQ_JUDGE_API_KEY" in judge["api_key"]
    assert "BBQ_JUDGE_MODEL" in judge["model"]
    assert judge["base_url"] == "${oc.env:BBQ_JUDGE_BASE_URL}"
    assert judge["api_key"] == "${oc.env:BBQ_JUDGE_API_KEY}"
    # Model is env-overridable but defaults to the calibrated judge.
    assert judge["model"] == "${oc.env:BBQ_JUDGE_MODEL,nvidia/google/gemma-4-31b-it}"
    assert "<api_key>" not in text
    assert "http://<host_url>" not in text


def test_prompt_contract_uses_existing_plural_metadata() -> None:
    text = (ROOT / "configs/verifier_prompt_templates.yaml").read_text(encoding="utf-8")
    prompts = yaml.safe_load(text)
    assert set(prompts) == {"answer_check", "explanation_quality_check"}
    assert "{{label}}" in text
    assert "{{actor_aliases_json}}" in text
    assert "{{protected_axis}}" in text
    assert "expected_answer" not in text
    assert "stereotyped_actor_id" not in text
    assert "actor_registry" not in text


def test_no_git_dependency_or_hash_pin() -> None:
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    assert not any("git+" in requirement or "917c3d" in requirement for requirement in dependencies)


def test_requirements_do_not_pin_nemo_gym() -> None:
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "nemo-gym[dev]" in metadata["project"]["dependencies"]
    assert metadata["tool"]["uv"]["sources"]["nemo-gym"] == {"path": "../..", "editable": True}
