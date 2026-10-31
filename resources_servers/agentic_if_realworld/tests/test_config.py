# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


@pytest.fixture
def generic_model_env(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS", raising=False)
    monkeypatch.delenv("AGENTIC_IF_POLICY_ENDPOINT", raising=False)
    monkeypatch.delenv("AGENTIC_IF_POLICY_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "env.yaml").write_text(
        "policy_base_url: https://generic-provider.invalid/v1\n"
        "policy_model_name: generic-model\n"
        "policy_api_key: generic-test-key\n"
        "config_regression_sentinel: loaded-from-env-yaml\n"
    )


@pytest.mark.parametrize("scoped_env", [False, True])
def test_adaptive_policy_ignores_generic_model_settings(generic_model_env, monkeypatch, scoped_env):
    endpoint, model = "http://localhost:8000/v1/chat/completions", "policy"
    if scoped_env:
        endpoint, model = "https://benchmark-provider.invalid/v1/chat/completions", "benchmark-model"
        monkeypatch.setenv("AGENTIC_IF_POLICY_ENDPOINT", endpoint)
        monkeypatch.setenv("AGENTIC_IF_POLICY_MODEL", model)
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {"config_paths": ["benchmarks/agentic_if_realworld/config.yaml"]}
            ),
            skip_load_from_cli=True,
            offline=True,
        )
    )
    assert resolved.config_regression_sentinel == "loaded-from-env-yaml"
    assert resolved.policy_base_url == "https://generic-provider.invalid/v1"
    assert resolved.policy_model_name == "generic-model"
    adapter = resolved.policy_model.responses_api_models.agentic_if_text
    assert adapter.endpoint == endpoint
    assert adapter.model == model


def test_native_profile_uses_stock_model_settings(generic_model_env, monkeypatch):
    monkeypatch.setenv("AGENTIC_IF_POLICY_ENDPOINT", "https://benchmark-provider.invalid/v1/chat/completions")
    monkeypatch.setenv("AGENTIC_IF_POLICY_MODEL", "benchmark-model")
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "benchmarks/agentic_if_realworld/native.yaml",
                        "responses_api_models/openai_model/configs/openai_model.yaml",
                    ]
                }
            ),
            skip_load_from_cli=True,
            offline=True,
        )
    )
    assert resolved.config_regression_sentinel == "loaded-from-env-yaml"
    adapter = resolved.policy_model.responses_api_models.openai_model
    assert adapter.openai_base_url == "https://generic-provider.invalid/v1"
    assert adapter.openai_model == "generic-model"
    assert "agentic_if_text" not in resolved.policy_model.responses_api_models
