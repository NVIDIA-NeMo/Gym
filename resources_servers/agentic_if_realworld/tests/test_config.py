# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from omegaconf import OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig


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
                {"config_paths": ["benchmarks/agentic_if_realworld/raw_receipt.yaml"]}
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


@pytest.mark.parametrize(
    "profile,model_type,base_field,model_field",
    [
        ("config.yaml", "vllm_model", "base_url", "model"),
        ("native.yaml", "vllm_model", "base_url", "model"),
        ("native.yaml", "openai_model", "openai_base_url", "openai_model"),
    ],
)
def test_profiles_use_stock_model_settings(
    generic_model_env, monkeypatch, profile, model_type, base_field, model_field
):
    monkeypatch.setenv("AGENTIC_IF_POLICY_ENDPOINT", "https://benchmark-provider.invalid/v1/chat/completions")
    monkeypatch.setenv("AGENTIC_IF_POLICY_MODEL", "benchmark-model")
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        f"benchmarks/agentic_if_realworld/{profile}",
                        f"responses_api_models/{model_type}/configs/{model_type}.yaml",
                    ]
                }
            ),
            skip_load_from_cli=True,
            offline=True,
        )
    )
    assert resolved.config_regression_sentinel == "loaded-from-env-yaml"
    adapter = resolved.policy_model.responses_api_models[model_type]
    assert adapter[base_field] == "https://generic-provider.invalid/v1"
    assert adapter[model_field] == "generic-model"
    assert "agentic_if_text" not in resolved.policy_model.responses_api_models
