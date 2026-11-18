# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from benchmarks.agentic_if_realworld import rollout_driver
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
    if profile == "config.yaml":
        assert adapter.max_num_tries == adapter.max_http_attempts == 1
        assert adapter.refit_output_budget_to_window is adapter.refit_output_budget_to_prompt is False
        assert adapter.sampling_overrides is None
        assert resolved.agentic_if_realworld_agent.responses_api_agents.simple_agent.max_model_retries == 0
        environment = resolved.agentic_if_realworld_environment.environment_servers.single_agent_turn_legacy
        assert environment.skip_verification is True
        assert not environment.resources_tool_transports and not environment.tool_servers


@pytest.mark.parametrize(
    "target,override",
    [
        ("model", {"max_num_tries": None}),
        ("model", {"refit_output_budget_to_window": True}),
        ("model", {"refit_output_budget_to_prompt": True}),
        ("model", {"sampling_overrides": {"max_tokens": 4}}),
        ("agent", {"max_model_retries": 1}),
    ],
)
def test_adaptive_rejects_hidden_retries_and_cap_changes(monkeypatch, target, override):
    monkeypatch.setenv("SYNTHETIC_JUDGE_KEY", "synthetic")
    model = {"max_num_tries": 1, "refit_output_budget_to_window": False}
    agent = {"max_model_retries": 0}
    (model if target == "model" else agent).update(override)
    settings = {
        "policy_model": {"responses_api_models": {"vllm_model": model}},
        "agentic_if_realworld_agent": {"responses_api_agents": {"simple_agent": agent}},
        "agentic_if_realworld_resources": {
            "resources_servers": {
                "agentic_if_realworld": {
                    "judge_endpoint": "http://synthetic.invalid/judge",
                    "judge_model": "synthetic-judge",
                    "judge_api_key_env": "SYNTHETIC_JUDGE_KEY",
                    "response_source": "gym_response",
                }
            }
        },
    }
    config = SimpleNamespace(num_repeats=1, fan_out=None, responses_create_params=None)
    with pytest.raises(ValueError, match="single-attempt policy calls"):
        asyncio.run(rollout_driver.run_rollout_collection(config, settings))
