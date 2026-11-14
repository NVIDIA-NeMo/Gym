# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


ROOT = Path(__file__).resolve().parents[3]
CATALOG = "benchmarks/opencode_multimodal_benchmarks"


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench"])
@pytest.mark.parametrize("mode", ["default", "openai", "vllm", "training"])
def test_catalog_pairings_validate_with_selected_model(benchmark, mode, monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-only")
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "https://sandbox.invalid")
    adapter = "openai_model" if mode == "openai" else "vllm_model"
    model_config = "vllm_model_for_training" if mode == "training" else adapter
    paths = [f"responses_api_models/{adapter}/configs/{model_config}.yaml", f"{CATALOG}/{benchmark}/config.yaml"]
    overrides = {
        "visual_agent": {"responses_api_agents": {"opencode_agent": {"sandbox_config": {"image": "test-image"}}}}
    }
    if mode in {"openai", "vllm"}:
        paths[-1] = f"{CATALOG}/{benchmark}/opencode.yaml"
    if mode == "vllm":
        paths.append(f"{CATALOG}/opencode_eval_vllm.yaml")
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [str(ROOT / p) for p in paths],
                    "policy_model_name": "test-model",
                    "policy_base_url": "http://127.0.0.1:9999/v1",
                    "policy_api_key": "test-only",
                    **overrides,
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            skip_exporters=True,
            offline=True,
        )
    )
    assert "policy_model" in {s.name for s in parser.filter_for_server_instance_configs(config)}
    assert set(config.policy_model.responses_api_models) == {adapter}
    env = SingleAgentTurnEnvironmentServerConfig.model_validate(
        OmegaConf.to_container(config.visual_environment.environment_servers.single_agent_turn, resolve=True)
        | {"name": "visual_environment"}
    )
    assert env.cleanup_timeout_seconds == 180
    assert env.default_episode_timeout_seconds is None
    assert env.resources_tool_transports == []
    assert env.agent_server.name == "visual_agent"
    assert env.resources_server.name == "opencode_multimodal_benchmarks"
    agent = config.visual_agent.responses_api_agents
    assert set(agent) == {"opencode_agent"}
    assert agent.opencode_agent.model_server.name == "policy_model"
    assert agent.opencode_agent.network_access == "model_only"
    assert agent.opencode_agent.image_input
    assert not agent.opencode_agent.token_id_capture
    assert agent.opencode_agent.opencode_config.permission.webfetch == "deny"
    assert agent.opencode_agent.opencode_config.permission.websearch == "deny"
    manifest = OmegaConf.load(ROOT / CATALOG / benchmark / "manifest.yaml")
    assert manifest.agent_server == "opencode_agent"
    resources = config.opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks
    assert resources.datasets[0].num_repeats == (5 if benchmark == "zerobench" else 1)
    if mode == "vllm":
        model = config.policy_model.responses_api_models.vllm_model
        assert not agent.opencode_agent.token_id_capture
        assert not model.return_token_id_information
        assert model.uses_reasoning_parser and model.uses_interleaved_reasoning
        assert model.chat_template_kwargs.truncate_history_thinking is False
        assert model.sampling_overrides.max_tokens is None
        assert model.sampling_overrides.max_completion_tokens is None
        assert model.extra_body is None and model.propagate_context_overflow_errors
    if mode == "training":
        assert config.policy_model.responses_api_models.vllm_model.return_token_id_information
