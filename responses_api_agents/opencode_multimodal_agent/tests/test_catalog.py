# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from responses_api_agents.opencode_multimodal_agent.app import OpenCodeMultimodalAgentConfig
from responses_api_agents.opencode_multimodal_agent.eval import OpenCodeMultimodalEvalConfig


ROOT = Path(__file__).resolve().parents[3]


def parse(monkeypatch, paths, overrides):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-placeholder")
    initial = OmegaConf.create(
        {
            "config_paths": paths,
            "policy_model_name": "vision",
            "policy_base_url": "http://model.invalid/v1",
            "policy_api_key": "none",
        }
        | overrides
    )
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench"])
def test_catalog_uses_one_owned_session_route(monkeypatch, benchmark):
    cfg = parse(
        monkeypatch,
        [
            f"benchmarks/opencode_multimodal_benchmarks/{benchmark}/opencode.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        {
            "visual_agent": {
                "responses_api_agents": {"opencode_multimodal_agent": {"sandbox_config": {"image": "test-image"}}}
            },
            "opencode_multimodal_benchmarks": {
                "resources_servers": {"opencode_multimodal_benchmarks": {"judge_model": "test-judge"}}
            },
        },
    )
    assert "opencode_agent" not in cfg
    assert "visual_opencode" not in cfg
    assert set(cfg.visual_agent.responses_api_agents) == {"opencode_multimodal_agent"}
    settings = cfg.visual_agent.responses_api_agents.opencode_multimodal_agent
    agent = OpenCodeMultimodalEvalConfig.model_validate(
        OmegaConf.to_container(settings, resolve=True) | {"name": "visual_agent"}
    )
    assert agent.command == "opencode" and agent.sandbox_provider == "sandbox"
    assert agent.system_prompt_mode == "replace"
    assert agent.final_answer_reserve is None and agent.tool_rewards is None
    assert agent.turn_budget.hard_limit == 65
    assert cfg.rollout_correlation
    assert cfg.environment_server_routes.visual_tasks == "visual_environment"
    environment = SingleAgentTurnEnvironmentServerConfig.model_validate(
        OmegaConf.to_container(cfg.visual_environment.environment_servers.single_agent_turn, resolve=True)
        | {"name": "visual_environment"}
    )
    assert environment.entrypoint == "opencode.py"
    assert environment.agent_server.name == "visual_agent"
    assert environment.resources_server.name == "opencode_multimodal_benchmarks"
    assert all(
        "legacy_agent" not in node.get("environment_servers", {}) for node in cfg.values() if OmegaConf.is_dict(node)
    )


def test_training_defaults_are_session_only(monkeypatch):
    cfg = parse(
        monkeypatch,
        [
            "responses_api_agents/opencode_multimodal_agent/configs/opencode_multimodal_agent.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        {
            "training_resources_server": "string_match",
            "opencode_multimodal_agent": {
                "responses_api_agents": {
                    "opencode_multimodal_agent": {
                        "sandbox_config": {"image": "test-image"},
                        "env": {"PATH": "/old/host/tools"},
                    }
                }
            },
        },
    )
    settings = cfg.opencode_multimodal_agent.responses_api_agents.opencode_multimodal_agent
    agent = OpenCodeMultimodalAgentConfig.model_validate(
        OmegaConf.to_container(settings, resolve=True) | {"name": "opencode_multimodal_agent"}
    )
    assert agent.env == {} and agent.sandbox_provider == "sandbox" and agent.token_id_capture
    assert cfg.token_id_capture.enabled
    assert set(cfg.opencode_multimodal_agent_environment_server.environment_servers) == {"single_agent_turn"}
    assert (
        cfg.opencode_multimodal_agent_environment_server.environment_servers.single_agent_turn.entrypoint
        == "opencode.py"
    )


@pytest.mark.parametrize(
    "agent_name,resource",
    [
        ("opencode_multimodal_agent", "math_with_judge"),
        ("opencode_string_match", "string_match"),
        ("opencode_mcqa", "mcqa"),
    ],
)
def test_training_verifier_profiles_resolve_independent_bindings(monkeypatch, agent_name, resource):
    cfg = parse(
        monkeypatch,
        [
            "responses_api_agents/opencode_multimodal_agent/configs/train_verifiers.yaml",
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
        ],
        {
            "opencode_multimodal_agent": {
                "responses_api_agents": {"opencode_multimodal_agent": {"sandbox_config": {"image": "test-image"}}}
            }
        },
    )
    settings = cfg[agent_name + "_environment_server"].environment_servers.single_agent_turn
    env = SingleAgentTurnEnvironmentServerConfig.model_validate(
        OmegaConf.to_container(settings, resolve=True) | {"name": agent_name + "_environment_server"}
    )
    assert env.agent_server.name == agent_name
    assert env.resources_server.name == resource
    agent = OpenCodeMultimodalAgentConfig.model_validate(
        OmegaConf.to_container(cfg[agent_name].responses_api_agents.opencode_multimodal_agent, resolve=True)
        | {"name": agent_name}
    )
    assert agent.sandbox_config["image"] == "test-image"
    assert agent.token_id_capture


def test_training_recipe_requires_explicit_verifier(monkeypatch):
    from omegaconf.errors import InterpolationToMissingValueError, MissingMandatoryValue

    with pytest.raises((MissingMandatoryValue, InterpolationToMissingValueError)):
        cfg = parse(
            monkeypatch,
            [
                "responses_api_agents/opencode_multimodal_agent/configs/opencode_multimodal_agent.yaml",
                "responses_api_models/vllm_model/configs/vllm_model.yaml",
            ],
            {
                "opencode_multimodal_agent": {
                    "responses_api_agents": {"opencode_multimodal_agent": {"sandbox_config": {"image": "test-image"}}}
                }
            },
        )
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
