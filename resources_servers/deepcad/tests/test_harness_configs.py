# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise stock input boundaries; this does not install or run native clients."""

from importlib import import_module
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollouts import RolloutCollectionHelper
from resources_servers.deepcad.tests.test_opencode_manifest import CONFIG, prepared


@pytest.mark.parametrize(
    "harness,class_name,boundary",
    [
        ("opencode", "OpenCodeAgent", "_session_input"),
        ("pi", "PiAgent", "_validate_request"),
        ("codex", "CodexAgent", "_sandbox_input"),
        ("claude_code", "ClaudeCodeAgent", "_validate_sandbox_request"),
        ("openclaw", "OpenClawAgent", "_sandbox_input"),
    ],
)
@pytest.mark.parametrize("validate_boundary", [False, True])
def test_mixed_config_routes_file_input_to_stock_session_boundary(
    tmp_path, monkeypatch, harness, class_name, boundary, validate_boundary
):
    root = CONFIG.parents[3]
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        str(root / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                        str(CONFIG.with_name("multiharness.yaml")),
                    ],
                    "default_host": "192.0.2.10",
                    "policy_base_url": "http://192.0.2.10:8000/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "validation-only",
                    "anthropic_api_key": "",
                    "error_on_agent_without_environment_server": True,
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    # Agent uploads run as root into private directories created by the image user.
    assert not config.agent_sandbox.docker.create.get("cap_drop", [])
    assert config.sandbox.docker.create.cap_drop == ["ALL"]
    assert config.agent_sandbox.docker.create.security_opt == ["no-new-privileges"]
    _, row = prepared(tmp_path, file_input=True, harness=harness)
    rows = RolloutCollectionHelper().preprocess_examples([row], global_config_dict=config)
    RolloutCollectionHelper._validate_environment_servers(rows, config)
    row = rows[0]
    env_name = f"deepcad_{harness}_environment_server"
    env = SingleAgentTurnLegacyEnvironmentServerConfig(
        name=env_name,
        **OmegaConf.to_container(
            config[env_name].environment_servers.single_agent_turn_legacy,
            resolve=True,
        ),
    )
    assert row["agent_ref"]["name"] == env.agent_server.name
    assert env.resources_server.name == "deepcad_opencode"
    assert env.resources_tool_transports == []
    assert (
        len([key for key, value in config.items() if isinstance(value, DictConfig) and "environment_servers" in value])
        == 5
    )
    from responses_api_models.vllm_model.app import VLLMModelConfig

    model = VLLMModelConfig(
        name="deepcad_policy_model",
        **OmegaConf.to_container(config.deepcad_policy_model.responses_api_models.vllm_model, resolve=True),
    )
    assert model.inline_tool_output_images
    assert env.agent_server.name == f"deepcad_{harness}_agent"
    assert "required_model_servers" not in config.get("model_call_sampling", {})
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    assert len(params.input) == 1
    serialized = params.model_dump(mode="json")
    assert params.input[0].role == "user"
    assert "input_image" not in str(serialized) and "data:image" not in str(serialized)
    if not validate_boundary:
        return
    module = import_module(f"responses_api_agents.{harness}_agent.app")
    values = OmegaConf.to_container(
        config[env.agent_server.name].responses_api_agents[harness + "_agent"], resolve=True
    )
    agent_config = getattr(module, class_name + "Config")(name=env.agent_server.name, **values)
    context_fields = {
        "opencode": "context_window",
        "pi": "context_window",
        "codex": "model_context_window",
        "openclaw": "context_window",
    }
    if harness in context_fields:
        assert getattr(agent_config, context_fields[harness]) == 131072
    if harness in {"opencode", "pi", "openclaw"}:
        assert agent_config.max_output_tokens == 32768
    if harness in {"pi", "openclaw"}:
        assert agent_config.image_input
    if harness == "pi":
        assert not agent_config.auto_compaction
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = config
    agent = getattr(module, class_name)(config=agent_config, server_client=client)
    getattr(agent, boundary)(params)


@pytest.mark.parametrize(
    "other_recipe,other_agent,harness,cad_first",
    [
        ("benchmarks/swebench/verified/opencode.yaml", "swebench_verified_opencode_agent", "opencode", False),
        ("benchmarks/swebench/verified/opencode.yaml", "swebench_verified_opencode_agent", "opencode", True),
    ],
)
def test_cad_blend_does_not_override_other_benchmark_agent(
    tmp_path, monkeypatch, other_recipe, other_agent, harness, cad_first
):
    root = CONFIG.parents[3]

    def resolve(recipes):
        return GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.create(
                    {
                        "config_paths": [
                            str(root / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                            *recipes,
                        ],
                        "default_host": "192.0.2.10",
                        "policy_base_url": "http://192.0.2.10:8000/v1",
                        "policy_api_key": "EMPTY",
                        "policy_model_name": "validation-only",
                        "anthropic_api_key": "",
                        "error_on_agent_without_environment_server": True,
                    }
                ),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )

    other = str(root / other_recipe)
    cad = str(CONFIG.with_name("multiharness.yaml"))
    baseline = resolve([other])
    cad_only = resolve([cad])
    combined = resolve([cad, other] if cad_first else [other, cad])
    # Compare actual resolved benchmark components, not only raw recipe keys.
    before = OmegaConf.to_container(baseline[other_agent], resolve=False)
    after = OmegaConf.to_container(combined[other_agent], resolve=False)
    assert after == before
    assert OmegaConf.to_container(combined.policy_model, resolve=True) == OmegaConf.to_container(
        baseline.policy_model, resolve=True
    )
    stock_model = combined.policy_model.responses_api_models.vllm_model
    cad_model = combined.deepcad_policy_model.responses_api_models.vllm_model
    assert not stock_model.get("inline_tool_output_images", False)
    assert (cad_model.base_url, cad_model.model, cad_model.api_key) == (
        stock_model.base_url,
        stock_model.model,
        stock_model.api_key,
    )
    cad_config = combined[f"deepcad_{harness}_agent"].responses_api_agents[harness + "_agent"]
    # Only the standalone OpenCode recipe binds its own resources server; the others are borrowed through the environment server.
    assert (cad_config.resources_server is None) == (harness != "opencode")
    assert cad_config.model_server.name == "deepcad_policy_model"
    assert cad_config.get("datasets") is None
    # The whole CAD agent block must equal its CAD-only resolve, not just a few keys.
    assert OmegaConf.to_container(combined[f"deepcad_{harness}_agent"], resolve=False) == OmegaConf.to_container(
        cad_only[f"deepcad_{harness}_agent"], resolve=False
    )
    assert (
        combined.deepcad_pi_environment_server.environment_servers.single_agent_turn_legacy.agent_server.name
        == "deepcad_pi_agent"
    )
    for name in ["opencode", "pi", "codex", "claude_code", "openclaw"]:
        env = combined[f"deepcad_{name}_environment_server"].environment_servers.single_agent_turn_legacy
        assert env.agent_server.name == f"deepcad_{name}_agent"
