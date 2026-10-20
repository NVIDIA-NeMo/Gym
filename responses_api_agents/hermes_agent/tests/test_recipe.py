# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import _environment_server_for_agent, _environment_servers_by_agent
from responses_api_agents.hermes_agent.app import HermesAgentConfig


@pytest.mark.parametrize("model_override", [None, "other-served-model"])
def test_readme_composes_independent_harness_and_benchmark(model_override: str | None) -> None:
    root = Path(__file__).parents[3]
    readme = (root / "responses_api_agents/hermes_agent/README.md").read_text()
    composition = re.search(r"```yaml\n(config_paths:.*?)\n```", readme, re.DOTALL)
    assert composition is not None
    initial = OmegaConf.merge(
        OmegaConf.create(composition.group(1)),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"policy_model_name": "served-policy-model"},
    )
    if model_override is not None:
        initial.hermes_agent.responses_api_agents.hermes_agent.model = model_override
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    agent = HermesAgentConfig(name="hermes_agent", **config.hermes_agent.responses_api_agents.hermes_agent)
    assert agent.resources_server is None
    assert agent.model == (model_override or "served-policy-model")
    assert agent.enabled_toolsets == ["terminal"]
    assert "datasets" not in config.hermes_agent.responses_api_agents.hermes_agent
    environment_name = _environment_server_for_agent(agent.name, _environment_servers_by_agent(config))
    assert environment_name == "single_agent_turn_legacy"
    environment = config[environment_name].environment_servers.single_agent_turn_legacy
    assert environment.resources_server.name == "swebench_pro_resources_server"
    assert environment.resources_tool_transports == []
    assert not any(name.startswith("swebench_pro_hermes") for name in config)


@pytest.mark.parametrize("model_override", [None, "other-served-model"])
def test_hermes_recipe_resolves_to_session_environment(model_override: str | None) -> None:
    recipe = Path(__file__).parents[3] / "benchmarks/swebench/pro/hermes.yaml"
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(recipe)])
    config = OmegaConf.merge(*configs, {"policy_model_name": "served-policy-model"})
    parser._recursively_swap_keys(config)
    if model_override is not None:
        config.swebench_pro_hermes_agent.responses_api_agents.hermes_agent.model = model_override
    assert config.get("environment_routing_mode", "agent") == "agent"
    environment_name = "swebench_pro_hermes"
    environment = SingleAgentTurnEnvironmentServerConfig(
        name=environment_name,
        host="localhost",
        port=8000,
        **OmegaConf.to_container(config[environment_name].environment_servers.single_agent_turn_legacy, resolve=True),
    )
    agent = HermesAgentConfig(
        name=environment.agent_server.name,
        host="localhost",
        port=8001,
        **OmegaConf.to_container(
            config[environment.agent_server.name].responses_api_agents.hermes_agent, resolve=True
        ),
    )
    assert agent.model == (model_override or "served-policy-model")
    assert agent.num_workers is None
    assert agent.session_close_retry_window_seconds == 300
    assert agent.resources_server.name == environment.resources_server.name
    assert agent.model_server.name == "policy_model"
    resources = config[environment.resources_server.name].resources_servers.swebench_pro
    assert _environment_server_for_agent(agent.name, _environment_servers_by_agent(config)) == environment_name
    assert resources.allowed_agents == ["hermes_agent"]
    assert resources.datasets[0].jsonl_fpath == "benchmarks/swebench/data/swebench_pro_benchmark.jsonl"
    assert resources.datasets[0].prepare_script == "benchmarks/swebench/pro/prepare.py"
