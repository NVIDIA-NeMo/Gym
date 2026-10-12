# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.rollout_collection import _environment_server_for_agent, _environment_servers_by_agent
from responses_api_agents.hermes_agent.app import HermesAgentConfig


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
