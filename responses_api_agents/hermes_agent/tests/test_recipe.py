# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.global_config import GlobalConfigDictParser


def test_hermes_recipe_resolves_to_session_environment() -> None:
    recipe = Path(__file__).parents[3] / "benchmarks/swebench/pro/hermes.yaml"
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(recipe)])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    assert config.environment_routing_mode == "taskset"
    environment_name = config.environment_server_routes["swebench_pro"]
    assert environment_name == "swebench_pro_hermes"
    environment = SingleAgentTurnEnvironmentServerConfig(
        name=environment_name,
        host="localhost",
        port=8000,
        **OmegaConf.to_container(config[environment_name].environment_servers.single_agent_turn, resolve=True),
    )
    agent = config[environment.agent_server.name].responses_api_agents.hermes_agent
    assert agent.num_workers == 1
    assert agent.resources_server.name == environment.resources_server.name
    assert agent.model_server.name == "policy_model"
    resources = config[environment.resources_server.name].resources_servers.swebench_pro
    assert resources.allowed_agents == ["hermes_agent"]
    assert resources.datasets[0].jsonl_fpath == "benchmarks/swebench/data/swebench_pro_benchmark.jsonl"
    assert resources.datasets[0].prepare_script == "benchmarks/swebench/pro/prepare.py"
