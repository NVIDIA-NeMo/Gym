# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE-bench Verified Hermes recipe wires one taskset route to a resources server and a sandbox-borrowing agent."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser
from resources_servers.swebench.app import SwebenchResourcesServerConfig


RECIPE = Path(__file__).parents[3] / "benchmarks/swebench/verified/hermes.yaml"


@pytest.fixture
def recipe():
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(RECIPE)])
    config = OmegaConf.merge(*configs, {"policy_model_name": "served-policy-model"})
    parser._recursively_swap_keys(config)
    return config


def test_recipe_routes_the_verified_taskset_to_a_session_environment(recipe) -> None:
    assert recipe.environment_routing_mode == "taskset"
    assert OmegaConf.to_container(recipe.environment_server_routes) == {
        "swebench:verified": "swebench_verified_hermes"
    }

    environment = SingleAgentTurnEnvironmentServerConfig(
        name="swebench_verified_hermes",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(recipe.swebench_verified_hermes.environment_servers.single_agent_turn, resolve=True),
    )
    assert environment.resources_server.name == "swebench_verified_hermes_resources_server"
    assert environment.agent_server.name == "swebench_verified_hermes_agent"
    assert not environment.resources_tool_transports

    block = OmegaConf.to_container(recipe[environment.resources_server.name].resources_servers.swebench, resolve=True)
    resources = SwebenchResourcesServerConfig(
        name=environment.resources_server.name, host="localhost", port=8002, **block
    )
    # The dataset's taskset is the route key, so its rows reach this environment.
    [dataset] = block["datasets"]
    assert dataset["taskset"] == "swebench:verified" and dataset["type"] == "benchmark"
    assert block["allowed_agents"] == ["hermes_agent"] and block["num_workers"] == 1
    assert resources.apply_anti_cheating is False
    # The session lives as long as the sandbox, so it must outlive the episode deadline the agent runs under.
    assert resources.sandbox_provider == "sandbox" and resources.sandbox_config["ttl_s"] == 25200
    assert environment.default_episode_timeout_seconds < resources.sandbox_config["ttl_s"]
    assert resources.sandbox_config["ready_timeout_s"] == 1200, "the override keeps the inherited sandbox settings"

    agent = recipe[environment.agent_server.name].responses_api_agents.hermes_agent
    assert agent.resources_server is None  # the agent borrows the resources server's task sandbox
    assert agent.model == "served-policy-model"
    assert list(agent.enabled_toolsets) == ["terminal"]
    assert agent.get("datasets") is None
