# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE-bench Verified OpenCode recipe keeps its task sandbox for an episode and its checkpoint pauses."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser
from resources_servers.swebench.app import SwebenchResourcesServerConfig
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig


RECIPE = Path(__file__).parents[3] / "benchmarks/swebench/verified/opencode.yaml"


@pytest.fixture
def recipe():
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(RECIPE)])
    config = OmegaConf.merge(*configs, {"policy_model_name": "served-policy-model"})
    parser._recursively_swap_keys(config)
    return config


def test_the_task_sandbox_outlives_the_episode_and_the_time_it_may_spend_paused(recipe) -> None:
    environment = SingleAgentTurnEnvironmentServerConfig(
        name="swebench_verified_opencode_single_agent_turn",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(
            recipe.swebench_verified_opencode_single_agent_turn.environment_servers.single_agent_turn, resolve=True
        ),
    )
    block = OmegaConf.to_container(recipe[environment.resources_server.name].resources_servers.swebench, resolve=True)
    resources = SwebenchResourcesServerConfig(
        name=environment.resources_server.name, host="localhost", port=8002, **block
    )
    assert block["allowed_agents"] == ["opencode_agent"]
    assert resources.sandbox_config["ready_timeout_s"] == 1200, "the override keeps the inherited sandbox settings"

    pause_allowance = OpenCodeAgentConfig.model_fields["checkpoint_pause_allowance_seconds"].default
    agent = recipe[environment.agent_server.name].responses_api_agents.opencode_agent
    assert agent.get("checkpoint_pause_allowance_seconds", pause_allowance) == pause_allowance
    # The session lives as long as the sandbox: it must outlive the episode and the time it may spend paused.
    assert environment.default_episode_timeout_seconds + pause_allowance < resources.sandbox_config["ttl_s"]
