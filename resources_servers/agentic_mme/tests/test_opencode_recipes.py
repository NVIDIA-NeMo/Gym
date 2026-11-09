# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode recipes end each episode before the sandbox the resources server lends expires."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser


BENCHMARKS = Path(__file__).parents[3] / "benchmarks/agentic_mme"
# Room after the episode deadline for the environment server's cleanup and the close of a slow episode.
MARGIN_S = 600


def load(recipe: str):
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(BENCHMARKS / f"{recipe}.yaml")])
    config = OmegaConf.merge(*configs, {"policy_model_name": "served-policy-model"})
    parser._recursively_swap_keys(config)
    return config


@pytest.mark.parametrize("recipe", ["opencode", "opencode_server_tools"])
def test_the_lent_sandbox_outlives_the_episode(recipe: str) -> None:
    config = load(recipe)
    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name="agentic_mme_environment_server",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(
            config.agentic_mme_environment_server.environment_servers.single_agent_turn_legacy, resolve=True
        ),
    )
    resources = OmegaConf.to_container(config.agentic_mme.resources_servers.agentic_mme, resolve=True)
    agent = OmegaConf.to_container(config.agentic_mme_opencode_agent.responses_api_agents.opencode_agent, resolve=True)

    deadline = environment.default_episode_timeout_seconds
    ttl = resources["sandbox_config"]["ttl_s"]
    # The resources session expires with the sandbox, so a verify after the TTL is masked as a lost session.
    assert deadline is not None and deadline + environment.cleanup_timeout_seconds + MARGIN_S <= ttl
    # OpenCode's own timeout and the session close fit inside the episode, with room for the seed and the judge.
    assert agent["timeout"] + agent["session_close_timeout_seconds"] < deadline
