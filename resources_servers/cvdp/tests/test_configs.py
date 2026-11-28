# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped configs, resolved offline by Gym's own parser."""

import pytest
from omegaconf import OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict


HARNESSES = {
    "hermes": "cvdp_agentic_hermes_agent",
    "opencode": "cvdp_agentic_opencode_agent",
    "claude_code": "cvdp_agentic_claude_code_agent",
}


def _config(*config_paths: str):
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": list(config_paths), "cvdp_sim_image": "/images/cvdp-sim.sif"},
                # The legacy configs' own settings.
                {"anthropic_model_name": "claude", "policy_api_key": "key", "policy_base_url": "http://model"},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def _server(config, name: str) -> dict:
    return OmegaConf.to_container(get_first_server_config_dict(config, name), resolve=True)


@pytest.mark.parametrize("harness", sorted(HARNESSES))
def test_each_harness_borrows_the_task_sandbox_of_the_cvdp_server(harness):
    config = _config(f"resources_servers/cvdp/configs/cvdp_agentic_{harness}.yaml")
    agent = _server(config, HARNESSES[harness])
    resources = _server(config, "cvdp_agentic_resources_server")
    environment = _server(config, f"cvdp_agentic_{harness}")

    assert agent["resources_server"]["name"] == "cvdp_agentic_resources_server"
    assert agent["model_server"]["name"] == "policy_model"
    assert environment["agent_server"]["name"] == HARNESSES[harness]
    assert environment["resources_server"]["name"] == "cvdp_agentic_resources_server"
    assert resources["oss_sim_image"] == "/images/cvdp-sim.sif"
    # Sessions live in one process: a second worker would receive verifies for sessions it never seeded.
    assert resources["num_workers"] == 1
    # The agent reattaches through the provider block the server names, which mounts the workspace it works in.
    sandbox = OmegaConf.to_container(config[resources["agent_sandbox_provider"]], resolve=True)
    assert sandbox["apptainer"]["create"]["mount_point"] == resources["agent_workdir"]
    # A provider that expires sandboxes keeps the task sandbox for the whole episode.
    assert resources["agent_sandbox_config"]["ttl_s"] >= environment["default_episode_timeout_seconds"]
    # Episodes are admitted no faster than the agent runs them (OpenCode's default is 8), and wait for admission
    # outside the episode clock.
    assert environment["max_concurrent_episodes"] <= agent.get("concurrency", 8)
    assert environment["queue_timeout_seconds"] >= environment["default_episode_timeout_seconds"]


def test_the_three_harness_configs_compose_on_one_resources_server():
    config = _config(*(f"resources_servers/cvdp/configs/cvdp_agentic_{harness}.yaml" for harness in HARNESSES))
    resources = _server(config, "cvdp_agentic_resources_server")

    for harness, agent_name in HARNESSES.items():
        agent = _server(config, agent_name)
        assert _server(config, f"cvdp_agentic_{harness}")["agent_server"]["name"] == agent_name
        assert agent["resources_server"]["name"] == "cvdp_agentic_resources_server"
        assert next(iter(config[agent_name]["responses_api_agents"])) in resources["allowed_agents"]


@pytest.mark.parametrize("harness", ["claude", "hermes", "opencode"])
def test_the_legacy_configs_still_run_cvdp_agent_on_the_relay(harness):
    name = f"cvdp_agent_generic_{harness}"
    config = _config(
        "resources_servers/cvdp/configs/cvdp.yaml", f"responses_api_agents/cvdp_agent/configs/{name}.yaml"
    )

    agent = _server(config, name)
    environment = _server(config, f"{name}_environment_server")

    assert agent["simple_agent"] is False
    assert environment["agent_server"]["name"] == name
