# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode recipes end each episode before the lent sandbox expires, and declare image input."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig


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
        **{
            "name": "agentic_mme_environment_server",
            "host": "localhost",
            "port": 8000,
            # The recipe's own host, 127.0.0.1, replaces the placeholder.
            **OmegaConf.to_container(
                config.agentic_mme_environment_server.environment_servers.single_agent_turn_legacy, resolve=True
            ),
        }
    )
    resources = OmegaConf.to_container(config.agentic_mme.resources_servers.agentic_mme, resolve=True)
    agent = OmegaConf.to_container(config.agentic_mme_opencode_agent.responses_api_agents.opencode_agent, resolve=True)

    deadline = environment.default_episode_timeout_seconds
    ttl = resources["sandbox_config"]["ttl_s"]
    # The resources session expires with the sandbox, so a verify after the TTL is masked as a lost session.
    assert deadline is not None and deadline + environment.cleanup_timeout_seconds + MARGIN_S <= ttl
    # OpenCode's own timeout and the session close fit inside the episode, with room for the seed and the judge.
    assert agent["timeout"] + agent["session_close_timeout_seconds"] < deadline


@pytest.mark.parametrize("recipe", ["opencode", "opencode_server_tools"])
def test_the_recipe_declares_image_input_for_tool_images(recipe: str) -> None:
    # Without it, OpenCode replaces an image a tool returns with an error text when the prompt has no image.
    agent = load(recipe).agentic_mme_opencode_agent.responses_api_agents.opencode_agent
    assert agent.image_input is True


def test_the_sandbox_image_carries_the_pinned_plugin_dependencies() -> None:
    """The image builds OpenCode's plugin dependencies from agentic_if's lockfile, to the digest the recipe pins."""
    root = Path(__file__).parents[3]
    agent = load("opencode").agentic_mme_opencode_agent.responses_api_agents.opencode_agent
    image = root / "resources_servers" / "agentic_mme" / "sandbox_image"
    for name in ("package.json", "package-lock.json"):
        assert (image / "opencode" / name).read_bytes() == (
            root / "benchmarks/agentic_if/opencode" / name
        ).read_bytes()
    dockerfile = (image / "Dockerfile").read_text()
    archive = Path(agent.sandbox_plugin_dependencies_path)
    assert f"COPY --from=plugin-archive /{archive.name} {archive.parent}/" in dockerfile
    # The build checks the archive against the pin, so the image and the recipe cannot drift apart.
    assert f"{agent.sandbox_plugin_dependencies_sha256}  /{archive.name}" in dockerfile


@pytest.mark.parametrize(
    ("recipe", "reachable"),
    [("opencode", {"policy_model"}), ("opencode_server_tools", {"policy_model", "agentic_mme"})],
)
def test_only_the_servers_the_sandbox_calls_take_a_launcher_s_host_address(recipe: str, reachable: set[str]) -> None:
    """The lent sandbox may reach every port of the hosts it is allowed, so other servers stay on loopback.

    NeMo RL sets default_host to the node's address. A server's own host wins over it, so the judge, the agent and the
    Environment Server (and the resources server, unless OpenCode calls its /mcp route) stay on 127.0.0.1.
    """
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {
                    "config_paths": [str(BENCHMARKS / f"{recipe}.yaml")],
                    "policy_model_name": "served-policy-model",
                    "default_host": "10.1.2.3",
                },
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    hosts = {
        instance.name: instance.get_inner_run_server_config_dict()["host"]
        for instance in parser.filter_for_server_instance_configs(config)
    }
    assert hosts == {name: "10.1.2.3" if name in reachable else "127.0.0.1" for name in hosts}
    assert set(hosts) >= {"agentic_mme_judge_model", "agentic_mme_opencode_agent", "agentic_mme_environment_server"}


@pytest.mark.parametrize("recipe", ["opencode", "opencode_server_tools"])
def test_episodes_are_admitted_at_the_agent_s_rate(recipe: str) -> None:
    # Each seed starts a sandbox, so admitting more episodes than OpenCode runs would leave sandboxes idle.
    config = load(recipe)
    pairing = config.agentic_mme_environment_server.environment_servers.single_agent_turn_legacy
    agent = config.agentic_mme_opencode_agent.responses_api_agents.opencode_agent
    assert pairing.max_concurrent_episodes == agent.concurrency
    assert pairing.queue_timeout_seconds is not None
