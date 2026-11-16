# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OfficeEnv's config resolves and starts its environment server without OfficeEnv installed: the simple agent on the
step loop, as shipped and as `--agent-type simple_agent` composes it, and an agent that keeps no agent session
relayed through legacy_agent."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


ROOT = Path(__file__).resolve().parents[3]
OFFICEENV = "resources_servers/officeenv/configs/officeenv.yaml"
MODEL = "responses_api_models/openai_model/configs/openai_model.yaml"
ENVIRONMENT = "officeenv_environment_server"


def _resolve(*paths: str) -> DictConfig:
    """The run config, as `gym env resolve` gives it for these config paths (`--agent-type` adds the agent's)."""
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [str(ROOT / path) for path in paths],
                    "policy_model_name": "test-model",
                    "policy_base_url": "http://127.0.0.1:9999/v1",
                    "policy_api_key": "test-only",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            skip_exporters=True,
            offline=True,
        )
    )


def _client(config: DictConfig) -> ServerClient:
    return ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=config)


def _step_loop(config: DictConfig) -> SingleAgentTurnLegacyEnvironmentServer:
    """OfficeEnv's environment server, built as it starts, which checks that its agent can be driven step by step."""
    block = config[ENVIRONMENT].environment_servers.single_agent_turn_legacy
    server_config = SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(
        OmegaConf.to_container(block, resolve=True) | {"name": ENVIRONMENT}
    )
    return SingleAgentTurnLegacyEnvironmentServer(config=server_config, server_client=_client(config))


def test_the_shipped_config_drives_the_simple_agent_one_step_at_a_time() -> None:
    config = _resolve(OFFICEENV, MODEL)

    server = _step_loop(config)

    assert (server.config.max_episode_steps, server.config.end_on_task_state) == (50, True)
    assert server.config.agent_server.name == "officeenv_simple_agent"
    assert server.checkpoint_mode_in_effect() == "restart_only"
    # The example rows route to it by taskset; flat rows reach it through its agent.
    assert config.environment_server_routes["officeenv:example"] == ENVIRONMENT


def test_selecting_the_simple_agent_keeps_the_step_loop() -> None:
    """`--agent-type simple_agent` adds the standalone template, which declares no max_steps: composition puts it in
    place of officeenv_simple_agent, and the step loop still drives it, one step per activation."""
    config = _resolve(OFFICEENV, MODEL, "responses_api_agents/simple_agent/configs/simple_agent.yaml")

    server = _step_loop(config)

    agent = config[server.config.agent_server.name].responses_api_agents.simple_agent
    assert agent.resources_server.name == "officeenv_resources_server"
    assert agent.get("max_steps") is None
    assert "simple_agent" not in config


def test_an_agent_without_agent_sessions_is_relayed_to_its_own_run() -> None:
    """The compaction agent keeps no agent session, so composition replaces the step loop's server with a legacy_agent
    relay to the composed agent's /run, which runs its own loop."""
    config = _resolve(
        OFFICEENV, MODEL, "responses_api_agents/simple_agent_with_compaction/configs/simple_agent_with_compaction.yaml"
    )

    servers = config[ENVIRONMENT].environment_servers
    assert list(servers) == ["legacy_agent"]
    relay_config = LegacyAgentEnvironmentServerConfig.model_validate(
        OmegaConf.to_container(servers.legacy_agent, resolve=True) | {"name": ENVIRONMENT}
    )
    relay = LegacyAgentEnvironmentServer(config=relay_config, server_client=_client(config))
    assert relay.config.agent_server.name == "officeenv_simple_agent_with_compaction"
    agent = config.officeenv_simple_agent_with_compaction.responses_api_agents.simple_agent_with_compaction
    assert (agent.resources_server.name, agent.model_server.name) == ("officeenv_resources_server", "policy_model")
