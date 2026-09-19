# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select environment-server composition for an environment's episode protocol."""

from pathlib import Path

from nemo_gym import component_search_roots
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.authoring import LoadedEnvironment
from nemo_gym.environment.runtime_composition import (
    ENVIRONMENT_ADAPTER_NAME,
    AgentRoleBinding,
    EpisodeProtocolRuntime,
)
from nemo_gym.single_agent_episode_types import SINGLE_AGENT_TASK_INPUT_CONTRACT


def create_episode_protocol_runtime(loaded: LoadedEnvironment) -> EpisodeProtocolRuntime:
    """Select the environment server that implements the declared episode protocol."""

    protocol = loaded.definition.episode_protocol
    if protocol == SINGLE_AGENT_TASK_INPUT_CONTRACT:
        return _single_agent_runtime(protocol)
    raise ConfigError(f"No environment server is registered for episode protocol {protocol!r}.")


def _single_agent_runtime(protocol: str) -> EpisodeProtocolRuntime:
    return EpisodeProtocolRuntime(
        config_paths=(_component_file("environment_servers/single_agent/configs/single_agent.yaml"),),
        environment_server_name="single_agent_environment_server",
        environment_server_config={
            "single_agent_environment_server": {
                "environment_servers": {
                    "single_agent": {
                        "resources_server": {
                            "type": "resources_servers",
                            "name": ENVIRONMENT_ADAPTER_NAME,
                        },
                        "agent_server": {
                            "type": "responses_api_agents",
                            "name": "agent",
                        },
                        "task_input_contract": protocol,
                    }
                }
            }
        },
        agent_roles={
            "agent": AgentRoleBinding(resources_server_name=ENVIRONMENT_ADAPTER_NAME),
        },
    )


def _component_file(relative_path: str) -> Path:
    for root in component_search_roots():
        candidate = root / relative_path
        if candidate.is_file():
            return candidate.resolve()
    raise ConfigError(f"Required NeMo Gym component config was not found: {relative_path}")


__all__ = ["create_episode_protocol_runtime"]
