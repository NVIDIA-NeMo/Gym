# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select Environment Server composition for an authored environment."""

from pathlib import Path

from nemo_gym import component_search_roots
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.authoring import DEFAULT_ENVIRONMENT_SERVER, LoadedEnvironment
from nemo_gym.environment.runtime_composition import (
    ENVIRONMENT_ADAPTER_NAME,
    AgentRoleBinding,
    EnvironmentServerRuntime,
)
from nemo_gym.single_agent_episode_types import SINGLE_AGENT_TASK_INPUT_CONTRACT


def create_environment_server_runtime(loaded: LoadedEnvironment) -> EnvironmentServerRuntime:
    """Resolve the Environment Server selected by an authored environment."""

    environment_server = loaded.definition.environment_server
    if environment_server == DEFAULT_ENVIRONMENT_SERVER:
        return _single_agent_runtime()
    raise ConfigError(f"No environment server is registered as {environment_server!r}.")


def _single_agent_runtime() -> EnvironmentServerRuntime:
    return EnvironmentServerRuntime(
        config_paths=(_component_file("environment_servers/single_agent/configs/single_agent.yaml"),),
        environment_server_name="single_agent_environment_server",
        task_input_contract=SINGLE_AGENT_TASK_INPUT_CONTRACT,
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
                        "task_input_contract": SINGLE_AGENT_TASK_INPUT_CONTRACT,
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


__all__ = ["create_environment_server_runtime"]
