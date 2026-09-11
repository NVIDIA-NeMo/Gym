# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.agents import resolve_agent as resolve_core_agent


_LEGACY_AGENTS = {
    "nemo_fabric": ("responses_api_agents.nemo_fabric_agent.app", "NeMoFabricAgent", "NeMoFabricAgentConfig"),
    "opencode": ("responses_api_agents.opencode_agent.app", "OpenCodeAgent", "OpenCodeAgentConfig"),
}


def resolve_agent(name: str) -> tuple[str, str, str, str]:
    if name not in _LEGACY_AGENTS:
        return resolve_core_agent(name)
    module, agent_class, config_class = _LEGACY_AGENTS[name]
    return module, agent_class, config_class, f"{name}_agent"
