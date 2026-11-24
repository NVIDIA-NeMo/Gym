# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter names used by the sandbox harness runner."""

_AGENTS = {
    "claude_code": ("responses_api_agents.claude_code_agent.app", "ClaudeCodeAgent", "ClaudeCodeAgentConfig"),
    "cline": ("responses_api_agents.cline_agent.app", "ClineAgent", "ClineAgentConfig"),
    "codex": ("responses_api_agents.codex_agent.app", "CodexAgent", "CodexAgentConfig"),
    "deepagents": ("responses_api_agents.nemo_fabric_agent.app", "NeMoFabricAgent", "NeMoFabricAgentConfig"),
    "hermes": ("responses_api_agents.hermes_agent.app", "HermesAgent", "HermesAgentConfig"),
    "kilocode": ("responses_api_agents.kilocode_agent.app", "KiloCodeAgent", "KiloCodeAgentConfig"),
    "mini_swe": ("responses_api_agents.nemo_fabric_agent.app", "NeMoFabricAgent", "NeMoFabricAgentConfig"),
    "openclaw": ("responses_api_agents.openclaw_agent.app", "OpenClawAgent", "OpenClawAgentConfig"),
    "opencode": ("responses_api_agents.opencode_agent.app", "OpenCodeAgent", "OpenCodeAgentConfig"),
    "pi": ("responses_api_agents.pi_agent.app", "PiAgent", "PiAgentConfig"),
    "prime": ("responses_api_agents.prime_agent.app", "PrimeAgent", "PrimeAgentConfig"),
    "simple_strands": (
        "responses_api_agents.simple_strands_agent.app",
        "SimpleStrandsAgent",
        "SimpleStrandsAgentConfig",
    ),
    "terminus_2": ("responses_api_agents.terminus_2_agent.app", "Terminus2Agent", "Terminus2AgentConfig"),
}

FABRIC_ADAPTERS = {
    "deepagents": "nvidia.fabric.langchain.deepagents",
    "mini_swe": "nvidia.fabric.mini-swe-agent",
}


def resolve_agent(name: str) -> tuple[str, str, str]:
    """Return the runner module, agent class, and config class for a harness name."""
    try:
        return _AGENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {name}") from exc
