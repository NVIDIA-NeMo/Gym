# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from nemo_gym.agents.claude_code import ClaudeCodeHarness
from nemo_gym.agents.cline import ClineHarness
from nemo_gym.agents.codex import CodexHarness
from nemo_gym.agents.config import AgentHarnessConfig, AgentModelConfig
from nemo_gym.agents.hermes import HermesHarness
from nemo_gym.agents.kilocode import KiloCodeHarness
from nemo_gym.agents.openclaw import OpenClawHarness
from nemo_gym.agents.pi import PiHarness
from nemo_gym.agents.prime import PrimeAgentHarness


if TYPE_CHECKING:
    from nemo_gym.agents.terminus_2 import Terminus2Harness


__all__ = [
    "AgentHarnessConfig",
    "AgentModelConfig",
    "ClaudeCodeHarness",
    "ClineHarness",
    "CodexHarness",
    "HermesHarness",
    "KiloCodeHarness",
    "OpenClawHarness",
    "PiHarness",
    "PrimeAgentHarness",
    "resolve_agent",
]


_AGENTS = {
    "claude_code": ("nemo_gym.agents.claude_code", "ClaudeCodeHarness", "AgentHarnessConfig"),
    "cline": ("nemo_gym.agents.cline", "ClineHarness", "AgentHarnessConfig"),
    "codex": ("nemo_gym.agents.codex", "CodexHarness", "AgentHarnessConfig"),
    "hermes": ("nemo_gym.agents.hermes", "HermesHarness", "AgentHarnessConfig"),
    "kilocode": ("nemo_gym.agents.kilocode", "KiloCodeHarness", "AgentHarnessConfig"),
    "openclaw": ("nemo_gym.agents.openclaw", "OpenClawHarness", "AgentHarnessConfig"),
    "pi": ("nemo_gym.agents.pi", "PiHarness", "AgentHarnessConfig"),
    "prime": ("nemo_gym.agents.prime", "PrimeAgentHarness", "AgentHarnessConfig"),
    "terminus_2": ("nemo_gym.agents.terminus_2", "Terminus2Harness", "AgentHarnessConfig"),
}


def resolve_agent(name: str) -> tuple[str, str, str, str]:
    try:
        module, agent_class, config_class = _AGENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {name}") from exc
    return module, agent_class, config_class, f"{name}_agent"


def __getattr__(name: str) -> Any:
    if name == "Terminus2Harness":
        try:
            from nemo_gym.agents.terminus_2 import Terminus2Harness
        except ModuleNotFoundError as exc:
            if exc.name == "tenacity" or (exc.name and exc.name.split(".", 1)[0] == "harbor"):
                raise ModuleNotFoundError(
                    "Terminus2Harness requires the 'terminus-2' extra: pip install 'nemo-gym[terminus-2]'"
                ) from exc
            raise

        return Terminus2Harness
    raise AttributeError(name)
