# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate harness version settings in native and wrapped agent configurations."""

import hashlib
from collections.abc import Mapping

from omegaconf import DictConfig


def harness_version_target(agent: str, config: DictConfig) -> tuple[str, DictConfig]:
    """Identify which agent consumes the version selector, including inside task wrappers."""
    if agent == "pi_sandboxed_agent":
        return "pi_agent", config
    if agent in {"harness_agent", "harness_exa_search", "anyswe_agent", "anyterminal_agent"}:
        kwargs = config.get("agent_kwargs")
        if isinstance(kwargs, DictConfig) and {"openclaw_version", "pi_version"} & kwargs.keys():
            if agent in {"harness_agent", "harness_exa_search"}:
                name = config.get("agent")
                if name in ("openclaw", "pi"):
                    return f"{name}_agent", kwargs
            else:
                module = config.get("agent_server_module")
                if module in ("responses_api_agents.openclaw_agent.app", "responses_api_agents.pi_agent.app"):
                    return module.split(".")[1], kwargs
    return agent, config


def portable_harness_pin(agent: str, settings: Mapping[str, object]) -> tuple[str, dict[str, str]]:
    """Return a cache directory suffix and installer environment for a versioned portable bundle."""
    if agent not in {"openclaw_agent", "pi_agent"}:
        return "", {}
    version = settings.get(f"{agent.removesuffix('_agent')}_version")
    if not isinstance(version, str) or not version:
        return "", {}
    env = {"OPENCLAW_SPEC": f"openclaw@{version}"} if agent == "openclaw_agent" else {"PI_VERSION": version}
    # Legacy Pi settings also accept npm specs, which may contain path separators.
    return "_" + hashlib.sha256(version.encode()).hexdigest()[:16], env
