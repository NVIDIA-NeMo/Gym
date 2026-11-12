# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Egress policy for a sandbox that an agent creates for its own session."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

from nemo_gym.agents.agent_tools import restricted_network_policy
from nemo_gym.agents.base_responses_api_agent import AgentSeedRejectedError
from nemo_gym.sandbox.providers.base import SandboxProvider, SandboxSpec


SandboxNetworkAccess = Literal["inherit", "model_only", "model_and_tools"]


def restrict_sandbox_network(
    spec: SandboxSpec,
    *,
    network_access: SandboxNetworkAccess,
    provider: SandboxProvider,
    model_base_url: str,
    tool_urls: Sequence[str] = (),
    required_tool_urls: Sequence[str] = (),
) -> SandboxSpec:
    """Return ``spec`` with egress limited to the model server and, for ``model_and_tools``, the tool servers.

    ``inherit`` leaves the provider's default network. Each allowed host is opened on every port and path, so
    co-hosted services stay reachable. The policy is fixed when the sandbox is created: it applies only to a
    sandbox the agent creates, never to a task sandbox borrowed from Resources.

    ``tool_urls`` are the MCP grants the harness keeps; ``required_tool_urls`` are those of them the episode requires.
    ``model_only`` blocks every tool host, so a required grant would fail only after the sandbox and the runtime
    exist; it is refused here instead.

    A configuration that cannot be enforced (an explicit provider policy as well, a provider without network
    policies, an endpoint the sandbox cannot reach, or a required tool grant under ``model_only``) raises
    ``AgentSeedRejectedError``, so the seed fails once with HTTP 422 instead of being retried; it is still a
    ``ValueError``.
    """
    if network_access == "inherit":
        return spec
    provider_options = spec.provider_options or {}
    if "network_policy" in provider_options:
        raise AgentSeedRejectedError("Set network_access or sandbox_config.provider_options.network_policy, not both")
    if network_access == "model_only" and required_tool_urls:
        raise AgentSeedRejectedError(
            "network_access model_only blocks the hosts of the required tool grants ("
            + ", ".join(required_tool_urls)
            + "); use model_and_tools"
        )
    urls = [model_base_url, *(tool_urls if network_access == "model_and_tools" else ())]
    try:
        policy = restricted_network_policy(provider.name, urls)
    except ValueError as error:
        raise AgentSeedRejectedError(str(error)) from error
    return replace(spec, provider_options={**provider_options, "network_policy": policy})
