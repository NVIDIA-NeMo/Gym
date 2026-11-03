# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Egress policy for a sandbox that an agent creates for its own session."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

from nemo_gym.agents.agent_tools import restricted_network_policy
from nemo_gym.sandbox.providers.base import SandboxSpec


SandboxNetworkAccess = Literal["inherit", "model_only", "model_and_tools"]


def restrict_sandbox_network(
    spec: SandboxSpec,
    *,
    network_access: SandboxNetworkAccess,
    provider_name: str,
    model_base_url: str,
    tool_urls: Sequence[str] = (),
) -> SandboxSpec:
    """Return ``spec`` with egress limited to the model server and, for ``model_and_tools``, the tool servers.

    ``inherit`` leaves the provider's default network. Each allowed host is opened on every port and path, so
    co-hosted services stay reachable. The policy is fixed when the sandbox is created: it applies only to a
    sandbox the agent creates, never to a task sandbox borrowed from Resources.
    """
    if network_access == "inherit":
        return spec
    if "network_policy" in spec.provider_options:
        raise ValueError("Set network_access or sandbox_config.provider_options.network_policy, not both")
    urls = [model_base_url, *(tool_urls if network_access == "model_and_tools" else ())]
    policy = restricted_network_policy(provider_name, urls)
    return replace(spec, provider_options={**spec.provider_options, "network_policy": policy})
