# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The egress a resources server allows a sandbox it starts, such as the one it creates for the agent."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from nemo_gym.client.server_client import get_server_url
from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.server_runtime.base_resources_server import ResourcesSeedRejectedError
from nemo_gym.utils.network_policy import restricted_network_policy


# ``inherit`` keeps the provider's network, ``no_network`` denies all egress, ``model_only`` allows only the model
# server, and ``model_and_tools`` also the servers whose tools the harness calls.
SandboxNetworkAccess = Literal["inherit", "no_network", "model_only", "model_and_tools"]


class SandboxNetwork(BaseModel):
    """What one kind of sandbox may reach: the code that starts it passes :meth:`policy` to ``SandboxConfig.spec``.

    ``model_server`` is the model server that a harness in the sandbox calls, and ``tool_servers`` the servers whose
    tools it calls over MCP. Every port of an allowed host is open, so other services on that host stay reachable.
    Only a provider that enforces network policies can restrict a sandbox; on any other, :meth:`policy` refuses rather
    than start a sandbox with open egress.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    access: SandboxNetworkAccess = "inherit"
    model_server: str | None = None
    tool_servers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _servers_match_access(self) -> "SandboxNetwork":
        if self.access in ("model_only", "model_and_tools") and not self.model_server:
            raise ValueError(f"network_access {self.access} needs model_server, the model server the sandbox calls")
        if self.access in ("inherit", "no_network") and self.model_server:
            raise ValueError("model_server applies only to network_access model_only or model_and_tools")
        if self.access == "model_and_tools" and not self.tool_servers:
            raise ValueError(
                "network_access model_and_tools needs tool_servers, the servers whose tools the harness calls"
            )
        if self.access != "model_and_tools" and self.tool_servers:
            raise ValueError("tool_servers applies only to network_access model_and_tools")
        return self

    def check(self, sandbox_config: SandboxConfig) -> None:
        """Refuse, at startup, a restriction that ``sandbox_config.provider_options`` would also set.

        Raises:
            ValueError: If this restricts the network and ``provider_options`` already holds a ``network_policy``.
        """
        if self.access != "inherit" and "network_policy" in sandbox_config.provider_options:
            raise ValueError("Set network_access or sandbox_config.provider_options.network_policy, not both")

    def policy(self, provider_config: Mapping[str, object]) -> dict[str, object] | None:
        """The ``network_policy`` for a sandbox of ``provider_config``, or ``None`` to keep the provider's network.

        Args:
            provider_config: The resolved single-key ``{provider_name: config}`` mapping.

        Raises:
            ResourcesSeedRejectedError: If the provider cannot enforce network policies or an allowed server
                advertises a loopback host. No retry can fix either, so a seed reports it as HTTP 422.
        """
        if self.access == "inherit":
            return None
        (provider_name,) = provider_config
        servers = [self.model_server, *self.tool_servers] if self.model_server else []
        try:
            return restricted_network_policy(provider_name, [get_server_url(server) for server in servers])
        except ValueError as error:
            raise ResourcesSeedRejectedError(f"Cannot start a {self.access} sandbox: {error}") from error
