# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Egress block for the sandbox a SWE resources server creates for the agent.

With open egress, RL agents learn to fetch a task's upstream fix (newer releases from package
registries, the fixed source from GitHub, through any language runtime), which a command
deny-list cannot stop. The SWE servers create the agent's sandbox in ``seed_session`` and lend it
to the agent, so the agent's own ``network_access`` cannot put a policy on it. These servers
attach the policy themselves, following the rules of ``LentSandboxes`` in
``nemo_gym/server_runtime/sandbox_lending.py``. Evaluation sandboxes keep their network.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, model_validator

from nemo_gym.client.server_client import get_server_url
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_runtime import ResourcesSeedRejectedError
from nemo_gym.utils.network_policy import restricted_network_policy


class AgentNetworkConfig(BaseModel):
    """Config fields that block egress from the agent's sandbox, except to the model server's host.

    ``enforce_agent_no_network`` (default off) gives the agent's sandbox a deny-by-default policy
    that allows only the host of ``sandbox_model_server``, on every port. Evaluation sandboxes are
    unaffected. Turning it on needs:

    - an OpenSandbox provider, which enforces network policies (checked at startup);
    - ``sandbox_model_server`` naming the model server the agent calls, the same server as the
      agent's ``model_server``, advertising a host the sandbox can reach (``use_absolute_ip=true``);
    - no ``network_policy`` in ``sandbox_config.provider_options``;
    - an agent that needs nothing else from the network: an OpenCode agent must use a preinstalled
      or uploaded OpenCode binary, since its default install downloads one inside the sandbox,
      and must leave its own ``network_access`` at ``inherit``.
    """

    sandbox_provider: str
    sandbox_config: dict[str, Any]
    enforce_agent_no_network: bool = False
    sandbox_model_server: ModelServerRef | None = None

    @model_validator(mode="after")
    def _check_agent_network(self) -> "AgentNetworkConfig":
        if self.enforce_agent_no_network:
            if self.sandbox_model_server is None:
                raise ValueError(
                    "enforce_agent_no_network needs sandbox_model_server, the model server the agent calls"
                )
            if "network_policy" in self.sandbox_config.get("provider_options", {}):
                raise ValueError(
                    "Set enforce_agent_no_network or sandbox_config.provider_options.network_policy, not both"
                )
        return self

    def check_agent_network_provider(self, named_configs: Mapping[str, Any]) -> None:
        """Raise ``ValueError`` at startup when the sandbox provider cannot enforce the egress block."""
        if self.enforce_agent_no_network:
            (provider_name,) = resolve_provider_config(self.sandbox_provider, named_configs)
            restricted_network_policy(provider_name, [])

    def sandbox_provider_options(self, provider_config: Mapping[str, Any], *, for_agent: bool) -> dict[str, Any]:
        """Return the provider options for a new sandbox, with the egress block on the agent's sandbox.

        A model server on a loopback host cannot be reached from the sandbox; that refuses the
        seed with ``ResourcesSeedRejectedError`` (HTTP 422), so it fails once instead of being retried.
        """
        options = dict(self.sandbox_config.get("provider_options", {}))
        if for_agent and self.enforce_agent_no_network:
            (provider_name,) = provider_config
            try:
                options["network_policy"] = restricted_network_policy(
                    provider_name, [get_server_url(self.sandbox_model_server.name)]
                )
            except ValueError as error:
                raise ResourcesSeedRejectedError(f"Cannot block the agent sandbox's egress: {error}") from error
        return options
