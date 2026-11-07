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
"""Egress policy for the agent's sandbox in the SWE resources servers.

With open egress, agents learn to fetch the task's upstream fix. The servers create the
agent's sandbox themselves, so the agent's own ``network_access`` cannot apply a policy there.
"""

from copy import deepcopy
from typing import Any, Mapping, cast

from nemo_gym.config import ModelServerRef
from nemo_gym.sandbox.agent_tools import restricted_network_policy, sandbox_server_url


def check_agent_network_config(provider_config: Mapping[str, Any], model_server: ModelServerRef | None) -> None:
    """Fail at server startup on an ``enforce_agent_no_network`` config that cannot work."""
    if model_server is None:
        raise ValueError(
            "enforce_agent_no_network requires sandbox_model_server: the agent in the sandbox must reach the policy model"
        )
    (provider_name,) = provider_config
    # Raises for providers without network-policy support.
    restricted_network_policy(provider_name, [])


def agent_network_policy(
    provider_config: Mapping[str, Any],
    model_server: ModelServerRef,
    configured: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deny all egress except ``model_server``'s host, or append that host to ``configured``.

    Raises ValueError for providers without network policies or an unreachable model host.
    """
    (provider_name,) = provider_config
    policy = restricted_network_policy(provider_name, [sandbox_server_url(model_server.name, require_reachable=True)])
    if configured is None:
        return policy
    merged = deepcopy(dict(configured))
    egress = merged.setdefault("egress", [])
    if not isinstance(egress, list):
        raise TypeError("provider_options.network_policy.egress must be a list")
    egress.extend(rule for rule in cast(list[dict[str, str]], policy["egress"]) if rule not in egress)
    return merged
