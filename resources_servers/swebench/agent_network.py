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
"""Egress policy for the sandbox a SWE resources server creates for the agent.

With open egress, RL agents learn to fetch a task's upstream fix (newer releases from
package registries, the fixed source from GitHub, through any language runtime), which a
command deny-list cannot stop. The SWE servers create the agent's sandbox in
``seed_session`` and hand it to the agent, so the agent's own ``network_access`` cannot
apply a policy there; this builds the same policy (deny by default, allow the policy model
server's host) for the server to attach. Grading sandboxes are created separately and keep
their network.
"""

from typing import Any, Mapping

from nemo_gym.config_types import ModelServerRef
from nemo_gym.sandbox.agent_tools import restricted_network_policy, sandbox_server_url


def agent_network_policy(provider_config: Mapping[str, Any], model_server: ModelServerRef | None) -> dict[str, object]:
    """Deny all egress except ``model_server``'s host; deny everything when it is ``None``.

    Raises ValueError for providers without network-policy support or a model server that
    does not advertise a sandbox-reachable host (fail closed rather than run with open egress).
    """
    (provider_name,) = provider_config
    urls = [sandbox_server_url(model_server.name, require_reachable=True)] if model_server else []
    return restricted_network_policy(provider_name, urls)
