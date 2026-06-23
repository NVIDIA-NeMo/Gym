# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider-neutral in-sandbox model-server egress primitive.

A self-driving agent (e.g. OpenHands) runs inside the sandbox and must reach the
Gym model server. This resolves a sandbox-reachable endpoint per provider and
injects only the minimal ``base_url``/``api_key``/``model`` via ``SandboxSpec.env``;
it deliberately does not serialize the whole global-config dict into the sandbox.

* apptainer: shares the host network namespace, so host loopback works.
* opensandbox: a distinct network namespace, so it requires a cluster-reachable
  Service/ingress URL. If one is not configured, egress is unavailable and the
  caller must declare the agent apptainer-only for that provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class ModelEgressUnavailable(RuntimeError):
    """Raised when no sandbox-reachable model endpoint can be resolved for a provider."""


@dataclass(frozen=True)
class ModelEndpoint:
    """A sandbox-reachable model-server endpoint.

    Attributes:
        base_url: The base URL the in-sandbox agent uses to reach the model server.
        api_key: Optional API key for authenticating to the model server.
        model: Optional model name to use.
    """

    base_url: str
    api_key: str = ""
    model: str = ""

    def to_sandbox_env(self) -> dict[str, str]:
        """Build the minimal set of environment variables to inject into the sandbox.

        Returns:
            dict[str, str]: Environment variables carrying the base URL and,
            when set, the API key and model name. The global config dict is
            never included.
        """
        env = {"OPENAI_BASE_URL": self.base_url, "NEMO_GYM_MODEL_BASE_URL": self.base_url}
        if self.api_key:
            env["OPENAI_API_KEY"] = self.api_key
        if self.model:
            env["NEMO_GYM_MODEL"] = self.model
        return env


def resolve(
    provider_name: str,
    model_server: Mapping[str, Any],
    *,
    host_loopback_url: str = "http://127.0.0.1:8000/v1",
    opensandbox_service_url: str | None = None,
) -> ModelEndpoint:
    """Resolve a sandbox-reachable model endpoint for a sandbox provider.

    Args:
        provider_name: The sandbox provider name (e.g. ``"apptainer"``,
            ``"opensandbox"``, ``"docker"``).
        model_server: Mapping describing the model server, read for the
            ``api_key``, ``model``, and ``base_url`` keys.
        host_loopback_url: Fallback URL used when the provider shares the host
            network namespace and no base URL is configured.
        opensandbox_service_url: Cluster-reachable Service/ingress URL used for
            the opensandbox provider when no other base URL is configured.

    Returns:
        ModelEndpoint: The resolved endpoint carrying the base URL, API key,
        and model name.

    Raises:
        ModelEgressUnavailable: If the opensandbox provider cannot resolve a
            cluster-reachable model-server URL (e.g. only loopback is available).
    """
    api_key = str(model_server.get("api_key", "") or "")
    model = str(model_server.get("model", "") or "")
    configured_base = str(model_server.get("base_url", "") or "")

    if provider_name == "apptainer":
        base_url = configured_base or host_loopback_url
    elif provider_name == "opensandbox":
        base_url = opensandbox_service_url or configured_base
        if not base_url or "127.0.0.1" in base_url or "localhost" in base_url:
            raise ModelEgressUnavailable(
                "opensandbox needs a cluster-reachable model-server URL (k8s Service/ingress); "
                "loopback is unreachable from the pod. Configure 'opensandbox_service_url', or "
                "run the agent with the apptainer or docker provider instead."
            )
    else:
        # docker / local: shares host network by default (host loopback reachable).
        base_url = configured_base or host_loopback_url

    return ModelEndpoint(base_url=base_url, api_key=api_key, model=model)
