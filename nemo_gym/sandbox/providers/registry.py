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

"""Provider registration utilities."""

from typing import TypeAlias

from nemo_gym.sandbox.config import SandboxProviderConfig
from nemo_gym.sandbox.providers.base import SandboxProvider


ProviderClass: TypeAlias = type[SandboxProvider]

G_PROVIDER_REGISTRY: dict[str, ProviderClass] = {}


def register_provider(name: str, provider_class: ProviderClass) -> None:
    """Register a sandbox provider class."""
    if not name:
        raise ValueError("Provider name must be non-empty")
    if name in G_PROVIDER_REGISTRY:
        raise ValueError(f"Sandbox provider {name!r} is already registered")
    G_PROVIDER_REGISTRY[name] = provider_class


def get_provider_class(name: str) -> ProviderClass:
    """Return a registered provider class."""
    try:
        return G_PROVIDER_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(G_PROVIDER_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown sandbox provider {name!r}. Available providers: {available}") from e


def create_provider(config: SandboxProviderConfig) -> SandboxProvider:
    """Instantiate a provider from ``env.sandbox.provider`` config."""
    provider_class = get_provider_class(config["name"])
    if "kwargs" in config:
        return provider_class(**config["kwargs"])
    return provider_class()


def list_providers() -> list[str]:
    """List registered provider names."""
    return sorted(G_PROVIDER_REGISTRY)


def _register_builtins() -> None:
    from nemo_gym.sandbox.providers.opensandbox import OpenSandboxProvider

    if "opensandbox" not in G_PROVIDER_REGISTRY:
        register_provider("opensandbox", OpenSandboxProvider)


_register_builtins()
