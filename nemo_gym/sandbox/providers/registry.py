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

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from nemo_gym.sandbox.providers.base import SandboxProvider


ProviderClass: TypeAlias = type[SandboxProvider]
ProviderLoader: TypeAlias = Callable[[], ProviderClass]

_PROVIDER_REGISTRY: dict[str, ProviderClass] = {}
_BUILTIN_PROVIDER_LOADERS: dict[str, ProviderLoader] = {}


def register_provider(name: str, provider_class: ProviderClass, *, override: bool = False) -> None:
    """Register a sandbox provider class."""
    if not name:
        raise ValueError("Provider name must be non-empty")
    if not override and (name in _PROVIDER_REGISTRY or name in _BUILTIN_PROVIDER_LOADERS):
        raise ValueError(f"Sandbox provider {name!r} is already registered")
    _PROVIDER_REGISTRY[name] = provider_class


def get_provider_class(name: str) -> ProviderClass:
    """Return a registered provider class."""
    try:
        return _PROVIDER_REGISTRY[name]
    except KeyError as e:
        loader = _BUILTIN_PROVIDER_LOADERS.get(name)
        if loader is not None:
            return loader()
        available = ", ".join(list_providers()) or "<none>"
        raise ValueError(f"Unknown sandbox provider {name!r}. Available providers: {available}") from e


def create_provider(config: Mapping[str, Any]) -> SandboxProvider:
    """Instantiate a provider from a single-key provider config."""
    if len(config) != 1:
        raise ValueError("Sandbox provider config must contain exactly one provider name")
    provider_name, provider_kwargs = next(iter(config.items()))
    if not isinstance(provider_name, str) or not provider_name:
        raise ValueError("Sandbox provider name must be a non-empty string")
    if provider_kwargs is None:
        provider_kwargs = {}
    if not isinstance(provider_kwargs, Mapping):
        raise TypeError(f"Sandbox provider {provider_name!r} config must be a mapping")

    provider_class = get_provider_class(provider_name)
    return provider_class(**dict(provider_kwargs))


def list_providers() -> list[str]:
    """List registered provider names."""
    return sorted({*_PROVIDER_REGISTRY, *_BUILTIN_PROVIDER_LOADERS})


def _load_opensandbox_provider() -> ProviderClass:
    from nemo_gym.sandbox.providers.opensandbox import OpenSandboxProvider

    return OpenSandboxProvider


def _load_apptainer_provider() -> ProviderClass:
    from nemo_gym.sandbox.providers.apptainer import ApptainerProvider

    return ApptainerProvider


_BUILTIN_PROVIDER_LOADERS["apptainer"] = _load_apptainer_provider
_BUILTIN_PROVIDER_LOADERS["opensandbox"] = _load_opensandbox_provider
