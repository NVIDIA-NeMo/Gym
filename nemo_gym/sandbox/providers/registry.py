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

"""Provider registration utilities.

Providers can be made available three ways, in lookup precedence order:

1. ``register_provider(name, cls)`` — explicit in-process registration.
2. Built-in loaders shipped with NeMo Gym (e.g. ``opensandbox``).
3. Python entry points in the ``nemo_gym.sandbox_providers`` group, so a separate
   package can publish a provider that becomes available on install. Declare one
   in that package's ``pyproject.toml``::

       [project.entry-points."nemo_gym.sandbox_providers"]
       my_provider = "my_pkg.provider:MyProvider"

On name collisions: two entry points sharing a name raise (selection would be
nondeterministic); an entry point shadowed by a higher-precedence built-in or
registered provider is warned and ignored.
"""

import logging
from collections.abc import Callable, Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import Any, TypeAlias

from nemo_gym.sandbox.providers.base import SandboxProvider


LOGGER = logging.getLogger(__name__)

ProviderClass: TypeAlias = type[SandboxProvider]
ProviderLoader: TypeAlias = Callable[[], ProviderClass]

ENTRY_POINT_GROUP = "nemo_gym.sandbox_providers"

_PROVIDER_REGISTRY: dict[str, ProviderClass] = {}
_BUILTIN_PROVIDER_LOADERS: dict[str, ProviderLoader] = {}
_ENTRY_POINT_LOADERS: dict[str, ProviderLoader] | None = None


def _entry_point_dist_name(ep: EntryPoint) -> str:
    dist = getattr(ep, "dist", None)
    return getattr(dist, "name", None) or "<unknown distribution>"


def _entry_point_loaders() -> dict[str, ProviderLoader]:
    """Discover provider loaders from installed entry points (cached).

    Raises if two distributions publish the same provider name, since lookup
    would otherwise pick one nondeterministically. Warns when an entry point is
    shadowed by a built-in or explicitly registered provider of the same name.
    """
    global _ENTRY_POINT_LOADERS
    if _ENTRY_POINT_LOADERS is None:
        loaders: dict[str, ProviderLoader] = {}
        dist_by_name: dict[str, str] = {}
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            dist_name = _entry_point_dist_name(ep)
            if ep.name in loaders:
                raise ValueError(
                    f"Duplicate sandbox provider entry point {ep.name!r} published by "
                    f"{dist_by_name[ep.name]!r} and {dist_name!r}. Rename one of them."
                )
            if ep.name in _BUILTIN_PROVIDER_LOADERS or ep.name in _PROVIDER_REGISTRY:
                LOGGER.warning(
                    f"Sandbox provider entry point {ep.name!r} from {dist_name!r} is shadowed by a "
                    f"built-in or registered provider of the same name and will not be used."
                )
            loaders[ep.name] = ep.load
            dist_by_name[ep.name] = dist_name
        _ENTRY_POINT_LOADERS = loaders
    return _ENTRY_POINT_LOADERS


def register_provider(name: str, provider_class: ProviderClass, *, override: bool = False) -> None:
    """Register a sandbox provider class."""
    if not name:
        raise ValueError("Provider name must be non-empty")
    if not override and (name in _PROVIDER_REGISTRY or name in _BUILTIN_PROVIDER_LOADERS):
        raise ValueError(f"Sandbox provider {name!r} is already registered")
    _PROVIDER_REGISTRY[name] = provider_class


def get_provider_class(name: str) -> ProviderClass:
    """Return a provider class by name (explicit > built-in > entry point)."""
    if name in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[name]
    loader = _BUILTIN_PROVIDER_LOADERS.get(name) or _entry_point_loaders().get(name)
    if loader is not None:
        return loader()
    available = ", ".join(list_providers()) or "<none>"
    raise ValueError(f"Unknown sandbox provider {name!r}. Available providers: {available}")


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
    """List available provider names from all sources."""
    return sorted({*_PROVIDER_REGISTRY, *_BUILTIN_PROVIDER_LOADERS, *_entry_point_loaders()})


def _load_opensandbox_provider() -> ProviderClass:
    from nemo_gym.sandbox.providers.opensandbox import OpenSandboxProvider

    return OpenSandboxProvider


def _load_apptainer_provider() -> ProviderClass:
    from nemo_gym.sandbox.providers.apptainer import ApptainerProvider

    return ApptainerProvider


def _load_ecs_fargate_provider() -> ProviderClass:
    from nemo_gym.sandbox.providers.ecs_fargate import EcsFargateProvider

    return EcsFargateProvider


_BUILTIN_PROVIDER_LOADERS["opensandbox"] = _load_opensandbox_provider
_BUILTIN_PROVIDER_LOADERS["apptainer"] = _load_apptainer_provider
_BUILTIN_PROVIDER_LOADERS["ecs_fargate"] = _load_ecs_fargate_provider
