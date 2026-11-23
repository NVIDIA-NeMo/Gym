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
"""Selecting a backend / session provider from config.

Both are chosen the same way the sandbox providers are (`nemo_gym.sandbox`): a
single-key mapping whose key is the registered name and whose value is that
implementation's kwargs.

    backend:
      remote_cdp:
        session_provider:
          static_cdp: {cdp_url: "http://127.0.0.1:9222"}

Session providers can be made available three ways, in lookup precedence order:

1. ``register_session_provider(name, cls)`` — explicit in-process registration.
2. Built-in loaders shipped with this environment (imported lazily, so an
   unselected provider never imports its SDK).
3. Python entry points in the ``nemo_gym.browser_session_providers`` group, so a
   separate package can publish a provider that becomes available on install::

       [project.entry-points."nemo_gym.browser_session_providers"]
       my_browser_cloud = "my_pkg.provider:MyProvider"
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any, Callable, Optional

from .base import BrowserBackend, BrowserSessionProvider
from .local_playwright import LocalPlaywrightBackend
from .remote_cdp import RemoteCDPBackend, StaticCDPProvider


LOGGER = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "nemo_gym.browser_session_providers"


# --------------------------------------------------------------------------- #
# session providers
# --------------------------------------------------------------------------- #
def _load_lexmount() -> type:
    """Import the in-tree example provider (see providers/lexmount/).

    Deliberately lazy and inside the loader: the environment must not import a
    third-party SDK unless the operator selected that provider.
    """
    try:
        from ..providers.lexmount.provider import LexmountSessionProvider
    except ImportError:  # app.py run as a script: no parent package
        from providers.lexmount.provider import LexmountSessionProvider  # type: ignore[no-redef]

    return LexmountSessionProvider


_PROVIDER_REGISTRY: dict[str, type] = {}
_BUILTIN_PROVIDER_LOADERS: dict[str, Callable[[], type]] = {
    "static_cdp": lambda: StaticCDPProvider,
    "lexmount": _load_lexmount,
}
_ENTRY_POINT_LOADERS: Optional[dict[str, Callable[[], type]]] = None


def _entry_point_loaders() -> dict[str, Callable[[], type]]:
    global _ENTRY_POINT_LOADERS
    if _ENTRY_POINT_LOADERS is None:
        loaders: dict[str, Callable[[], type]] = {}
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            if ep.name in _BUILTIN_PROVIDER_LOADERS or ep.name in _PROVIDER_REGISTRY:
                LOGGER.warning(
                    "Browser session provider entry point %r is shadowed by a built-in or "
                    "registered provider of the same name and will not be used.",
                    ep.name,
                )
            loaders[ep.name] = ep.load
        _ENTRY_POINT_LOADERS = loaders
    return _ENTRY_POINT_LOADERS


def register_session_provider(name: str, provider_class: type, *, override: bool = False) -> None:
    """Register a browser session provider class."""
    if not name:
        raise ValueError("Provider name must be non-empty")
    if not override and (name in _PROVIDER_REGISTRY or name in _BUILTIN_PROVIDER_LOADERS):
        raise ValueError(f"Browser session provider {name!r} is already registered")
    _PROVIDER_REGISTRY[name] = provider_class


def list_session_providers() -> list[str]:
    return sorted({*_PROVIDER_REGISTRY, *_BUILTIN_PROVIDER_LOADERS, *_entry_point_loaders()})


def get_session_provider_class(name: str) -> type:
    """Return a provider class by name (explicit > built-in > entry point)."""
    if name in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[name]
    loader = _BUILTIN_PROVIDER_LOADERS.get(name) or _entry_point_loaders().get(name)
    if loader is not None:
        return loader()
    available = ", ".join(list_session_providers()) or "<none>"
    raise ValueError(f"Unknown browser session provider {name!r}. Available providers: {available}")


def create_session_provider(config: Mapping[str, Any]) -> BrowserSessionProvider:
    """Instantiate a provider from a single-key `{name: kwargs}` mapping."""
    name, kwargs = _single_key(config, "browser session provider")
    return get_session_provider_class(name)(**kwargs)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def _build_local_playwright(session_metadata: dict[str, str], **kwargs) -> BrowserBackend:
    return LocalPlaywrightBackend(session_metadata=session_metadata, **kwargs)


def _build_remote_cdp(session_metadata: dict[str, str], **kwargs) -> BrowserBackend:
    provider_config = kwargs.pop("session_provider", None)
    if provider_config is None:
        raise ValueError(
            "backend `remote_cdp` requires a `session_provider` block, e.g. "
            "{static_cdp: {cdp_url: 'http://127.0.0.1:9222'}}"
        )
    return RemoteCDPBackend(
        create_session_provider(provider_config),
        session_metadata=session_metadata,
        **kwargs,
    )


_BACKEND_BUILDERS: dict[str, Callable[..., BrowserBackend]] = {
    "local_playwright": _build_local_playwright,
    "remote_cdp": _build_remote_cdp,
}


def register_backend(name: str, builder: Callable[..., BrowserBackend], *, override: bool = False) -> None:
    """Register a backend builder: `builder(session_metadata, **config_kwargs)`."""
    if not name:
        raise ValueError("Backend name must be non-empty")
    if not override and name in _BACKEND_BUILDERS:
        raise ValueError(f"Browser backend {name!r} is already registered")
    _BACKEND_BUILDERS[name] = builder


def list_backends() -> list[str]:
    return sorted(_BACKEND_BUILDERS)


def create_backend(
    config: Mapping[str, Any],
    *,
    session_metadata: Optional[dict[str, str]] = None,
) -> BrowserBackend:
    """Instantiate a backend from a single-key `{name: kwargs}` mapping."""
    name, kwargs = _single_key(config, "browser backend")
    builder = _BACKEND_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown browser backend {name!r}. Available backends: {', '.join(list_backends())}")
    return builder(dict(session_metadata or {}), **kwargs)


def _single_key(config: Mapping[str, Any], what: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(config, Mapping) or len(config) != 1:
        raise ValueError(f"{what} config must be a mapping with exactly one name, got {config!r}")
    name, kwargs = next(iter(config.items()))
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what} name must be a non-empty string")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"{what} {name!r} config must be a mapping, got {type(kwargs).__name__}")
    return name, dict(kwargs)
