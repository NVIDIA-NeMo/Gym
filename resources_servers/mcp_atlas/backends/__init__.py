# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend registry for the MCP-Atlas resources server.

A backend is selected by the ``backend`` config key. Lookup order: ``register_backend`` registrations, the
built-in backends shipped here (``scale_live``), then a module of this package named after the backend that
registers it on import, so a deployment can add a backend without editing the server. An unknown name is a
configuration error that lists the backends available.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable

from .base import MCPAtlasBackend


BackendClass = type[MCPAtlasBackend]

_BACKENDS: dict[str, BackendClass] = {}
_BUILTIN_LOADERS: dict[str, Callable[[], BackendClass]] = {}
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def register_backend(name: str, backend_class: BackendClass, *, override: bool = False) -> None:
    """Make ``backend_class`` available as ``backend: <name>``. Registering a taken name raises unless overridden."""
    if not _NAME_PATTERN.match(name or ""):
        raise ValueError(f"MCP-Atlas backend name must be a lowercase identifier, got {name!r}")
    if not override and (name in _BACKENDS or name in _BUILTIN_LOADERS):
        raise ValueError(f"MCP-Atlas backend {name!r} is already registered")
    _BACKENDS[name] = backend_class


def list_backends() -> list[str]:
    """The names a config may select without importing anything else."""
    return sorted({*_BACKENDS, *_BUILTIN_LOADERS})


def get_backend(name: str) -> BackendClass:
    """Return the backend class for ``name``. Raises ValueError naming the available backends for an unknown one."""
    if name in _BACKENDS:
        return _BACKENDS[name]
    loader = _BUILTIN_LOADERS.get(name)
    if loader is not None:
        return loader()
    if _NAME_PATTERN.match(name or ""):
        try:
            importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise
        else:
            if name in _BACKENDS:
                return _BACKENDS[name]
            raise ValueError(f"module {__name__}.{name} did not register the MCP-Atlas backend {name!r}")
    raise ValueError(f"Unknown MCP-Atlas backend {name!r}. Available backends: {', '.join(list_backends())}")


def _load_scale_live() -> BackendClass:
    return importlib.import_module(f"{__name__}.scale_live").ScaleLiveBackend


_BUILTIN_LOADERS["scale_live"] = _load_scale_live
