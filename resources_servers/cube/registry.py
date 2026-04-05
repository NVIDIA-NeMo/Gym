# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Name → :class:`CubeEnvironmentBase` subclass registry for YAML ``config.env_domain``.

This module maps string keys (e.g. ``osworld``) to implementation classes and builds instances.
Concrete behavior lives under :mod:`resources_servers.cube.domains`.
"""

from __future__ import annotations

from typing import Dict, Type

from resources_servers.cube.domains.base import CubeEnvironmentBase


_REGISTERED_DOMAIN_CLASSES: Dict[str, Type[CubeEnvironmentBase]] = {}
_builtin_domains_registered: bool = False


def register_domain(name: str, cls: Type[CubeEnvironmentBase]) -> None:
    """Register a :class:`CubeEnvironmentBase` subclass for YAML ``env_domain: <name>``."""
    _REGISTERED_DOMAIN_CLASSES[name] = cls


def instantiate_domain(name: str) -> CubeEnvironmentBase:
    """Return a new instance of the class registered for ``name``."""
    _ensure_builtin_domains()
    cls = _REGISTERED_DOMAIN_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown cube.resources_servers domain: {name!r}")
    return cls()


def _ensure_builtin_domains() -> None:
    global _builtin_domains_registered
    if _builtin_domains_registered:
        return
    from resources_servers.cube.domains.osworld import OSWorldEnvironment

    _REGISTERED_DOMAIN_CLASSES.setdefault("osworld", OSWorldEnvironment)
    _builtin_domains_registered = True
