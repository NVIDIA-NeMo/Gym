# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful BrowserGym resource server for web-agent benchmarks."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from resources_servers.browsergym_web.app import BrowserGymWebResourcesServer
    from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig


__all__ = ["BrowserGymWebResourcesServer", "BrowserGymWebResourcesServerConfig"]


def __getattr__(name: str) -> Any:
    """Keep schema imports dependency-light while preserving public exports."""

    if name == "BrowserGymWebResourcesServer":
        from resources_servers.browsergym_web.app import BrowserGymWebResourcesServer

        return BrowserGymWebResourcesServer
    if name == "BrowserGymWebResourcesServerConfig":
        from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig

        return BrowserGymWebResourcesServerConfig
    raise AttributeError(name)
