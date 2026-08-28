# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful BrowserGym resource server for web-agent benchmarks."""

from resources_servers.browsergym_web.app import BrowserGymWebResourcesServer
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig


__all__ = ["BrowserGymWebResourcesServer", "BrowserGymWebResourcesServerConfig"]
