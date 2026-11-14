# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated visual-browser resource server for WebVoyager."""

from resources_servers.visual_browser.app import VisualBrowserResourcesServer
from resources_servers.visual_browser.config import VisualBrowserResourcesServerConfig


__all__ = ["VisualBrowserResourcesServer", "VisualBrowserResourcesServerConfig"]
