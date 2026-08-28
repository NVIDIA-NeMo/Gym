# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated native-browser resource server for WebVoyager."""

from resources_servers.webvoyager_browser.app import WebVoyagerBrowserResourcesServer
from resources_servers.webvoyager_browser.config import WebVoyagerBrowserResourcesServerConfig


__all__ = ["WebVoyagerBrowserResourcesServer", "WebVoyagerBrowserResourcesServerConfig"]
