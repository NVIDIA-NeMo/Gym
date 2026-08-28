# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated Gym browser resource server for native WebVoyager sessions."""

from __future__ import annotations

from nemo_gym.web.resources_server import WebResourcesServer
from resources_servers.webvoyager_browser.config import WebVoyagerBrowserResourcesServerConfig
from resources_servers.webvoyager_browser.session_manager import WebVoyagerBrowserSessionManager


class WebVoyagerBrowserResourcesServer(WebResourcesServer):
    config: WebVoyagerBrowserResourcesServerConfig

    def make_session_manager(self) -> WebVoyagerBrowserSessionManager:
        return WebVoyagerBrowserSessionManager(self.config)


if __name__ == "__main__":
    WebVoyagerBrowserResourcesServer.run_webserver()
