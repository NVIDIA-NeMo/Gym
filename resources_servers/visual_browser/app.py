# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated Gym visual-browser resource server for WebVoyager sessions."""

from __future__ import annotations

from nemo_gym.web.resources_server import WebResourcesServer
from resources_servers.visual_browser.config import VisualBrowserResourcesServerConfig
from resources_servers.visual_browser.session_manager import VisualBrowserSessionManager


class VisualBrowserResourcesServer(WebResourcesServer):
    config: VisualBrowserResourcesServerConfig

    def make_session_manager(self) -> VisualBrowserSessionManager:
        return VisualBrowserSessionManager(self.config)


if __name__ == "__main__":
    VisualBrowserResourcesServer.run_webserver()
