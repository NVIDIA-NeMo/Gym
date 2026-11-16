# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated Gym visual-browser resource server for WebVoyager sessions."""

from __future__ import annotations

from nemo_gym.web import WebResourcesServer
from resources_servers.visual_browser.config import VisualBrowserResourcesServerConfig
from resources_servers.visual_browser.session_manager import VisualBrowserSessionManager


class VisualBrowserResourcesServer(WebResourcesServer):
    ray_enabled = False
    config: VisualBrowserResourcesServerConfig

    def multiple_workers_unsupported_reason(self) -> str | None:
        return (
            "PyAutoGUI drives the one X display, and seed moves the session cookie to an identity "
            "that session routing does not follow, so later steps and /close can reach a worker without the browser"
        )

    def make_session_manager(self) -> VisualBrowserSessionManager:
        return VisualBrowserSessionManager(self.config)


if __name__ == "__main__":
    VisualBrowserResourcesServer.run_webserver()
