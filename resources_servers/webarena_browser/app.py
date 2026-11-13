# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym resource server for WebArena visual-browser sessions."""

from __future__ import annotations

from nemo_gym.web import WebResourcesServer
from resources_servers.webarena_browser.config import WebArenaBrowserResourcesServerConfig
from resources_servers.webarena_browser.session_manager import WebArenaBrowserSessionManager


class WebArenaBrowserResourcesServer(WebResourcesServer):
    ray_enabled = False
    config: WebArenaBrowserResourcesServerConfig

    def multiple_workers_unsupported_reason(self) -> str | None:
        return (
            "PyAutoGUI drives the one X display, seed moves the session cookie to an identity "
            "that session routing does not follow, so later steps and /close can reach a worker without the browser"
        )

    def make_session_manager(self) -> WebArenaBrowserSessionManager:
        return WebArenaBrowserSessionManager(self.config)


if __name__ == "__main__":
    WebArenaBrowserResourcesServer.run_webserver()
