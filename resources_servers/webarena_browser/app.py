# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym resource server for WebArena visual-browser sessions."""

from __future__ import annotations

from nemo_gym.web.resources_server import WebResourcesServer
from resources_servers.webarena_browser.config import WebArenaBrowserResourcesServerConfig
from resources_servers.webarena_browser.session_manager import WebArenaBrowserSessionManager


class WebArenaBrowserResourcesServer(WebResourcesServer):
    config: WebArenaBrowserResourcesServerConfig

    def make_session_manager(self) -> WebArenaBrowserSessionManager:
        return WebArenaBrowserSessionManager(self.config)


if __name__ == "__main__":
    WebArenaBrowserResourcesServer.run_webserver()
