# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful BrowserGym resource server for WebArena-family rollouts."""

from __future__ import annotations

from nemo_gym.web.resources_server import WebResourcesServer
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig
from resources_servers.browsergym_web.session_manager import BrowserGymSessionManager


class BrowserGymWebResourcesServer(WebResourcesServer):
    """Own live BrowserGym environments using the common web HTTP contract."""

    config: BrowserGymWebResourcesServerConfig

    def make_session_manager(self) -> BrowserGymSessionManager:
        return BrowserGymSessionManager(self.config)


if __name__ == "__main__":
    BrowserGymWebResourcesServer.run_webserver()
