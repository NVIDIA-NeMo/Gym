# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.osworld.app import OSWorldResourcesServer
from resources_servers.osworld.config import OSWorldResourcesServerConfig


def test_health_is_public_and_session_routes_require_bearer_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_OSWORLD_RESOURCES_TOKEN", "test-token")
    config = OSWorldResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="osworld",
        auth_token_env="TEST_OSWORLD_RESOURCES_TOKEN",
        require_auth=True,
        cache_dir=str(tmp_path / "cache"),
        state_dir=str(tmp_path / "state"),
        cleanup_orphans_on_start=False,
        discover_workers=False,
        workers=[
            {
                "name": "worker-a",
                "remote_host": "worker-a.example",
                "data_host": "10.0.0.8",
            }
        ],
    )
    server = OSWorldResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )
    manager = MagicMock()
    manager.start = AsyncMock()
    manager.stop = AsyncMock()
    manager.health = AsyncMock(
        return_value={
            "status": "ok",
            "deployment_id": "osworld-decoupled",
            "sessions": 0,
            "workers": [],
        }
    )
    server._manager = manager

    with TestClient(server.setup_webserver()) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        unauthorized = client.get("/session")
        assert unauthorized.status_code == 401
        assert unauthorized.json() == {"detail": "invalid bearer token"}

    manager.start.assert_awaited_once()
    manager.stop.assert_awaited_once()
