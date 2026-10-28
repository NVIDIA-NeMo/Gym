# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.osworld.app import OSWorldResourcesServer
from resources_servers.osworld.config import OSWorldResourcesServerConfig
from resources_servers.osworld.models import OSWorldEvaluateResponse, OSWorldVerifyRequest
from resources_servers.osworld.session_manager import SessionNotFoundError


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


def _verify_server(tmp_path, evaluate: AsyncMock) -> OSWorldResourcesServer:
    config = OSWorldResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="osworld",
        cache_dir=str(tmp_path / "cache"),
        state_dir=str(tmp_path / "state"),
        cleanup_orphans_on_start=False,
        discover_workers=False,
        workers=[{"name": "worker-a", "remote_host": "worker-a.example", "data_host": "10.0.0.8"}],
    )
    server = OSWorldResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._manager = MagicMock(evaluate=evaluate, close_session=AsyncMock(return_value=True))
    return server


def _verify_request() -> OSWorldVerifyRequest:
    return OSWorldVerifyRequest(
        responses_create_params={"input": []},
        response={
            "id": "resp",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
    )


async def test_a_scored_task_is_not_a_failure(tmp_path) -> None:
    server = _verify_server(tmp_path, AsyncMock(return_value=OSWorldEvaluateResponse(score=0.0)))

    result = await server.verify(SimpleNamespace(session={SESSION_ID_KEY: "s1"}), _verify_request())

    assert (result.reward, result.mask_sample, result.failure_kind, result.failure_reason) == (0.0, False, None, None)
    server._manager.close_session.assert_awaited_once_with("s1")


async def test_a_failed_evaluation_is_masked_as_a_verifier_error(tmp_path) -> None:
    server = _verify_server(tmp_path, AsyncMock(side_effect=RuntimeError("evaluator crashed")))

    result = await server.verify(SimpleNamespace(session={SESSION_ID_KEY: "s1"}), _verify_request())

    dumped = result.model_dump()
    assert (dumped["reward"], dumped["osworld_score"], dumped["mask_sample"]) == (0.0, 0.0, True)
    assert dumped["failure_kind"] == "verifier_error"
    assert dumped["failure_reason"] == "OSWorld evaluation raised RuntimeError: evaluator crashed"
    server._manager.close_session.assert_awaited_once_with("s1")


async def test_a_session_missing_at_verify_is_masked_as_lost(tmp_path) -> None:
    server = _verify_server(tmp_path, AsyncMock(side_effect=SessionNotFoundError("s1")))

    result = await server.verify(SimpleNamespace(session={SESSION_ID_KEY: "s1"}), _verify_request())

    dumped = result.model_dump()
    assert (dumped["reward"], dumped["osworld_score"], dumped["mask_sample"]) == (0.0, 0.0, True)
    assert dumped["failure_kind"] == "session_lost"
    assert "s1" in dumped["failure_reason"]
    server._manager.close_session.assert_awaited_once_with("s1")
