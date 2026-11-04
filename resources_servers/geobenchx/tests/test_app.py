# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.geobenchx.app import GeoBenchXResourcesServer, GeoBenchXResourcesServerConfig
from resources_servers.geobenchx.prepare_data import TASKS, dataset_row


TASK_ID = next(iter(TASKS))


class FakePool:
    def __init__(self, reset_reply: dict[str, Any] | None = None, release_reply: dict[str, Any] | None = None):
        self.reset_reply = reset_reply or {"ok": True}
        self.release_reply = release_reply or {"ok": True}
        self.released: list[str] = []

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def reset(self, session_id: str) -> dict[str, Any]:
        return self.reset_reply

    async def release(self, session_id: str) -> dict[str, Any]:
        self.released.append(session_id)
        return self.release_reply

    async def call(self, session_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "result": "done"}


def _server(pool: FakePool | None = None) -> GeoBenchXResourcesServer:
    server = GeoBenchXResourcesServer(
        config=GeoBenchXResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="geobenchx"),
        server_client=MagicMock(spec=ServerClient),
    )
    server._upstream = pool
    return server


def _verify_body() -> dict[str, Any]:
    row = dataset_row(TASKS[TASK_ID])
    response = {
        "id": "response",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    return {**row, "response": response}


def test_verify_releases_the_rollout_session() -> None:
    server = _server()
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json={"task_id": TASK_ID}).status_code == 200
    client.post("/load_geodata", json={"geodataset": "Countries", "output_geodataframe_name": "c"})
    assert len(server._sessions) == 1
    assert client.post("/verify", json=_verify_body()).status_code == 200
    assert server._sessions == {}


def test_closed_sessions_are_released_and_cannot_be_seeded_again() -> None:
    pool = FakePool()
    server = _server(pool)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json={"task_id": TASK_ID}).status_code == 200
    close = {"resources_session_id": "r", "episode_id": {"rollout_id": "rollout", "attempt": 0}}
    assert client.post("/close_session", json=close).status_code == 200
    assert client.post("/close_session", json=close).status_code == 200
    assert len(pool.released) == 2
    assert client.post("/seed_session", json={"task_id": TASK_ID}).status_code == 409


def test_failed_release_on_close_is_reported() -> None:
    server = _server(FakePool(release_reply={"ok": False, "result": "Error: tool worker failed to start"}))
    client = TestClient(server.setup_webserver())
    client.post("/seed_session", json={"task_id": TASK_ID})
    close = {"resources_session_id": "r", "episode_id": {"rollout_id": "rollout", "attempt": 0}}
    assert client.post("/close_session", json=close).status_code == 503
    # A failed close is retryable: the session is not recorded as closed.
    assert server._closed_sessions == set()


def test_seed_fails_loudly_when_no_worker_can_hold_the_session() -> None:
    server = _server(FakePool(reset_reply={"ok": False, "result": "Error: tool worker is at its session capacity."}))
    client = TestClient(server.setup_webserver())
    seeded = client.post("/seed_session", json={"task_id": TASK_ID})
    assert seeded.status_code == 503
    assert "capacity" in seeded.text


async def test_retired_sessions_are_released() -> None:
    pool = FakePool()
    server = _server(pool)
    await server.retire_session_state("retired")
    assert pool.released == ["retired"]
