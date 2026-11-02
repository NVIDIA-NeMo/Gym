# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from resources_servers.example_session_state_mgmt.app import StatefulCounterResourcesServerConfig
from resources_servers.example_session_state_mgmt.nooa_app import NOOAStatefulCounterResourcesServer


def test_resource_session_preserves_retries_and_cleans_counter_state() -> None:
    config = StatefulCounterResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="counter")
    server = NOOAStatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    with TestClient(server.setup_webserver()) as client:
        seed = {
            "resources_session_id": "counter-resource-session",
            "episode_id": {"rollout_id": "rollout"},
            "task_id": {"taskset": "counter", "task_id": "one"},
            "task_data": {"initial_count": 3, "expected_count": 6},
        }
        assert client.post("/seed_session", json=seed).status_code == 200
        assert client.post("/increment_counter", json={"count": 3}).status_code == 200
        assert client.post("/seed_session", json=seed).status_code == 200
        assert client.post("/get_counter_value").json() == {"count": 6}
        changed = seed | {"task_data": {"initial_count": 0}}
        assert client.post("/seed_session", json=changed).status_code == 409
        close = {"resources_session_id": "counter-resource-session", "episode_id": {"rollout_id": "rollout"}}
        wrong = close | {"episode_id": {"rollout_id": "another"}}
        assert client.post("/close_session", json=wrong).status_code == 409
        assert client.post("/get_counter_value").json() == {"count": 6}
        assert client.post("/close_session", json=close).status_code == 200
        assert client.post("/close_session", json=close).status_code == 200
        assert not server.session_id_to_counter
        assert client.post("/seed_session", json=seed).status_code == 409
        assert client.post("/get_counter_value").status_code == 409


def test_closed_sessions_are_remembered_for_the_close_retry_window(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    config = StatefulCounterResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="counter", session_close_retry_window_seconds=60
    )
    server = NOOAStatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    with TestClient(server.setup_webserver()) as client:
        seed = {
            "resources_session_id": "counter-resource-session",
            "episode_id": {"rollout_id": "rollout"},
            "task_id": {"taskset": "counter", "task_id": "one"},
            "task_data": {"initial_count": 3, "expected_count": 6},
        }
        close = {"resources_session_id": "counter-resource-session", "episode_id": {"rollout_id": "rollout"}}
        assert client.post("/seed_session", json=seed).status_code == 200
        assert client.post("/close_session", json=close).status_code == 200

        clock[0] = 59.0
        assert client.post("/seed_session", json=seed).status_code == 409
        other_episode = close | {"episode_id": {"rollout_id": "another"}}
        assert client.post("/close_session", json=other_episode).status_code == 409

        # Past the window the close is forgotten, as every other worker has forgotten it once the routing claim ends.
        clock[0] = 60.0
        assert len(server._closed_sessions) == 0
        # The stale cookie from the first seed must not bring the counter back.
        assert client.post("/increment_counter", json={"count": 1}).status_code == 409
        assert client.post("/get_counter_value").status_code == 409
        assert not server.session_id_to_counter
        assert client.post("/close_session", json=other_episode).status_code == 200
        assert len(server._closed_sessions) == 1

        clock[0] = 120.0
        assert len(server._closed_sessions) == 0
        assert client.post("/seed_session", json=seed).status_code == 200
        assert client.post("/get_counter_value").json() == {"count": 3}


def test_tool_routes_refuse_a_session_that_was_never_seeded() -> None:
    config = StatefulCounterResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="counter")
    server = NOOAStatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/increment_counter", json={"count": 1}).status_code == 409
        assert client.post("/get_counter_value").status_code == 409
        assert not server.session_id_to_counter
