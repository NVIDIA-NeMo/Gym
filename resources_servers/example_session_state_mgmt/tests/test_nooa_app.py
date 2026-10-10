# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.example_session_state_mgmt.app import StatefulCounterResourcesServerConfig
from resources_servers.example_session_state_mgmt.nooa_app import NOOAStatefulCounterResourcesServer


def test_native_session_preserves_retries_and_cleans_state() -> None:
    config = StatefulCounterResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="counter")
    server = NOOAStatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    with TestClient(server.setup_webserver()) as client:
        seed = {
            "resources_session_id": "native-counter",
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
        close = {"resources_session_id": "native-counter", "episode_id": {"rollout_id": "rollout"}}
        wrong = close | {"episode_id": {"rollout_id": "another"}}
        assert client.post("/close_session", json=wrong).status_code == 409
        assert client.post("/get_counter_value").json() == {"count": 6}
        assert client.post("/close_session", json=close).status_code == 200
        assert client.post("/close_session", json=close).status_code == 200
        assert not server.session_id_to_counter
        assert client.post("/seed_session", json=seed).status_code == 409
        assert client.post("/get_counter_value").status_code == 409
