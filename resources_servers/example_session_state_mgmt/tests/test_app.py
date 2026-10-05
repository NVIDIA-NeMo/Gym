# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from http.cookiejar import CookieJar
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.example_session_state_mgmt.app import (
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = StatefulCounterResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = StatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        app = server.setup_webserver()

        class StatelessCookieJar(CookieJar):
            def extract_cookies(self, response: object, request: object) -> None:
                # Sessions must come from explicit request cookies, not earlier responses.
                pass

        client = TestClient(app, cookies=StatelessCookieJar())

        # Check that we are at 0
        response = client.post("/get_counter_value")
        initial_request_cookies = response.cookies
        assert response.json() == {"count": 0}
        response = client.post("/increment_counter", json={"count": 2}, cookies=initial_request_cookies)
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", cookies=initial_request_cookies)
        assert response.json() == {"count": 2}

        # Start a new session i.e. don't pass cookies
        response = client.post("/increment_counter", json={"count": 4})
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", cookies=response.cookies)
        assert response.json() == {"count": 4}
        response = client.post("/increment_counter", json={"count": 3}, cookies=response.cookies)
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", cookies=response.cookies)
        assert response.json() == {"count": 7}

        response = client.post("/get_counter_value", cookies=initial_request_cookies)
        assert response.json() == {"count": 2}

    @staticmethod
    def _server() -> StatefulCounterResourcesServer:
        config = StatefulCounterResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="counter")
        return StatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_follows_the_session_contract(self) -> None:
        check_resources_session_contract(
            self._server().setup_webserver(),
            ResourcesSeedSessionRequest(
                resources_session_id="contract-session",
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="example", task_id="0"),
                task_data={"initial_count": 3, "expected_count": 6},
            ),
            keeps_state=True,
        )

    def test_environment_server_session_counts_from_the_seeded_value(self) -> None:
        server = self._server()
        client = TestClient(server.setup_webserver())
        episode_id = EpisodeId(rollout_id="rollout")

        seed = client.post(
            "/seed_session",
            json=ResourcesSeedSessionRequest(
                resources_session_id="resources-session",
                episode_id=episode_id,
                task_id=TaskId(taskset="example", task_id="0"),
                task_data={"initial_count": 3, "expected_count": 6},
            ).model_dump(mode="json"),
        )
        assert seed.json()["resources_session_id"] == "resources-session"
        # The seed cookie, which the agent gets in its tool grant, reaches the seeded counter.
        cookies = seed.cookies
        client.post("/increment_counter", json={"count": 1}, cookies=cookies)
        client.post("/increment_counter", json={"count": 2}, cookies=cookies)
        assert client.post("/get_counter_value", cookies=cookies).json() == {"count": 6}

        close = client.post(
            "/close_session",
            json=ResourcesCloseSessionRequest(
                resources_session_id="resources-session", episode_id=episode_id
            ).model_dump(mode="json"),
        )
        assert close.status_code == 200
        assert "resources-session" not in server.session_id_to_counter
        # A late tool call or seed cannot recreate the closed session.
        assert client.post("/increment_counter", json={"count": 1}, cookies=cookies).status_code == 409
        late_seed = client.post(
            "/seed_session",
            json=ResourcesSeedSessionRequest(
                resources_session_id="resources-session",
                episode_id=episode_id,
                task_id=TaskId(taskset="example", task_id="0"),
                task_data={"initial_count": 3, "expected_count": 6},
            ).model_dump(mode="json"),
        )
        assert late_seed.status_code == 409
        assert "resources-session" not in server.session_id_to_counter
