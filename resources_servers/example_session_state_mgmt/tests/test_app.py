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

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from resources_servers.example_session_state_mgmt.app import (
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
    StatefulCounterSeedSessionRequest,
)


def _server() -> StatefulCounterResourcesServer:
    config = StatefulCounterResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return StatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class StatelessCookieJar(CookieJar):
    def extract_cookies(self, response: object, request: object) -> None:
        # Sessions must come from explicit request cookies, not earlier responses.
        pass


class TestApp:
    def test_each_session_counts_from_its_own_seed(self) -> None:
        client = TestClient(_server().setup_webserver(), cookies=StatelessCookieJar())

        first = client.post("/seed_session", json={"initial_count": 0}).cookies
        assert client.post("/increment_counter", json={"count": 2}, cookies=first).json() == {"success": True}
        assert client.post("/get_counter_value", json={}, cookies=first).json() == {"count": 2}

        second = client.post("/seed_session", json={"initial_count": 4}).cookies
        assert client.post("/increment_counter", json={"count": 3}, cookies=second).json() == {"success": True}
        assert client.post("/get_counter_value", json={}, cookies=second).json() == {"count": 7}
        assert client.post("/get_counter_value", json={}, cookies=first).json() == {"count": 2}

        # A session that was never seeded holds no counter.
        assert client.post("/get_counter_value", json={}).status_code == 404

    def test_verify_reads_the_count_the_tools_left(self) -> None:
        client = TestClient(_server().setup_webserver())
        client.post("/seed_session", json={"initial_count": 3})
        client.post("/increment_counter", json={"count": 4})
        response = NeMoGymResponse(
            id="response",
            created_at=0,
            model="model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        ).model_dump(mode="json")

        def reward(expected_count: int) -> float:
            body = {
                "responses_create_params": {"input": "count"},
                "response": response,
                "expected_count": expected_count,
            }
            return client.post("/verify", json=body).json()["reward"]

        assert reward(7) == 1.0
        assert reward(3) == 0.0

    async def test_export_leaves_out_a_session_the_server_no_longer_holds(self) -> None:
        server = _server()
        await server.initialize_session_state("live", StatefulCounterSeedSessionRequest(initial_count=3))

        # Made-up state for the dropped session would restore a wrong counter.
        assert await server.export_session_states(["live", "dropped"], checkpoint_id="c") == {"live": {"count": 3}}
