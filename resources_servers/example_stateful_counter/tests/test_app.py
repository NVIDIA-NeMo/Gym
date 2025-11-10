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
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from httpx import Cookies

from nemo_gym.server_utils import ServerClient
from resources_servers.example_stateful_counter.app import (
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
        client = TestClient(app)

        class StatelessCookies(Cookies):
            def extract_cookies(self, response):
                pass

        client._cookies = StatelessCookies(client._cookies)

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
