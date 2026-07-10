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

from nemo_gym.mcp_test_utils import http_tool_names, mcp_call, mcp_list_tools
from nemo_gym.server_utils import ServerClient
from resources_servers.example_single_tool_call.app import (
    SimpleWeatherResourcesServer,
    SimpleWeatherResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleWeatherResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        SimpleWeatherResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestDualTransport:
    """The @gym_tool migration must keep the HTTP wire contract and add the MCP surface."""

    def _server(self) -> SimpleWeatherResourcesServer:
        config = SimpleWeatherResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="example_single_tool_call"
        )
        return SimpleWeatherResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_http_wire_contract_is_unchanged(self) -> None:
        with TestClient(self._server().setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            # The exact request/response bytes simple_agent and the dataset rows rely on.
            resp = client.post("/get_weather", json={"city": "sf"})
            assert resp.status_code == 200
            assert resp.json() == {"city": "sf", "weather_description": "The weather in sf is cold."}
            # Error path: a missing required field is rejected by validation, as before.
            assert client.post("/get_weather", json={}).status_code == 422

    def test_mcp_lists_and_calls_the_same_tool(self) -> None:
        with TestClient(self._server().setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            tools = mcp_list_tools(client)
            assert set(tools) == {"get_weather"}
            assert set(tools["get_weather"]["inputSchema"]["properties"]) == {"city"}

            result = mcp_call(client, "get_weather", {"city": "sf"})
            assert result.get("isError") is not True
            assert result["structuredContent"] == {"city": "sf", "weather_description": "The weather in sf is cold."}

    def test_transport_parity(self) -> None:
        app = self._server().setup_webserver()
        assert http_tool_names(app) == {"get_weather"}
