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

from nemo_gym.mcp_test_utils import http_tool_names, mcp_call, mcp_list_tools
from nemo_gym.server_utils import ServerClient
from resources_servers.example_multi_step.app import (
    ExampleMultiStepResourcesServer,
    ExampleMultiStepResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = ExampleMultiStepResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        ExampleMultiStepResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestDualTransport:
    """The @gym_tool migration must keep the HTTP wire contract and add the MCP surface."""

    def _server(self) -> ExampleMultiStepResourcesServer:
        config = ExampleMultiStepResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="example_multi_step"
        )
        return ExampleMultiStepResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_http_wire_contract_is_unchanged(self) -> None:
        from fastapi.testclient import TestClient

        with TestClient(self._server().setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            # The exact request/response bytes simple_agent and the dataset rows rely on.
            resp = client.post("/get_synonym_value", json={"synonym": "Arid"})
            assert resp.status_code == 200
            assert resp.content == b'{"synonym_value":384}'  # sum(map(ord, "Arid"))

            resp = client.post("/extract_synonym_values", json={"synonym_values": [384, 279]})
            assert resp.status_code == 200
            assert resp.content == b'{"success":true}'

            # Error paths: missing/mistyped required fields are rejected by validation, as before.
            assert client.post("/get_synonym_value", json={}).status_code == 422
            resp = client.post("/extract_synonym_values", json={"synonym_values": "not-a-list"})
            assert resp.status_code == 422
            assert resp.json()["detail"][0]["loc"] == ["body", "synonym_values"]

    def test_mcp_lists_and_calls_the_same_tools(self) -> None:
        from fastapi.testclient import TestClient

        with TestClient(self._server().setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            tools = mcp_list_tools(client)
            assert set(tools) == {"get_synonym_value", "extract_synonym_values"}
            assert set(tools["get_synonym_value"]["inputSchema"]["properties"]) == {"synonym"}
            assert set(tools["extract_synonym_values"]["inputSchema"]["properties"]) == {"synonym_values"}

            result = mcp_call(client, "get_synonym_value", {"synonym": "Arid"})
            assert result.get("isError") is not True
            assert result["structuredContent"] == {"synonym_value": 384}

    def test_transport_parity(self) -> None:
        app = self._server().setup_webserver()
        assert http_tool_names(app) == {"get_synonym_value", "extract_synonym_values"}
