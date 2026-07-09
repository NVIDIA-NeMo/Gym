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

import pytest
from fastapi.testclient import TestClient
from httpx import Cookies

from nemo_gym.server_utils import ServerClient
from resources_servers.example_session_state_mgmt.app import (
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)


pytest.importorskip("mcp")

RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
TOKEN_HEADER = "X-NeMo-Gym-Session-Token"


def _server() -> StatefulCounterResourcesServer:
    config = StatefulCounterResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="example_session_state_mgmt",
    )
    return StatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _rpc(client, method: str, params: dict, token: str | None = None, rpc_id: int = 1):
    headers = dict(RPC_HEADERS)
    if token is not None:
        headers[TOKEN_HEADER] = token
    return client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params},
        follow_redirects=False,
    )


class TestApp:
    def test_sessions_are_isolated_and_persistent(self) -> None:
        server = _server()

        app = server.setup_webserver()
        client = TestClient(app)

        class StatelessCookies(Cookies):
            def extract_cookies(self, response):
                pass

        client._cookies = StatelessCookies(client._cookies)

        # Check that we are at 0
        response = client.post("/get_counter_value", json={})
        initial_request_cookies = response.cookies
        assert response.json() == {"count": 0}
        response = client.post("/increment_counter", json={"count": 2}, cookies=initial_request_cookies)
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", json={}, cookies=initial_request_cookies)
        assert response.json() == {"count": 2}

        # Start a new session i.e. don't pass cookies
        response = client.post("/increment_counter", json={"count": 4})
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", json={}, cookies=response.cookies)
        assert response.json() == {"count": 4}
        response = client.post("/increment_counter", json={"count": 3}, cookies=response.cookies)
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", json={}, cookies=response.cookies)
        assert response.json() == {"count": 7}

        response = client.post("/get_counter_value", json={}, cookies=initial_request_cookies)
        assert response.json() == {"count": 2}


class TestDualTransport:
    """The @gym_tool migration must keep the HTTP wire contract and add the MCP surface."""

    def test_http_replay_byte_equal(self) -> None:
        """The exact request/response bytes simple_agent and the dataset rows rely on."""
        with TestClient(_server().setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/seed_session", json={"initial_count": 3})
            assert resp.status_code == 200

            resp = client.post("/increment_counter", json={"count": 2})
            assert resp.status_code == 200
            assert resp.content == b'{"success":true}'

            resp = client.post("/get_counter_value", json={})
            assert resp.status_code == 200
            assert resp.content == b'{"count":5}'

    def test_http_error_paths_byte_equal(self) -> None:
        """Validation failures must serialize exactly as the pre-migration hand-written route did."""
        with TestClient(_server().setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/increment_counter", json={})
            assert resp.status_code == 422
            assert (
                resp.content
                == b'{"detail":[{"type":"missing","loc":["body","count"],"msg":"Field required","input":{}}]}'
            )

            resp = client.post("/increment_counter", json={"count": "abc"})
            assert resp.status_code == 422
            assert resp.content == (
                b'{"detail":[{"type":"int_parsing","loc":["body","count"],'
                b'"msg":"Input should be a valid integer, unable to parse string as an integer","input":"abc"}]}'
            )

    def test_seed_session_carries_mcp_metadata(self) -> None:
        """Harness endpoint /seed_session is auto-augmented with the per-rollout MCP block."""
        with TestClient(_server().setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            body = client.post("/seed_session", json={"initial_count": 3}).json()
            assert body["mcp"]["url_path"] == "/mcp"
            assert TOKEN_HEADER in body["mcp"]["headers"]

            # The adopted signature keeps body validation live: initial_count stays required.
            assert client.post("/seed_session", json={}).status_code == 422

    def test_mcp_round_trip_shares_session_state_with_http(self) -> None:
        """tools/list + tools/call over raw JSON-RPC; the token resolves the seeded cookie session."""
        with TestClient(_server().setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            token = client.post("/seed_session", json={"initial_count": 40}).json()["mcp"]["headers"][TOKEN_HEADER]

            listing = _rpc(client, "tools/list", {}, token=token)
            assert listing.status_code == 200, listing.text
            tools = {tool["name"]: tool for tool in listing.json()["result"]["tools"]}
            assert set(tools) == {"increment_counter", "get_counter_value"}
            # session_id is injected by the base and never model-visible.
            assert set(tools["increment_counter"]["inputSchema"]["properties"]) == {"count"}
            assert tools["get_counter_value"]["inputSchema"].get("properties", {}) == {}

            called = _rpc(
                client, "tools/call", {"name": "increment_counter", "arguments": {"count": 2}}, token=token, rpc_id=2
            )
            result = called.json()["result"]
            assert result.get("isError") is not True
            assert result["structuredContent"] == {"success": True}

            # HTTP (cookie) and MCP (token) resolve the SAME per-rollout counter.
            assert client.post("/get_counter_value", json={}).json() == {"count": 42}
            called = _rpc(client, "tools/call", {"name": "get_counter_value", "arguments": {}}, token=token, rpc_id=3)
            assert called.json()["result"]["structuredContent"] == {"count": 42}

    def test_transport_parity(self) -> None:
        app = _server().setup_webserver()
        non_tool_paths = {"/seed_session", "/verify", "/aggregate_metrics", "/mcp", "/{tool_name}"}
        http_tools = {
            route.path.lstrip("/")
            for route in app.router.routes
            if getattr(route, "path", None)
            and "POST" in (getattr(route, "methods", None) or set())
            and route.path not in non_tool_paths
        }
        assert http_tools == {"increment_counter", "get_counter_value"}

        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            listing = _rpc(client, "tools/list", {})
            mcp_tools = {tool["name"] for tool in listing.json()["result"]["tools"]}
        assert mcp_tools == http_tools
