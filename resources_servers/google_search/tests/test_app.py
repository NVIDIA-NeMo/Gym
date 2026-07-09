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
import json
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.google_search.app import (
    GoogleSearchResourcesServer,
    GoogleSearchResourcesServerConfig,
    box_parser,
)


pytest.importorskip("trafilatura")

RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _make_server() -> GoogleSearchResourcesServer:
    config = GoogleSearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="google_search",
        google_api_key="dummy_key",  # pragma: allowlist secret
        google_cx="dummy_cx",
    )
    return GoogleSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _rpc(client, method: str, params: Optional[dict] = None, rpc_id: int = 1):
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if method.startswith("notifications/"):
        payload["params"] = params or {}
    else:
        payload["id"] = rpc_id
        payload["params"] = params or {}
    return client.post("/mcp", headers=RPC_HEADERS, json=payload, follow_redirects=False)


def _handshake(client) -> None:
    resp = _rpc(
        client,
        "initialize",
        {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
    )
    assert resp.status_code == 200, resp.text
    resp = _rpc(client, "notifications/initialized")
    assert resp.status_code in (200, 202)


class TestApp:
    def test_sanity(self) -> None:
        config = GoogleSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            google_api_key="dummy_key",  # pragma: allowlist secret
            google_cx="dummy_cx",
        )
        GoogleSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_box_parser_valid_content(self) -> None:
        """Test box_parser with valid boxed content"""
        # Test basic boxed content
        result = box_parser("The answer is \\boxed{42}")
        assert result == "42"

        # Test with complex content
        result = box_parser("After calculation: \\boxed{x + y = 10}")
        assert result == "x + y = 10"

        # Test with no boxed content
        result = box_parser("No boxed content here")
        assert result is None

        # Test with empty string
        result = box_parser("")
        assert result is None


class _FakeSearchResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class TestHTTPReplay:
    """Byte-equal HTTP replay for the /search and /browse tool routes, including error paths."""

    def test_search_success_body_is_byte_identical(self, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        payload = {"items": [{"title": "hit", "link": "https://example.com"}]}
        captured = {}

        def fake_get(url, params=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            captured["timeout"] = timeout
            return _FakeSearchResponse(payload)

        monkeypatch.setattr("resources_servers.google_search.app.requests.get", fake_get)

        server = _make_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/search", json={"query": "acoustics"})
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/json"
            assert resp.json() == {"search_results": json.dumps(payload)}
        # The google API is still called with the config credentials + the flat query arg.
        assert captured["url"] == "https://www.googleapis.com/customsearch/v1"
        assert captured["params"] == {"key": "dummy_key", "cx": "dummy_cx", "q": "acoustics"}
        assert captured["timeout"] == 10

    def test_search_error_path_body_is_byte_identical(self, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        def boom(url, params=None, timeout=None):
            raise RuntimeError("network down")

        monkeypatch.setattr("resources_servers.google_search.app.requests.get", boom)

        server = _make_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/search", json={"query": "x"})
            assert resp.status_code == 200
            assert resp.json() == {"search_results": "Error: Unexpected error - network down"}

    def test_browse_success_body_is_byte_identical(self, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.setattr(
            "resources_servers.google_search.app.trafilatura.fetch_url", lambda url: "<html>raw</html>"
        )
        monkeypatch.setattr("resources_servers.google_search.app.trafilatura.extract", lambda html: "clean text")

        server = _make_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/browse", json={"url": "https://example.com"})
            assert resp.status_code == 200
            assert resp.json() == {"page_content": "clean text"}

    def test_browse_no_html_path_is_byte_identical(self, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.setattr("resources_servers.google_search.app.trafilatura.fetch_url", lambda url: None)

        server = _make_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/browse", json={"url": "https://example.com"})
            assert resp.status_code == 200
            assert resp.json() == {"page_content": "No HTML found"}

    def test_search_route_validates_body(self) -> None:
        from fastapi.testclient import TestClient

        server = _make_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            assert client.post("/search", json={}).status_code == 422
            assert client.post("/browse", json={}).status_code == 422


class TestMCPRoundTrip:
    """MCP JSON-RPC round-trip: tools/list advertises both tools, tools/call dispatches search."""

    def test_tools_list_and_call(self, monkeypatch) -> None:
        pytest.importorskip("mcp")
        from fastapi.testclient import TestClient

        payload = {"items": [{"title": "hit"}]}
        monkeypatch.setattr(
            "resources_servers.google_search.app.requests.get",
            lambda url, params=None, timeout=None: _FakeSearchResponse(payload),
        )

        server = _make_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            _handshake(client)

            listed = _rpc(client, "tools/list", rpc_id=2)
            assert listed.status_code == 200, listed.text
            tools = {tool["name"]: tool for tool in listed.json()["result"]["tools"]}
            assert set(tools) == {"search", "browse"}
            assert set(tools["search"]["inputSchema"]["properties"]) == {"query"}
            assert set(tools["browse"]["inputSchema"]["properties"]) == {"url"}

            called = _rpc(client, "tools/call", {"name": "search", "arguments": {"query": "q"}}, rpc_id=3)
            assert called.status_code == 200, called.text
            result = called.json()["result"]
            assert result.get("isError") is not True
            assert result["structuredContent"] == {"search_results": json.dumps(payload)}


class TestTransportParity:
    """MCP tool names match HTTP tool routes (minus non-tool endpoints)."""

    NON_TOOL_PATHS = {"/seed_session", "/verify", "/aggregate_metrics", "/mcp", "/{tool_name}"}

    def test_tool_sets_identical_across_transports(self) -> None:
        pytest.importorskip("mcp")
        from fastapi.testclient import TestClient

        server = _make_server()
        app = server.setup_webserver()
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            _handshake(client)
            listed = _rpc(client, "tools/list", rpc_id=2)
            mcp_names = {tool["name"] for tool in listed.json()["result"]["tools"]}

            http_names = set()
            for route in app.router.routes:
                path = getattr(route, "path", None)
                methods = getattr(route, "methods", None) or set()
                if path and "POST" in methods and path not in self.NON_TOOL_PATHS:
                    http_names.add(path.lstrip("/"))

            assert mcp_names == http_names == {"search", "browse"}
