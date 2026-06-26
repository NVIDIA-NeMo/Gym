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
from fastapi import Request

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    MCPResourcesServer,
    MCPServerMetadata,
    MCPSessionError,
    SimpleResourcesServer,
    gym_tool,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


class TestBaseResourcesServer:
    def test_sanity(self) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")

        class TestSimpleResourcesServer(SimpleResourcesServer):
            async def verify(self, body):
                pass

        agent = TestSimpleResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        agent.setup_webserver()


class TestMCPResourcesServer:
    def test_mounts_mcp_endpoint_with_normal_gym_endpoints(self) -> None:
        pytest.importorskip("mcp")
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_mcp_resources_server")

        class TestMCPServer(MCPResourcesServer):
            def register_mcp_tools(self, mcp):
                @mcp.tool()
                def ping() -> str:
                    return "pong"

            async def verify(self, body):
                pass

        server = TestMCPServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        paths = {getattr(route, "path", None) for route in app.routes}

        assert "/seed_session" in paths
        assert "/verify" in paths
        assert "/aggregate_metrics" in paths
        assert "/mcp" in paths

    def test_build_mcp_session_metadata_maps_token_to_session_id(self) -> None:
        pytest.importorskip("mcp")
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_mcp_resources_server")

        class TestMCPServer(MCPResourcesServer):
            def register_mcp_tools(self, mcp):
                pass

            async def verify(self, body):
                pass

        server = TestMCPServer(config=config, server_client=MagicMock(spec=ServerClient))
        request = MagicMock(spec=Request)
        request.session = {SESSION_ID_KEY: "gym-session-1"}

        metadata = server.build_mcp_session_metadata(request)
        token = metadata.headers["X-NeMo-Gym-Session-Token"]

        assert metadata.server_name == "test_mcp_resources_server"
        assert metadata.url_path == "/mcp"
        # The signed token round-trips back to the session id (no server-side storage).
        from nemo_gym.base_resources_server import _MCP_SESSION_TOKEN

        ctx = _MCP_SESSION_TOKEN.set(token)
        try:
            assert server.require_mcp_session_id() == "gym-session-1"
        finally:
            _MCP_SESSION_TOKEN.reset(ctx)

    def test_missing_mcp_session_token_raises(self) -> None:
        pytest.importorskip("mcp")
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_mcp_resources_server")

        class TestMCPServer(MCPResourcesServer):
            def register_mcp_tools(self, mcp):
                pass

            async def verify(self, body):
                pass

        server = TestMCPServer(config=config, server_client=MagicMock(spec=ServerClient))

        with pytest.raises(MCPSessionError):
            server.require_mcp_session_id()

    def test_invalid_mcp_session_token_raises(self) -> None:
        pytest.importorskip("mcp")
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_mcp_resources_server")

        class TestMCPServer(MCPResourcesServer):
            def register_mcp_tools(self, mcp):
                pass

            async def verify(self, body):
                pass

        server = TestMCPServer(config=config, server_client=MagicMock(spec=ServerClient))

        from nemo_gym.base_resources_server import _MCP_SESSION_TOKEN

        context_token = _MCP_SESSION_TOKEN.set("bad-token")
        try:
            with pytest.raises(MCPSessionError):
                server.require_mcp_session_id()
        finally:
            _MCP_SESSION_TOKEN.reset(context_token)

    def test_mcp_endpoint_accepts_non_loopback_host(self) -> None:
        """Regression: the MCP SDK's default DNS-rebinding protection returns HTTP 421 for any
        non-loopback Host header, which breaks multi-node/absolute-IP deployments. MCPResourcesServer
        must disable it so server-to-server MCP calls keep working off-loopback."""
        pytest.importorskip("mcp")
        from fastapi.testclient import TestClient

        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_mcp_resources_server")

        class TestMCPServer(MCPResourcesServer):
            def register_mcp_tools(self, mcp):
                @mcp.tool()
                def ping() -> str:
                    return "pong"

            async def verify(self, body):
                pass

        server = TestMCPServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()

        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            resp = client.post(
                "/mcp",
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                    # A routable, non-loopback host as seen on a multi-node deployment.
                    "Host": "10.20.30.40:8000",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "ping", "arguments": {}},
                },
                follow_redirects=False,
            )

        assert resp.status_code != 421, "MCP endpoint rejected a non-loopback Host (DNS-rebinding protection)"
        assert resp.status_code == 200
        assert resp.json()["result"]["structuredContent"]["result"] == "pong"


class _GymToolSeedResponse(BaseSeedSessionResponse):
    mcp: MCPServerMetadata


class _GymToolServer(MCPResourcesServer):
    """Exercises the @gym_tool auto-registration: a session-bound tool and a stateless one."""

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> _GymToolSeedResponse:
        return _GymToolSeedResponse(mcp=self.build_mcp_session_metadata(request))

    @gym_tool
    def echo(self, session_id: str, text: str) -> str:
        """Echo text tagged with the session id."""
        return f"{session_id}:{text}"

    @gym_tool
    def add(self, a: int, b: int) -> int:
        """Add two numbers (stateless — no session_id)."""
        return a + b

    async def verify(self, body):
        pass


def _gym_tool_server() -> _GymToolServer:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_gym_tool_server")
    return _GymToolServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestGymToolAutoRegistration:
    def test_decorated_methods_auto_register_over_mcp_with_session_hidden(self) -> None:
        pytest.importorskip("mcp")
        from fastapi.testclient import TestClient

        server = _gym_tool_server()
        app = server.setup_webserver()
        rpc_headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            token = client.post("/seed_session", json={}).json()["mcp"]["headers"]["X-NeMo-Gym-Session-Token"]

            listing = client.post(
                "/mcp",
                headers=rpc_headers,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                follow_redirects=False,
            )
            tools = {t["name"]: t for t in listing.json()["result"]["tools"]}
            assert set(tools) == {"echo", "add"}
            # session_id is injected, never surfaced to the model
            assert set(tools["echo"]["inputSchema"]["properties"]) == {"text"}
            assert set(tools["add"]["inputSchema"]["properties"]) == {"a", "b"}
            assert tools["echo"]["description"] == "Echo text tagged with the session id."

            # the session-bound tool resolves session_id from the token
            echoed = client.post(
                "/mcp",
                headers={**rpc_headers, "X-NeMo-Gym-Session-Token": token},
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"text": "hi"}},
                },
                follow_redirects=False,
            )
            assert echoed.json()["result"]["structuredContent"]["result"].endswith(":hi")

            # the stateless tool needs no token
            summed = client.post(
                "/mcp",
                headers={**rpc_headers, "X-NeMo-Gym-Session-Token": token},
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "add", "arguments": {"a": 2, "b": 3}},
                },
                follow_redirects=False,
            )
            assert summed.json()["result"]["structuredContent"]["result"] == 5

    def test_missing_token_surfaces_as_clean_tool_error(self) -> None:
        """A session-bound tool called without a token must come back as a clean MCP tool error
        (HTTP 200, isError) — not an HTTP 401, and without leaking the raw status into the message."""
        pytest.importorskip("mcp")
        from fastapi.testclient import TestClient

        server = _gym_tool_server()
        app = server.setup_webserver()
        rpc_headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            resp = client.post(
                "/mcp",
                headers=rpc_headers,  # note: no X-NeMo-Gym-Session-Token
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"text": "hi"}},
                },
                follow_redirects=False,
            )

        assert resp.status_code == 200  # MCP/JSON-RPC: transport succeeds; the failure is in the body
        result = resp.json()["result"]
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "X-NeMo-Gym-Session-Token" in text  # clean, specific message
        assert "401" not in text  # no leaked HTTP status code

    def test_rejects_reserved_tool_name(self) -> None:
        pytest.importorskip("mcp")
        server = _gym_tool_server()
        with pytest.raises(ValueError, match="reserved endpoint name"):
            server._register_gym_tool(MagicMock(), "aggregate_metrics", lambda **_: None)

    def test_rejects_request_parameter(self) -> None:
        pytest.importorskip("mcp")
        server = _gym_tool_server()

        def needs_request(request: Request, city: str) -> str:
            return city

        with pytest.raises(ValueError, match="must not take a 'request' parameter"):
            server._register_gym_tool(MagicMock(), "bad_tool", needs_request)
