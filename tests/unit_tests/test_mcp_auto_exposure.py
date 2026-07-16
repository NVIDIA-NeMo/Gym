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
"""Unit tests for nemo_gym.mcp_auto_exposure — MCP auto-exposure of resources-server tool routes.

Self-contained: synthetic SimpleResourcesServer subclasses with a handful of routes exercise the
engine through the real /mcp endpoint via TestClient. No external server dependencies.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel


pytest.importorskip("mcp")

from nemo_gym.base_resources_server import BaseResourcesServerConfig, SimpleResourcesServer  # noqa: E402
from nemo_gym.mcp_auto_exposure import (  # noqa: E402
    TOKEN_HEADER,
    bind_route,
    install_auto_exposure,
    maybe_auto_expose,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient  # noqa: E402


RPC_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


class EchoBody(BaseModel):
    value: str


class Store(SimpleResourcesServer):
    """A typed tool, a dict-body tool, and a raw-body PlainTextResponse catch-all dispatcher."""

    expose_tools_over_mcp: ClassVar[bool] = True
    session_state: dict[str, list] = {}

    async def verify(self, body):
        pass

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.post("/append")
        async def append(body: EchoBody, request: Request):
            """Append a value to this session's list and return it."""
            sid = request.session[SESSION_ID_KEY]
            self.session_state.setdefault(sid, []).append(body.value)
            return {"values": self.session_state[sid]}

        @app.post("/raw_step")
        async def raw_step(body: dict, request: Request):
            # dict body: FastAPI passes the parsed JSON through unvalidated.
            _ = request.session[SESSION_ID_KEY]
            return {"echo": body}

        @app.post("/{tool_name}")
        async def dispatch(tool_name: str, request: Request) -> PlainTextResponse:
            # raw-body catch-all: reads request.json(), returns PlainTextResponse.
            args = await request.json()
            return PlainTextResponse(json.dumps({"tool": tool_name, "args": args}))

        return app

    def mcp_tool_inventory(self) -> list[dict]:
        return [{"name": "lookup", "input_schema": {"type": "object", "additionalProperties": True}}]


def _server(cls=Store, name="store") -> SimpleResourcesServer:
    cfg = BaseResourcesServerConfig(host="", port=0, entrypoint="", name=name)
    return cls(config=cfg, server_client=MagicMock(spec=ServerClient))


def _seed(client: TestClient) -> str:
    """POST /seed_session, return the MCP session token."""
    resp = client.post("/seed_session", json={})
    return resp.json()["mcp"]["headers"][TOKEN_HEADER]


def _rpc(client: TestClient, method: str, params: dict | None = None, token: str | None = None, rid: int = 1) -> dict:
    headers = dict(RPC_HEADERS)
    if token:
        headers[TOKEN_HEADER] = token
    body = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", headers=headers, json=body).json()


def _handshake(client: TestClient) -> None:
    _rpc(
        client,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    )
    client.post("/mcp", headers=RPC_HEADERS, json={"jsonrpc": "2.0", "method": "notifications/initialized"})


def _list(client: TestClient, token: str | None = None) -> list[dict]:
    return _rpc(client, "tools/list", {}, token=token, rid=2)["result"]["tools"]


def _call(client: TestClient, name: str, args: dict, token: str | None = None) -> dict:
    return _rpc(client, "tools/call", {"name": name, "arguments": args}, token=token, rid=3)["result"]


# ==================================================================================================
# The flag gate + mounting
# ==================================================================================================


def test_flag_off_does_not_mount_mcp():
    class Plain(Store):
        expose_tools_over_mcp: ClassVar[bool] = False

    server = _server(Plain)
    app = server.setup_webserver()
    assert maybe_auto_expose(server, app) is None
    assert "/mcp" not in {getattr(r, "path", None) for r in app.routes}


def test_flag_on_mounts_mcp_and_harvests_tools():
    server = _server()
    app = server.setup_webserver()
    tools = maybe_auto_expose(server, app)
    assert tools is not None
    assert "/mcp" in {getattr(r, "path", None) for r in app.routes}
    # typed + dict + inventory tools; the catch-all itself is not a tool
    assert {"append", "raw_step", "lookup"} <= set(tools)
    assert "{tool_name}" not in " ".join(tools)


# ==================================================================================================
# tools/list + tools/call over the real /mcp endpoint
# ==================================================================================================


def test_tools_list_advertises_typed_schema():
    server = _server()
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        token = _seed(client)
        _handshake(client)
        tools = {t["name"]: t for t in _list(client, token)}
        assert sorted(tools["append"]["inputSchema"]["properties"]) == ["value"]
        assert tools["append"]["description"].startswith("Append a value")


def test_direct_dispatch_runs_handler_and_shares_session_with_http_door():
    server = _server()
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        token = _seed(client)
        _handshake(client)
        # HTTP door (cookie) then MCP (token) — same seeded session id, so state accumulates.
        client.post("/append", json={"value": "a"})
        result = _call(client, "append", {"value": "b"}, token=token)
        assert result.get("isError") is not True
        assert json.loads(result["content"][0]["text"])["values"] == ["a", "b"]


def test_dict_body_tool_dispatches_direct():
    server = _server()
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        token = _seed(client)
        _handshake(client)
        result = _call(client, "raw_step", {"anything": [1, 2]}, token=token)
        assert json.loads(result["content"][0]["text"])["echo"] == {"anything": [1, 2]}


def test_raw_body_catchall_dispatches_and_unwraps_plaintext():
    server = _server()
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        token = _seed(client)
        _handshake(client)
        result = _call(client, "lookup", {"q": "iron"}, token=token)
        payload = json.loads(result["content"][0]["text"])
        assert payload == {"tool": "lookup", "args": {"q": "iron"}}


def test_allowed_tools_filters_list_and_gates_call():
    server = _server()
    app = server.setup_webserver()
    install_auto_exposure(server, app, allowed_tools=["append"])
    with TestClient(app) as client:
        token = _seed(client)
        _handshake(client)
        assert {t["name"] for t in _list(client, token)} == {"append"}
        blocked = _call(client, "raw_step", {}, token=token)
        assert blocked["isError"] is True and "not allowed" in blocked["content"][0]["text"]


def test_error_mapping():
    server = _server()
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        token = _seed(client)
        _handshake(client)
        # unknown tool
        r = _call(client, "nope", {}, token=token)
        assert r["isError"] is True and "Unknown tool" in r["content"][0]["text"]
        # missing token
        r = _call(client, "append", {"value": "x"}, token=None)
        assert r["isError"] is True and TOKEN_HEADER in r["content"][0]["text"]
        # invalid token
        r = _call(client, "append", {"value": "x"}, token="garbage")
        assert r["isError"] is True and "Invalid" in r["content"][0]["text"]
        # malformed args -> the handler's own 422
        r = _call(client, "append", {"wrong": "field"}, token=token)
        assert r["isError"] is True and "422" in r["content"][0]["text"]


# ==================================================================================================
# Refusal: shapes/servers direct dispatch cannot reproduce raise loudly at startup
# ==================================================================================================


def test_refuses_custom_middleware():
    server = _server()
    app = server.setup_webserver()

    @app.middleware("http")
    async def audit(request, call_next):
        return await call_next(request)

    with pytest.raises(ValueError, match="non-Gym middleware"):
        install_auto_exposure(server, app)


def test_refuses_dependency_injection_handler():
    server = _server()
    app = server.setup_webserver()

    def gate() -> bool:
        return True

    @app.post("/gated")
    async def gated(ok: bool = Depends(gate)):
        return {"ok": ok}

    with pytest.raises(ValueError, match="cannot be dispatched directly"):
        install_auto_exposure(server, app)


# ==================================================================================================
# The detector's annotation resolution (regression: factory-set __signature__ must win)
# ==================================================================================================


def test_bind_route_honors_factory_signature_over_annotations():
    import inspect

    from fastapi.routing import APIRoute

    app = FastAPI()

    async def handler(body: Any, request: Request):  # __annotations__ say Any
        return {}

    # A factory rewrites __signature__ with the REAL body model (the newton_bench pattern).
    handler.__signature__ = inspect.Signature(
        [
            inspect.Parameter("body", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=EchoBody),
            inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request),
        ]
    )
    app.post("/factory")(handler)
    route = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == "/factory")
    outcome = bind_route(route)
    assert outcome.binding is not None
    assert outcome.binding.body_model is EchoBody  # the signature won, not Any
