# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ast
import inspect
import json
import logging
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.routing import Mount


pytest.importorskip("mcp")

from nemo_gym.base_resources_server import (  # noqa: E402
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
    normalize_tool_name,
)
from nemo_gym.mcp_auto_exposure import (  # noqa: E402
    TOKEN_HEADER,
    bind_route,
    install_auto_exposure,
    maybe_auto_expose,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient, SimpleServer  # noqa: E402


RPC_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


class EchoBody(BaseModel):
    value: str


class OtherBody(BaseModel):
    other: str


class PublicView(BaseModel):
    shown: str


class Store(SimpleResourcesServer):
    """A typed tool, a dict-body tool, and a raw-body PlainTextResponse catch-all dispatcher."""

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
            _ = request.session[SESSION_ID_KEY]
            return {"echo": body}

        @app.post("/{tool_name}")
        async def dispatch(tool_name: str, request: Request) -> PlainTextResponse:
            args = await request.json()
            return PlainTextResponse(json.dumps({"tool": tool_name, "args": args}))

        return app

    def mcp_tools(self, harvested, catchall):
        return harvested + [catchall.tool("lookup", {"type": "object", "additionalProperties": True})]


class Shapes(SimpleResourcesServer):
    """One route per handler shape that direct dispatch must reproduce (or map to the right error)."""

    async def verify(self, body):
        pass

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.post("/typed_dict_body")
        async def typed_dict_body(body: dict[str, Any]):
            return {"echo": body}

        @app.post("/model_and_raw")
        async def model_and_raw(body: EchoBody, request: Request):
            raw = await request.json()
            return {"model": body.value, "raw": raw}

        @app.post("/sync_tool")
        def sync_tool(body: EchoBody):
            return {"upper": body.value.upper()}

        @app.post("/filtered", response_model=PublicView)
        async def filtered(body: EchoBody):
            return {"shown": body.value, "secret": "leak"}  # pragma: allowlist secret

        @app.post("/explode")
        async def explode():
            raise RuntimeError("kaboom")

        @app.post("/teapot")
        async def teapot():
            raise HTTPException(status_code=418, detail="short and stout")

        @app.post("/bad_status")
        async def bad_status() -> PlainTextResponse:
            return PlainTextResponse("nope", status_code=400)

        @app.post("/plain_ok")
        async def plain_ok() -> PlainTextResponse:
            return PlainTextResponse("plain hello")

        return app


def _server(cls=Store, name="store", expose=True) -> SimpleResourcesServer:
    cfg = BaseResourcesServerConfig(host="", port=0, entrypoint="", name=name, expose_tools_over_mcp=expose)
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


@contextmanager
def _mcp(server_cls=Store, name="store"):
    """Install auto-exposure, start the app, seed a session, and hand back (client, token)."""
    server = _server(server_cls, name)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        token = _seed(client)
        _handshake(client)
        yield client, token


def _payload(result: dict) -> Any:
    assert result.get("isError") is not True, result
    return json.loads(result["content"][0]["text"])


# ==================================================================================================
# The flag gate + mounting
# ==================================================================================================


def test_flag_off_does_not_mount_mcp():
    server = _server(expose=False)
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


def test_refuses_server_that_already_mounts_mcp():
    server = _server()
    app = server.setup_webserver()

    async def existing_mcp(scope, receive, send):  # a hand-rolled /mcp mount
        pass

    app.router.routes.append(Mount("/mcp", app=existing_mcp))
    with pytest.raises(ValueError, match="already serves /mcp"):
        install_auto_exposure(server, app)


def test_server_utils_imports_mcp_auto_exposure_lazily():
    # Agents and models import server_utils; the MCP SDK must not come along for the ride.
    code = (
        "import sys\n"
        "import nemo_gym.server_utils\n"
        "assert 'nemo_gym.mcp_auto_exposure' not in sys.modules, 'mcp_auto_exposure imported eagerly'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_run_webserver_guards_maybe_auto_expose_behind_flag():
    """run_webserver spins up ray + config, so assert on its AST: both the lazy import and the
    maybe_auto_expose call must sit inside the expose_tools_over_mcp guard."""
    module_tree = ast.parse(inspect.getsource(sys.modules[SimpleServer.__module__]))
    tree = next(
        node
        for cls in ast.walk(module_tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "SimpleServer"
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_webserver"
    )
    call_guards: list[list[str]] = []
    import_guards: list[list[str]] = []

    def visit(node: ast.AST, guards: list[str]) -> None:
        if isinstance(node, ast.Call):
            fn = node.func
            fn_name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if fn_name == "maybe_auto_expose":
                call_guards.append(list(guards))
        if isinstance(node, ast.ImportFrom) and node.module == "nemo_gym.mcp_auto_exposure":
            import_guards.append(list(guards))
        for child in ast.iter_child_nodes(node):
            child_guards = guards
            if isinstance(node, ast.If) and child in node.body:
                child_guards = guards + [ast.unparse(node.test)]
            visit(child, child_guards)

    visit(tree, [])
    assert call_guards, "run_webserver no longer calls maybe_auto_expose"
    assert import_guards, "run_webserver no longer imports mcp_auto_exposure lazily"
    for guards in call_guards + import_guards:
        assert any("expose_tools_over_mcp" in g for g in guards), guards


# ==================================================================================================
# tools/list + tools/call over the real /mcp endpoint
# ==================================================================================================


def test_tools_list_advertises_typed_schema():
    with _mcp() as (client, token):
        tools = {t["name"]: t for t in _list(client, token)}
        assert sorted(tools["append"]["inputSchema"]["properties"]) == ["value"]
        assert tools["append"]["description"].startswith("Append a value")


def test_direct_dispatch_runs_handler_and_shares_session_with_plain_http_route():
    with _mcp() as (client, token):
        # Plain HTTP route (cookie) then MCP (token) — same seeded session id, so state accumulates.
        client.post("/append", json={"value": "a"})
        result = _call(client, "append", {"value": "b"}, token=token)
        assert result.get("isError") is not True
        assert json.loads(result["content"][0]["text"])["values"] == ["a", "b"]


def test_dict_body_tool_dispatches_direct():
    with _mcp() as (client, token):
        result = _call(client, "raw_step", {"anything": [1, 2]}, token=token)
        assert json.loads(result["content"][0]["text"])["echo"] == {"anything": [1, 2]}


def test_raw_body_catchall_dispatches_and_unwraps_plaintext():
    with _mcp() as (client, token):
        result = _call(client, "lookup", {"q": "iron"}, token=token)
        payload = json.loads(result["content"][0]["text"])
        assert payload == {"tool": "lookup", "args": {"q": "iron"}}


def test_error_mapping():
    with _mcp() as (client, token):
        r = _call(client, "nope", {}, token=token)
        assert r["isError"] is True and "Unknown tool" in r["content"][0]["text"]
        r = _call(client, "append", {"value": "x"}, token=None)
        assert r["isError"] is True and TOKEN_HEADER in r["content"][0]["text"]
        r = _call(client, "append", {"value": "x"}, token="garbage")
        assert r["isError"] is True and "Invalid" in r["content"][0]["text"]
        # malformed args -> the handler's own 422
        r = _call(client, "append", {"wrong": "field"}, token=token)
        assert r["isError"] is True and "422" in r["content"][0]["text"]


# ==================================================================================================
# Direct-dispatch parity: each handler shape returns what the plain HTTP route would have
# ==================================================================================================


def test_optional_body_model_is_refused_at_install():
    # Optional[Model] = None has no proven-equivalent direct dispatch, so exposure refuses by name.
    class OptionalBody(Store):
        def setup_webserver(self) -> FastAPI:
            app = super().setup_webserver()

            @app.post("/opt_body")
            async def opt_body(body: Optional[EchoBody] = None):
                return {"got": None if body is None else body.value}

            return app

    server = _server(OptionalBody)
    with pytest.raises(ValueError, match="opt_body"):
        install_auto_exposure(server, server.setup_webserver())


def test_optional_body_model_bind_route_refuses_with_reason():
    async def opt_body(body: Optional[EchoBody] = None):
        pass

    bound = bind_route(_stub_route(opt_body))
    assert bound.binding is None
    assert any("union/optional body param" in r for r in bound.reasons), bound.reasons


def test_parameterized_dict_body_dispatches():
    with _mcp(Shapes, "shapes") as (client, token):
        payload = _payload(_call(client, "typed_dict_body", {"k": [1, 2]}, token=token))
        assert payload == {"echo": {"k": [1, 2]}}


def test_body_model_handler_can_also_read_request_json():
    # The fabricated Request must carry the same bytes the body model was validated from.
    with _mcp(Shapes, "shapes") as (client, token):
        payload = _payload(_call(client, "model_and_raw", {"value": "x"}, token=token))
        assert payload == {"model": "x", "raw": {"value": "x"}}


def test_sync_def_handler_dispatches_correctly():
    with _mcp(Shapes, "shapes") as (client, token):
        payload = _payload(_call(client, "sync_tool", {"value": "ab"}, token=token))
        assert payload == {"upper": "AB"}


def test_defaulted_query_param_is_refused_at_install():
    class Defaulted(Store):
        def setup_webserver(self):
            app = super().setup_webserver()

            @app.post("/with_default")
            async def with_default(body: EchoBody, limit: int = 3):
                return {"value": body.value, "limit": limit}

            return app

    server = _server(Defaulted)
    app = server.setup_webserver()
    with pytest.raises(ValueError, match="with_default"):
        install_auto_exposure(server, app)


def test_response_model_filters_extra_fields():
    with _mcp(Shapes, "shapes") as (client, token):
        payload = _payload(_call(client, "filtered", {"value": "v"}, token=token))
        assert payload == {"shown": "v"}


def test_unexpected_handler_exception_maps_to_is_error():
    with _mcp(Shapes, "shapes") as (client, token):
        r = _call(client, "explode", {}, token=token)
        assert r["isError"] is True
        text = r["content"][0]["text"]
        assert "HTTP 500" in text and "kaboom" in text


def test_http_exception_keeps_status_and_detail():
    with _mcp(Shapes, "shapes") as (client, token):
        r = _call(client, "teapot", {}, token=token)
        assert r["isError"] is True
        text = r["content"][0]["text"]
        assert "HTTP 418" in text and "short and stout" in text


def test_non_2xx_response_maps_to_is_error():
    with _mcp(Shapes, "shapes") as (client, token):
        r = _call(client, "bad_status", {}, token=token)
        assert r["isError"] is True
        text = r["content"][0]["text"]
        assert "HTTP 400" in text and "nope" in text


def test_non_json_2xx_response_passes_through_as_text():
    with _mcp(Shapes, "shapes") as (client, token):
        r = _call(client, "plain_ok", {}, token=token)
        assert r.get("isError") is not True
        assert r["content"][0]["text"] == "plain hello"


def test_sequential_calls_keep_sessions_isolated_and_ordered():
    server = _server()
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        token_a = _seed(client)
        client.cookies.clear()  # a fresh seed mints a distinct session id
        token_b = _seed(client)
        _handshake(client)
        for v in ("a1", "a2"):
            _call(client, "append", {"value": v}, token=token_a)
        _call(client, "append", {"value": "b1"}, token=token_b)
        ra = _payload(_call(client, "append", {"value": "a3"}, token=token_a))
        rb = _payload(_call(client, "append", {"value": "b2"}, token=token_b))
        assert ra["values"] == ["a1", "a2", "a3"]
        assert rb["values"] == ["b1", "b2"]


# ==================================================================================================
# Session tokens: tokenless callers, garbage tokens
# ==================================================================================================


def test_tokenless_and_garbage_token_list_all_tools():
    with _mcp() as (client, _token):
        full = {t["name"] for t in _list(client, token=None)}
        assert {"append", "raw_step", "lookup"} <= full
        # tools/list treats the token as optional, so a bad one degrades to tokenless, not an error
        assert {t["name"] for t in _list(client, token="garbage")} == full


# ==================================================================================================
# Per-session tool restriction: mcp_allowed_tools_for_session(seed_body)
# ==================================================================================================


class SessionScoped(Store):
    def mcp_allowed_tools_for_session(self, seed_body: dict) -> Optional[list[str]]:
        return seed_body.get("allowed_tools")


def test_session_hook_returning_none_mints_unrestricted_token():
    with _mcp(SessionScoped) as (client, token):  # _seed posts {} -> hook returns None
        assert {"append", "raw_step", "lookup"} <= {t["name"] for t in _list(client, token)}
        payload = _payload(_call(client, "raw_step", {"k": 1}, token=token))
        assert payload == {"echo": {"k": 1}}


def test_session_hook_restricts_that_sessions_token():
    server = _server(SessionScoped)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        resp = client.post("/seed_session", json={"allowed_tools": ["append"]})
        token = resp.json()["mcp"]["headers"][TOKEN_HEADER]
        _handshake(client)
        assert {t["name"] for t in _list(client, token)} == {"append"}
        blocked = _call(client, "raw_step", {}, token=token)
        assert blocked["isError"] is True and "not allowed" in blocked["content"][0]["text"]
        assert _payload(_call(client, "append", {"value": "x"}, token=token))["values"] == ["x"]


def test_session_hook_error_fails_seed_request_not_silent_unrestricted():
    class BrokenHook(Store):
        def mcp_allowed_tools_for_session(self, seed_body: dict) -> Optional[list[str]]:
            raise RuntimeError("hook boom")

    server = _server(BrokenHook)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/seed_session", json={})
        assert resp.status_code >= 500
        assert TOKEN_HEADER not in resp.text


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


def test_dispatcher_ignoring_catchall_soft_warns_instead_of_raising(caplog):
    class NoInventoryDispatcher(SimpleResourcesServer):
        async def verify(self, body):
            pass

        def setup_webserver(self) -> FastAPI:
            app = super().setup_webserver()

            @app.post("/{tool_name}")
            async def dispatch(tool_name: str, request: Request):
                return {}

            return app

    server = _server(NoInventoryDispatcher, "dispatcher")  # default mcp_tools ignores the catch-all
    with caplog.at_level(logging.WARNING):
        tools = install_auto_exposure(server, server.setup_webserver())
    assert tools == {}
    assert "catch-all" in caplog.text


def test_refuses_reserved_tool_name():
    class ReservedInventory(Store):
        def mcp_tools(self, harvested, catchall):
            return harvested + [catchall.tool("verify")]

    server = _server(ReservedInventory)
    with pytest.raises(ValueError, match="reserved"):
        install_auto_exposure(server, server.setup_webserver())


class Excluding(SimpleResourcesServer):
    async def verify(self, body):
        pass

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.post("/append")
        async def append(body: EchoBody):
            return {"value": body.value}

        @app.post("/end_session")
        async def end_session(body: EchoBody):
            return {"ended": body.value}

        return app

    def mcp_tools(self, harvested, catchall):
        return [t for t in harvested if t.name != "end_session"]


def test_excluded_route_is_not_a_tool_but_plain_http_still_works():
    with _mcp(Excluding, "excl") as (client, token):
        assert {t["name"] for t in _list(client, token)} == {"append"}
        r = _call(client, "end_session", {"value": "x"}, token=token)
        assert r["isError"] is True and "Unknown tool" in r["content"][0]["text"]
        resp = client.post("/end_session", json={"value": "x"})
        assert resp.status_code == 200 and resp.json() == {"ended": "x"}


def test_excluded_route_with_depends_param_does_not_refuse():
    # A route dropped by mcp_tools() is never required to be dispatchable, so its Depends param
    # (which direct dispatch cannot reproduce) does not refuse exposure of the surviving tools.
    class ExcludedDepends(Excluding):
        def setup_webserver(self) -> FastAPI:
            app = super().setup_webserver()

            @app.post("/gated")
            async def gated(ok: bool = Depends(lambda: True)):
                return {"ok": ok}

            return app

        def mcp_tools(self, harvested, catchall):
            return [t for t in harvested if t.name not in ("end_session", "gated")]

    server = _server(ExcludedDepends, "excl")
    tools = install_auto_exposure(server, server.setup_webserver())
    assert set(tools) == {"append"}


def test_undispatchable_route_keeps_typed_schema_in_harvested_list():
    # The body model survives a failed bind, so an mcp_tools() override inspecting the harvested
    # list sees the real schema even for a tool it must drop.
    harvested_schemas: dict[str, dict] = {}

    class DroppedTyped(Excluding):
        def setup_webserver(self) -> FastAPI:
            app = super().setup_webserver()

            @app.post("/gated")
            async def gated(body: EchoBody, ok: bool = Depends(lambda: True)):
                return {"ok": ok}

            return app

        def mcp_tools(self, harvested, catchall):
            harvested_schemas.update({t.name: t.tool.inputSchema for t in harvested})
            return [t for t in harvested if t.name not in ("end_session", "gated")]

    server = _server(DroppedTyped, "excl")
    tools = install_auto_exposure(server, server.setup_webserver())
    assert set(tools) == {"append"}
    assert sorted(harvested_schemas["gated"]["properties"]) == ["value"]


def test_refuses_duplicate_tool_name():
    class DuplicateInventory(Store):
        def mcp_tools(self, harvested, catchall):
            return harvested + [catchall.tool("append")]

    server = _server(DuplicateInventory)
    with pytest.raises(ValueError, match="Duplicate MCP tool name"):
        install_auto_exposure(server, server.setup_webserver())


def test_refuses_nested_tool_route():
    class NestedRoute(SimpleResourcesServer):
        async def verify(self, body):
            pass

        def setup_webserver(self) -> FastAPI:
            app = super().setup_webserver()

            @app.post("/a/b")
            async def nested(body: EchoBody):
                return {}

            return app

    server = _server(NestedRoute, "nested")
    with pytest.raises(ValueError, match="does not match"):
        install_auto_exposure(server, server.setup_webserver())


def test_missing_seed_session_raises_a_clear_error():
    server = _server()
    app = server.setup_webserver()
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/seed_session"]
    with pytest.raises(ValueError, match="seed_session"):
        install_auto_exposure(server, app)


# ==================================================================================================
# The detector: bind_route classification of accepted and refused shapes
# ==================================================================================================


def _stub_route(endpoint, path="/t", response_model=None):
    """bind_route only reads endpoint/path/response_model, so a stub covers shapes FastAPI itself
    would reject at registration."""
    return SimpleNamespace(endpoint=endpoint, path=path, response_model=response_model)


def test_bind_route_refusal_reasons():
    async def var_args(*args):
        pass

    async def two_models(a: EchoBody, b: OtherBody):
        pass

    async def ambiguous_union(body: EchoBody | str):
        pass

    async def di_default(ok: bool = Depends(lambda: True)):
        pass

    async def bare_required(x: int):
        pass

    cases = {
        "*args/**kwargs": var_args,
        "multiple body models": two_models,
        "union/optional body param": ambiguous_union,
        "DI marker default": di_default,
        "unsupported required param": bare_required,
    }
    for expected, endpoint in cases.items():
        bound = bind_route(_stub_route(endpoint))
        assert bound.binding is None, expected
        assert any(expected in reason for reason in bound.reasons), (expected, bound.reasons)


def test_bind_route_refuses_defaulted_query_param():
    async def handler(body: EchoBody, limit: int = 5):
        pass

    bound = bind_route(_stub_route(handler))
    assert bound.binding is None
    assert any("defaulted query param" in r for r in bound.reasons), bound.reasons


def test_silently_wrong_shapes_are_classified_not_degraded():
    """The shapes that would dispatch wrongly if misclassified: response_model is recorded for
    filtering, and a sync (def) handler is recorded as is_coroutine=False."""
    server = _server(Shapes, "shapes")
    app = server.setup_webserver()
    routes = {r.path: r for r in app.routes if isinstance(r, APIRoute)}

    filt = bind_route(routes["/filtered"]).binding
    assert filt is not None and filt.return_model is PublicView

    sync = bind_route(routes["/sync_tool"]).binding
    assert sync is not None and sync.is_coroutine is False


# ==================================================================================================
# The detector's annotation resolution (regression: factory-set __signature__ must win)
# ==================================================================================================


def test_bind_route_honors_factory_signature_over_annotations():
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
    bound = bind_route(route)
    assert bound.binding is not None, bound.reasons
    assert bound.binding.body_model is EchoBody


# ==================================================================================================
# Verify-time tool-name normalization (scoring-only, gated on expose_tools_over_mcp)
# ==================================================================================================


def _verify_body(names: list[str]) -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "x"}]},
        "response": {
            "id": "resp_x",
            "created_at": 0.0,
            "model": "m",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {"type": "function_call", "name": n, "arguments": "{}", "call_id": f"c{i}"}
                for i, n in enumerate(names)
            ],
        },
    }


def test_verify_normalizes_mcp_namespaced_tool_names():
    """MCP-driven rollouts record tool calls as mcp__<server>__<tool>; verify must see bare names,
    but the echoed response must keep what the model emitted (transport provenance preserved)."""
    seen: dict[str, list] = {}

    class Recorder(Store):
        async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
            seen["names"] = [o.name for o in body.response.output if o.type == "function_call"]
            return BaseVerifyResponse(**body.model_dump(), reward=1.0)

    server = _server(Recorder, name="store")  # _server defaults expose_tools_over_mcp = True in the config
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        emitted = ["mcp__store__append", "raw_step", "mcp__other__tool"]
        resp = client.post("/verify", json=_verify_body(emitted))
        assert resp.status_code == 200, resp.text
        echoed = [o["name"] for o in resp.json()["response"]["output"] if o["type"] == "function_call"]
    # verify SAW: this server's prefix stripped, bare names untouched, other servers' prefixes left alone
    assert seen["names"] == ["append", "raw_step", "mcp__other__tool"]
    assert echoed == emitted


def test_litmus_pattern_reregistered_verify_is_still_normalized():
    """Servers that strip and re-register /verify (litmus_agent pattern) get the install-time wrap
    on their own handler: it sees bare names, and the response restores the emitted ones."""
    seen: dict[str, list] = {}

    class LitmusStore(Store):
        def mcp_tools(self, harvested, catchall):  # the catch-all is dropped below, so expose only typed routes
            return harvested

        def setup_webserver(self) -> FastAPI:
            app = super().setup_webserver()

            async def verify_and_cleanup(body: BaseVerifyRequest) -> BaseVerifyResponse:
                seen["names"] = [o.name for o in body.response.output if o.type == "function_call"]
                return BaseVerifyResponse(**body.model_dump(), reward=1.0)

            # The catch-all is dropped too: it would shadow the re-appended /verify (litmus has none).
            app.router.routes[:] = [
                r for r in app.router.routes if getattr(r, "path", None) not in ("/verify", "/{tool_name}")
            ]
            app.post("/verify")(verify_and_cleanup)
            return app

    server = _server(LitmusStore, name="store")
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        emitted = ["mcp__store__append", "raw_step", "mcp__other__tool"]
        resp = client.post("/verify", json=_verify_body(emitted))
        assert resp.status_code == 200, resp.text
        echoed = [o["name"] for o in resp.json()["response"]["output"] if o["type"] == "function_call"]
    assert seen["names"] == ["append", "raw_step", "mcp__other__tool"]
    assert echoed == emitted


def test_flag_on_server_without_verify_route_refuses_at_install():
    server = _server()
    app = server.setup_webserver()
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/verify"]
    with pytest.raises(ValueError, match="/verify route"):
        install_auto_exposure(server, app)


def test_verify_does_not_normalize_when_mcp_exposure_off():
    """Flag off (the default for every existing benchmark): verify is byte-identical, no rewrite."""
    seen: dict[str, list] = {}

    class Plain(Store):
        async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
            seen["names"] = [o.name for o in body.response.output if o.type == "function_call"]
            return BaseVerifyResponse(**body.model_dump(), reward=1.0)

    server = _server(Plain, name="store", expose=False)
    app = server.setup_webserver()
    with TestClient(app) as client:
        emitted = ["mcp__store__append", "raw_step"]
        resp = client.post("/verify", json=_verify_body(emitted))
        assert resp.status_code == 200, resp.text
    assert seen["names"] == emitted


def test_normalize_tool_name_without_server_name_strips_first_namespace():
    assert normalize_tool_name("mcp__store__append") == "append"
    # only the first separator is the namespace boundary; the tool's own underscores survive
    assert normalize_tool_name("mcp__store__ns__tool") == "ns__tool"
    # no tool part after the prefix -> not a namespaced name, unchanged
    assert normalize_tool_name("mcp__dangling") == "mcp__dangling"
    assert normalize_tool_name("plain") == "plain"
    # with a server name, only that server's prefix is stripped
    assert normalize_tool_name("mcp__store__append", "store") == "append"
    assert normalize_tool_name("mcp__other__append", "store") == "mcp__other__append"
