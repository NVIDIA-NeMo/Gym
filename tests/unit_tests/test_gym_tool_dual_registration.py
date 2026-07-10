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
"""Dual-registration test suite: every gym_tool is served over BOTH transports (HTTP POST + MCP).

Covers the E-tests from the single-entry-point design review: the full JSON-RPC handshake (E1),
raw-argument fidelity for dict/model-schema tools (E1b), nested-model parity across transports (E3),
Annotated/BeforeValidator/Field fidelity (E4), the allowed_tools claim (E5), the unknown-tool
catch-all (E6), seed_session auto-augmentation (E7), the contract long tail (E8), and transport
parity (E9). All MCP traffic runs through TestClient (in-memory ASGI) — no network sockets.
"""

import json
import logging
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, BeforeValidator, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    SimpleResourcesServer,
    gym_tool,
)
from nemo_gym.mcp_test_utils import (
    TOKEN_HEADER,
    assert_transport_parity,
    mcp_call,
    mcp_handshake,
    mcp_list_tools,
    mcp_result_payload,
    seed_token,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


pytest.importorskip("mcp")

SEND_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "recipient": {"type": "string", "description": "Who to message"},
        "message": {"type": "string"},
    },
    "required": ["recipient", "message"],
    "additionalProperties": False,
}


class Point(BaseModel):
    x: int
    y: int


class LocateResponse(BaseModel):
    param_type: str
    x: int


class RepeatBody(BaseModel):
    count: int
    label: str


def _parse_int_list(value: Any) -> Any:
    if isinstance(value, str):
        return [int(part) for part in value.split(",")]
    return value


class _DualServer(SimpleResourcesServer):
    """One server exercising all three input_schema modes plus session state."""

    session_state: dict[str, Any] = Field(default_factory=dict)
    observed_raw_args: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        def send_message(session_id: str, **arguments: Any) -> dict:
            self.observed_raw_args = dict(arguments)
            return {"delivered": True, "observed": dict(arguments)}

        gym_tool(
            send_message,
            name="send_message",
            description="Send a message.",
            input_schema=SEND_MESSAGE_SCHEMA,
            owner=self,
        )

        def repeat(body: RepeatBody) -> str:
            return body.label * body.count

        gym_tool(repeat, name="repeat", description="Repeat a label.", input_schema=RepeatBody, owner=self)

    @gym_tool
    async def bump(self, session_id: str, amount: int) -> dict:
        """Increase this session's counter."""
        counter = self.session_state.setdefault(session_id, 0) + amount
        self.session_state[session_id] = counter
        return {"session_id": session_id, "counter": counter}

    @gym_tool
    async def locate(self, point: Point) -> LocateResponse:
        """Report the type the nested param actually arrived as."""
        return LocateResponse(param_type=type(point).__name__, x=getattr(point, "x", -1))

    @gym_tool
    def greet(self, name: str) -> str:
        """Sync tool returning a bare string."""
        return f"hello {name}"

    @gym_tool
    async def coerce(
        self,
        values: Annotated[list[int], BeforeValidator(_parse_int_list)],
        detail: Annotated[str, Field(description="Detail marker")] = "d",
    ) -> dict:
        """Annotated params: validator coercion + Field description must survive both transports."""
        return {"values": values, "detail": detail}

    async def verify(self, body: BaseVerifyRequest):
        pass


class _ToolLessServer(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest):
        pass


def _make(server_cls, name: str = "dual_server"):
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name=name)
    return server_cls(config=config, server_client=MagicMock(spec=ServerClient))


class TestFullHandshake:
    """E1/E1b: the full JSON-RPC lifecycle, including raw-argument fidelity for schema tools."""

    def test_initialize_list_call_lifecycle(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            token = seed_token(client)
            mcp_handshake(client, token=token)

            tools = mcp_list_tools(client, token=token)
            assert set(tools) == {"bump", "locate", "greet", "coerce", "send_message", "repeat"}
            # The dict schema is advertised byte-for-byte verbatim.
            assert tools["send_message"]["inputSchema"] == SEND_MESSAGE_SCHEMA
            # The model-class schema is the model's own JSON schema (fields as top-level properties).
            assert set(tools["repeat"]["inputSchema"]["properties"]) == {"count", "label"}
            # session_id is never model-visible.
            assert "session_id" not in tools["bump"]["inputSchema"]["properties"]

            result = mcp_call(client, "bump", {"amount": 5}, token=token)
            assert result.get("isError") is not True
            assert mcp_result_payload(result)["counter"] == 5

    def test_dict_schema_tool_receives_out_of_schema_args_verbatim(self) -> None:
        """The exact case FastMCP validation silently corrupts: undeclared arg names must survive."""
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            token = seed_token(client)
            result = mcp_call(client, "send_message", {"to": "alice", "message": "hi"}, token=token)
            assert result.get("isError") is not True
            assert server.observed_raw_args == {"to": "alice", "message": "hi"}

    def test_model_schema_tool_validates_and_passes_instance(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            ok = mcp_call(client, "repeat", {"count": 3, "label": "ab"})
            assert ok.get("isError") is not True
            assert ok["content"][0]["text"] == "ababab"

            bad = mcp_call(client, "repeat", {"count": "not-an-int", "label": "ab"})
            assert bad["isError"] is True
            assert "count" in bad["content"][0]["text"]


class TestCrossTransportParity:
    """E3/E4 + the promoted cookie-and-token-same-session test."""

    def test_nested_model_param_arrives_as_model_on_both_transports(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            http = client.post("/locate", json={"point": {"x": 3, "y": 4}})
            assert http.status_code == 200, http.text
            assert http.json()["param_type"] == "Point"

            mcp = mcp_call(client, "locate", {"point": {"x": 3, "y": 4}})
            assert mcp.get("isError") is not True
            assert mcp["structuredContent"]["param_type"] == "Point"

    def test_annotated_validator_and_description_survive_both_transports(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            # BeforeValidator coerces the comma string on the HTTP body model...
            http = client.post("/coerce", json={"values": "1,2,3"})
            assert http.status_code == 200, http.text
            assert http.json()["values"] == [1, 2, 3]

            # ...and on the MCP argument model; the Field description reaches the MCP schema.
            tools = mcp_list_tools(client)
            assert tools["coerce"]["inputSchema"]["properties"]["detail"]["description"] == "Detail marker"
            mcp = mcp_call(client, "coerce", {"values": "4,5"})
            assert mcp.get("isError") is not True
            assert mcp_result_payload(mcp)["values"] == [4, 5]

    def test_cookie_and_token_resolve_the_same_session(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            token = seed_token(client)

            http = client.post("/bump", json={"amount": 5})
            assert http.status_code == 200, http.text
            http_sid, counter = http.json()["session_id"], http.json()["counter"]
            assert counter == 5

            mcp = mcp_call(client, "bump", {"amount": 3}, token=token)
            assert mcp_result_payload(mcp)["session_id"] == http_sid
            assert mcp_result_payload(mcp)["counter"] == 8


class TestAllowedToolsClaim:
    """E5: the signed allowed_tools claim filters tools/list AND gates tools/call (MCP only)."""

    def _restricted_token(self, server) -> str:
        request = MagicMock(spec=Request)
        request.session = {SESSION_ID_KEY: "restricted-session"}
        return server.build_mcp_session_metadata(request, allowed_tools=["bump"]).headers[TOKEN_HEADER]

    def test_list_filtered_and_call_gated(self) -> None:
        server = _make(_DualServer)
        token = self._restricted_token(server)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            assert set(mcp_list_tools(client, token=token)) == {"bump"}

            blocked = mcp_call(client, "greet", {"name": "eve"}, token=token)
            assert blocked["isError"] is True
            assert "not available" in blocked["content"][0]["text"]

            # Status-quo HTTP behavior: the claim narrows the per-session MCP view only.
            assert client.post("/greet", json={"name": "eve"}).status_code == 200

            allowed = mcp_call(client, "bump", {"amount": 2}, token=token)
            assert allowed.get("isError") is not True
            assert mcp_result_payload(allowed)["session_id"] == "restricted-session"

    def test_legacy_bare_token_and_no_token_list_everything(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            bare_token = seed_token(client)  # bare payload: no allowed_tools claim
            assert len(mcp_list_tools(client, token=bare_token)) == 6
            assert len(mcp_list_tools(client, token=None)) == 6


class TestUnknownToolCatchall:
    """E6: catch-all ordering, idempotence, and the informative default miss-path."""

    def test_default_404_lists_available_tools(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/no_such_tool", json={})
            assert resp.status_code == 404
            assert "Available tools" in resp.json()["error"]
            assert "bump" in resp.json()["error"]

    def test_catchall_absent_on_tool_less_servers(self) -> None:
        server = _make(_ToolLessServer, name="tool_less")
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/no_such_tool", json={})
            assert resp.status_code == 404
            assert resp.json() == {"detail": "Not Found"}  # FastAPI stock 404, no catch-all mounted

    def test_catchall_does_not_shadow_subclass_routes_and_registers_once(self) -> None:
        class _WithExtraRoute(_DualServer):
            def setup_webserver(self):
                app = super().setup_webserver()
                app.post("/extra")(self._extra)
                return app

            async def _extra(self) -> dict:
                return {"extra": True}

        server = _make(_WithExtraRoute)
        app = server.setup_webserver()
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            # The catch-all lands after the subclass route (registered post-super) — no shadowing.
            assert client.post("/extra", json={}).json() == {"extra": True}
            assert client.post("/nope", json={}).status_code == 404
        # Repeated installation attempts must not accumulate duplicates. (A second full lifespan
        # cycle on one app is impossible anyway — the SDK's session manager is single-use — so the
        # guard is exercised directly.)
        server._ensure_unknown_tool_catchall(app)
        server._ensure_unknown_tool_catchall(app)
        catchalls = [r for r in app.router.routes if getattr(r, "name", None) == "_gym_unknown_tool_catchall"]
        assert len(catchalls) == 1

    def test_override_preserves_historical_bytes(self) -> None:
        class _SoftErrorServer(_DualServer):
            async def handle_unknown_tool(self, tool_name: str, request: Request):
                return {"results": f"Error executing tool '{tool_name}': unknown"}

        server = _make(_SoftErrorServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/no_such_tool", json={})
            assert resp.status_code == 200
            assert resp.json() == {"results": "Error executing tool 'no_such_tool': unknown"}


class TestSeedSessionAutoAugment:
    """E7: the seed wrapper matrix, asserted over the wire (response_model must not strip the key)."""

    def test_inherited_default_seed_gets_mcp_injected(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            body = client.post("/seed_session", json={}).json()
            assert body["mcp"]["url_path"] == "/mcp"
            assert TOKEN_HEADER in body["mcp"]["headers"]

    def test_override_with_extra_fields_keeps_both(self) -> None:
        class _SeedBody(BaseSeedSessionRequest):
            task_id: str

        class _Custom(_DualServer):
            async def seed_session(self, request: Request, body: _SeedBody) -> BaseSeedSessionResponse:
                return {"greeting": "hello", "task_id": body.task_id}  # type: ignore[return-value]

        server = _make(_Custom)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            body = client.post("/seed_session", json={"task_id": "task-42"}).json()
            assert body["greeting"] == "hello"
            assert body["task_id"] == "task-42"
            assert "mcp" in body

            # The adopted signature keeps FastAPI body validation live through the wrapper.
            assert client.post("/seed_session", json={}).status_code == 422

    def test_existing_mcp_value_is_never_clobbered(self) -> None:
        custom_mcp = {"server_name": "mine", "url_path": "/mcp", "transport": "http", "headers": {"k": "v"}}

        class _OwnMCP(_DualServer):
            async def seed_session(self, request: Request, body: BaseSeedSessionRequest):
                return {"mcp": custom_mcp, "note": "mine"}

        server = _make(_OwnMCP)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            body = client.post("/seed_session", json={}).json()
            assert body["mcp"] == custom_mcp
            assert body["note"] == "mine"

    def test_tool_less_server_seed_response_is_unchanged(self) -> None:
        server = _make(_ToolLessServer, name="tool_less")
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            body = client.post("/seed_session", json={}).json()
            assert body == {}  # plain method, no wrapper, no injected key


class TestContractLongTail:
    """E8: str->text/plain, 422s, session_id hygiene, lazy gate, deletion, schema names."""

    def test_str_return_is_text_plain_over_http(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/greet", json={"name": "sam"})
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/plain")
            assert resp.text == "hello sam"

    def test_typed_route_validates_body(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            assert client.post("/bump", json={}).status_code == 422
            assert client.post("/bump", json={"amount": "NaN"}).status_code == 422

    def test_session_id_not_injectable_from_http_payload(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            client.post("/seed_session", json={})
            resp = client.post("/bump", json={"amount": 1, "session_id": "ATTACKER"})
            # Typed body model has no session_id field -> 422 (undeclared key with strict body)
            # or, if tolerated, the cookie session must win. Either way ATTACKER never reaches state.
            if resp.status_code == 200:
                assert resp.json()["session_id"] != "ATTACKER"
            assert "ATTACKER" not in server.session_state

            resp = client.post("/send_message", json={"session_id": "ATTACKER", "message": "x"})
            assert resp.status_code == 200
            assert "session_id" not in server.observed_raw_args

    def test_tool_less_server_has_no_mcp_surface(self) -> None:
        server = _make(_ToolLessServer, name="tool_less")
        app = server.setup_webserver()
        paths = {getattr(route, "path", None) for route in app.routes}
        assert "/mcp" not in paths
        assert {"/seed_session", "/verify", "/aggregate_metrics"} <= paths

    def test_mcp_resources_server_is_fully_deleted(self) -> None:
        with pytest.raises(ImportError):
            from nemo_gym.base_resources_server import MCPResourcesServer  # noqa: F401

    def test_duplicate_tool_names_rejected(self) -> None:
        server = _make(_DualServer)

        def bump(session_id: str, **arguments: Any) -> str:
            return "dup"

        gym_tool(bump, name="bump", input_schema={"type": "object"}, owner=server)
        with pytest.raises(ValueError, match="Duplicate gym_tool name"):
            server.setup_webserver()

    def test_reserved_runtime_tool_name_rejected(self) -> None:
        server = _make(_DualServer)

        def verify_impersonator(**arguments: Any) -> str:
            return "nope"

        gym_tool(verify_impersonator, name="verify", input_schema={"type": "object"}, owner=server)
        with pytest.raises(ValueError, match="reserved endpoint name"):
            server.setup_webserver()

    def test_synth_body_model_names_are_namespaced_in_openapi(self) -> None:
        server = _make(_DualServer)
        app = server.setup_webserver()
        schemas = app.openapi()["components"]["schemas"]
        assert "bump__GymToolBody" in schemas


class TestTransportParity:
    """E9: the tested invariant — unfiltered tools/list == HTTP tool routes == expected inventory."""

    EXPECTED = {"bump", "locate", "greet", "coerce", "send_message", "repeat"}

    def test_tool_sets_identical_across_transports(self) -> None:
        server = _make(_DualServer)
        app = server.setup_webserver()
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            assert_transport_parity(app, client, self.EXPECTED)

    def test_manual_mcp_only_tool_triggers_parity_warning(self, caplog) -> None:
        class _ManualServer(_DualServer):
            def register_mcp_tools(self, mcp):
                super().register_mcp_tools(mcp)

                @mcp.tool()
                def ghost() -> str:
                    return "mcp-only"

        server = _make(_ManualServer)
        with caplog.at_level(logging.WARNING, logger="nemo_gym.base_resources_server"):
            server.setup_webserver()
        assert any("ghost" in record.getMessage() for record in caplog.records)


class TestModeCoverage:
    """The remaining mode branches: BaseModel-mode over HTTP, validate=True, return shapes."""

    def test_model_schema_tool_over_http_validates_like_a_handwritten_route(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            ok = client.post("/repeat", json={"count": 2, "label": "xy"})
            assert ok.status_code == 200
            assert ok.text == "xyxy"  # str return -> text/plain
            assert client.post("/repeat", json={"count": "bad", "label": "xy"}).status_code == 422

    def test_validated_dict_tool_gates_both_transports_but_passes_raw(self) -> None:
        class _Validated(_DualServer):
            def model_post_init(self, context: Any) -> None:
                super().model_post_init(context)

                async def strict_echo(**arguments: Any) -> list:
                    return sorted(arguments)

                gym_tool(
                    strict_echo,
                    name="strict_echo",
                    input_schema={
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                    },
                    validate=True,
                    owner=self,
                )

        server = _make(_Validated)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            # Shallow validation gates the declared property's type...
            assert client.post("/strict_echo", json={"n": "not-an-int"}).status_code == 422
            bad = mcp_call(client, "strict_echo", {"n": "not-an-int"})
            assert bad["isError"] is True
            # ...but undeclared args still pass through raw (list return -> JSON text block).
            ok = client.post("/strict_echo", json={"n": 1, "extra": "kept"})
            assert ok.status_code == 200
            assert ok.json() == ["extra", "n"]
            mcp_ok = mcp_call(client, "strict_echo", {"n": 1, "extra": "kept"})
            assert mcp_ok.get("isError") is not True
            assert json.loads(mcp_ok["content"][0]["text"]) == ["extra", "n"]

    def test_dict_tool_rejects_non_object_body_and_tolerates_empty(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            assert client.post("/send_message", json=[1, 2]).status_code == 422  # non-object JSON
            resp = client.post("/send_message", content=b"", headers={"Content-Type": "application/json"})
            assert resp.status_code == 200  # genuinely empty body -> {} -> closure sees no args
            assert server.observed_raw_args == {}

    def test_dict_tool_rejects_malformed_body_without_dispatching(self) -> None:
        """A non-empty but unparseable body is a client 422, not a coerce-to-{} dispatch (regression
        guard: the pre-migration typed-body route returned 422 and never ran the tool)."""
        server = _make(_DualServer)
        server.observed_raw_args = {"sentinel": True}
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/send_message", content=b"{not json", headers={"Content-Type": "application/json"})
            assert resp.status_code == 422
            assert server.observed_raw_args == {"sentinel": True}  # tool was NOT dispatched

    def test_raw_tool_model_return_renders_structured_content(self) -> None:
        class _ModelReturn(_DualServer):
            def model_post_init(self, context: Any) -> None:
                super().model_post_init(context)

                def locate_raw(**arguments: Any) -> LocateResponse:
                    return LocateResponse(param_type="raw", x=9)

                gym_tool(locate_raw, name="locate_raw", input_schema={"type": "object"}, owner=self)

        server = _make(_ModelReturn)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            result = mcp_call(client, "locate_raw", {})
            assert result["structuredContent"] == {"param_type": "raw", "x": 9}
            http = client.post("/locate_raw", json={})
            assert http.status_code == 200
            assert http.json() == {"param_type": "raw", "x": 9}

    def test_model_schema_tool_with_wrong_arity_is_rejected_at_setup(self) -> None:
        server = _make(_DualServer)

        def two_params(a: RepeatBody, b: RepeatBody) -> str:
            return "nope"

        gym_tool(two_params, name="two_params", input_schema=RepeatBody, owner=server)
        with pytest.raises(ValueError, match="exactly one"):
            server.setup_webserver()

    def test_gym_tool_requires_a_name(self) -> None:
        import functools

        with pytest.raises(ValueError, match="requires name="):
            gym_tool(functools.partial(lambda x: x, 1))

    def test_metadata_mints_session_id_when_session_is_empty(self) -> None:
        server = _make(_DualServer)
        request = MagicMock(spec=Request)
        request.session = {}
        metadata = server.build_mcp_session_metadata(request)
        assert request.session[SESSION_ID_KEY]  # freshly minted and stored
        assert TOKEN_HEADER in metadata.headers

    def test_garbage_token_lists_everything(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            assert len(mcp_list_tools(client, token="garbage-token")) == 6

    def test_header_middleware_passes_non_http_scopes_through(self) -> None:
        import asyncio

        from nemo_gym.base_resources_server import _MCPHeaderSessionMiddleware

        seen = {}

        async def inner(scope, receive, send):
            seen["scope"] = scope["type"]

        middleware = _MCPHeaderSessionMiddleware(inner)
        asyncio.get_event_loop().run_until_complete(middleware({"type": "lifespan"}, None, None))
        assert seen["scope"] == "lifespan"

    def test_unresolvable_annotation_falls_back_gracefully(self) -> None:
        server = _make(_DualServer)

        def tool_with_bad_hint(x: "NotARealType") -> str:  # noqa: F821
            return "ok"

        # get_type_hints raises NameError; the request-param check must fall back, not crash.
        server._check_no_request_param("bad_hint_tool", tool_with_bad_hint)


class TestMCPErrorSurface:
    def test_session_tool_without_token_is_clean_tool_error_for_raw_tools(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            result = mcp_call(client, "send_message", {"recipient": "a", "message": "b"}, token=None)
            assert result["isError"] is True
            assert TOKEN_HEADER in result["content"][0]["text"]

    def test_tool_exception_becomes_is_error_not_transport_error(self) -> None:
        class _Exploding(_DualServer):
            @gym_tool
            async def explode(self) -> str:
                """Always raises."""
                raise RuntimeError("kaboom")

        server = _make(_Exploding)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            result = mcp_call(client, "explode", {})
            assert result["isError"] is True
            assert "kaboom" in result["content"][0]["text"]

    def test_raw_dict_result_renders_structured_content(self) -> None:
        server = _make(_DualServer)
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            token = seed_token(client)
            result = mcp_call(client, "send_message", {"recipient": "a", "message": "b"}, token=token)
            assert result["structuredContent"]["delivered"] is True
            # The text block is valid JSON of the same payload (SDK renders both).
            assert json.loads(result["content"][0]["text"])["delivered"] is True
