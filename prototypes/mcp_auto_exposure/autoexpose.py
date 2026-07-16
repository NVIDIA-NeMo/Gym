# SPDX-License-Identifier: Apache-2.0
"""Brian-design auto-exposure base (spike).

Contract (R1-R4):
  R1  Env authors write plain FastAPI routes exactly as on origin/main. No decorators.
  R2  Handlers keep their ``request: Request`` params and read ``request.session[SESSION_ID_KEY]``
      — byte-identical handler code. This module never touches handler functions.
  R3  Every non-basic POST route is exposed as an MCP tool automatically. Parameterized catch-all
      routes cannot be single tools; dispatcher servers override ONE function,
      ``mcp_tool_inventory()``, returning the tool inventory (names + schemas); those tools
      dispatch by replaying ``POST /<name>`` which the existing catch-all route matches.
  R4  The HTTP door is untouched except for two additive deltas, both reported by the checks:
      the ``/seed_session`` response gains an ``mcp`` key, and ``/mcp`` (a new path) exists.

Mechanism (the replay bridge):
  MCP tools/call -> verify the signed session token (header ``X-NeMo-Gym-Session-Token``) ->
  mint the starlette SessionMiddleware cookie for that session id (Gym owns the itsdangerous
  secret) -> re-issue the call as an INTERNAL ASGI request through the app's OWN middleware
  stack (SessionMiddleware populates request.session; the unmodified handler runs) -> map the
  HTTP response to an MCP result (2xx JSON -> content + structuredContent; anything else ->
  isError carrying the HTTP status and body text).

MCP-side implementation choice: the official SDK's LOW-LEVEL ``mcp.server.lowlevel.Server`` +
``StreamableHTTPSessionManager`` (stateless, json_response). Why:
  * tools here are dynamic (harvested from routes / returned by an inventory hook) with
    hand-built JSON schemas — lowlevel serves that first-class via list_tools/call_tool
    handlers, no function-introspection to fight and no private attributes to patch
    (FastMCP would need ``_tool_manager`` surgery for dynamic tools);
  * the SDK populates ``server.request_context.request`` with the actual starlette Request of
    the POST /mcp envelope, so the session-token header is read directly — no ASGI middleware
    + ContextVar bridge needed;
  * zero new dependency: Gym already ships the ``mcp`` package.
"""

from __future__ import annotations

import inspect
import json
import logging
from base64 import b64encode
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, get_type_hints
from uuid import uuid4

import mcp.types as types
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.routing import APIRoute
from itsdangerous import BadSignature, TimestampSigner, URLSafeSerializer
from mcp.server.lowlevel import Server as _LowLevelMCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from nemo_gym.server_utils import SESSION_ID_KEY, SimpleServer


LOG = logging.getLogger("autoexpose")

# Mirrors nemo_gym.base_resources_server (kept literal so this base stands alone).
TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
TOKEN_SALT = "nemo-gym-mcp-session-token"
MCP_METADATA_KEY = "mcp"
MCP_URL_PATH = "/mcp"

# "Basic" routes are infrastructure, not tools (R3). Docs/openapi routes are GET-only and are
# excluded by the POST filter; the /mcp endpoint itself is excluded by path.
BASIC_PATHS = frozenset({"/seed_session", "/verify", "/aggregate_metrics", MCP_URL_PATH})

PERMISSIVE_SCHEMA: dict = {"type": "object", "additionalProperties": True}


# ==================================================================================================
# The httpx-free internal ASGI invoker
# ==================================================================================================


async def asgi_call(
    app: Any, method: str, path: str, headers: dict[str, str], body: bytes = b""
) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    """Issue one in-process HTTP request through the app's FULL ASGI stack. No httpx anywhere."""
    raw_headers = tuple((k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items())
    return await _asgi_call_prepared(app, method.upper(), path, path.encode("utf-8"), raw_headers, body)


async def _asgi_call_prepared(
    app: Any, method: str, path: str, raw_path: bytes, base_headers: tuple, body: bytes
) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    """``asgi_call`` core with the per-call encodes hoisted out (perf/bench_interleaved.py).

    ``base_headers`` is a tuple of pre-encoded ``(name, value)`` byte pairs WITHOUT content-length
    (appended here from the actual body). Only IMMUTABLE objects (a tuple of bytes pairs, the
    raw_path bytes) are shared between calls — the scope dict, its nested "asgi" dict, the headers
    list, and "state" are built fresh per call, so a middleware that mutates its scope can never
    leak state into a later replay (verified by perf/bench_safe_wins.py proofs P3/P4).
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": raw_path,
        "query_string": b"",
        "root_path": "",
        "headers": [*base_headers, (b"content-length", str(len(body)).encode("latin-1"))],
        "client": ("127.0.0.1", 0),
        "server": ("internal-mcp-replay", 80),
        "state": {},
    }

    body_sent = False

    async def receive() -> dict:
        nonlocal body_sent
        if body_sent:
            return {"type": "http.disconnect"}
        body_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    status: Optional[int] = None
    response_headers: list[tuple[bytes, bytes]] = []
    chunks: list[bytes] = []

    async def send(message: dict) -> None:
        nonlocal status, response_headers
        if message["type"] == "http.response.start":
            status = message["status"]
            response_headers = list(message.get("headers", []))
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    try:
        await app(scope, receive, send)
    except BaseException:
        # Starlette's ServerErrorMiddleware sends its 500 response BEFORE re-raising (so real
        # servers can log). If a response completed, surface it exactly like the HTTP door would.
        if status is None:
            raise
    assert status is not None, "ASGI app returned no response"
    return status, response_headers, b"".join(chunks)


# ==================================================================================================
# Session-cookie mint (Gym owns the SessionMiddleware secret; forge what it would have set)
# ==================================================================================================


def mint_session_cookie(secret_key: str, cookie_name: str, session_id: str) -> str:
    """Build the exact Cookie header value starlette's SessionMiddleware would verify."""
    data = b64encode(json.dumps({SESSION_ID_KEY: session_id}).encode("utf-8"))
    signed = TimestampSigner(str(secret_key)).sign(data).decode("utf-8")
    return f"{cookie_name}={signed}"


# ==================================================================================================
# Route harvesting (R3) — name = path, schema = route.body_field's model, description = docstring
# ==================================================================================================


@dataclass
class ToolEntry:
    name: str
    path: str  # the POST path the MCP call replays
    tool: types.Tool
    raw_path: bytes = b""  # pre-encoded ``path`` for the replay scope (derived, immutable)

    def __post_init__(self) -> None:
        if not self.raw_path:
            self.raw_path = self.path.encode("utf-8")


def _route_input_schema(route: APIRoute) -> dict:
    body_field = getattr(route, "body_field", None)
    if body_field is None:
        return dict(PERMISSIVE_SCHEMA)  # honest fallback: the handler takes no request body
    # FastAPI <=0.11x exposes the body model as ModelField.type_; newer _compat.v2 ModelFields
    # carry it on field_info.annotation. Accept either.
    model = getattr(body_field, "type_", None)
    if model is None:
        model = getattr(getattr(body_field, "field_info", None), "annotation", None)
    if model is None:
        # A body model EXISTS but neither known spelling resolved it — a FastAPI internal layout
        # this code does not handle. Crash rather than silently advertise a typed tool as
        # permissive: a wrong schema poisons rollouts invisibly, a startup crash is a small fix.
        raise RuntimeError(
            f"Cannot extract the body model for route {route.path!r}: body_field is present but "
            "has neither .type_ nor .field_info.annotation. FastAPI's internals changed; update "
            "_route_input_schema()."
        )
    if isinstance(model, type) and issubclass(model, BaseModel):
        return model.model_json_schema()
    return dict(PERMISSIVE_SCHEMA)  # honest fallback: the body is a scalar/non-model type


def harvest_route_tools(app: FastAPI) -> tuple[list[ToolEntry], list[APIRoute]]:
    """Scan app.routes; return (typed tool entries, parameterized catch-all routes)."""
    entries: list[ToolEntry] = []
    catchalls: list[APIRoute] = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or "POST" not in (route.methods or set()):
            continue
        if route.path in BASIC_PATHS:
            continue
        if "{" in route.path:
            catchalls.append(route)
            continue
        name = route.path.lstrip("/")
        description = (route.description or route.summary or "").strip() or None
        entries.append(
            ToolEntry(
                name=name,
                path=route.path,
                tool=types.Tool(name=name, description=description, inputSchema=_route_input_schema(route)),
            )
        )
    return entries, catchalls


# ==================================================================================================
# /seed_session augmentation: wrap (never edit) the route endpoint so the response gains the
# signed session token under the "mcp" key. Mirrors _build_seed_session_endpoint on the
# mcp-dual-registration branch; the wrapped route is swapped IN PLACE to preserve route order
# (a dispatcher's catch-all must keep matching after /seed_session).
# ==================================================================================================


def _wrap_seed_session(app: FastAPI, mint_metadata: Callable[[Request], dict]) -> None:
    idx, route = next(
        (i, r) for i, r in enumerate(app.router.routes) if isinstance(r, APIRoute) and r.path == "/seed_session"
    )
    method = route.endpoint
    signature = inspect.signature(method)
    hints = get_type_hints(method)
    request_param_name = next(
        (n for n, p in signature.parameters.items() if hints.get(n, p.annotation) is Request),
        None,
    )
    params = [p.replace(annotation=hints.get(n, p.annotation)) for n, p in signature.parameters.items()]
    passthrough = tuple(signature.parameters)
    if request_param_name is None:
        request_param_name = "request"
        params = [
            inspect.Parameter("request", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request),
            *params,
        ]

    async def seed_session_endpoint(**kwargs: Any) -> JSONResponse:
        request: Request = kwargs[request_param_name]
        result = method(**{k: kwargs[k] for k in passthrough})
        if inspect.isawaitable(result):
            result = await result
        payload = jsonable_encoder(result)
        if isinstance(payload, dict) and MCP_METADATA_KEY not in payload:
            payload[MCP_METADATA_KEY] = mint_metadata(request)
        return JSONResponse(payload)

    seed_session_endpoint.__name__ = "seed_session"
    seed_session_endpoint.__signature__ = inspect.Signature(parameters=params)
    seed_session_endpoint.__annotations__ = {p.name: p.annotation for p in params}

    app.post("/seed_session")(seed_session_endpoint)
    new_route = app.router.routes.pop()  # the route just appended by app.post
    app.router.routes[idx] = new_route  # in-place swap keeps ordering vs catch-all routes


# ==================================================================================================
# The auto-exposure installer
# ==================================================================================================


def install_auto_exposure(
    server: SimpleServer, app: FastAPI, allowed_tools: Optional[list[str]] = None
) -> dict[str, ToolEntry]:
    """Post-registration hook: harvest routes, wire /seed_session metadata, mount /mcp.

    ``server`` is ANY Gym SimpleServer built exactly as on origin/main; ``app`` is the FastAPI app
    its unmodified ``setup_webserver()`` returned. Dispatcher servers (one catch-all route backing
    many tools) override one function: ``mcp_tool_inventory(self) -> list[dict]`` with items
    ``{"name": ..., "input_schema": {...}, "description": ...}`` (Brian's 2.d). Returns the tool
    map for introspection.
    """
    secret = server.get_session_middleware_key()  # Gym convention: secret == session cookie name
    serializer = URLSafeSerializer(secret, salt=TOKEN_SALT)

    # ---- tool inventory (R3) ---------------------------------------------------------------
    entries, catchall_routes = harvest_route_tools(app)

    # Whether a catch-all backs real tools is author knowledge no classifier can recover
    # (workplace's /{path} backs 27 tools; finance's /{tool_name} only returns error strings).
    # Authors declare toolless catch-alls explicitly so the missing-inventory warning below
    # stays meaningful. A declared path that matches no harvested catch-all is a hard error —
    # a typo here would silently re-enable the very blindness the declaration exists to remove.
    declared_toolless = frozenset(getattr(server, "mcp_toolless_catchall_paths", ()) or ())
    unknown_declared = declared_toolless - {route.path for route in catchall_routes}
    if unknown_declared:
        raise ValueError(
            f"mcp_toolless_catchall_paths on {type(server).__name__} names route(s) "
            f"{sorted(unknown_declared)} but the app's catch-all routes are "
            f"{sorted(route.path for route in catchall_routes)}. Fix the declaration."
        )
    undeclared_catchalls = [route for route in catchall_routes if route.path not in declared_toolless]

    inventory_fn = getattr(server, "mcp_tool_inventory", None)
    if inventory_fn is not None:
        for item in inventory_fn():
            entries.append(
                ToolEntry(
                    name=item["name"],
                    path="/" + item["name"],  # dispatch through the existing catch-all route
                    tool=types.Tool(
                        name=item["name"],
                        description=item.get("description"),
                        inputSchema=item.get("input_schema") or dict(PERMISSIVE_SCHEMA),
                    ),
                )
            )
    elif undeclared_catchalls:
        LOG.warning(
            "%s has parameterized catch-all route(s) %s but no mcp_tool_inventory() override; "
            "any tools behind them are NOT exposed over MCP. Add the override (R3), or declare "
            "them toolless via mcp_toolless_catchall_paths to silence this warning.",
            type(server).__name__,
            [r.path for r in undeclared_catchalls],
        )

    tool_map: dict[str, ToolEntry] = {}
    for entry in entries:
        if entry.name in tool_map:
            raise ValueError(f"Duplicate MCP tool name {entry.name!r} (route harvest vs inventory override)")
        tool_map[entry.name] = entry

    # ---- /seed_session -> signed session token (mint mirrors base_resources_server) ----------
    def mint_metadata(request: Request) -> dict:
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            session_id = str(uuid4())
            request.session[SESSION_ID_KEY] = session_id
        payload: Any = session_id if allowed_tools is None else {"sid": session_id, "tools": list(allowed_tools)}
        return {
            "server_name": server.config.name or type(server).__name__,
            "url_path": MCP_URL_PATH,
            "transport": "http",
            "headers": {TOKEN_HEADER: serializer.dumps(payload)},
        }

    _wrap_seed_session(app, mint_metadata)

    # ---- the MCP server: tools/list + tools/call (the replay bridge) -------------------------
    mcp_server = _LowLevelMCPServer(server.config.name or type(server).__name__)

    # Perf caches (perf/bench_interleaved.py). Both are keyed by per-session values, so they grow
    # one small entry per rollout session — the same rate as any Gym server's own session state.
    #   claims_cache: serializer.loads is deterministic per token; verify the HMAC once per
    #     session instead of on every tools/call (~6-10 us/call saved). Failures are NOT cached,
    #     so invalid tokens take exactly today's path.
    #   session_headers_cache: the minted session cookie is deterministic per (secret, session
    #     id); mint + encode the replay's request headers once per session (~14 us/call saved).
    #     Caching is MORE main-faithful than re-minting: a real HTTP client re-sends the cookie
    #     bytes it was last handed, it does not re-mint per call. (SessionMiddleware max_age is
    #     14 days — a cached cookie only goes stale for a session idle that long, at which point
    #     the HTTP door's cookie would have expired identically.)
    claims_cache: dict[str, Any] = {}
    session_headers_cache: dict[str, tuple] = {}

    def session_claims(required: bool = True) -> tuple[Optional[str], Optional[frozenset]]:
        ctx_request = mcp_server.request_context.request  # starlette Request of the POST /mcp
        token = ctx_request.headers.get(TOKEN_HEADER) if ctx_request is not None else None
        if not token:
            if required:
                raise ValueError(f"Missing {TOKEN_HEADER} for Gym MCP tool call.")
            return None, None
        payload = claims_cache.get(token)
        if payload is None:
            try:
                payload = serializer.loads(token)
            except BadSignature:
                if required:
                    raise ValueError("Invalid Gym MCP session token.")
                return None, None
            claims_cache[token] = payload
        if isinstance(payload, dict):
            allowed = payload.get("tools")
            return payload.get("sid"), None if allowed is None else frozenset(allowed)
        return payload, None

    @mcp_server.list_tools()
    async def list_tools() -> list[types.Tool]:
        _, allowed = session_claims(required=False)
        return [e.tool for e in tool_map.values() if allowed is None or e.name in allowed]

    # validate_input=False: argument validation stays where it always was — the FastAPI route.
    # Malformed args therefore surface as the HTTP door's own 422 body, mapped to isError below.
    @mcp_server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict):
        entry = tool_map.get(name)
        if entry is None:
            raise ValueError(f"Unknown tool: {name!r}. Available tools: {sorted(tool_map)}")
        session_id, allowed = session_claims(required=True)
        if allowed is not None and name not in allowed:
            raise ValueError(f"Tool {name!r} is not allowed for this session.")

        base_headers = session_headers_cache.get(session_id)
        if base_headers is None:
            # Byte-for-byte the headers asgi_call would build from the previous per-call dict
            # (same names, same order, content-length appended per body by _asgi_call_prepared).
            base_headers = (
                (b"content-type", b"application/json"),
                (b"cookie", mint_session_cookie(secret, secret, session_id).encode("latin-1")),
                (b"host", b"internal-mcp-replay"),
            )
            session_headers_cache[session_id] = base_headers

        status, _, resp_body = await _asgi_call_prepared(
            app,
            "POST",
            entry.path,
            entry.raw_path,
            base_headers,
            json.dumps(arguments or {}).encode("utf-8"),
        )
        text = resp_body.decode("utf-8", errors="replace")
        if not 200 <= status < 300:
            raise ValueError(f"HTTP {status} from POST {entry.path}: {text}")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [types.TextContent(type="text", text=text)]
        if isinstance(parsed, dict):
            return [types.TextContent(type="text", text=text)], parsed
        return [types.TextContent(type="text", text=text)]

    # ---- mount the streamable-http transport at /mcp -----------------------------------------
    manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=True,
        # Mirrors the Gym rationale: endpoint is token-gated; Host/Origin checks would 421
        # off-loopback multi-node access.
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    class _MCPEndpoint:
        async def __call__(self, scope, receive, send):
            await manager.handle_request(scope, receive, send)

    endpoint = _MCPEndpoint()
    # Insert at the FRONT so a dispatcher's catch-all POST /{path} cannot shadow POST /mcp.
    app.router.routes.insert(0, Route(MCP_URL_PATH, endpoint, include_in_schema=False))
    app.router.routes.insert(1, Mount(MCP_URL_PATH, app=endpoint))

    main_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_wrapper(app_: FastAPI):
        async with manager.run():
            async with main_lifespan(app_) as state:
                yield state

    app.router.lifespan_context = lifespan_wrapper
    return tool_map
