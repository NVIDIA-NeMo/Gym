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
"""Serve an unmodified resources server's FastAPI tool routes over MCP.

A resources server sets ``expose_tools_over_mcp = True`` and its plain ``POST /<tool>`` routes are
advertised and callable over an MCP ``/mcp`` endpoint — no decorators, no handler changes. The
handlers keep their ``request: Request`` parameter and their ``request.session[SESSION_ID_KEY]``
reads exactly as written; this module never touches them.

``run_webserver`` calls :func:`maybe_auto_expose` after building the app, so exposure is automatic
for any server that sets the flag. Dispatcher servers (one catch-all route backing many tools, whose
per-tool schemas live in data) additionally override one method, ``mcp_tool_inventory()``.

Dispatch, chosen per route at startup:
  * DIRECT (default): the frozen handler runs exactly ONCE, invoked with a fabricated ``Request``
    whose ``.session`` is materialized directly — no middleware, no routing, no second app pass.
  * REPLAY (fallback): where the detector cannot prove direct == a real HTTP request (an author's
    custom middleware, or a handler shape direct dispatch does not reproduce), the call is re-issued
    as an internal in-process HTTP request through the full app stack. Correctness never depends on
    the fast path.

MCP-side engine: the official SDK's public low-level ``mcp.server.lowlevel.Server`` +
``StreamableHTTPSessionManager`` — no private-attribute access.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from base64 import b64encode
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints
from uuid import uuid4

import mcp.types as types
from aiohttp import ClientResponseError
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.routing import APIRoute
from itsdangerous import BadSignature, TimestampSigner, URLSafeSerializer
from mcp.server.lowlevel import Server as _LowLevelMCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from nemo_gym.server_utils import SESSION_ID_KEY


LOG = logging.getLogger(__name__)

# Mirrors nemo_gym.base_resources_server's MCP session-token scheme (same secret + salt derivation).
TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
TOKEN_SALT = "nemo-gym-mcp-session-token"
MCP_METADATA_KEY = "mcp"
MCP_URL_PATH = "/mcp"

# Infrastructure routes are never tools. GET docs/openapi are excluded by the POST filter below;
# /mcp is excluded by path.
BASIC_PATHS = frozenset({"/seed_session", "/verify", "/aggregate_metrics", MCP_URL_PATH})

PERMISSIVE_SCHEMA: dict = {"type": "object", "additionalProperties": True}

# Path-template params from the public route.path string ("/{tool_name}", "/items/{id:int}").
_PATH_PARAM_RE = re.compile(r"{([^}:]+)(?::[^}]*)?}")

# Middleware whose dispatch lives in these modules is Gym's own stack (SessionMiddleware +
# add_session_id + the exception middleware) — its effect is replicated by direct dispatch, so its
# absence there is compensated, not lost.
_GYM_MIDDLEWARE_MODULES = frozenset({"nemo_gym.server_utils"})


# ==================================================================================================
# The signed session token (mirrors base_resources_server) + the session cookie the replay path mints
# ==================================================================================================


def mint_session_cookie(secret_key: str, cookie_name: str, session_id: str) -> str:
    """Build the exact Cookie header value starlette's SessionMiddleware would verify (replay path)."""
    data = b64encode(json.dumps({SESSION_ID_KEY: session_id}).encode("utf-8"))
    signed = TimestampSigner(str(secret_key)).sign(data).decode("utf-8")
    return f"{cookie_name}={signed}"


# ==================================================================================================
# The detector: bind_route (route-level) + audit_middleware (server-level)
# ==================================================================================================


@dataclass
class DirectBinding:
    """Everything needed to invoke one frozen handler directly, resolved once at startup."""

    endpoint: Callable
    path: str
    request_params: tuple[str, ...] = ()
    body_param: Optional[str] = None
    body_model: Optional[type[BaseModel]] = None
    path_param: Optional[str] = None  # catch-all routes: the str param bound per tool
    defaulted_params: tuple[str, ...] = ()
    return_model: Optional[type[BaseModel]] = None
    needs_raw_body: bool = False  # handler reads ``await request.json()`` (no body model)
    body_is_dict: bool = False  # handler declares ``body: dict`` — FastAPI passes the parsed JSON through


@dataclass
class BindOutcome:
    binding: Optional[DirectBinding]  # None -> this route must be REPLAY-dispatched
    reasons: list[str] = field(default_factory=list)
    body_model: Optional[type[BaseModel]] = None  # for the tools/list schema, even when binding is None


def bind_route(route: APIRoute) -> BindOutcome:
    """Classify one route's handler signature for direct dispatch. Public introspection only.

    Annotation resolution matches FastAPI's own: ``inspect.signature`` first (it honors a
    factory-set ``__signature__`` — some servers rewrite it with the real body model while
    ``__annotations__`` still says ``Any``), falling back to ``get_type_hints`` only for deferred
    string annotations (``from __future__ import annotations``).
    """
    endpoint = route.endpoint
    reasons: list[str] = []
    try:
        hints = get_type_hints(endpoint)
    except Exception:  # unresolvable forward refs; only fatal if a needed annotation is a string
        hints = {}
    signature = inspect.signature(endpoint)
    path_params = set(_PATH_PARAM_RE.findall(route.path))

    def resolve(name: str, raw: Any) -> Any:
        if isinstance(raw, str):  # deferred annotation — get_type_hints is the resolver
            return hints.get(name, raw)
        if raw is inspect.Parameter.empty:
            return hints.get(name, raw)
        return raw  # concrete object on the signature wins (FastAPI reads the signature too)

    request_params: list[str] = []
    body_param: Optional[str] = None
    body_model: Optional[type[BaseModel]] = None
    body_is_dict = False
    path_param: Optional[str] = None
    defaulted: list[str] = []

    for name, param in signature.parameters.items():
        annotation = resolve(name, param.annotation)
        if isinstance(annotation, str):
            reasons.append(f"unresolvable string annotation on {name!r}: {annotation!r}")
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            reasons.append(f"*args/**kwargs parameter {name!r}")
            continue
        if annotation is Request:
            request_params.append(name)
            continue
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if body_param is not None:
                reasons.append(f"multiple body models ({body_param!r}, {name!r})")
                continue
            body_param, body_model = name, annotation
            continue
        if annotation is dict:
            # ``body: dict`` — FastAPI parses the JSON body and passes the dict through with no
            # validation. Direct equivalent: pass ``arguments`` as-is.
            if body_param is not None:
                reasons.append(f"multiple body params ({body_param!r}, {name!r})")
                continue
            body_param, body_is_dict = name, True
            continue
        if name in path_params:
            if annotation not in (str, inspect.Parameter.empty):
                reasons.append(f"non-str path param {name!r}: {annotation!r}")
            else:
                path_param = name
            continue
        if param.default is not inspect.Parameter.empty:
            # FastAPI treats these as query params; MCP calls carry no query string, so the HTTP door
            # would hand the handler the default too — matching direct behavior. EXCEPT DI markers
            # (Depends/Security), which the HTTP door would resolve.
            default_type = f"{type(param.default).__module__}.{type(param.default).__name__}"
            if default_type.startswith("fastapi."):
                reasons.append(f"DI marker default on {name!r}: {default_type}")
            else:
                defaulted.append(name)
            continue
        reasons.append(f"unsupported required param {name!r}: {annotation!r}")

    ret = resolve("return", signature.return_annotation)
    return_model = ret if isinstance(ret, type) and issubclass(ret, BaseModel) else None

    if reasons:
        return BindOutcome(None, reasons, body_model)
    return BindOutcome(
        DirectBinding(
            endpoint=endpoint,
            path=route.path,
            request_params=tuple(request_params),
            body_param=body_param,
            body_model=body_model,
            path_param=path_param,
            defaulted_params=tuple(defaulted),
            return_model=return_model,
            needs_raw_body=body_param is None and bool(request_params),
            body_is_dict=body_is_dict,
        ),
        [],
        body_model,
    )


def audit_middleware(app: FastAPI) -> list[str]:
    """Return the names of NON-Gym middleware installed on the app (empty == direct-safe).

    Any non-Gym middleware means an env author added per-request behavior that direct dispatch would
    silently skip, so the whole server falls back to replay. Each entry is a
    ``starlette.middleware.Middleware`` data holder: ``.cls`` is the class, ``.kwargs`` its
    constructor kwargs (``dispatch=fn`` for ``@app.middleware("http")`` functions).
    """
    custom: list[str] = []
    for m in app.user_middleware:
        cls = m.cls
        if f"{cls.__module__}.{cls.__name__}" == "starlette.middleware.sessions.SessionMiddleware":
            continue  # Gym's SessionMiddleware — replaced by a materialized session on direct dispatch
        dispatch = m.kwargs.get("dispatch")
        if dispatch is not None and getattr(dispatch, "__module__", None) in _GYM_MIDDLEWARE_MODULES:
            continue  # Gym's add_session_id / exception middleware
        custom.append(f"{cls.__module__}.{cls.__name__}")
    return custom


# ==================================================================================================
# Direct invocation: fabricate the Request, call the frozen handler ONCE
# ==================================================================================================


class DirectDispatchError(Exception):
    """Wraps a handler-visible failure so call_tool maps it to the same isError text replay produces."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status} (direct): {detail}")
        self.status = status
        self.detail = detail


def _make_receive(body: bytes):
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def call_direct(
    app: FastAPI, binding: DirectBinding, session_id: str, arguments: dict, path_value: Optional[str] = None
) -> Any:
    """Invoke the frozen handler once and return its JSON-able payload.

    Replicates what skipping Gym's own stack would otherwise lose: SessionMiddleware + add_session_id
    become ``scope["session"] = {SESSION_ID_KEY: sid}`` (handlers only READ request.session); the
    exception middleware's status-carrying text is reproduced by pre-formatting HTTPException /
    ValidationError / ClientResponseError into DirectDispatchError.
    """
    kwargs: dict[str, Any] = {}
    if binding.path_param is not None:
        kwargs[binding.path_param] = path_value if path_value is not None else ""
    if binding.body_model is not None:
        try:
            kwargs[binding.body_param] = binding.body_model.model_validate(arguments)
        except ValidationError as e:
            raise DirectDispatchError(422, json.dumps(jsonable_encoder(e.errors()))) from e
    elif binding.body_is_dict:
        kwargs[binding.body_param] = dict(arguments or {})  # FastAPI's dict-body pass-through
    if binding.request_params:
        raw = json.dumps(arguments or {}).encode("utf-8") if binding.needs_raw_body else b""
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": binding.path if path_value is None else "/" + path_value,
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 0),
            "server": ("internal-mcp-direct", 80),
            "state": {},
            "app": app,
            # SessionMiddleware's documented effect, materialized for this rollout's session id.
            "session": {SESSION_ID_KEY: session_id},
        }
        request = Request(scope, _make_receive(raw))
        for name in binding.request_params:
            kwargs[name] = request

    try:
        result = binding.endpoint(**kwargs)
        if inspect.isawaitable(result):
            result = await result
    except StarletteHTTPException as e:  # fastapi.HTTPException subclasses this
        raise DirectDispatchError(e.status_code, str(e.detail)) from e
    except ClientResponseError as e:
        detail = getattr(e, "response_content", None)
        raise DirectDispatchError(500, f"Hit an exception calling an inner server: {detail or e}") from e

    if isinstance(result, Response):  # e.g. a handler returning PlainTextResponse
        text = bytes(result.body).decode("utf-8", errors="replace")
        if not 200 <= result.status_code < 300:
            raise DirectDispatchError(result.status_code, text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if binding.return_model is not None and not isinstance(result, binding.return_model):
        result = binding.return_model.model_validate(result)  # parity with response_model filtering
    return jsonable_encoder(result)


# ==================================================================================================
# The replay fallback: one in-process HTTP request through the app's full ASGI stack (httpx-free)
# ==================================================================================================


async def _replay(app: FastAPI, path: str, raw_path: bytes, base_headers: tuple, body: bytes) -> tuple[int, bytes]:
    """Issue one internal HTTP request through the full app stack. Only immutable objects (the
    pre-encoded header tuple, raw_path) are shared between calls; the scope and its nested dicts are
    built fresh per call, so a scope-mutating middleware cannot leak state into a later replay."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
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
    status: Optional[int] = None
    chunks: list[bytes] = []

    async def send(message: dict) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    try:
        await app(scope, _make_receive(body), send)
    except BaseException:
        # Starlette's ServerErrorMiddleware sends its 500 before re-raising; surface a completed one.
        if status is None:
            raise
    assert status is not None, "ASGI app returned no response"
    return status, b"".join(chunks)


# ==================================================================================================
# Harvest: one walk over app.routes -> the tool map (advertisement + dispatch plan per tool)
# ==================================================================================================


@dataclass
class MCPTool:
    name: str
    tool: types.Tool  # the tools/list advertisement
    mode: str  # "direct" | "replay"
    replay_path: str  # POST path the replay fallback targets
    raw_path: bytes  # pre-encoded replay_path
    binding: Optional[DirectBinding] = None  # set when mode == "direct"
    path_value: Optional[str] = None  # catch-all tools: value bound to the path param
    reasons: list[str] = field(default_factory=list)  # why replay, when mode == "replay"


def _schema_for(body_model: Optional[type[BaseModel]]) -> dict:
    return body_model.model_json_schema() if body_model is not None else dict(PERMISSIVE_SCHEMA)


def harvest_tools(app: FastAPI, server: Any) -> dict[str, MCPTool]:
    """Scan app.routes once; return {tool name -> MCPTool}. Also runs the server-level middleware gate.

    Dispatcher servers (one catch-all route backing many data-defined tools) override
    ``mcp_tool_inventory(self) -> list[dict]`` returning ``{"name", "input_schema", "description"}``
    items; those tools dispatch through the catch-all with its path param bound to the tool name.
    Catch-alls that back no tools are declared via ``mcp_toolless_catchall_paths``.
    """
    custom_middleware = audit_middleware(app)
    server_mode = "replay" if custom_middleware else "direct"

    typed_routes: dict[str, APIRoute] = {}
    catchall_routes: list[APIRoute] = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or "POST" not in (route.methods or set()):
            continue
        if route.path in BASIC_PATHS:
            continue
        if "{" in route.path:
            catchall_routes.append(route)
            continue
        typed_routes[route.path.lstrip("/")] = route

    # A catch-all backs tools (workplace's /{path}) or only returns errors (finance's /{tool_name});
    # only the author knows. Declaring toolless keeps the missing-inventory warning meaningful; a
    # declaration naming no real catch-all is a hard error (a typo would re-hide the tools it guards).
    declared_toolless = frozenset(getattr(server, "mcp_toolless_catchall_paths", ()) or ())
    unknown_declared = declared_toolless - {r.path for r in catchall_routes}
    if unknown_declared:
        raise ValueError(
            f"mcp_toolless_catchall_paths on {type(server).__name__} names route(s) {sorted(unknown_declared)} "
            f"but the app's catch-all routes are {sorted(r.path for r in catchall_routes)}. Fix the declaration."
        )

    def make(
        name: str, description: Optional[str], schema: dict, route: Optional[APIRoute], path_value: Optional[str]
    ) -> MCPTool:
        binding, reasons = None, ["no route to bind"]
        if route is not None:
            outcome = bind_route(route)
            binding, reasons = outcome.binding, outcome.reasons
        mode = "direct" if (server_mode == "direct" and binding is not None) else "replay"
        replay_path = "/" + name
        return MCPTool(
            name=name,
            tool=types.Tool(name=name, description=description, inputSchema=schema),
            mode=mode,
            replay_path=replay_path,
            raw_path=replay_path.encode("utf-8"),
            binding=binding,
            path_value=path_value,
            reasons=[] if mode == "direct" else (reasons or [f"custom middleware: {custom_middleware}"]),
        )

    tools: dict[str, MCPTool] = {}
    for name, route in typed_routes.items():
        outcome = bind_route(route)
        description = (route.description or route.summary or "").strip() or None
        # schema comes from the SAME resolution that decides dispatch (no separate route.body_field read)
        tools[name] = make(name, description, _schema_for(outcome.body_model), route, None)

    inventory_fn = getattr(server, "mcp_tool_inventory", None)
    if inventory_fn is not None:
        inventory_catchalls = [r for r in catchall_routes if r.path not in declared_toolless]
        catch_route = inventory_catchalls[0] if inventory_catchalls else None
        for item in inventory_fn():
            name = item["name"]
            if name in tools:
                raise ValueError(f"Duplicate MCP tool name {name!r} (route harvest vs inventory override)")
            schema = item.get("input_schema") or dict(PERMISSIVE_SCHEMA)
            tools[name] = make(name, item.get("description"), schema, catch_route, name)
    else:
        undeclared = [r for r in catchall_routes if r.path not in declared_toolless]
        if undeclared:
            LOG.warning(
                "%s has parameterized catch-all route(s) %s but no mcp_tool_inventory() override; any tools "
                "behind them are NOT exposed over MCP. Add the override, or declare them toolless via "
                "mcp_toolless_catchall_paths.",
                type(server).__name__,
                [r.path for r in undeclared],
            )

    direct_n = sum(1 for t in tools.values() if t.mode == "direct")
    LOG.info(
        "%s MCP: %d tools (%d direct, %d replay), server_mode=%s%s",
        type(server).__name__,
        len(tools),
        direct_n,
        len(tools) - direct_n,
        server_mode,
        f", custom middleware {custom_middleware}" if custom_middleware else "",
    )
    return tools


# ==================================================================================================
# /seed_session augmentation: wrap (never edit) the endpoint so its response gains the signed token
# ==================================================================================================


def _wrap_seed_session(app: FastAPI, mint_metadata: Callable[[Request], dict]) -> None:
    idx, route = next(
        (i, r) for i, r in enumerate(app.router.routes) if isinstance(r, APIRoute) and r.path == "/seed_session"
    )
    method = route.endpoint
    signature = inspect.signature(method)
    hints = get_type_hints(method)
    request_param_name = next(
        (n for n, p in signature.parameters.items() if hints.get(n, p.annotation) is Request), None
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
# The installer + the flag-gated automatic entry point
# ==================================================================================================


def maybe_auto_expose(server: Any, app: FastAPI) -> Optional[dict[str, MCPTool]]:
    """Install MCP auto-exposure iff the server opts in (``expose_tools_over_mcp = True``).

    Called by ``run_webserver`` after the app is fully built, so every route is present. Returns the
    tool map (for tests/introspection), or None when the server did not opt in.
    """
    if not getattr(server, "expose_tools_over_mcp", False):
        return None
    return install_auto_exposure(server, app)


def install_auto_exposure(server: Any, app: FastAPI, allowed_tools: Optional[list[str]] = None) -> dict[str, MCPTool]:
    """Harvest the tool routes, wire the /seed_session token, and mount the /mcp endpoint.

    ``server`` is any resources server built exactly as on main; ``app`` is the FastAPI app its
    unmodified ``setup_webserver()`` returned. Returns the tool map.
    """
    secret = server.get_session_middleware_key()  # Gym convention: the token secret == the cookie name
    serializer = URLSafeSerializer(secret, salt=TOKEN_SALT)
    tools = harvest_tools(app, server)

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

    mcp_server = _LowLevelMCPServer(server.config.name or type(server).__name__)

    # Per-session caches: verify the token HMAC once per session; mint the replay cookie once per
    # session. Both grow one small entry per rollout, like any server's own session state.
    claims_cache: dict[str, Any] = {}
    replay_headers_cache: dict[str, tuple] = {}

    def session_claims(required: bool = True) -> tuple[Optional[str], Optional[frozenset]]:
        ctx_request = mcp_server.request_context.request  # the POST /mcp starlette Request
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
        return [t.tool for t in tools.values() if allowed is None or t.name in allowed]

    def _to_result(payload: Any):
        # dict -> text + structuredContent; str -> text; other JSON -> text. JSONResponse renders
        # the door's exact success bytes.
        if isinstance(payload, dict):
            return [types.TextContent(type="text", text=JSONResponse(payload).body.decode("utf-8"))], payload
        if isinstance(payload, str):
            return [types.TextContent(type="text", text=payload)]
        return [types.TextContent(type="text", text=JSONResponse(payload).body.decode("utf-8"))]

    @mcp_server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict):
        tool = tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name!r}. Available tools: {sorted(tools)}")
        session_id, allowed = session_claims(required=True)
        if allowed is not None and name not in allowed:
            raise ValueError(f"Tool {name!r} is not allowed for this session.")

        if tool.mode == "direct":
            try:
                payload = await call_direct(app, tool.binding, session_id, arguments, path_value=tool.path_value)
            except DirectDispatchError as exc:
                raise ValueError(f"HTTP {exc.status} from POST /{name}: {exc.detail}")
            return _to_result(payload)

        # replay fallback
        base_headers = replay_headers_cache.get(session_id)
        if base_headers is None:
            base_headers = (
                (b"content-type", b"application/json"),
                (b"cookie", mint_session_cookie(secret, secret, session_id).encode("latin-1")),
                (b"host", b"internal-mcp-replay"),
            )
            replay_headers_cache[session_id] = base_headers
        status, body = await _replay(
            app, tool.replay_path, tool.raw_path, base_headers, json.dumps(arguments or {}).encode("utf-8")
        )
        text = body.decode("utf-8", errors="replace")
        if not 200 <= status < 300:
            raise ValueError(f"HTTP {status} from POST {tool.replay_path}: {text}")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [types.TextContent(type="text", text=text)]
        if isinstance(parsed, dict):
            return [types.TextContent(type="text", text=text)], parsed
        return [types.TextContent(type="text", text=text)]

    manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=True,
        # The endpoint is token-gated; Host/Origin checks would 421 off-loopback multi-node access.
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
    return tools
