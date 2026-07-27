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
"""Serve an unmodified resources server's FastAPI tool routes over MCP.

A resources server sets ``expose_tools_over_mcp: true`` in its config and its plain ``POST /<tool>`` routes are
advertised and callable over an MCP ``/mcp`` endpoint — no decorators, no handler changes. The
handlers keep their ``request: Request`` parameter and their ``request.session[SESSION_ID_KEY]``
reads exactly as written; this module never touches them.

``run_webserver`` calls :func:`maybe_auto_expose` after building the app, so exposure is automatic
for any server that sets the flag. A server tailors what it exposes by overriding one method,
``mcp_tools(harvested, catchall)``: the default returns the auto-harvested typed POST routes, and an
override may filter them (exclude a route), append catch-all-backed tools (dispatcher servers whose
per-tool schemas live in data: ``harvested + [catchall.tool(name, input_schema, description)]``), or
return ``None``/``[]`` to expose nothing. A server can narrow one rollout's token to a subset of
tools by overriding ``mcp_allowed_tools_for_session(seed_body)``; the token minted by that
/seed_session response then lists and calls only those tools.

Dispatch is direct: the route's handler runs exactly once per MCP call, invoked with a fabricated
``Request`` whose ``.session`` is materialized directly — no middleware, no routing, no second app
pass. Where that cannot be proven equivalent to a real HTTP request (the server installs custom
middleware, or a handler uses a shape direct dispatch does not reproduce — FastAPI dependency
injection, multiple body models, ...), exposure refuses loudly at startup, naming the route and the
reason: a wrong dispatch would corrupt rollouts silently, while a startup error is a small fix.

MCP-side engine: the official SDK's public low-level ``mcp.server.lowlevel.Server`` +
``StreamableHTTPSessionManager`` — no private-attribute access.

How to read this file
---------------------
Two timelines run here, and every function belongs to exactly one. Keep them apart while reading.

STARTUP (runs once at boot, per opted-in server):
    ``maybe_auto_expose`` (flag gate) -> ``install_auto_exposure``, which (after refusing a server
    that already serves /mcp):
      * ``harvest_tools`` — walks the app's POST routes, calls ``bind_route`` on each, and builds the
        ``{tool name: MCPTool}`` map (advertisement + a direct binding per tool);
      * wraps endpoints — ``_wrap_seed_session`` makes the /seed_session response hand the client a
        session token; ``_wrap_verify`` normalizes MCP-namespaced tool-call names for scoring only;
      * mounts /mcp — registers the ``list_tools``/``call_tool`` handlers and attaches the SDK ASGI app.

PER-CALL (runs on every MCP ``tools/call``):
    a POST /mcp request -> ``session_claims`` reads the token (session id + any allow-list) ->
    ``call_direct(binding)`` runs the route's own handler exactly once with a fabricated ``Request``
    and returns its JSON-able payload.

Reading order: start with the dataclasses ``DirectBinding``, ``BindResult``, and ``MCPTool``, then ``bind_route``
(startup classify), then ``call_direct`` (per-call dispatch), then ``harvest_tools`` (startup build),
then ``install_auto_exposure`` (startup wire-up). Everything else is a helper for those five.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints
from uuid import uuid4

import mcp.types as types
from aiohttp import ClientResponseError
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.routing import APIRoute
from itsdangerous import BadSignature, URLSafeSerializer
from mcp.server.lowlevel import Server as _LowLevelMCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from nemo_gym.base_resources_server import (
    _MCP_TOKEN_SALT,
    NEMO_GYM_MCP_METADATA_KEY,
    NEMO_GYM_MCP_SESSION_TOKEN_HEADER,
    RESERVED_MCP_TOOL_NAMES,
    MCPServerMetadata,
)
from nemo_gym.server_utils import SESSION_ID_KEY


LOG = logging.getLogger(__name__)

# Alias for readability; the canonical value lives in base_resources_server because the header is
# wire contract — claude_code_agent reads the same constant from the /seed_session metadata.
TOKEN_HEADER = NEMO_GYM_MCP_SESSION_TOKEN_HEADER

MCP_URL_PATH = "/mcp"

PERMISSIVE_SCHEMA: dict = {"type": "object", "additionalProperties": True}

# MCP clients reject tool names outside this alphabet, and verify-time name normalization
# (mcp__<server>__<tool>) cannot round-trip them.
_MCP_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")

# Path-template params from the public route.path string ("/{tool_name}", "/items/{id:int}").
_PATH_PARAM_RE = re.compile(r"{([^}:]+)(?::[^}]*)?}")

# Gym's function-based middleware (add_session_id + exception middleware); SessionMiddleware is matched by class name.
_GYM_MIDDLEWARE_MODULES = frozenset({"nemo_gym.server_utils"})


# ==================================================================================================
# STARTUP — detect & bind routes: bind_route classifies one handler signature for direct dispatch
# ==================================================================================================


@dataclass
class DirectBinding:
    """Everything needed to invoke one route handler directly: resolved once at STARTUP by bind_route,
    read per-call by call_direct."""

    endpoint: Callable
    path: str
    request_params: tuple[str, ...] = ()
    body_param: Optional[str] = None
    body_model: Optional[type[BaseModel]] = None
    path_param: Optional[str] = None  # catch-all routes: the str param bound per tool
    return_model: Optional[type[BaseModel]] = None
    body_is_dict: bool = False  # handler declares ``body: dict`` — FastAPI passes the parsed JSON through
    is_coroutine: bool = False  # sync (def) handlers go to a threadpool, as FastAPI would send them


@dataclass(frozen=True)
class BindResult:
    """bind_route's verdict for one route.

    ``binding`` is None for a handler shape that is not directly dispatchable, with ``reasons``
    saying why. ``body_model`` is resolved even when ``binding`` is None, so the harvested
    tools/list schema stays typed for a route the ``mcp_tools()`` override may drop.
    """

    binding: Optional[DirectBinding]
    reasons: list[str]
    body_model: Optional[type[BaseModel]]


def bind_route(route: APIRoute) -> BindResult:
    """Classify one route's handler signature for direct dispatch (see :class:`BindResult`).
    Public introspection only.

    Annotation resolution matches FastAPI's own: ``inspect.signature`` first (it honors a
    factory-set ``__signature__`` — some servers rewrite it with the real body model while
    ``__annotations__`` still says ``Any``), falling back to ``get_type_hints`` only for deferred
    string annotations (``from __future__ import annotations``).

    Pure (no app state is mutated), and called once per route: by ``harvest_tools`` for each typed
    POST route, and by ``_CatchAll.tool`` the first time an override mints a catch-all-backed tool.
    This is the deferred-validation pattern: a route the ``mcp_tools()`` override drops is never
    required to be dispatchable, so bind failures must surface only for tools actually exposed —
    harvest keeps the ``binding`` (``None`` for an undispatchable route) and the ``reasons`` on the
    ``MCPTool``, and ``_validate_tools`` raises with those stored reasons iff such a tool is exposed.
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
        return raw  # concrete object on the signature wins (FastAPI reads the signature too)

    request_params: list[str] = []
    body_param: Optional[str] = None
    body_model: Optional[type[BaseModel]] = None
    body_is_dict = False
    path_param: Optional[str] = None

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
        if get_origin(annotation) in (Union, UnionType):
            # ``body: Optional[Model]``/``Model | None`` reaches FastAPI as a body param, but direct
            # MCP dispatch has no proven-equivalent unwrapping for it, so refuse rather than guess.
            members = [a for a in get_args(annotation) if a is not type(None)]
            if any(isinstance(m, type) and issubclass(m, BaseModel) for m in members):
                reasons.append(f"union/optional body param {name!r} is not supported over MCP: {annotation!r}")
                continue
        if annotation is dict or get_origin(annotation) is dict:
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
            default_type = f"{type(param.default).__module__}.{type(param.default).__name__}"
            if default_type.startswith("fastapi."):
                reasons.append(f"DI marker default on {name!r}: {default_type}")
            else:
                reasons.append(f"defaulted query param {name!r} is not supported over MCP: {annotation!r}")
            continue
        reasons.append(f"unsupported required param {name!r}: {annotation!r}")

    # FastAPI filters responses with an explicit ``response_model=`` when given, so it wins over the
    # return annotation for parity filtering.
    if isinstance(route.response_model, type) and issubclass(route.response_model, BaseModel):
        return_model = route.response_model
    else:
        ret = resolve("return", signature.return_annotation)
        return_model = ret if isinstance(ret, type) and issubclass(ret, BaseModel) else None

    if reasons:
        return BindResult(binding=None, reasons=reasons, body_model=body_model)
    return BindResult(
        binding=DirectBinding(
            endpoint=endpoint,
            path=route.path,
            request_params=tuple(request_params),
            body_param=body_param,
            body_model=body_model,
            path_param=path_param,
            return_model=return_model,
            body_is_dict=body_is_dict,
            is_coroutine=inspect.iscoroutinefunction(inspect.unwrap(endpoint)),
        ),
        reasons=[],
        body_model=body_model,
    )


# ==================================================================================================
# PER-CALL — dispatch a frozen handler: fabricate the Request, call the route handler once
# ==================================================================================================


class DirectDispatchError(Exception):
    """Wraps a handler-visible failure, carrying the HTTP status and detail the plain route would have
    returned, so call_tool can surface the same text in the MCP isError result."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status} (direct): {detail}")
        self.status = status
        self.detail = detail


async def call_direct(
    app: FastAPI, binding: DirectBinding, session_id: str, arguments: dict, path_value: Optional[str] = None
) -> Any:
    """Invoke the handler resolved at startup once and return its JSON-able payload.

    Replicates what skipping Gym's own stack would otherwise lose: SessionMiddleware + add_session_id
    become ``scope["session"] = {SESSION_ID_KEY: sid}`` (handlers only read request.session); the
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
        # The body is always the serialized arguments — even when a body model exists — because a
        # handler may take the model and still read ``await request.json()``, which over HTTP would
        # see the same bytes FastAPI validated the model from.
        raw = json.dumps(arguments or {}).encode("utf-8")
        headers = [(b"content-type", b"application/json")]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": binding.path if path_value is None else "/" + path_value,
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 0),
            "server": ("internal-mcp-direct", 80),
            "state": {},
            "app": app,
            "session": {SESSION_ID_KEY: session_id},
        }

        async def receive() -> dict:
            return {"type": "http.request", "body": raw, "more_body": False}

        request = Request(scope, receive)
        for name in binding.request_params:
            kwargs[name] = request

    try:
        if binding.is_coroutine:
            result = await binding.endpoint(**kwargs)
        else:
            # FastAPI runs sync (def) handlers in a threadpool; do the same so one blocking tool
            # does not stall every concurrent rollout on this event loop.
            result = await run_in_threadpool(binding.endpoint, **kwargs)
        # Not a double-await: the call above resolves one level. A sync (def) handler can itself
        # return a coroutine unawaited — e.g. a thin def wrapper whose body is `return some_async(...)`
        # — and the threadpool hands that coroutine straight back, so await it here. A plain value is
        # not awaitable and skips this.
        if inspect.isawaitable(result):
            result = await result
    except StarletteHTTPException as e:  # fastapi.HTTPException subclasses this
        raise DirectDispatchError(e.status_code, str(e.detail)) from e
    except ClientResponseError as e:
        detail = getattr(e, "response_content", None)
        raise DirectDispatchError(500, f"Hit an exception calling an inner server: {detail or e}") from e
    except Exception as e:
        # Over plain HTTP, Gym's exception middleware logs the traceback and returns repr(e) with a
        # 500; reproduce both so the model reads the same text and the server keeps evidence.
        LOG.exception("Unhandled exception dispatching %s directly over MCP", binding.path)
        raise DirectDispatchError(500, repr(e)) from e

    if isinstance(result, Response):  # e.g. a handler returning PlainTextResponse
        text = bytes(result.body).decode("utf-8", errors="replace")
        if not 200 <= result.status_code < 300:
            raise DirectDispatchError(result.status_code, text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if binding.return_model is not None:
        # FastAPI's response_model filtering dumps the returned object and re-validates it against
        # the declared model; skipping this when isinstance already matches would leak subclass
        # fields the plain HTTP route hides.
        data = result.model_dump() if isinstance(result, BaseModel) else result
        try:
            result = binding.return_model.model_validate(data)
        except ValidationError as e:
            raise DirectDispatchError(500, json.dumps(jsonable_encoder(e.errors()))) from e
    return jsonable_encoder(result)


# ==================================================================================================
# STARTUP — routes to tool map: one walk over app.routes -> {name: MCPTool} (advertisement + binding)
# ==================================================================================================


@dataclass
class MCPTool:
    """Holds one exposed tool: its tools/list advertisement plus how to dispatch it (binding +
    optional catch-all path value); built during STARTUP harvesting, read per-call by call_tool."""

    name: str
    tool: types.Tool  # the tools/list advertisement
    binding: Optional[DirectBinding] = None  # how to invoke the route handler directly; None -> unbindable
    path_value: Optional[str] = None  # catch-all tools: value bound to the path param
    path: Optional[str] = None  # source route path, for bind-failure error messages
    reasons: tuple[str, ...] = ()  # bind-failure reasons from harvest time, read by _validate_tools


class _CatchAll:
    """Holds the single parameterized catch-all route; built during STARTUP harvesting and handed to
    ``mcp_tools()`` overrides so they can mint catch-all-backed tools.

    ``tool(name, input_schema, description)`` binds one MCP tool to that route with its path param set
    to ``name`` (workplace's ``POST /{path}`` pattern), reusing the same direct-dispatch binding.
    """

    def __init__(self, server: Any, route: APIRoute):
        self.server = server
        self.route = route
        self._binding: Optional[DirectBinding] = None

    def tool(self, name: str, input_schema: Optional[dict] = None, description: Optional[str] = None) -> MCPTool:
        if self._binding is None:
            # First catch-all-backed tool for this route: bind it once and cache; on failure the
            # reasons drive the error (see bind_route's deferred-validation note).
            bound = bind_route(self.route)
            if bound.binding is None:
                raise ValueError(
                    f"{type(self.server).__name__} catch-all route {self.route.path!r} cannot be dispatched "
                    f"directly: {'; '.join(bound.reasons)}. Direct MCP dispatch does not reproduce this handler shape."
                )
            self._binding = bound.binding
        return MCPTool(
            name=name,
            tool=types.Tool(name=name, description=description, inputSchema=input_schema or dict(PERMISSIVE_SCHEMA)),
            binding=self._binding,
            path_value=name,
            path=self.route.path,
        )


def harvest_tools(app: FastAPI, server: Any) -> dict[str, MCPTool]:
    """Scan app.routes once; return {tool name -> MCPTool}. Also runs the server-level middleware gate.

    Each non-parameterized typed POST route becomes a harvested tool. The single parameterized
    catch-all route (if any) is offered to the server via ``mcp_tools(harvested, catchall)``, whose
    return value is the final tool list — the default returns ``harvested`` unchanged; an override may
    filter it, append ``catchall.tool(...)`` entries, or return ``None``/``[]`` to expose nothing.
    """
    custom_middleware: list[str] = []
    for m in app.user_middleware:
        cls = m.cls
        if f"{cls.__module__}.{cls.__name__}" == "starlette.middleware.sessions.SessionMiddleware":
            continue  # Gym's SessionMiddleware — replaced by a materialized session on direct dispatch
        dispatch = m.kwargs.get("dispatch")
        if dispatch is not None and getattr(dispatch, "__module__", None) in _GYM_MIDDLEWARE_MODULES:
            continue  # Gym's add_session_id / exception middleware
        custom_middleware.append(f"{cls.__module__}.{cls.__name__}")
    if custom_middleware:
        raise ValueError(
            f"{type(server).__name__} installs non-Gym middleware {custom_middleware}, which direct MCP "
            "dispatch would silently skip. Remove the middleware, or leave expose_tools_over_mcp off in the config."
        )

    harvested: list[MCPTool] = []
    catchall_routes: list[APIRoute] = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or "POST" not in (route.methods or set()):
            continue
        # Never tools. GET docs/openapi are excluded by the POST filter above; /mcp by path.
        if route.path.lstrip("/") in RESERVED_MCP_TOOL_NAMES or route.path == MCP_URL_PATH:
            continue
        if "{" in route.path:
            catchall_routes.append(route)
            continue
        name = route.path.lstrip("/")
        # Keep the binding + reasons; a None binding (undispatchable) only errors later, and only if
        # the override actually exposes this tool. See bind_route's deferred-validation note. The
        # schema comes from body_model, which survives a failed bind, so overrides see it typed.
        bound = bind_route(route)
        schema = bound.body_model.model_json_schema() if bound.body_model is not None else dict(PERMISSIVE_SCHEMA)
        description = (route.description or route.summary or "").strip() or None
        harvested.append(
            MCPTool(
                name=name,
                tool=types.Tool(name=name, description=description, inputSchema=schema),
                binding=bound.binding,
                path=route.path,
                reasons=tuple(bound.reasons),
            )
        )

    if len(catchall_routes) > 1:
        raise ValueError(
            f"{type(server).__name__} has multiple parameterized catch-all routes "
            f"{sorted(r.path for r in catchall_routes)}; MCP auto-exposure cannot tell which backs the tools. "
            "Collapse them to one, or leave expose_tools_over_mcp off in the config."
        )
    catchall = _CatchAll(server, catchall_routes[0]) if catchall_routes else None

    tools = _validate_tools(server, server.mcp_tools(harvested, catchall))

    if catchall is not None and not any(t.path_value is not None for t in tools.values()):
        LOG.warning(
            "%s has a parameterized catch-all route %r but no exposed MCP tool dispatches through it; tools "
            "behind that route are not callable over MCP (rollouts needing them would score 0). Override "
            "mcp_tools() to add catch-all-backed tools via harvested + [catchall.tool(...)].",
            type(server).__name__,
            catchall.route.path,
        )

    LOG.info("%s MCP: exposing %d tool(s) over direct dispatch", type(server).__name__, len(tools))
    return tools


def _validate_tools(server: Any, selected: Optional[list[MCPTool]]) -> dict[str, MCPTool]:
    """Validate the final tool list from ``mcp_tools()``: legal name, not reserved, unique, dispatchable."""
    tools: dict[str, MCPTool] = {}
    for tool in selected or []:
        name = tool.name
        if not _MCP_TOOL_NAME_RE.fullmatch(name):
            raise ValueError(
                f"{type(server).__name__} exposes MCP tool name {name!r}, which does not match "
                "^[A-Za-z0-9_-]+$; MCP clients reject such names and verify-time normalization cannot "
                "round-trip them. Rename the route or tool, or leave expose_tools_over_mcp off in the config."
            )
        if name in RESERVED_MCP_TOOL_NAMES:
            raise ValueError(
                f"{type(server).__name__} exposes MCP tool {name!r}, which collides with a reserved endpoint "
                f"name {sorted(RESERVED_MCP_TOOL_NAMES)}; rename the tool."
            )
        if name in tools:
            raise ValueError(f"Duplicate MCP tool name {name!r} in {type(server).__name__}.mcp_tools().")
        if tool.binding is None:
            # Exposed but unbindable: fail startup with the reasons recorded at harvest time,
            # naming the route and why (see bind_route's deferred-validation note).
            raise ValueError(
                f"{type(server).__name__} tool {name!r} (route {tool.path!r}) cannot be dispatched "
                f"directly: {'; '.join(tool.reasons)}. Direct MCP dispatch does not reproduce this handler shape."
            )
        tools[name] = tool
    return tools


# ==================================================================================================
# STARTUP — wrap /seed_session and /verify: wrap (never edit) the endpoints the app currently holds
# ==================================================================================================


def _take_route(app: FastAPI, path: str, why: str) -> tuple[int, APIRoute]:
    found = next(
        ((i, r) for i, r in enumerate(app.router.routes) if isinstance(r, APIRoute) and r.path == path),
        None,
    )
    if found is None:
        raise ValueError(f"expose_tools_over_mcp requires a {path} route ({why}), but the app has none.")
    return found


def _swap_route(app: FastAPI, idx: int, path: str, endpoint: Callable) -> None:
    app.post(path)(endpoint)
    app.router.routes[idx] = app.router.routes.pop()  # in-place swap keeps ordering vs catch-all routes


def _wrap_seed_session(app: FastAPI, mint_metadata: Callable[[Request, dict], dict]) -> None:
    """Replace the /seed_session route with a wrapper that appends the MCP session token to its response.

    The wrapper is a ``**kwargs`` function, but FastAPI decides which dependencies to inject by reading
    the endpoint's *signature* — so ``**kwargs`` alone would receive nothing. The fix is signature
    surgery: build a parameter list that (a) presents the original handler's params (with string
    annotations resolved to real types, so FastAPI still validates the body model) and (b) guarantees
    a ``Request`` param, since the wrapper needs the live Request to read the session and raw body when
    minting the token. That list is then stamped onto ``__signature__``/``__annotations__`` so FastAPI
    injects exactly those arguments into ``**kwargs``; ``passthrough`` records the original param names
    to forward to the real handler (the injected Request, if added, is not among them).
    """
    idx, route = _take_route(app, "/seed_session", "its response carries the MCP session token to the agent")
    method = route.endpoint
    signature = inspect.signature(method)
    hints = get_type_hints(method)
    # Does the original handler already declare a Request param? If so, reuse it; the wrapper reads
    # the Request out of that same name instead of injecting a second one.
    request_param_name = next(
        (n for n, p in signature.parameters.items() if hints.get(n, p.annotation) is Request), None
    )
    # The handler's own params, with deferred string annotations resolved so FastAPI sees real types.
    params = [p.replace(annotation=hints.get(n, p.annotation)) for n, p in signature.parameters.items()]
    passthrough = tuple(signature.parameters)  # names to forward verbatim to the original handler
    if request_param_name is None:
        # No Request declared: prepend one, under a name no real handler param can collide with.
        request_param_name = "__nemo_gym_request"
        params = [
            inspect.Parameter(request_param_name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request),
            *params,
        ]

    # FastAPI filters the response through the route's response_model; the wrapper must do the same
    # or subclass-only fields the original route hides would leak into the wrapped response.
    response_model = route.response_model
    if not (isinstance(response_model, type) and issubclass(response_model, BaseModel)):
        response_model = None

    async def seed_session_endpoint(**kwargs: Any) -> JSONResponse:
        request: Request = kwargs[request_param_name]
        result = method(**{k: kwargs[k] for k in passthrough})
        if inspect.isawaitable(result):
            result = await result
        if response_model is not None:
            data = result.model_dump() if isinstance(result, BaseModel) else result
            result = response_model.model_validate(data)
        payload = jsonable_encoder(result)
        if isinstance(payload, dict) and NEMO_GYM_MCP_METADATA_KEY not in payload:
            # request.body() returns FastAPI's cached bytes; the stream was consumed validating the body model.
            raw_body = await request.body()
            try:
                seed_body = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError:
                seed_body = {}
            if not isinstance(seed_body, dict):
                seed_body = {}
            payload[NEMO_GYM_MCP_METADATA_KEY] = mint_metadata(request, seed_body)
        return JSONResponse(payload)

    # Stamp the fabricated signature onto the **kwargs wrapper so FastAPI injects exactly `params`.
    seed_session_endpoint.__name__ = "seed_session"
    seed_session_endpoint.__signature__ = inspect.Signature(parameters=params)
    seed_session_endpoint.__annotations__ = {p.name: p.annotation for p in params}

    _swap_route(app, idx, "/seed_session", seed_session_endpoint)


def _wrap_verify(app: FastAPI, server: Any) -> None:
    """Wrap the app's current /verify endpoint so MCP-namespaced tool-call names are normalized for
    scoring only. Verification runs against a deep copy with bare names; the reward response's
    echoed names are restored to what the model emitted (matched by call_id), so persisted rollout
    artifacts keep transport provenance. Wrapping whatever handler the route holds at install time
    covers servers that strip and re-register /verify with their own handler.
    """
    idx, route = _take_route(app, "/verify", "its tool-call names are normalized for scoring")
    endpoint = route.endpoint

    def _function_calls(container: Any) -> list:
        return [
            item
            for item in (getattr(getattr(container, "response", None), "output", None) or [])
            if getattr(item, "type", None) == "function_call"
        ]

    def _locate_trajectory_arg(kwargs: dict) -> Optional[tuple[str, Any]]:
        """Find the single argument that carries the trajectory (has function_call output items).

        Verify signatures vary ((body), (request, body), ...), so the trajectory argument is found by
        content, not name; FastAPI always invokes endpoints with keyword arguments. Returns
        ``(name, container)``, or ``None`` when no argument carries a trajectory.
        """
        for name, value in kwargs.items():
            if _function_calls(value):
                return name, value
        return None

    @functools.wraps(endpoint)
    async def verify_normalized(**kwargs: Any) -> Any:
        located = _locate_trajectory_arg(kwargs)
        if located is None:
            result = endpoint(**kwargs)
            return await result if inspect.isawaitable(result) else result
        key, container = located

        emitted = {item.call_id: item.name for item in _function_calls(container)}
        normalized = container.model_copy(deep=True)
        for item in _function_calls(normalized):
            item.name = server.normalize_tool_name(item.name)
        kwargs[key] = normalized

        result = endpoint(**kwargs)
        if inspect.isawaitable(result):
            result = await result

        for item in _function_calls(result):
            if item.call_id in emitted:
                item.name = emitted[item.call_id]
        return result

    _swap_route(app, idx, "/verify", verify_normalized)


# ==================================================================================================
# STARTUP — mount /mcp (flag-gated entry point + installer); PER-CALL handlers (session_claims,
# list_tools, call_tool) are defined here but run once per MCP request
# ==================================================================================================


def _mint_session_metadata(server: Any, serializer: URLSafeSerializer, request: Request, seed_body: dict) -> dict:
    """Mint the session token embedded in a /seed_session response (see ``_wrap_seed_session``).

    Assigns the rollout its session id, asks the server for any per-session tool narrowing, and
    signs ``sid`` + allow-list into the token the agent replays on every ``tools/call``.
    """
    session_id = request.session.get(SESSION_ID_KEY)
    if not session_id:
        session_id = str(uuid4())
        request.session[SESSION_ID_KEY] = session_id
    # A raising hook propagates and fails the seed request — no token is minted past a broken hook.
    session_allowed = server.mcp_allowed_tools_for_session(seed_body)
    payload = {"sid": session_id, "tools": session_allowed}
    return MCPServerMetadata(
        server_name=server.config.name or type(server).__name__,
        url_path=MCP_URL_PATH,
        transport="http",
        headers={NEMO_GYM_MCP_SESSION_TOKEN_HEADER: serializer.dumps(payload)},
    ).model_dump()


def _parse_session_token(
    serializer: URLSafeSerializer, token: Optional[str], required: bool
) -> tuple[Optional[str], Optional[frozenset]]:
    """Decode a Gym MCP session token into ``(session id, token allow-list)``.

    ``required`` callers (tools/call) raise on a missing/forged token; optional callers (tools/list)
    fall back to ``(None, None)``. A ``tools`` of None means the session is unrestricted.
    """
    if not token:
        if required:
            raise ValueError(f"Missing {NEMO_GYM_MCP_SESSION_TOKEN_HEADER} for Gym MCP tool call.")
        return None, None
    try:
        # Verified per call: caching claims per token would grow one entry per rollout with nothing to evict it.
        payload = serializer.loads(token)
    except BadSignature:
        if required:
            raise ValueError("Invalid Gym MCP session token.")
        return None, None
    allowed = payload.get("tools")
    return payload["sid"], None if allowed is None else frozenset(allowed)


def _to_result(payload: Any):
    """Per the SDK's call_tool contract, a ``(content, structuredContent)`` tuple populates both
    CallToolResult fields while a bare content list leaves structuredContent unset — and only a JSON
    object maps to structuredContent."""
    text = payload if isinstance(payload, str) else JSONResponse(payload).body.decode("utf-8")
    content = [types.TextContent(type="text", text=text)]
    return (content, payload) if isinstance(payload, dict) else content


def maybe_auto_expose(server: Any, app: FastAPI) -> Optional[dict[str, MCPTool]]:
    """Install MCP auto-exposure iff the server opts in (``expose_tools_over_mcp: true`` in the config).

    Called by ``run_webserver`` after the app is fully built, so every route is present. Returns the
    tool map (for tests/introspection), or None when the server did not opt in.
    """
    if not getattr(getattr(server, "config", None), "expose_tools_over_mcp", False):
        return None
    return install_auto_exposure(server, app)


def install_auto_exposure(server: Any, app: FastAPI) -> dict[str, MCPTool]:
    """Harvest the tool routes, wire the /seed_session token, and mount the /mcp endpoint.

    ``server`` is any resources server built exactly as on main; ``app`` is the FastAPI app its
    unmodified ``setup_webserver()`` returned. Returns the tool map.
    """
    # A second /mcp inserted at the front would shadow a pre-existing /mcp mount
    # and silently drop its tools.
    preexisting_mcp = [
        r for r in app.router.routes if isinstance(r, (Route, Mount)) and getattr(r, "path", None) == MCP_URL_PATH
    ]
    if preexisting_mcp:
        raise ValueError(
            f"{type(server).__name__} already serves {MCP_URL_PATH}, which conflicts with MCP auto-exposure "
            "on the same server. Remove the hand-rolled /mcp mount and rely on expose_tools_over_mcp."
        )

    secret = server.get_session_middleware_key()
    serializer = URLSafeSerializer(secret, salt=_MCP_TOKEN_SALT)
    tools = harvest_tools(app, server)

    mint_metadata = functools.partial(_mint_session_metadata, server, serializer)
    _wrap_seed_session(app, mint_metadata)
    _wrap_verify(app, server)

    mcp_server = _LowLevelMCPServer(server.config.name or type(server).__name__)

    # mcp_server.request_context is a contextvar the SDK populates only while one of the handlers
    # below is on the stack — exactly when session_claims runs, which is why it stays local.
    def session_claims(required: bool = True) -> tuple[Optional[str], Optional[frozenset]]:
        ctx_request = mcp_server.request_context.request  # the POST /mcp starlette Request
        token = ctx_request.headers.get(NEMO_GYM_MCP_SESSION_TOKEN_HEADER) if ctx_request is not None else None
        return _parse_session_token(serializer, token, required)

    @mcp_server.list_tools()
    async def list_tools() -> list[types.Tool]:
        _, allowed = session_claims(required=False)
        return [t.tool for t in tools.values() if allowed is None or t.name in allowed]

    @mcp_server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict):
        tool = tools.get(name)
        if tool is None:
            # Models hallucinate tool names, so answer the miss cheaply with a plain isError result
            # instead of raising. The SDK still refreshes its tool cache and logs one warning per
            # unknown name before this handler runs; that happens in its wrapper, out of our reach.
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text=f"Unknown tool: {name!r}. Available tools: {sorted(tools)}")
                ],
                isError=True,
            )
        session_id, allowed = session_claims(required=True)
        if allowed is not None and name not in allowed:
            raise ValueError(f"Tool {name!r} is not allowed for this session.")
        try:
            payload = await call_direct(app, tool.binding, session_id, arguments, path_value=tool.path_value)
        except DirectDispatchError as exc:
            raise ValueError(f"HTTP {exc.status} from POST /{name}: {exc.detail}")
        return _to_result(payload)

    manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=True,
        # The agent reaches this endpoint server-to-server via the resources server's resolved host
        # (a routable IP/hostname on multi-node runs); enabling the SDK's Host/Origin allowlist would
        # mean enumerating every such host. DNS-rebinding protection defends browsers, which never
        # talk to this endpoint; disabling it is not what makes the endpoint safe.
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    class _MCPEndpoint:
        async def __call__(self, scope, receive, send):
            await manager.handle_request(scope, receive, send)

    endpoint = _MCPEndpoint()
    # Insert at the front so a dispatcher's catch-all POST /{path} cannot shadow POST /mcp.
    #
    # Two registrations for the same ASGI endpoint, because clients address /mcp two ways and one
    # registration alone would miss the other:
    #   * Route("/mcp") matches the bare path exactly ("/mcp", no trailing segment).
    #   * Mount("/mcp") matches "/mcp" plus any subpath ("/mcp/", "/mcp/messages", ...).
    # Both forward to the same _MCPEndpoint, so the SDK app receives the request whether a client
    # posts to the exact path or to a subpath, with no 307 redirect in between.
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
