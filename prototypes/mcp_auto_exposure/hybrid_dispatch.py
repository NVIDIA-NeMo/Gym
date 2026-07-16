# SPDX-License-Identifier: Apache-2.0
"""Hybrid dispatch (spike): direct handler invocation where provably safe, replay where not.

Track: WHAT DIRECT DISPATCH GIVES UP, AND THE HYBRID SAFETY NET.

Direct dispatch = call the frozen origin/main handler function ONCE, with a fabricated
starlette Request whose scope carries a pre-populated session. No second pass through the
app's middleware/routing/validation stack (that second pass is Design B's replay bridge).

The DETECTOR decides per server and per route, at startup, whether direct dispatch is
provably equivalent to a replay. Two independent gates:

  Gate 1 (server-level): the app's middleware stack must contain ONLY Gym's own middleware
      (SessionMiddleware + the nemo_gym.server_utils @app.middleware functions). Any other
      middleware means an env author installed per-request behavior that direct dispatch
      would silently skip -> the WHOLE server falls back to replay.
  Gate 2 (route-level): every handler parameter must be one of the shapes real Gym handlers
      use — `request: Request`, one pydantic BaseModel body param, a str path param, or a
      defaulted param whose default is not a Depends/Security marker. Anything else (true
      DI, BackgroundTasks, UploadFile, ...) -> THAT route falls back to replay; the rest of
      the server stays direct.

Public-surface inventory of THIS module (the honest list):
  PUBLIC:  inspect.signature / typing.get_type_hints on the endpoint; pydantic
           model_validate / ValidationError; fastapi.encoders.jsonable_encoder;
           starlette Request(scope, receive) constructor; Response.body/.status_code;
           starlette.middleware.Middleware .cls/.kwargs (documented data-holder);
           fastapi/starlette HTTPException fields.
  SEMI:    app.user_middleware (stable Starlette attribute since 0.13, not in the docs
           index; the productized alternative is fully public — Gym builds the app, so
           SimpleServer.setup_webserver can wrap the instance's public add_middleware to
           RECORD anything an env author adds after Gym's own stack);
           route.endpoint / route.path / route.methods (Starlette Route attributes mirroring
           documented constructor args);
           scope["session"] (the key SessionMiddleware documents itself as providing;
           request.session is the public reader — the WRITE side has no public spelling).
  NOT USED: fastapi.dependencies.utils.solve_dependencies, Dependant, route.body_field,
           route.dependant — none of FastAPI's DI internals are touched for dispatch.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints

from aiohttp import ClientResponseError
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import Response

from nemo_gym.server_utils import SESSION_ID_KEY


# Middleware whose dispatch function lives in these modules is Gym's own stack
# (add_session_id, exception_handling_middleware) — replicated by the dispatcher, so its
# absence under direct dispatch is compensated, not lost.
GYM_MIDDLEWARE_MODULES = frozenset({"nemo_gym.server_utils"})
_PATH_PARAM_RE = re.compile(r"{([^}:]+)(?::[^}]+)?}")


# ==================================================================================================
# Gate 1: middleware audit (server-level)
# ==================================================================================================


def audit_middleware(app: FastAPI) -> list[str]:
    """Return the names of NON-Gym middleware installed on the app (empty == direct-safe).

    ``app.user_middleware`` is the semi-public read (see module docstring for the fully
    public productization: wrap the app instance's public ``add_middleware`` inside
    ``SimpleServer.setup_webserver`` and record what env authors add).
    Each entry is a ``starlette.middleware.Middleware`` data holder: ``.cls`` is the
    middleware class, ``.kwargs`` its constructor kwargs (``dispatch=fn`` for
    ``@app.middleware("http")`` functions).
    """
    custom: list[str] = []
    for m in app.user_middleware:
        cls = m.cls
        if f"{cls.__module__}.{cls.__name__}" == "starlette.middleware.sessions.SessionMiddleware":
            continue  # Gym's SessionMiddleware — replaced by scope["session"] seeding
        dispatch = m.kwargs.get("dispatch")
        if dispatch is not None and getattr(dispatch, "__module__", None) in GYM_MIDDLEWARE_MODULES:
            continue  # Gym's add_session_id / exception_handling_middleware
        custom.append(f"{cls.__module__}.{cls.__name__}")
    return custom


# ==================================================================================================
# Gate 2: route binding (route-level)
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
    needs_raw_body: bool = False  # handler reads await request.json() (no body model)
    body_is_dict: bool = False  # handler declares ``body: dict`` — FastAPI passes the parsed
    #                             JSON through unvalidated; direct dispatch passes ``arguments``


@dataclass
class BindOutcome:
    binding: Optional[DirectBinding]
    reasons: list[str] = field(default_factory=list)  # non-empty == replay this route


def bind_route(route: APIRoute) -> BindOutcome:
    """Classify one route's handler signature. PUBLIC introspection only.

    Annotation resolution matches FastAPI's own: ``inspect.signature`` first (it honors a
    factory-set ``__signature__`` — newton_bench rewrites __signature__ with the real per-module
    body model while ``__annotations__`` still says ``Any``), falling back to ``get_type_hints``
    only for deferred string annotations (``from __future__ import annotations``).
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
            # validation (openenv's non-MCP /step). Direct equivalent: pass ``arguments`` as-is.
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
            # FastAPI treats these as query params; MCP calls carry no query string, so the
            # HTTP door would hand the handler the default too — matching direct behavior.
            # EXCEPT DI markers (Depends/Security), which the HTTP door would resolve.
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
        return BindOutcome(None, reasons)
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
    )


# ==================================================================================================
# Direct invocation: fabricate the Request, call the frozen handler ONCE
# ==================================================================================================


def _make_receive(body: bytes):
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


class DirectDispatchError(Exception):
    """Wraps handler-visible failures with the same semantics the replay bridge's isError has."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status} (direct) from {'POST'}: {detail}")
        self.status = status
        self.detail = detail


async def call_direct(
    app: FastAPI,
    binding: DirectBinding,
    session_id: str,
    arguments: dict,
    path_value: Optional[str] = None,
    cookie_header: Optional[bytes] = None,
) -> Any:
    """Invoke the frozen handler once. Returns the JSON-able response payload.

    Replicates what skipping Gym's own stack would otherwise lose:
      * SessionMiddleware + add_session_id  -> scope["session"] = {SESSION_ID_KEY: sid}
        (handlers only ever READ request.session[SESSION_ID_KEY] — grep-verified);
      * exception_handling_middleware       -> exceptions propagate to the MCP SDK's
        call_tool wrapper, which already maps ANY Exception to isError(str(e)); we
        pre-format HTTPException/ValidationError/ClientResponseError so the text carries
        the same status information the replay bridge's isError text carries;
      * the session cookie the HTTP door would carry -> pass ``cookie_header`` (the minted,
        per-session-cached SessionMiddleware cookie) so a handler that forwards
        ``request.cookies`` downstream (the CLAUDE.md stateful-env pattern; today only
        AGENT servers do this, no resources server does) sees exactly what replay sees.
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
        headers = [(b"content-type", b"application/json")]
        if cookie_header is not None:
            headers.append((b"cookie", cookie_header))
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
            # SessionMiddleware's documented effect, materialized directly: the session the
            # cookie round-trip would have produced for this rollout's session id.
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
        # Gym's exception_handling_middleware special-case, replicated (its message shape).
        detail = getattr(e, "response_content", None)
        raise DirectDispatchError(500, f"Hit an exception calling an inner server: {detail or e}") from e

    if isinstance(result, Response):  # litmus_agent / ns_tools return PlainTextResponse
        text = bytes(result.body).decode("utf-8", errors="replace")
        if not 200 <= result.status_code < 300:
            raise DirectDispatchError(result.status_code, text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if binding.return_model is not None and not isinstance(result, binding.return_model):
        # Parity with FastAPI's return-annotation-as-response_model filtering.
        result = binding.return_model.model_validate(result)
    return jsonable_encoder(result)


# ==================================================================================================
# The hybrid plan: detector output + one call surface
# ==================================================================================================


@dataclass
class ToolPlan:
    name: str
    mode: str  # "direct" | "replay"
    path: str
    binding: Optional[DirectBinding] = None
    path_value: Optional[str] = None  # catch-all tools: value for the path param
    reasons: list[str] = field(default_factory=list)


@dataclass
class HybridPlan:
    server_mode: str  # "direct" | "replay" (Gate 1)
    custom_middleware: list[str]
    tools: dict[str, ToolPlan]

    def summary(self) -> dict:
        return {
            "server_mode": self.server_mode,
            "custom_middleware": self.custom_middleware,
            "tools": {n: {"mode": t.mode, "reasons": t.reasons} for n, t in self.tools.items()},
        }


def plan_hybrid(
    app: FastAPI,
    typed_tool_paths: dict[str, str],
    catchall_tools: Optional[dict[str, str]] = None,  # tool name -> catch-all route path
) -> HybridPlan:
    """The DETECTOR. Runs once at startup; every decision is inspectable in .summary()."""
    custom = audit_middleware(app)
    server_mode = "replay" if custom else "direct"
    routes_by_path: dict[str, APIRoute] = {
        r.path: r for r in app.routes if isinstance(r, APIRoute) and "POST" in (r.methods or set())
    }

    tools: dict[str, ToolPlan] = {}

    def add(name: str, route_path: str, path_value: Optional[str]) -> None:
        route = routes_by_path[route_path]
        if server_mode == "replay":
            tools[name] = ToolPlan(name, "replay", route_path, reasons=[f"custom middleware: {custom}"])
            return
        outcome = bind_route(route)
        if outcome.binding is None:
            tools[name] = ToolPlan(name, "replay", route_path, reasons=outcome.reasons)
        else:
            tools[name] = ToolPlan(name, "direct", route_path, binding=outcome.binding, path_value=path_value)

    for name, route_path in typed_tool_paths.items():
        add(name, route_path, None)
    for name, route_path in (catchall_tools or {}).items():
        add(name, route_path, name)
    return HybridPlan(server_mode=server_mode, custom_middleware=custom, tools=tools)


async def hybrid_call(
    app: FastAPI,
    plan: HybridPlan,
    replay_fn: Callable,  # async (path: str, session_id: str, arguments: dict) -> payload
    name: str,
    session_id: str,
    arguments: dict,
) -> Any:
    tool = plan.tools[name]
    if tool.mode == "direct":
        return await call_direct(app, tool.binding, session_id, arguments, path_value=tool.path_value)
    replay_path = tool.path if tool.path_value is None else "/" + tool.path_value
    return await replay_fn(replay_path, session_id, arguments)
