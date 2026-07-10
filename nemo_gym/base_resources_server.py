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
import functools
import inspect
import json
import logging
from abc import abstractmethod
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional, get_type_hints
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel, PrivateAttr, ValidationError, create_model
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.server_utils import SESSION_ID_KEY, BaseRunServerInstanceConfig, BaseServer, SimpleServer


LOG = logging.getLogger(__name__)

NEMO_GYM_MCP_SESSION_TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
NEMO_GYM_MCP_METADATA_KEY = "mcp"
_MCP_SESSION_TOKEN: ContextVar[Optional[str]] = ContextVar("nemo_gym_mcp_session_token", default=None)
# Salt namespacing the signed MCP session token, so it can't be confused with another signer
# that happens to share the same session-middleware secret.
_MCP_TOKEN_SALT = "nemo-gym-mcp-session-token"


class MCPSessionError(Exception):
    """A Gym MCP tool call lacked a valid per-rollout session token.

    Deliberately not an HTTP error: MCP runs over JSON-RPC, so the transport returns HTTP 200 and
    surfaces this to the client as a tool error (``isError: true``). An HTTP status code raised here
    would never reach the caller, so we raise a plain error with a clear message instead.
    """


# Names a gym_tool may not use, because they collide with the resources server's own
# endpoints (and would silently shadow them on HTTP while still registering as MCP tools).
RESERVED_MCP_TOOL_NAMES = frozenset({"verify", "seed_session", "aggregate_metrics", "mcp"})


class GymToolSpec(BaseModel):
    """Declaration attached to a callable by ``gym_tool``; drives both transports.

    ``input_schema`` decides where the tool's advertised schema comes from:
    ``None`` — introspect the callable's typed parameters (signature = schema);
    a JSON-schema ``dict`` — advertise it verbatim over MCP and pass call arguments through raw;
    a Pydantic model class — the model's fields are the schema, arguments validate into an instance.
    """

    name: str
    description: Optional[str] = None
    input_schema: Optional[Any] = None
    validate_input: bool = False


@dataclass(slots=True)
class _RawToolEntry:
    """Precomputed dispatch record for a dict/model-schema gym tool, built once at setup so the MCP
    hot path (tools/list, tools/call) rebuilds nothing per request."""

    fn: Callable
    model: Optional[type[BaseModel]]  # the Pydantic input model (model schema) or None (dict schema)
    body_param: Optional[str]  # name of the single non-session param (model schema) or None
    validator: Optional[type[BaseModel]]  # shallow validator (dict schema + validate=True) or None
    inject_session: bool
    is_coro: bool
    tool: Any  # precomputed mcp.types.Tool advertised by tools/list


def gym_tool(
    fn: Optional[Callable] = None,
    /,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    input_schema: Optional[Any] = None,
    validate: bool = False,
    owner: Optional["SimpleResourcesServer"] = None,
):
    """Declare a callable as a Gym tool, served over both transports: MCP and HTTP ``POST /<name>``.

    This is the single tool-registration entry point. Two binding times, one API:

    - Decorator on a typed method (bare or with kwargs)::

        @gym_tool
        async def get_weather(self, session_id: str, city: str) -> GetWeatherResponse: ...

    - Runtime call for tools that only exist after startup (registries, discovered tool sets),
      typically from ``model_post_init``::

        gym_tool(closure, name=tool.name, description=tool.description,
                 input_schema=tool.input_schema, owner=self)

    Contract (both transports):
    - ``session_id``: declare a ``session_id: str`` parameter when the tool reads or writes
      per-rollout state; leave it out for stateless tools. The base fills it in itself — from the
      session cookie on HTTP, from the signed session token on MCP — so it never appears in the
      model-facing schema, and a ``session_id`` key sent in the payload is ignored.
    - Never take a ``request`` parameter: on the MCP path there is no FastAPI ``Request`` object to
      pass (the MCP library calls the function directly with the JSON-RPC arguments).
    - ``input_schema=None`` (the default): the schema is derived from the function's typed
      parameters, so write them flat — ``def get_weather(self, city: str)`` gives callers the
      argument shape ``{"city": ...}`` on both transports. Do not wrap the arguments in a single
      Pydantic parameter (``def get_weather(self, body: WeatherArgs)``): the schema derives from
      the parameter list, so MCP callers would have to send ``{"body": {"city": ...}}`` while HTTP
      callers send the flat form — one tool, two conflicting argument shapes.
    - ``input_schema=<dict>``: for tools whose JSON schema already exists as data (registries,
      discovered tool sets). The dict is advertised over MCP exactly as given, and call arguments
      reach the callable unmodified on both transports — re-validating against a derived model
      would silently drop argument names the schema doesn't declare. ``validate=True`` adds a
      shallow type check (422 on mismatch) for the HTTP route.
    - ``input_schema=<Pydantic model class>``: for tools that already had a hand-written request
      model. The model's fields become the schema (callers still send flat arguments), the base
      validates them into an instance, and the callable receives that instance as its single
      parameter besides ``session_id`` — the same object a hand-written FastAPI route received, so
      a migrated route keeps its exact validation behavior.
    - Return values: ``str`` is served as ``text/plain`` over HTTP; models and dicts as JSON.
    - Sync callables run in a threadpool on both transports so they cannot stall the event loop.
    - Error shapes differ per transport because the protocols do. MCP is JSON-RPC, which separates
      transport success from tool failure: an exception becomes a tool result with
      ``isError: true`` inside an HTTP 200, which the client hands to the model as tool output.
      Plain HTTP has only status codes, so the same exception surfaces as a 500 with ``repr(e)``.
      Session presence differs the same way: MCP without a token is a clean tool error, while HTTP
      without a cookie simply mints a fresh (unseeded) session id — stateful tools should keep a
      seeded-session guard in their body.
    """

    def apply(func: Callable) -> Callable:
        tool_name = name or getattr(func, "__name__", None)
        if not tool_name:
            raise ValueError("gym_tool requires name= for callables without a __name__.")
        func.__gym_tool__ = GymToolSpec(
            name=tool_name,
            description=description if description is not None else ((func.__doc__ or "").strip() or None),
            input_schema=input_schema,
            validate_input=validate,
        )
        if owner is not None:
            owner._dynamic_gym_tools.append(func)
        return func

    return apply if fn is None else apply(fn)


def normalize_tool_name(name: str, server_name: Optional[str] = None) -> str:
    """Map a trajectory tool-call name to the server's bare tool name.

    HTTP-driven agents record bare tool names ("email_reply_email"); MCP-native agents (e.g.
    Claude Code) record them namespaced per server ("mcp__workplace_assistant__email_reply_email").
    Verifiers that compare trajectory names against dataset/ground-truth vocabulary should
    normalize first so rollouts score identically on both transports. Non-namespaced names pass
    through unchanged. When ``server_name`` is given, only that server's prefix is stripped
    (robust to tool names that themselves contain double underscores).
    """
    if not name.startswith("mcp__"):
        return name
    if server_name is not None:
        prefix = f"mcp__{server_name}__"
        return name[len(prefix) :] if name.startswith(prefix) else name
    _, sep, tool = name[len("mcp__") :].partition("__")
    return tool if sep else name


class BaseResourcesServerConfig(BaseRunServerInstanceConfig):
    pass


class BaseResourcesServer(BaseServer):
    config: BaseResourcesServerConfig


class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float


class BaseSeedSessionRequest(BaseModel):
    pass


class BaseSeedSessionResponse(BaseModel):
    pass


class MCPServerMetadata(BaseModel):
    """Metadata returned from /seed_session for per-rollout Gym MCP access."""

    server_name: str
    url_path: str = "/mcp"
    transport: str = "http"
    headers: dict[str, str]


class _MCPHeaderSessionMiddleware:
    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        token = Headers(scope=scope).get(NEMO_GYM_MCP_SESSION_TOKEN_HEADER)
        context_token = _MCP_SESSION_TOKEN.set(token)
        try:
            await self.app(scope, receive, send)
        finally:
            _MCP_SESSION_TOKEN.reset(context_token)


class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    """Resources server base: /seed_session, /verify, /aggregate_metrics — and, lazily, Gym tools.

    Declare tools with ``gym_tool`` (see its docstring for the full contract). Each declared tool is
    served over both transports: an HTTP ``POST /<name>`` route and an MCP tool on ``mcp_url_path``.
    The MCP endpoint (and the ``mcp`` package import) only exists when the server declares at least
    one tool or overrides ``register_mcp_tools``; a tool-less server's app is unchanged.

    Per-rollout sessions thread through both transports to the same session id: ``/seed_session``
    responses are auto-augmented with :class:`MCPServerMetadata` (a signed session token the agent
    sends back as the ``X-NeMo-Gym-Session-Token`` header), and HTTP calls carry the session cookie.
    Pass ``allowed_tools`` to :meth:`build_mcp_session_metadata` to restrict which tools an MCP
    session can list and call.
    """

    config: BaseResourcesServerConfig

    mcp_url_path: str = "/mcp"

    _dynamic_gym_tools: list = PrivateAttr(default_factory=list)
    _raw_dispatch: dict = PrivateAttr(default_factory=dict)
    _gym_tool_names: tuple = PrivateAttr(default=())
    _cached_token_serializer: Any = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        gym_tools = self._collect_gym_tools()
        mcp_overridden = type(self).register_mcp_tools is not SimpleResourcesServer.register_mcp_tools
        mcp_enabled = bool(gym_tools) or mcp_overridden

        # Tool-less servers keep the exact pre-MCP app: plain seed_session, no /mcp, no mcp import.
        app.post("/seed_session")(self._build_seed_session_endpoint() if mcp_enabled else self.seed_session)
        app.post("/verify")(self.verify)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

        if mcp_enabled:
            self._setup_mcp(app, gym_tools)

        return app

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        return BaseSeedSessionResponse()

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Compute aggregate metrics from verify responses.

        RewardProfiler provides baseline stats. Override compute_metrics() and/or
        get_key_metrics() for benchmark-specific customization.
        """
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )

    # --------------------------------------------------------------------------------------------
    # Gym tool collection and MCP setup
    # --------------------------------------------------------------------------------------------

    def _collect_gym_tools(self) -> list[tuple[GymToolSpec, Callable]]:
        """Gather every gym_tool declaration: decorated class methods + runtime-registered callables."""
        tools: list[tuple[GymToolSpec, Callable]] = []
        for attr_name, func in inspect.getmembers(type(self), predicate=inspect.isfunction):
            spec = getattr(func, "__gym_tool__", None)
            if spec is not None:
                tools.append((spec, getattr(self, attr_name)))
        # Runtime-registered callables always carry the spec (gym_tool sets it before appending).
        tools.extend((func.__gym_tool__, func) for func in self._dynamic_gym_tools)

        seen: set[str] = set()
        for spec, _ in tools:
            if spec.name in RESERVED_MCP_TOOL_NAMES:
                raise ValueError(
                    f"@gym_tool method {spec.name!r} collides with a reserved endpoint name "
                    f"{sorted(RESERVED_MCP_TOOL_NAMES)}; rename the tool."
                )
            if spec.name in seen:
                raise ValueError(f"Duplicate gym_tool name {spec.name!r}; tool names must be unique per server.")
            seen.add(spec.name)
        return tools

    def _setup_mcp(self, app: FastAPI, gym_tools: list[tuple[GymToolSpec, Callable]]) -> None:
        try:
            from mcp.server.fastmcp import FastMCP
            from mcp.server.transport_security import TransportSecuritySettings
        except ImportError as exc:  # pragma: no cover - exercised only without the optional runtime dependency
            raise RuntimeError("Gym tools require the official MCP Python SDK. Install the 'mcp' package.") from exc

        mcp = FastMCP(
            self.config.name or self.__class__.__name__,
            stateless_http=True,
            json_response=True,
            streamable_http_path="/",
            # The MCP SDK enables DNS-rebinding protection by default, which only accepts loopback
            # Host headers and returns HTTP 421 for anything else. Gym mounts this endpoint for
            # server-to-server access: the agent reaches it via the resources server's resolved host,
            # which is a routable IP/hostname when use_absolute_ip=True (required for multi-node runs).
            # The endpoint is already gated by the per-rollout X-NeMo-Gym-Session-Token, so we disable
            # Host/Origin validation to keep MCP tool calls working off-loopback.
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )

        # Precompute everything the MCP hot path would otherwise rebuild per request — the validator,
        # the advertised inputSchema, and the types.Tool object — so tools/list and tools/call reuse
        # them (mirrors the HTTP twin in _register_http_gym_tool, which already precomputes at setup).
        self._raw_dispatch = {}
        for spec, fn in gym_tools:
            if spec.input_schema is not None:
                self._check_no_request_param(spec.name, fn)
                self._raw_dispatch[spec.name] = self._build_raw_tool_entry(spec, fn)

        self.register_mcp_tools(mcp)
        self._install_mcp_list_handler(mcp)
        self._install_mcp_call_handler(mcp)

        for spec, fn in gym_tools:
            self._register_http_gym_tool(app, spec, fn)

        self._gym_tool_names = tuple(sorted(spec.name for spec, _ in gym_tools))
        self._warn_on_transport_parity_gap(mcp)

        main_app_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app: FastAPI):
            # The catch-all must land after every subclass-added route, and exactly once across
            # repeated lifespan cycles (TestClient re-entry) — hence registered here, guarded.
            self._ensure_unknown_tool_catchall(app)
            async with mcp.session_manager.run():
                async with main_app_lifespan(app) as maybe_state:
                    yield maybe_state

        app.router.lifespan_context = lifespan_wrapper
        mcp_app = mcp.streamable_http_app()
        streamable_http_route = next(route for route in mcp_app.routes if getattr(route, "path", None) == "/")

        # Mounting serves the slash-suffixed path; this exact route avoids relying on client redirects.
        app.router.routes.append(
            Route(
                self.mcp_url_path,
                _MCPHeaderSessionMiddleware(streamable_http_route.endpoint),
                include_in_schema=False,
            )
        )
        app.mount(self.mcp_url_path, _MCPHeaderSessionMiddleware(mcp_app))

    def register_mcp_tools(self, mcp: Any) -> None:
        """Register this server's typed gym_tool declarations as MCP tools.

        Override for manual control of the MCP surface; to add manual ``@mcp.tool()`` functions on
        top of the auto-registered ones, call ``super().register_mcp_tools(mcp)`` first. Note that
        manual registrations are MCP-only: HTTP routes are driven solely by gym_tool declarations,
        so hand-registered MCP tools get no HTTP twin (the base logs a warning). Tools declared with
        an explicit ``input_schema`` (dict or model) are dispatched by the base directly and do not
        pass through FastMCP registration.
        """
        for spec, method in self._collect_gym_tools():
            if spec.input_schema is None:
                self._register_gym_tool(mcp, spec.name, method, description=spec.description)

    def _register_gym_tool(self, mcp: Any, name: str, method: Any, description: Optional[str] = None) -> None:
        """Register one typed gym_tool callable as a FastMCP tool.

        Builds a wrapper whose signature mirrors the method's parameters minus ``session_id`` (so the
        session id stays out of the model-visible input schema) and injects the resolved Gym session id
        at call time. Enforces the ``@gym_tool`` constraints.
        """
        if name in RESERVED_MCP_TOOL_NAMES:
            raise ValueError(
                f"@gym_tool method {name!r} collides with a reserved endpoint name "
                f"{sorted(RESERVED_MCP_TOOL_NAMES)}; rename the tool."
            )
        self._check_no_request_param(name, method)

        signature = inspect.signature(method)
        hints = get_type_hints(method, include_extras=True)
        inject_session = "session_id" in signature.parameters

        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def wrapper(**kwargs: Any) -> Any:
                if inject_session:
                    kwargs["session_id"] = self.require_mcp_session_id()
                return await method(**kwargs)
        else:

            @functools.wraps(method)
            async def wrapper(**kwargs: Any) -> Any:
                if inject_session:
                    kwargs["session_id"] = self.require_mcp_session_id()
                # Offload blocking sync tools to a thread so they don't stall the event loop
                # (which would otherwise block every concurrent rollout in this worker).
                return await run_in_threadpool(method, **kwargs)

        # Mirror the method's parameters (with resolved annotations) minus session_id, so FastMCP builds
        # the tool's input schema from real types even under ``from __future__ import annotations``.
        visible_params = [
            param.replace(annotation=hints.get(param_name, param.annotation))
            for param_name, param in signature.parameters.items()
            if param_name != "session_id"
        ]
        wrapper.__signature__ = signature.replace(
            parameters=visible_params,
            return_annotation=hints.get("return", signature.return_annotation),
        )
        wrapper.__annotations__ = {k: v for k, v in hints.items() if k != "session_id"}
        mcp.add_tool(wrapper, name=name, description=description or (method.__doc__ or "").strip() or None)

    def _check_no_request_param(self, name: str, method: Callable) -> None:
        signature = inspect.signature(method)
        try:
            hints = get_type_hints(method)
        except Exception:
            hints = {}
        for param_name, param in signature.parameters.items():
            if param_name == "request" or hints.get(param_name, param.annotation) is Request:
                raise ValueError(
                    f"@gym_tool method {name!r} must not take a 'request' parameter; there is no FastAPI "
                    "Request on the MCP path. Declare a 'session_id: str' parameter to access the Gym session."
                )

    # --------------------------------------------------------------------------------------------
    # MCP low-level handlers: session-aware tools/list, raw-argument tools/call
    # --------------------------------------------------------------------------------------------

    def _install_mcp_list_handler(self, mcp: Any) -> None:
        """Re-register the low-level tools/list handler: raw-schema tools + per-session filtering.

        FastMCP's own listing only knows FastMCP-registered (typed/manual) tools and has no
        per-request filter; this handler appends dict/model-schema gym tools (schemas advertised
        verbatim) and applies the session token's ``allowed_tools`` claim when present.

        NOTE: reaches into the MCP SDK private attr ``mcp._mcp_server`` (also in
        ``_install_mcp_call_handler``; and ``mcp._tool_manager`` in ``_warn_on_transport_parity_gap``).
        The ``mcp`` dependency is pinned to a tested range for this reason; the dual-registration test
        suite drives the full JSON-RPC handshake so an SDK layout change fails loudly in CI.
        """

        @mcp._mcp_server.list_tools()
        async def _gym_list_tools() -> list:
            # Reuse the precomputed types.Tool objects (schemas were generated once at setup); only the
            # per-session allowed_tools filter runs per request.
            tools = list(await mcp.list_tools())
            tools.extend(entry.tool for entry in self._raw_dispatch.values())
            _, allowed = self._mcp_session_claims(required=False)
            if allowed is not None:
                tools = [tool for tool in tools if tool.name in allowed]
            return tools

    def _install_mcp_call_handler(self, mcp: Any) -> None:
        """Re-register the low-level tools/call handler: claim gate + raw-argument dispatch.

        Dict/model-schema tools must not pass through FastMCP's argument validation — it silently
        drops every argument name not present in its synthesized signature. This handler enforces
        the ``allowed_tools`` claim for all tools (hiding alone does not block), dispatches
        raw-schema tools with the caller's arguments verbatim, and delegates typed tools to FastMCP.
        """

        @mcp._mcp_server.call_tool(validate_input=False)
        async def _gym_call_tool(name: str, arguments: Optional[dict]) -> Any:
            _, allowed = self._mcp_session_claims(required=False)
            if allowed is not None and name not in allowed:
                raise MCPSessionError(f"Tool {name!r} is not available for this session.")
            entry = self._raw_dispatch.get(name)
            if entry is None:
                return await mcp.call_tool(name, arguments or {})
            result = await self._call_raw_gym_tool(entry, arguments or {})
            return self._to_call_tool_return(result)

    def _build_raw_tool_entry(self, spec: GymToolSpec, fn: Callable) -> "_RawToolEntry":
        """Precompute (once, at setup) everything a raw dict/model-schema tool call needs.

        Hoists the per-request work out of the MCP hot path: the validated-instance model (model
        schema) or the shallow JSON-schema validator (dict schema + validate=True), the advertised
        inputSchema, the single body-param name, and the sync/async + session-injection flags.
        """
        import mcp.types as types

        signature = inspect.signature(fn)
        schema = spec.input_schema
        body_param: Optional[str] = None
        validator: Optional[type[BaseModel]] = None
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            input_schema = schema.model_json_schema()
            body_params = [p for p in signature.parameters if p != "session_id"]
            if len(body_params) != 1:
                raise ValueError(
                    f"gym_tool {spec.name!r} with a model input_schema must take exactly one "
                    f"non-session parameter (the validated instance); got {body_params}."
                )
            body_param = body_params[0]
        else:
            input_schema = schema
            if spec.validate_input:
                validator = self._model_from_json_schema(spec.name, schema)
        return _RawToolEntry(
            fn=fn,
            model=schema if body_param is not None else None,
            body_param=body_param,
            validator=validator,
            inject_session="session_id" in signature.parameters,
            is_coro=inspect.iscoroutinefunction(fn),
            tool=types.Tool(name=spec.name, description=spec.description, inputSchema=input_schema),
        )

    async def _call_raw_gym_tool(self, entry: "_RawToolEntry", arguments: dict) -> Any:
        """Invoke a dict/model-schema gym tool with raw arguments (MCP call path)."""
        if entry.model is not None:
            kwargs: dict[str, Any] = {entry.body_param: entry.model.model_validate(arguments)}
        else:
            if entry.validator is not None:
                entry.validator.model_validate(arguments)
            kwargs = dict(arguments)
            # The session id is never injectable from the model-facing payload.
            kwargs.pop("session_id", None)

        if entry.inject_session:
            kwargs["session_id"] = self.require_mcp_session_id()

        if entry.is_coro:
            return await entry.fn(**kwargs)
        return await run_in_threadpool(entry.fn, **kwargs)

    @staticmethod
    def _to_call_tool_return(result: Any) -> Any:
        """Convert a raw tool's return value for the low-level SDK result normalizer.

        The SDK renders a dict as structuredContent (+ a JSON text block) and a ContentBlock list as
        plain content — but both ``str`` and ``BaseModel`` define ``__iter__``, so returning them
        bare corrupts the result (char-exploded text / a spurious validation error). Convert first.
        """
        import mcp.types as types

        if isinstance(result, BaseModel):
            return result.model_dump(mode="json")
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            return [types.TextContent(type="text", text=result)]
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    def _warn_on_transport_parity_gap(self, mcp: Any) -> None:
        registry = getattr(getattr(mcp, "_tool_manager", None), "_tools", None)
        if registry is None:  # pragma: no cover - internal SDK layout changed; warning is best-effort
            return
        mcp_only = set(registry) - set(self._gym_tool_names)
        if mcp_only:
            LOG.warning(
                "MCP-only tools with no HTTP route (manual @mcp.tool() registrations?): %s. "
                "gym_tool declarations are served over both transports; manual MCP registrations are not.",
                sorted(mcp_only),
            )

    # --------------------------------------------------------------------------------------------
    # HTTP twin registration
    # --------------------------------------------------------------------------------------------

    def _register_http_gym_tool(self, app: FastAPI, spec: GymToolSpec, fn: Callable) -> None:
        """Register the HTTP ``POST /<name>`` twin of a gym tool (session id from the cookie)."""
        name = spec.name
        signature = inspect.signature(fn)
        inject_session = "session_id" in signature.parameters
        is_coro = inspect.iscoroutinefunction(fn)

        async def invoke(kwargs: dict, request: Request) -> Any:
            if inject_session:
                kwargs["session_id"] = request.session[SESSION_ID_KEY]
            result = await fn(**kwargs) if is_coro else await run_in_threadpool(fn, **kwargs)
            return PlainTextResponse(result) if isinstance(result, str) else result

        schema = spec.input_schema
        if schema is None:
            hints = get_type_hints(fn, include_extras=True)
            body_model = self._synth_body_model(name, signature, hints)

            async def http_handler(body: Any, request: Request) -> Any:
                # Shallow one-level unpack: nested models must stay model instances (a deep
                # body.model_dump() would hand the tool dicts where MCP hands it models).
                kwargs = {field: getattr(body, field) for field in type(body).model_fields}
                return await invoke(kwargs, request)

            # Pin annotations so FastAPI introspects `body` as the request body even if a future
            # maintainer adds `from __future__ import annotations` to this module. The return
            # annotation is pinned only for model returns, so response filtering matches a
            # hand-written route; str returns are served as text/plain instead.
            annotations: dict[str, Any] = {"body": body_model, "request": Request}
            return_hint = hints.get("return")
            if isinstance(return_hint, type) and issubclass(return_hint, BaseModel):
                annotations["return"] = return_hint
            http_handler.__annotations__ = annotations
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            # _setup_mcp built the _RawToolEntry (incl. the arity check) before this registration.
            body_param = self._raw_dispatch[name].body_param

            async def http_handler(body: Any, request: Request) -> Any:
                return await invoke({body_param: body}, request)

            http_handler.__annotations__ = {"body": schema, "request": Request}
        else:
            validator = self._raw_dispatch[name].validator

            async def http_handler(request: Request) -> Any:
                body_bytes = await request.body()
                if not body_bytes.strip():
                    # A genuinely empty body is tolerated as no-args (the pre-existing dispatchers did
                    # the same via exclude_unset on an all-optional body model).
                    raw: Any = {}
                else:
                    try:
                        raw = json.loads(body_bytes)
                    except Exception:
                        # A non-empty but malformed body is a client error — reject it rather than
                        # silently dispatching the tool on empty args (which would, e.g., step an env
                        # on garbage input). Matches the pre-migration typed-body 422.
                        return JSONResponse(
                            status_code=422, content={"error": f"Tool {name!r} expects a valid JSON object body."}
                        )
                if not isinstance(raw, dict):
                    return JSONResponse(
                        status_code=422, content={"error": f"Tool {name!r} expects a JSON object body."}
                    )
                if validator is not None:
                    try:
                        validator.model_validate(raw)
                    except ValidationError as exc:
                        return JSONResponse(status_code=422, content={"error": str(exc)})
                kwargs = dict(raw)
                # The session id is never injectable from the model-facing payload.
                kwargs.pop("session_id", None)
                return await invoke(kwargs, request)

        app.post(f"/{name}")(http_handler)

    def _synth_body_model(self, name: str, signature: inspect.Signature, hints: dict) -> type[BaseModel]:
        """Synthesize the HTTP body model from a typed tool's visible (non-session) parameters."""
        fields: dict[str, Any] = {}
        for param_name, param in signature.parameters.items():
            if param_name in ("self", "session_id"):
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD, param.POSITIONAL_ONLY):
                raise ValueError(
                    f"@gym_tool {name!r}: parameter {param_name!r} ({param.kind.description}) is unsupported "
                    "for typed tools; use keyword-compatible typed params or declare input_schema explicitly."
                )
            annotation = hints.get(param_name, param.annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param_name] = (annotation, default)
        # Namespaced model name: FastAPI de-duplicates identical OpenAPI schema names by mangling.
        return create_model(f"{name}__GymToolBody", **fields)

    def _model_from_json_schema(self, name: str, schema: dict) -> type[BaseModel]:
        """Shallow Pydantic model from a JSON-schema dict (top-level property types only)."""
        type_map = {"string": str, "integer": int, "number": float, "boolean": bool, "object": dict, "array": list}
        fields: dict[str, Any] = {}
        required = set(schema.get("required", []))
        for prop_name, prop in schema.get("properties", {}).items():
            python_type = type_map.get(prop.get("type", "string"), str)
            fields[prop_name] = (python_type, ...) if prop_name in required else (Optional[python_type], None)
        return create_model(f"{name}__GymToolSchema", **fields)

    # --------------------------------------------------------------------------------------------
    # Unknown-tool catch-all and seed_session auto-augmentation
    # --------------------------------------------------------------------------------------------

    async def handle_unknown_tool(self, tool_name: str, request: Request) -> Any:
        """Answer a POST to an unregistered tool path (only mounted on tool-bearing servers).

        The default is a 404 whose body lists the available tools — agents feed response bodies back
        to the model as tool output, so the model can recover with a valid name. Override to preserve
        a server's historical error bytes (e.g. the 200-with-error-string contracts).
        """
        available = ", ".join(self._gym_tool_names)
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown tool {tool_name!r}. Available tools: {available}"},
        )

    def _ensure_unknown_tool_catchall(self, app: FastAPI) -> None:
        if any(getattr(route, "name", None) == "_gym_unknown_tool_catchall" for route in app.router.routes):
            return

        async def _gym_unknown_tool_catchall(tool_name: str, request: Request) -> Any:
            return await self.handle_unknown_tool(tool_name, request)

        app.add_api_route(
            "/{tool_name}",
            _gym_unknown_tool_catchall,
            methods=["POST"],
            include_in_schema=False,
            name="_gym_unknown_tool_catchall",
        )

    def _build_seed_session_endpoint(self) -> Callable:
        """Wrap seed_session so its response always carries the MCP metadata block.

        The wrapper adopts the subclass's signature (synthesizing a ``request`` parameter when
        absent) so FastAPI body binding is unchanged, then injects :data:`NEMO_GYM_MCP_METADATA_KEY`
        with setdefault semantics. It returns a plain JSONResponse: adopting the method's
        ``-> BaseSeedSessionResponse`` annotation would make FastAPI's response filtering silently
        strip the injected key, so the return annotation is deliberately not pinned here — the exact
        opposite of the tool routes above.
        """
        method = self.seed_session
        signature = inspect.signature(method)
        try:
            hints = get_type_hints(method)
        except Exception:
            hints = {}

        request_param_name = next(
            (
                param_name
                for param_name, param in signature.parameters.items()
                if param_name == "request" or hints.get(param_name, param.annotation) is Request
            ),
            None,
        )

        params = [
            param.replace(annotation=hints.get(param_name, param.annotation))
            for param_name, param in signature.parameters.items()
        ]
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
            if isinstance(payload, dict) and NEMO_GYM_MCP_METADATA_KEY not in payload:
                payload[NEMO_GYM_MCP_METADATA_KEY] = jsonable_encoder(self.build_mcp_session_metadata(request))
            return JSONResponse(payload)

        seed_session_endpoint.__name__ = "seed_session"
        seed_session_endpoint.__signature__ = inspect.Signature(parameters=params)
        seed_session_endpoint.__annotations__ = {param.name: param.annotation for param in params}
        return seed_session_endpoint

    # --------------------------------------------------------------------------------------------
    # Stateless signed session token (shared secret across workers; no server-side token storage)
    # --------------------------------------------------------------------------------------------

    def build_mcp_session_metadata(
        self, request: Request, allowed_tools: Optional[list[str]] = None
    ) -> MCPServerMetadata:
        """Mint the per-rollout MCP metadata (signed session token) from the HTTP session.

        ``allowed_tools`` restricts which tools this session can list and call over MCP; the claim
        rides inside the signed token, so it needs no server-side state and is consistent across
        workers. HTTP routes are not restricted by this claim (status-quo HTTP behavior).
        """
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            session_id = str(uuid4())
            request.session[SESSION_ID_KEY] = session_id

        payload: Any = session_id if allowed_tools is None else {"sid": session_id, "tools": list(allowed_tools)}
        return MCPServerMetadata(
            server_name=self.config.name or self.__class__.__name__,
            url_path=self.mcp_url_path,
            headers={NEMO_GYM_MCP_SESSION_TOKEN_HEADER: self._mcp_token_serializer().dumps(payload)},
        )

    def _mcp_token_serializer(self) -> URLSafeSerializer:
        # Stateless signed token: the session-middleware secret is derived deterministically from the
        # server class + config name, so any worker can verify a token another worker signed. This needs
        # no per-worker token storage (it works with num_workers > 1, and there is nothing to evict).
        # Cached because the key is deterministic and this is on the per-tool-call path.
        if self._cached_token_serializer is None:
            self._cached_token_serializer = URLSafeSerializer(self.get_session_middleware_key(), salt=_MCP_TOKEN_SALT)
        return self._cached_token_serializer

    def normalize_tool_name(self, name: str) -> str:
        """Strip this server's MCP namespace from a trajectory tool-call name (see module function)."""
        return normalize_tool_name(name, self.config.name or self.__class__.__name__)

    def require_mcp_session_id(self) -> str:
        session_id, _ = self._mcp_session_claims(required=True)
        return session_id

    def _mcp_session_claims(self, required: bool = True) -> tuple[Optional[str], Optional[frozenset]]:
        """Resolve (session_id, allowed_tools) from the signed token header.

        Accepts both token payload shapes: the legacy bare session-id string and the dict form
        ``{"sid": ..., "tools": [...]}`` minted when ``allowed_tools`` was passed.
        """
        token = _MCP_SESSION_TOKEN.get()
        if not token:
            if required:
                raise MCPSessionError(f"Missing {NEMO_GYM_MCP_SESSION_TOKEN_HEADER} for Gym MCP tool call.")
            return None, None
        try:
            payload = self._mcp_token_serializer().loads(token)
        except BadSignature as exc:
            if required:
                raise MCPSessionError("Invalid Gym MCP session token.") from exc
            return None, None
        if isinstance(payload, dict):
            allowed = payload.get("tools")
            return payload.get("sid"), None if allowed is None else frozenset(allowed)
        return payload, None
