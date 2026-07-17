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
from abc import abstractmethod
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Optional, get_type_hints
from uuid import uuid4

from fastapi import FastAPI, Request
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.routing import Route

from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.server_utils import SESSION_ID_KEY, BaseRunServerInstanceConfig, BaseServer, SimpleServer


NEMO_GYM_MCP_SESSION_TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
NEMO_GYM_MCP_METADATA_KEY = "mcp"
_MCP_SESSION_TOKEN: ContextVar[Optional[str]] = ContextVar("nemo_gym_mcp_session_token", default=None)
# Salt namespacing the signed MCP session token, so it can't be confused with another signer
# that happens to share the same session-middleware secret.
_MCP_TOKEN_SALT = "nemo-gym-mcp-session-token"


class MCPSessionError(Exception):
    """A Gym MCP tool call lacked a valid per-rollout session token.

    Deliberately not an HTTP error: MCP runs over JSON-RPC, so FastMCP returns HTTP 200 and surfaces
    this to the client as a tool error (``isError: true``). An HTTP status code raised here would
    never reach the caller, so we raise a plain error with a clear message instead.
    """


# Names a @gym_tool method may not use, because they collide with the resources server's own
# endpoints (and would silently shadow them on HTTP while still registering as MCP tools).
RESERVED_MCP_TOOL_NAMES = frozenset({"verify", "seed_session", "aggregate_metrics", "mcp"})


def gym_tool(fn):
    """Mark a resources-server method as a tool to auto-expose over MCP.

    The method is registered as an MCP tool named after the method, and its MCP input schema is
    derived from the method's typed parameters. Declare a ``session_id: str`` parameter to receive
    the per-rollout Gym session id; it is injected automatically (from the hidden session token) and
    hidden from the tool's input schema. The method must NOT take a ``request`` parameter — there is
    no FastAPI ``Request`` on the MCP path; use ``session_id`` instead. Both sync and async methods
    are supported.
    """
    fn.__gym_tool__ = True
    return fn


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
    config: BaseResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/seed_session")(self.seed_session)
        app.post("/verify")(self.verify)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

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


class MCPResourcesServer(SimpleResourcesServer):
    """SimpleResourcesServer variant that also exposes Gym-owned MCP tools.

    Subclasses decorate tool methods with ``@gym_tool`` (the default ``register_mcp_tools``
    auto-registers them; override only for manual control) and call ``build_mcp_session_metadata``
    from ``seed_session`` to hand the agent a per-rollout token. A ``@gym_tool`` method receives the
    Gym session by declaring a ``session_id`` parameter, which the base resolves from that token (a
    stateless signed value) so tool calls share the session id used by /seed_session and /verify.
    """

    mcp_url_path: str = "/mcp"

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        try:
            from mcp.server.fastmcp import FastMCP
            from mcp.server.transport_security import TransportSecuritySettings
        except ImportError as exc:  # pragma: no cover - exercised only without the optional runtime dependency
            raise RuntimeError(
                "MCPResourcesServer requires the official MCP Python SDK. Install the 'mcp' package."
            ) from exc

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
        self.register_mcp_tools(mcp)

        main_app_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app: FastAPI):
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
        return app

    def register_mcp_tools(self, mcp: Any) -> None:
        """Auto-register methods decorated with ``@gym_tool`` as MCP tools.

        Subclasses can either rely on this default (just decorate tool methods with ``@gym_tool``) or
        override it for full manual control. To add manual ``@mcp.tool()`` functions on top of the
        auto-registered ones, call ``super().register_mcp_tools(mcp)`` first.
        """
        for name, func in inspect.getmembers(type(self), predicate=inspect.isfunction):
            if getattr(func, "__gym_tool__", False):
                self._register_gym_tool(mcp, name, getattr(self, name))

    def _register_gym_tool(self, mcp: Any, name: str, method: Any) -> None:
        """Register one bound ``@gym_tool`` method as an MCP tool.

        Builds a wrapper whose signature mirrors the method's parameters minus ``session_id`` (so the
        session id stays out of the model-visible input schema) and injects the resolved Gym session id
        at call time. Enforces the ``@gym_tool`` constraints.
        """
        if name in RESERVED_MCP_TOOL_NAMES:
            raise ValueError(
                f"@gym_tool method {name!r} collides with a reserved endpoint name "
                f"{sorted(RESERVED_MCP_TOOL_NAMES)}; rename the tool."
            )

        signature = inspect.signature(method)
        hints = get_type_hints(method)
        for param_name, param in signature.parameters.items():
            if param_name == "request" or hints.get(param_name, param.annotation) is Request:
                raise ValueError(
                    f"@gym_tool method {name!r} must not take a 'request' parameter; there is no FastAPI "
                    "Request on the MCP path. Declare a 'session_id: str' parameter to access the Gym session."
                )

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
        mcp.add_tool(wrapper, name=name, description=(method.__doc__ or "").strip() or None)

    def build_mcp_session_metadata(self, request: Request) -> MCPServerMetadata:
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            session_id = str(uuid4())
            request.session[SESSION_ID_KEY] = session_id

        return MCPServerMetadata(
            server_name=self.config.name or self.__class__.__name__,
            url_path=self.mcp_url_path,
            headers={NEMO_GYM_MCP_SESSION_TOKEN_HEADER: self._mcp_token_serializer().dumps(session_id)},
        )

    def _mcp_token_serializer(self) -> URLSafeSerializer:
        # Stateless signed token: the session-middleware secret is derived deterministically from the
        # server class + config name, so any worker can verify a token another worker signed. This needs
        # no per-worker token storage (it works with num_workers > 1, and there is nothing to evict).
        return URLSafeSerializer(self.get_session_middleware_key(), salt=_MCP_TOKEN_SALT)

    def require_mcp_session_id(self) -> str:
        token = _MCP_SESSION_TOKEN.get()
        if not token:
            raise MCPSessionError(f"Missing {NEMO_GYM_MCP_SESSION_TOKEN_HEADER} for Gym MCP tool call.")
        try:
            return self._mcp_token_serializer().loads(token)
        except BadSignature as exc:
            raise MCPSessionError("Invalid Gym MCP session token.") from exc
