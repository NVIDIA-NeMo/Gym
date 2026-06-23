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
from abc import abstractmethod
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, PrivateAttr
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

    Subclasses implement ``register_mcp_tools`` and call
    ``build_mcp_session_metadata`` from ``seed_session``. MCP tools can then
    call ``require_mcp_session_id`` to resolve the hidden per-rollout token to
    the same Gym session id used by /seed_session and /verify.
    """

    mcp_url_path: str = "/mcp"
    _mcp_session_id_by_token: dict[str, str] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:  # pragma: no cover - exercised only without the optional runtime dependency
            raise RuntimeError(
                "MCPResourcesServer requires the official MCP Python SDK. Install the 'mcp' package."
            ) from exc

        mcp = FastMCP(
            self.config.name or self.__class__.__name__,
            stateless_http=True,
            json_response=True,
            streamable_http_path="/",
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

    @abstractmethod
    def register_mcp_tools(self, mcp: Any) -> None:
        pass

    def build_mcp_session_metadata(self, request: Request) -> MCPServerMetadata:
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            session_id = str(uuid4())
            request.session[SESSION_ID_KEY] = session_id

        token = uuid4().hex
        self._mcp_session_id_by_token[token] = session_id
        return MCPServerMetadata(
            server_name=self.config.name or self.__class__.__name__,
            url_path=self.mcp_url_path,
            headers={NEMO_GYM_MCP_SESSION_TOKEN_HEADER: token},
        )

    def require_mcp_session_id(self) -> str:
        token = _MCP_SESSION_TOKEN.get()
        if not token:
            raise HTTPException(
                status_code=401,
                detail=f"Missing {NEMO_GYM_MCP_SESSION_TOKEN_HEADER} header for Gym MCP tool call.",
            )

        session_id = self._mcp_session_id_by_token.get(token)
        if not session_id:
            raise HTTPException(status_code=401, detail="Invalid Gym MCP session token.")
        return session_id
