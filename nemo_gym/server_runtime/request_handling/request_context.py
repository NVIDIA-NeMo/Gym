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
"""Middleware that gives each request its own copy of the context variables Gym sets."""

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

from nemo_gym.core.episode import rollout_context


try:
    from opentelemetry import context as _otel_context
except ImportError:  # the telemetry extra is optional; without the SDK there is no span context to isolate
    _otel_context = None


class RequestContextIsolationMiddleware:
    """Start every request from a clean context, whatever task uvicorn started it from.

    uvicorn starts a *pipelined* request's task from inside the previous request's task
    (``on_response_complete`` → ``loop.create_task``), so that task inherits the previous
    request's contextvars. The OTel ASGI instrumentation, which runs outermost, has the previous
    request's server span current at that point; for the new request it sees a current span,
    skips ``traceparent`` extraction, and files the request as an INTERNAL child of the previous
    one, under the previous rollout's trace. Pipelining happens whenever a client reuses a
    kept-alive connection before this server has sent the terminating body message, which
    Starlette's ``BaseHTTPMiddleware`` sends late.

    Installed outside the instrumentor (see ``install_request_context_isolation``), this resets
    the OTel context for the request's duration, so the instrumentor extracts the inbound
    ``traceparent``. It also clears the rollout id. Today that is a guard only: Gym's
    ``BaseHTTPMiddleware`` layers run the rollout middleware, ``/run`` and the handlers in a
    child task, so a rollout id set there does not reach the task uvicorn copies; the guard
    keeps that true if those layers are ever replaced by plain ASGI middleware.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        otel_token = _otel_context.attach(_otel_context.Context()) if _otel_context is not None else None
        try:
            with rollout_context(None):
                await self.app(scope, receive, send)
        finally:
            if otel_token is not None:
                _otel_context.detach(otel_token)


def install_request_context_isolation(app: FastAPI) -> None:
    """Wrap the complete middleware stack, instrumentation included, in context isolation.

    `add_middleware` is not enough: the FastAPI instrumentor installs its middleware by
    overriding `build_middleware_stack`, so it always ends up outside anything added that way.
    Wrapping the build the same way, after instrumentation, puts isolation outside the
    instrumentor, which is the only place it works: the instrumentor must see an empty context
    so it extracts `traceparent`, and the handlers must still see the server span the
    instrumentor then makes current. Call it after `instrument_app_for_telemetry`.
    """
    if getattr(app.state, "nemo_gym_request_context_isolation_installed", False):
        return
    app.state.nemo_gym_request_context_isolation_installed = True
    build_inner = app.build_middleware_stack

    def build_middleware_stack() -> ASGIApp:
        return RequestContextIsolationMiddleware(build_inner())

    app.build_middleware_stack = build_middleware_stack  # type: ignore[method-assign]
    app.middleware_stack = None
