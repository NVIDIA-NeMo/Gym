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

import asyncio
from contextvars import Context, copy_context

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

from nemo_gym.core.episode_context import rollout_context
from nemo_gym.core.model_call_sampling import model_sampling_context


try:
    from opentelemetry import context as _otel_context
except ImportError:  # the telemetry extra is optional; without the SDK there is no span context to isolate
    _otel_context = None


class RequestContextIsolationMiddleware:
    """Serve every request in a fresh copy of the server's context, whatever task uvicorn started it from.

    uvicorn starts a *pipelined* request's task from inside the previous request's task (``on_response_complete`` →
    ``loop.create_task``).
    That task inherits the previous request's contextvars.
    Gym's middleware and routes run in the request's own task.
    So a value a route sets and never resets, such as a resources server's per-request state, would reach the next
    request on the same connection.
    Pipelining happens whenever a client reuses a kept-alive connection before this server has sent the terminating
    body message.

    Each request therefore runs in a task of its own, with a copy of the context captured when the middleware was
    installed.
    That copy also starts with no OTel context, no rollout id and no model sampling segment.
    The OTel ASGI instrumentation runs inside this middleware, and must see no current span.
    Only then does it extract the inbound ``traceparent`` rather than file the request as a child of the previous one.
    """

    def __init__(self, app: ASGIApp, *, base_context: Context) -> None:
        self.app = app
        self.base_context = base_context

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Awaiting the task directly cancels it with this one, and re-raises what it raises.
        await asyncio.create_task(self._serve(scope, receive, send), context=self.base_context.copy())

    async def _serve(self, scope: Scope, receive: Receive, send: Send) -> None:
        otel_token = _otel_context.attach(_otel_context.Context()) if _otel_context is not None else None
        try:
            with rollout_context(None), model_sampling_context(None):
                await self.app(scope, receive, send)
        finally:
            if otel_token is not None:
                _otel_context.detach(otel_token)


def install_request_context_isolation(app: FastAPI) -> None:
    """Wrap the complete middleware stack, instrumentation included, in context isolation.

    `add_middleware` is not enough.
    The FastAPI instrumentor installs its middleware by overriding `build_middleware_stack`, so it always ends up
    outside anything added that way.
    Wrapping the build the same way, after instrumentation, puts isolation outside the instrumentor, which is the only
    place it works.
    The instrumentor must see an empty context so it extracts `traceparent`, and the handlers must still see the server
    span the instrumentor then makes current.
    Call it after `instrument_app_for_telemetry`, and before the server starts, because the context it captures then is
    the one every request starts from.
    """
    if getattr(app.state, "nemo_gym_request_context_isolation_installed", False):
        return
    app.state.nemo_gym_request_context_isolation_installed = True
    build_inner = app.build_middleware_stack
    # Captured now, not when the stack is first built, which happens inside the first request.
    base_context = copy_context()

    def build_middleware_stack() -> ASGIApp:
        return RequestContextIsolationMiddleware(build_inner(), base_context=base_context)

    app.build_middleware_stack = build_middleware_stack  # type: ignore[method-assign]
    app.middleware_stack = None
