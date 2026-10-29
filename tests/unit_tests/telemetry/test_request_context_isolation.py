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
"""A request on a reused connection must not inherit the previous request's context.

uvicorn queues a request that arrives before the previous response's terminating body
message as *pipelined* and starts its task from inside the previous request's task, so the
new task inherits that request's contextvars. The OTel ASGI instrumentation then sees a
current span, skips `traceparent` extraction, and files the request as an INTERNAL child of
the previous one. Starlette's `BaseHTTPMiddleware`, which Gym's servers use, sends the
terminating message late enough for clients to reuse the connection in between.

The server here reproduces that window deliberately (a middleware that splits the final body
and waits before the terminating message), the client reuses one connection from several
coroutines with distinct `traceparent` headers, and the test checks every server span's
parent. The control test shows the window really does mis-parent spans without the fix.

The handler here also leaves a rollout id behind on purpose. With Gym's real stack that id
does not leak (the `BaseHTTPMiddleware` layers run handlers in a child task); this app has no
such layer, so the check exercises the guard the middleware keeps for stacks without one.
"""

import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from fastapi import FastAPI

from nemo_gym.core import current_rollout_id
from nemo_gym.core.episode import _ROLLOUT_ID
from nemo_gym.server_runtime.request_handling.request_context import install_request_context_isolation
from tests.unit_tests.telemetry.conftest import requires_lens
from tests.unit_tests.telemetry.test_propagation import (
    traces,  # noqa: F401  (fixture: real Lens providers, in-memory exporter)
)


pytestmark = requires_lens

WINDOW_SECONDS = 0.02
COROUTINES = 8
CALLS_PER_COROUTINE = 12


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _LateTerminatingBody:
    """Send the terminating empty body message after a pause, as BaseHTTPMiddleware does."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def split_send(message):
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                await send({"type": "http.response.body", "body": message.get("body", b""), "more_body": True})
                await asyncio.sleep(WINDOW_SECONDS)
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            else:
                await send(message)

        await self.app(scope, receive, split_send)


def _app(isolate: bool) -> FastAPI:
    from nemo.lens.contrib.fastapi import instrument_fastapi

    app = FastAPI()

    @app.post("/v1/responses")
    async def responses():
        from opentelemetry import trace

        inherited = current_rollout_id()
        _ROLLOUT_ID.set("leaked-by-handler")  # left behind on purpose; see the module docstring
        # What Gym's own instrumentation does inside a handler: open a span that must sit under
        # the request's server span, which is only true if isolation runs *outside* the
        # instrumentor and the instrumentor's server span is current in here.
        with trace.get_tracer("test").start_as_current_span("handler.work"):
            await asyncio.sleep(0.005)
        return {"inherited_rollout_id": inherited}

    app.add_middleware(_LateTerminatingBody)
    instrument_fastapi(app)
    if isolate:
        install_request_context_isolation(app)
    return app


def _serve(app: FastAPI, port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", http="httptools")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline:  # pragma: no cover - only on a wedged machine
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    return server, thread


def _traceparent(trace_id: int, span_id: int) -> str:
    return f"00-{trace_id:032x}-{span_id:016x}-01"


async def _hammer_one_connection(port: int) -> list:
    """Several coroutines share one keep-alive connection; each request names its own parent."""
    inherited = []
    async with ClientSession(
        connector=TCPConnector(limit=1, limit_per_host=1), timeout=ClientTimeout(total=60)
    ) as session:

        async def one(i: int) -> None:
            for k in range(CALLS_PER_COROUTINE):
                headers = {"traceparent": _traceparent(0x1000 + i, 0x100 * (i + 1) + k + 1)}
                async with session.post(f"http://127.0.0.1:{port}/v1/responses", json={}, headers=headers) as r:
                    inherited.append((await r.json())["inherited_rollout_id"])

        await asyncio.gather(*(one(i) for i in range(COROUTINES)))
    return inherited


def _server_spans(spans):
    return [s for s in spans if s.name == "POST /v1/responses"]


@pytest.fixture(params=[True, False], ids=["isolated", "control"])
def served(request, traces):
    port = _free_port()
    server, thread = _serve(_app(isolate=request.param), port)
    try:
        yield request.param, port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def test_requests_on_a_reused_connection_keep_their_own_parent(served, traces):
    isolated, port = served
    inherited = await _hammer_one_connection(port)

    # The last server span ends only after its terminating body message, after the client already
    # has the response, so give the exporter a moment to see every span.
    deadline = time.monotonic() + 5
    while len(_server_spans(traces())) < COROUTINES * CALLS_PER_COROUTINE and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    all_spans = traces()
    spans = _server_spans(all_spans)
    assert len(spans) == COROUTINES * CALLS_PER_COROUTINE
    expected = {0x100 * (i + 1) + k + 1 for i in range(COROUTINES) for k in range(CALLS_PER_COROUTINE)}
    own_parent = [s for s in spans if s.parent is not None and s.parent.span_id in expected]
    wrong_kind = [s for s in spans if s.kind.name != "SERVER"]
    server_ids = {s.context.span_id for s in spans}
    handler_spans = [s for s in all_spans if s.name == "handler.work"]
    handler_under_server = [s for s in handler_spans if s.parent is not None and s.parent.span_id in server_ids]

    if isolated:
        assert len(own_parent) == len(spans), "every server span must be parented to its request's traceparent"
        assert not wrong_kind
        assert all(value is None for value in inherited), "the rollout-id guard must hold without a child-task layer"
        assert len(handler_spans) == len(spans) and len(handler_under_server) == len(spans), (
            "handlers must still run under the server span: isolation has to wrap the instrumentor, not sit inside it"
        )
    else:
        # The control: the same server without the isolation middleware mis-parents some requests, which is
        # the failure this test exists to catch. If this ever stops holding, the window closed upstream and the
        # middleware can go.
        assert len(own_parent) < len(spans), "expected the late terminating body to mis-parent some requests"
        assert wrong_kind
        assert any(value is not None for value in inherited)


def _layers(app: FastAPI) -> list:
    """Class names of the middleware stack from the outside in."""
    names = []
    layer = app.build_middleware_stack()
    while layer is not None and not isinstance(layer, FastAPI) and len(names) < 20:
        names.append(type(layer).__name__)
        layer = getattr(layer, "app", None)
    return names


def test_isolation_installed_after_the_instrumentor_wraps_it():
    """The order `run_webserver` uses: instrument, then isolate. Isolation must be the outermost
    layer and the instrumentor's middleware must sit inside it; revision one of this change had
    them the other way round because `add_middleware` cannot get outside the instrumentor."""
    from nemo.lens.contrib.fastapi import instrument_fastapi

    app = FastAPI()
    app.add_middleware(_LateTerminatingBody)
    instrument_fastapi(app)
    install_request_context_isolation(app)

    layers = _layers(app)
    assert layers[0] == "RequestContextIsolationMiddleware", layers
    otel = next(i for i, name in enumerate(layers) if "OpenTelemetry" in name or "ExceptionHandler" in name)
    assert otel > 0 and otel < layers.index("_LateTerminatingBody"), layers


def test_isolation_installed_before_the_instrumentor_disables_instrumentation():
    """Documents why the install order matters. The instrumentor wraps whatever build it finds and
    insists on Starlette's own outermost layer; finding ours instead, it logs an error and skips
    instrumenting the app, so a server would have isolation and no server spans at all."""
    from nemo.lens.contrib.fastapi import instrument_fastapi

    app = FastAPI()
    install_request_context_isolation(app)
    instrument_fastapi(app)

    layers = _layers(app)
    assert layers[0] == "RequestContextIsolationMiddleware", layers
    assert not any("OpenTelemetry" in name or "ExceptionHandler" in name for name in layers), layers


def test_isolation_installs_once():
    app = FastAPI()
    install_request_context_isolation(app)
    install_request_context_isolation(app)
    assert _layers(app).count("RequestContextIsolationMiddleware") == 1
