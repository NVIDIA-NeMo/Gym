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
"""Cross-process trace propagation — the load-bearing test of this integration.

One rollout crosses several Gym servers, and the whole point of instrumenting Gym is that
those hops end up in **one trace with correct parent/child edges**. Everything else is
supporting work.

Two real FastAPI apps are served over real HTTP on two ports, so `traceparent` genuinely
travels over the wire and is genuinely extracted by the receiver's ASGI instrumentation.
The two apps share this test process (spawning interpreters from a unit test would be
slow and flaky), but nothing about the propagation path is faked: the header is written by
`server_utils.request`, parsed by the FastAPI instrumentor, and the resulting parent link
is what gets asserted.
"""

import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from aiohttp import ClientSession
from fastapi import FastAPI

from nemo_gym import server_utils
from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.telemetry.conftest import requires_lens


#: These exercise the telemetry-enabled path, which needs nemo-lens. The absent-lens path
#: is covered by test_fallbacks.py, which runs either way.
pytestmark = requires_lens


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _reset_global_tracer_provider():
    """Clear OTel's set-once global TracerProvider so a test can install its own."""
    from opentelemetry import trace
    from opentelemetry.util._once import Once

    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()


@pytest.fixture
def traces(monkeypatch):
    """Real nemo-lens providers exporting into memory.

    Goes through `nemo.lens.setup_telemetry` rather than building a provider by hand, so
    the test also covers lens's propagator registration — without the W3C propagator
    installed, `inject_context` writes nothing and this whole test would pass vacuously
    against a broken setup.
    """
    from nemo.lens import NemoLensConfig, setup_telemetry
    from nemo.lens.state import set_enabled_span_groups
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    # OpenTelemetry installs the global TracerProvider exactly once per process and
    # silently ignores later calls, so without this reset every test after the first would
    # keep the first test's exporter and see zero spans. `setup_telemetry(_allow_reinit=True)`
    # bypasses only nemo-lens's own guard, not OTel's.
    _reset_global_tracer_provider()

    exporter = InMemorySpanExporter()
    config = NemoLensConfig(
        enabled=True,
        service_name="nemo-gym-test",
        export_strategy="all_ranks",
        span_groups="all",
        # Traces only. Leaving metrics on would stand up a PeriodicExportingMetricReader
        # pointed at the default OTLP endpoint and fill the run with connection errors.
        metrics_enabled=False,
        _span_group_cls=GymSpanGroup,
    )
    handle = setup_telemetry(config, rank=0, world_size=1, span_exporter=exporter, _allow_reinit=True)
    set_enabled_span_groups(GymSpanGroup.resolve("all"))
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", handle)
    monkeypatch.setattr(telemetry_setup, "_INITIALISED", True)

    def finished_spans():
        trace.get_tracer_provider().force_flush()
        return exporter.get_finished_spans()

    yield finished_spans

    handle.shutdown()
    _reset_global_tracer_provider()


@pytest.fixture
def loop_local_aiohttp_client(monkeypatch):
    """Give `server_utils.request` a session bound to whichever loop calls it.

    Gym's real global client is created once against the process's event loop. Here the
    two servers run in their own threads with their own loops, so a single shared session
    would raise. This keeps the code under test untouched and only swaps how it obtains a
    session.
    """
    sessions: dict = {}

    def get_client():
        loop = asyncio.get_running_loop()
        if loop not in sessions:
            sessions[loop] = ClientSession()
        return sessions[loop]

    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", get_client)
    yield
    for loop, session in sessions.items():
        if not loop.is_closed():
            asyncio.run_coroutine_threadsafe(session.close(), loop).result(timeout=5)


def _serve(app: FastAPI, port: int):
    """Run *app* on *port* in a background thread; return the uvicorn server."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline:  # pragma: no cover - only on a wedged machine
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    return server, thread


@pytest.fixture
def two_gym_servers(traces, loop_local_aiohttp_client):
    """An 'agent' server that calls a 'model' server through `server_utils.request`."""
    from nemo.lens.contrib.fastapi import instrument_fastapi

    port_model = _free_port()
    port_agent = _free_port()

    model_app = FastAPI()

    @model_app.get("/work")
    async def work():
        return {"ok": True}

    agent_app = FastAPI()

    @agent_app.get("/call")
    async def call():
        response = await server_utils.request("GET", f"http://127.0.0.1:{port_model}/work")
        return {"status": response.status}

    instrument_fastapi(model_app)
    instrument_fastapi(agent_app)

    model_server, model_thread = _serve(model_app, port_model)
    agent_server, agent_thread = _serve(agent_app, port_agent)
    try:
        yield f"http://127.0.0.1:{port_agent}/call"
    finally:
        for server, thread in ((agent_server, agent_thread), (model_server, model_thread)):
            server.should_exit = True
            thread.join(timeout=10)


def _by_kind(spans, kind_name):
    return [span for span in spans if span.kind.name == kind_name]


async def _drive_one_request(url: str) -> None:
    """Call the agent server from outside any span, as a real client would."""
    async with ClientSession() as session:
        async with session.get(url) as response:
            assert response.status == 200, await response.text()


async def test_one_request_across_two_servers_is_one_trace(two_gym_servers, traces):
    """The definition of done: one trace id across the agent -> model hop."""
    await _drive_one_request(two_gym_servers)

    spans = traces()
    described = [(span.name, span.kind.name) for span in spans]

    # Assert the spans exist *before* counting trace ids: with a single span the
    # one-trace-id assertion is trivially true, so on its own it would pass against an
    # integration that emits nothing at all.
    assert len(_by_kind(spans, "SERVER")) == 2, f"expected a span from each server, got {described}"
    assert len(_by_kind(spans, "CLIENT")) == 1, f"expected the outgoing request span, got {described}"

    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1, (
        f"the hop produced {len(trace_ids)} traces instead of one — context did not cross the boundary. "
        f"spans={described}"
    )


async def test_the_receiving_server_span_is_a_child_of_the_calling_client_span(two_gym_servers, traces):
    """The parent edge, not just a shared trace id.

    A shared trace id with a broken parent chain still renders as one trace but loses the
    causality — which is the thing a distributed trace is for.
    """
    await _drive_one_request(two_gym_servers)
    spans = traces()

    client_spans = _by_kind(spans, "CLIENT")
    server_spans = _by_kind(spans, "SERVER")
    assert len(client_spans) == 1, f"expected one CLIENT span, got {[s.name for s in client_spans]}"
    assert len(server_spans) == 2, f"expected two SERVER spans, got {[s.name for s in server_spans]}"
    # (The ASGI instrumentor also emits INTERNAL "http send" spans; they are not asserted on.)

    client = client_spans[0]
    downstream = next(s for s in server_spans if "/work" in s.name)
    upstream = next(s for s in server_spans if "/call" in s.name)

    assert downstream.parent is not None, "the model server's span is a root — traceparent was not extracted"
    assert downstream.parent.span_id == client.context.span_id, (
        "the model server's span is not parented to the agent's outgoing request span"
    )
    assert client.parent is not None and client.parent.span_id == upstream.context.span_id, (
        "the outgoing request span is not parented to the agent's inbound request span"
    )


async def test_the_incoming_edge_carries_a_traceparent_header(two_gym_servers, traces):
    """Assert the mechanism, not only its effect.

    If this fails while the tests above pass, something other than W3C propagation is
    joining the spans and the trace would not survive a real process boundary.
    """
    await _drive_one_request(two_gym_servers)
    spans = traces()

    client = _by_kind(spans, "CLIENT")[0]
    downstream = next(s for s in _by_kind(spans, "SERVER") if "/work" in s.name)

    assert downstream.context.trace_id == client.context.trace_id
    assert downstream.context.span_id != client.context.span_id


async def test_client_span_records_method_and_status(two_gym_servers, traces):
    await _drive_one_request(two_gym_servers)
    client = _by_kind(traces(), "CLIENT")[0]

    assert client.attributes["http.request.method"] == "GET"
    assert client.attributes["http.response.status_code"] == 200


async def test_client_span_url_attribute_drops_the_query_string(traces, loop_local_aiohttp_client, monkeypatch):
    """Query strings carry API keys and task content; they must not reach a span."""
    from nemo.lens.contrib.fastapi import instrument_fastapi

    port = _free_port()
    app = FastAPI()

    @app.get("/work")
    async def work():
        return {"ok": True}

    instrument_fastapi(app)
    server, thread = _serve(app, port)
    try:
        async with ClientSession():
            await server_utils.request("GET", f"http://127.0.0.1:{port}/work?api_key=secret&task=sensitive")
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    client = _by_kind(traces(), "CLIENT")[0]
    assert "secret" not in client.attributes["url.full"]
    assert "sensitive" not in client.attributes["url.full"]
    assert client.attributes["url.full"].endswith("/work")


async def test_no_spans_when_the_http_client_group_is_disabled(two_gym_servers, traces):
    """Disabling `http_client` must remove the client span — the group has to mean something."""
    from nemo.lens.state import set_enabled_span_groups

    set_enabled_span_groups(frozenset())
    await _drive_one_request(two_gym_servers)

    assert _by_kind(traces(), "CLIENT") == [], "a client span was emitted with every span group disabled"
