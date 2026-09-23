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

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientOSError, ClientSession, TCPConnector, web

from nemo_gym import server_utils
from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.telemetry.conftest import requires_lens


pytestmark = requires_lens


def _reset_global_tracer_provider() -> None:
    from opentelemetry import trace
    from opentelemetry.util._once import Once

    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()


@pytest.fixture
def traces(monkeypatch):
    from nemo.lens import NemoLensConfig, setup_telemetry
    from nemo.lens.state import set_enabled_span_groups
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    _reset_global_tracer_provider()
    exporter = InMemorySpanExporter()
    config = NemoLensConfig(
        enabled=True,
        service_name="nemo-gym-connection-pool-test",
        export_strategy="all_ranks",
        span_groups="all",
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


@asynccontextmanager
async def _serve(handler):
    app = web.Application()
    app.router.add_get("/work", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/work"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _client(monkeypatch, *, limit: int, limit_per_host: int):
    session = ClientSession(
        connector=TCPConnector(limit=limit, limit_per_host=limit_per_host),
        trace_configs=[server_utils._connection_queue_trace_config()],
    )
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: session)
    try:
        yield session
    finally:
        await session.close()


async def _get(url: str) -> None:
    response = await server_utils.request("GET", url)
    assert response.status == 200
    await response.read()


def _client_spans(traces):
    return [span for span in traces() if span.kind.name == "CLIENT"]


async def test_no_queue_is_recorded_explicitly(traces, monkeypatch):
    async def immediate(_request):
        return web.json_response({"ok": True})

    async with _serve(immediate) as url, _client(monkeypatch, limit=2, limit_per_host=2):
        await _get(url)

    span = _client_spans(traces)[0]
    assert span.attributes["nemo.gym.http.connection_pool.queued"] is False
    assert span.attributes["nemo.gym.http.connection_pool.queue_events"] == 0
    assert span.attributes["nemo.gym.http.connection_pool.queue_duration_ms"] == 0.0
    assert span.attributes["nemo.gym.http.connection_pool.pressure"] == "none"


async def test_per_host_limit_records_queue_wait(traces, monkeypatch):
    async def delayed(_request):
        await asyncio.sleep(0.03)
        return web.json_response({"ok": True})

    async with _serve(delayed) as url, _client(monkeypatch, limit=4, limit_per_host=1):
        await asyncio.gather(*(_get(url) for _ in range(3)))

    spans = _client_spans(traces)
    queued = [span for span in spans if span.attributes["nemo.gym.http.connection_pool.queued"]]
    assert len(queued) == 2
    assert all(span.attributes["nemo.gym.http.connection_pool.queue_events"] >= 1 for span in queued)
    assert all(span.attributes["nemo.gym.http.connection_pool.queue_duration_ms"] > 0 for span in queued)
    assert all(span.attributes["nemo.gym.http.connection_pool.pressure"] == "per_host" for span in queued)


async def test_total_limit_records_queue_across_destinations(traces, monkeypatch):
    async def delayed(_request):
        await asyncio.sleep(0.03)
        return web.json_response({"ok": True})

    async with (
        _serve(delayed) as first_url,
        _serve(delayed) as second_url,
        _client(monkeypatch, limit=1, limit_per_host=2),
    ):
        await asyncio.gather(_get(first_url), _get(second_url))

    spans = _client_spans(traces)
    assert sum(span.attributes["nemo.gym.http.connection_pool.queued"] for span in spans) == 1


async def test_cancelled_queue_wait_is_recorded(traces, monkeypatch):
    release = asyncio.Event()
    handler_entered = asyncio.Event()
    queue_started = asyncio.Event()
    original_queued = server_utils._ConnectionQueueTraceContext.queued

    def queued_callback(context, pressure="unknown"):
        original_queued(context, pressure)
        queue_started.set()

    monkeypatch.setattr(server_utils._ConnectionQueueTraceContext, "queued", queued_callback)

    async def blocked(_request):
        handler_entered.set()
        await release.wait()
        return web.json_response({"ok": True})

    async with _serve(blocked) as url, _client(monkeypatch, limit=1, limit_per_host=1):
        occupying = asyncio.create_task(_get(url))
        await asyncio.wait_for(handler_entered.wait(), timeout=1)
        queued = asyncio.create_task(_get(url))
        await asyncio.wait_for(queue_started.wait(), timeout=1)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        await occupying

    spans = _client_spans(traces)
    cancelled_span = next(span for span in spans if span.attributes["nemo.gym.http.connection_pool.queued"])
    assert cancelled_span.attributes["nemo.gym.http.connection_pool.queued"] is True
    assert cancelled_span.attributes["nemo.gym.http.connection_pool.queue_duration_ms"] > 0


async def test_queue_telemetry_failure_is_swallowed_without_retrying(traces, monkeypatch):
    """A raising telemetry hook must not alter the response, and must not cost an extra attempt.

    The previous version of this test only counted CLIENT spans, which the pre-existing retry
    loop produces either way; it passed with the whole feature reverted.
    """
    response = SimpleNamespace(status=200)
    calls = []

    async def request(**kwargs):
        calls.append(kwargs)
        context = server_utils._CONNECTION_QUEUE_TRACE_CONTEXT.get()
        context.queued("per_host")
        return response

    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: SimpleNamespace(request=request))
    monkeypatch.setattr(
        server_utils._ConnectionQueueTraceContext,
        "released",
        lambda _self: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )

    actual = await server_utils.request("GET", "http://example.test/work")

    assert actual is response
    # The failure must be swallowed where it happens, not retried around.
    assert len(calls) == 1
    span = _client_spans(traces)[0]
    assert span.attributes["http.response.status_code"] == 200


async def test_total_limit_pressure_is_attributed(traces, monkeypatch):
    """A wait caused by the aggregate limit is reported as `total`, not `per_host`."""
    release = asyncio.Event()

    async def blocked(_request):
        await release.wait()
        return web.json_response({"ok": True})

    # limit_per_host far above limit, so only the aggregate limit can bind.
    async with _serve(blocked) as url, _client(monkeypatch, limit=1, limit_per_host=50):
        occupying = asyncio.create_task(_get(url))
        await asyncio.sleep(0.05)
        waiting = asyncio.create_task(_get(url))
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(occupying, waiting)

    queued = [s for s in _client_spans(traces) if s.attributes["nemo.gym.http.connection_pool.queued"]]
    assert queued
    assert all(s.attributes["nemo.gym.http.connection_pool.pressure"] == "total" for s in queued)


async def test_attributes_are_omitted_when_callbacks_are_not_installed(traces, monkeypatch):
    """An uninstrumented session must not claim `queued=False`, which is indistinguishable
    from a genuinely unqueued request."""
    response = SimpleNamespace(status=200)

    async def request(**_kwargs):
        return response

    # A real ClientSession with no trace configs, unlike the SimpleNamespace doubles elsewhere.
    session = ClientSession(connector=TCPConnector(limit=1), trace_configs=[])
    session.request = request  # type: ignore[method-assign]
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: session)
    try:
        assert await server_utils.request("GET", "http://example.test/work") is response
    finally:
        await session.close()

    span = _client_spans(traces)[0]
    assert "nemo.gym.http.connection_pool.queued" not in span.attributes
    assert "nemo.gym.http.connection_pool.pressure" not in span.attributes


async def test_retries_share_one_span_and_accumulate_queue_waits(traces, monkeypatch):
    response = SimpleNamespace(status=200)
    calls = []

    async def request(**kwargs):
        calls.append(kwargs)
        context = server_utils._CONNECTION_QUEUE_TRACE_CONTEXT.get()
        context.queued()
        await asyncio.sleep(0.001)
        context.released()
        if len(calls) == 1:
            raise ClientOSError()
        return response

    client = SimpleNamespace(request=request)
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: client)
    monkeypatch.setattr(server_utils.asyncio, "sleep", AsyncMock())

    actual = await server_utils.request("GET", "http://example.test/work", _max_connection_retries=2)

    assert actual is response
    assert len(calls) == 2
    assert calls[0]["headers"]["traceparent"] == calls[1]["headers"]["traceparent"]
    span = _client_spans(traces)[0]
    assert span.attributes["nemo.gym.http.connection_pool.queue_events"] == 2
    assert span.attributes["nemo.gym.http.connection_pool.queue_duration_ms"] > 0
