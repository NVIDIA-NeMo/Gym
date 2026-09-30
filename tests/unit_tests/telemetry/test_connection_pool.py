# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from contextlib import asynccontextmanager

import pytest
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from nemo_gym import server_utils
from nemo_gym.telemetry import connection_pool, gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup


pytest.importorskip("opentelemetry.sdk.metrics")

QUEUE_DURATION = gym_metrics.HTTP_CONNECTION_POOL_QUEUE_DURATION_INSTRUMENT
CONNECT_TOTAL = gym_metrics.HTTP_CONNECTION_POOL_CONNECT_INSTRUMENT
CONSTRAINT = gym_metrics.HTTP_CONNECTION_POOL_QUEUE_CONSTRAINT_ATTRIBUTE
OUTCOME = gym_metrics.HTTP_CONNECTION_POOL_ATTEMPT_OUTCOME_ATTRIBUTE
SERVER = gym_metrics.HTTP_SERVER_NAME_ATTRIBUTE


@pytest.fixture
def collected_metrics(monkeypatch):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("connection-pool-test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    monkeypatch.setattr(connection_pool, "is_metrics_exporting", lambda: True)
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY", True)
    gym_metrics._reset_for_testing()
    connection_pool._CONNECT_COUNTS.clear()

    def collect():
        data = reader.get_metrics_data()
        result = {}
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    result[metric.name] = list(metric.data.data_points)
        return result

    yield collect
    provider.shutdown()


@asynccontextmanager
async def _serve_on(host, handler):
    app = web.Application()
    app.router.add_get("/work", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://{host}:{port}/work"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def _serve(handler):
    async with _serve_on("127.0.0.1", handler) as url:
        yield url


@asynccontextmanager
async def _client(monkeypatch, *, limit: int, limit_per_host: int):
    session = ClientSession(
        connector=connection_pool.QueueTimedTCPConnector(limit=limit, limit_per_host=limit_per_host),
    )
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: session)
    try:
        yield session
    finally:
        await session.close()


async def _get(url: str, **kwargs) -> None:
    response = await server_utils.request("GET", url, _server_name="model", **kwargs)
    assert response.status == 200
    await response.read()


def _points(collected_metrics):
    return collected_metrics().get(QUEUE_DURATION, [])


def _expanded_attribute(points, name):
    return sorted(value for point in points for value in [point.attributes[name]] * point.count)


async def test_no_queue_records_no_histogram_sample(collected_metrics, monkeypatch):
    async def immediate(_request):
        return web.json_response({"ok": True})

    async with _serve(immediate) as url, _client(monkeypatch, limit=2, limit_per_host=2):
        await _get(url)

    assert _points(collected_metrics) == []
    (connect_point,) = collected_metrics()[CONNECT_TOTAL]
    assert connect_point.value == 1
    assert connect_point.attributes == {SERVER: "model"}


async def test_connect_counter_survives_connector_replacement(collected_metrics, monkeypatch):
    async def immediate(_request):
        return web.json_response({"ok": True})

    async with _serve(immediate) as url:
        async with _client(monkeypatch, limit=2, limit_per_host=2):
            await _get(url)
        async with _client(monkeypatch, limit=2, limit_per_host=2):
            await _get(url)

    (connect_point,) = collected_metrics()[CONNECT_TOTAL]
    assert connect_point.value == 2


async def test_per_host_limit_records_queue_wait(collected_metrics, monkeypatch):
    async def delayed(_request):
        await asyncio.sleep(0.03)
        return web.json_response({"ok": True})

    async with _serve(delayed) as url, _client(monkeypatch, limit=4, limit_per_host=1):
        await asyncio.gather(*(_get(url) for _ in range(3)))

    points = _points(collected_metrics)
    assert _expanded_attribute(points, CONSTRAINT) == ["per_host", "per_host"]
    queued = [point for point in points if point.attributes[CONSTRAINT] == "per_host"]
    assert sum(point.count for point in queued) == 2
    assert sum(point.sum for point in queued) > 0
    assert all(
        list(point.explicit_bounds) == list(gym_metrics.HTTP_CONNECTION_POOL_QUEUE_DURATION_BOUNDARIES_MS)
        for point in points
    )


async def test_multi_destination_waits_are_attributed_to_the_binding_limit(collected_metrics, monkeypatch):
    release = asyncio.Event()
    entered = asyncio.Queue()
    constraints = asyncio.Queue()
    original = connection_pool._connector_queue_constraint

    def recording(session):
        constraint = original(session)
        constraints.put_nowait(constraint)
        return constraint

    monkeypatch.setattr(connection_pool, "_connector_queue_constraint", recording)

    async def blocked(request):
        entered.put_nowait(request.host)
        await release.wait()
        return web.json_response({"ok": True})

    async with (
        _serve_on("127.0.0.1", blocked) as url_a,
        _serve_on("127.0.0.2", blocked) as url_b,
        _client(monkeypatch, limit=2, limit_per_host=1),
    ):
        tasks = [asyncio.create_task(_get(url_a))]
        await asyncio.wait_for(entered.get(), timeout=2)
        tasks.append(asyncio.create_task(_get(url_a)))
        assert await asyncio.wait_for(constraints.get(), timeout=2) == "per_host"
        tasks.append(asyncio.create_task(_get(url_b)))
        await asyncio.wait_for(entered.get(), timeout=2)
        tasks.append(asyncio.create_task(_get(url_b)))
        assert await asyncio.wait_for(constraints.get(), timeout=2) == "total"
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)

    assert _expanded_attribute(_points(collected_metrics), CONSTRAINT) == ["per_host", "total"]


async def test_queued_timeout_and_retry_record_separate_attempts(collected_metrics, monkeypatch):
    release = asyncio.Event()
    entered = asyncio.Event()
    queued = asyncio.Queue()
    original = connection_pool._connector_queue_constraint

    def recording(session):
        constraint = original(session)
        queued.put_nowait(constraint)
        return constraint

    monkeypatch.setattr(connection_pool, "_connector_queue_constraint", recording)

    async def blocked(_request):
        entered.set()
        await release.wait()
        return web.json_response({"ok": True})

    async with _serve(blocked) as url, _client(monkeypatch, limit=1, limit_per_host=1):
        occupying = asyncio.create_task(_get(url))
        await asyncio.wait_for(entered.wait(), timeout=1)

        waiting = asyncio.create_task(_get(url, timeout=ClientTimeout(connect=0.1), _max_connection_retries=2))
        assert await asyncio.wait_for(queued.get(), timeout=1) == "total"
        assert await asyncio.wait_for(queued.get(), timeout=1) == "total"
        release.set()
        await waiting
        await occupying

    points = _points(collected_metrics)
    assert _expanded_attribute(points, OUTCOME) == ["abandoned", "ok"]
    timeout_point = next(point for point in points if point.attributes[OUTCOME] == "abandoned")
    assert 50 <= timeout_point.sum <= 250
    assert timeout_point.attributes[CONSTRAINT] == "total"
    retried_point = next(
        point for point in points if point.attributes[OUTCOME] == "ok" and point.attributes[CONSTRAINT] == "total"
    )
    assert retried_point.sum > 0


async def test_cancelled_queue_wait_is_recorded(collected_metrics, monkeypatch):
    release = asyncio.Event()
    entered = asyncio.Event()
    queued = asyncio.Event()
    original = connection_pool._connector_queue_constraint

    def recording(session):
        queued.set()
        return original(session)

    monkeypatch.setattr(connection_pool, "_connector_queue_constraint", recording)

    async def blocked(_request):
        entered.set()
        await release.wait()
        return web.json_response({"ok": True})

    async with _serve(blocked) as url, _client(monkeypatch, limit=1, limit_per_host=1):
        occupying = asyncio.create_task(_get(url))
        await asyncio.wait_for(entered.wait(), timeout=1)
        waiting = asyncio.create_task(_get(url))
        await asyncio.wait_for(queued.wait(), timeout=1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        release.set()
        await occupying

    cancelled = next(point for point in _points(collected_metrics) if point.attributes[OUTCOME] == "abandoned")
    assert cancelled.sum > 0


async def test_metric_failure_does_not_change_response(collected_metrics, monkeypatch):
    async def delayed(_request):
        await asyncio.sleep(0.03)
        return web.json_response({"ok": True})

    monkeypatch.setattr(
        connection_pool,
        "record_http_connection_pool_queue_duration",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("metrics failed")),
    )
    async with _serve(delayed) as url, _client(monkeypatch, limit=1, limit_per_host=1):
        await asyncio.gather(_get(url), _get(url))


def test_queue_wait_override_matches_aiohttp_signature():
    base = inspect.signature(TCPConnector._wait_for_available_connection)
    override = inspect.signature(connection_pool.QueueTimedTCPConnector._wait_for_available_connection)
    assert [(name, parameter.kind, parameter.default) for name, parameter in base.parameters.items()] == [
        (name, parameter.kind, parameter.default) for name, parameter in override.parameters.items()
    ]


def test_unlimited_total_can_only_queue_on_per_host_limit():
    connector = object.__new__(TCPConnector)
    connector._limit = 0
    connector._acquired = set()
    assert connection_pool._connector_queue_constraint(connector) == "per_host"


def test_connector_introspection_failure_is_unknown():
    class _BrokenAcquired:
        def __len__(self):
            raise RuntimeError("aiohttp internals changed")

    connector = object.__new__(TCPConnector)
    connector._limit = 1
    connector._acquired = _BrokenAcquired()
    assert connection_pool._connector_queue_constraint(connector) == "unknown"
