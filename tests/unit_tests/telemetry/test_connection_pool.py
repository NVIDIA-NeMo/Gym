# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import pytest
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from omegaconf import DictConfig

from nemo_gym import server_utils
from nemo_gym.telemetry import connection_pool, gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup


pytest.importorskip("opentelemetry.sdk.metrics")

QUEUE_DURATION = gym_metrics.HTTP_CONNECTION_POOL_QUEUE_DURATION_INSTRUMENT
CONNECT_TOTAL = gym_metrics.HTTP_CONNECTION_POOL_CONNECT_INSTRUMENT
CONSTRAINT = gym_metrics.HTTP_CONNECTION_POOL_QUEUE_CONSTRAINT_ATTRIBUTE
OUTCOME = gym_metrics.HTTP_CONNECTION_POOL_QUEUE_OUTCOME_ATTRIBUTE
SERVER = gym_metrics.HTTP_DESTINATION_SERVER_NAME_ATTRIBUTE


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
    monkeypatch.setattr(connection_pool, "is_metrics_exporter_active", lambda: True)
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


@asynccontextmanager
async def _lazy_client(monkeypatch, *, metrics_enabled: bool = True, workers: int = 1, aggregate_limit: int = 4):
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY", False)
    monkeypatch.setattr(connection_pool, "is_metrics_exporter_active", lambda: metrics_enabled)
    monkeypatch.setattr(server_utils, "get_nemo_gym_fastapi_num_workers", lambda: workers)
    monkeypatch.setattr(server_utils, "is_nemo_gym_fastapi_worker", lambda: True)
    monkeypatch.setattr(
        server_utils,
        "get_global_config_dict",
        lambda **_kwargs: {
            "global_aiohttp_connector_limit": aggregate_limit,
            "global_aiohttp_connector_limit_per_host": aggregate_limit,
        },
    )
    try:
        yield
    finally:
        if server_utils._GLOBAL_AIOHTTP_CLIENT is not None:
            await server_utils._GLOBAL_AIOHTTP_CLIENT.close()
            monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)


async def _get(url: str, **kwargs) -> None:
    try:
        response = await server_utils.request("GET", url, _server_name="model", **kwargs)
        assert response.status == 200
        await response.read()
    finally:
        assert connection_pool._SERVER_NAME.get() == "external"


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


async def test_client_span_is_kept_when_queue_metrics_are_enabled(collected_metrics, monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _group: True)

    async def immediate(_request):
        return web.json_response({"ok": True})

    try:
        async with _lazy_client(monkeypatch), _serve(immediate) as url:
            for count in (1, 2):
                await _get(url)
                assert isinstance(
                    server_utils._GLOBAL_AIOHTTP_CLIENT.connector, connection_pool.QueueTimedTCPConnector
                )
                spans = exporter.get_finished_spans()
                assert len(spans) == count
                assert all(span.kind == trace.SpanKind.CLIENT and span.name == "HTTP GET" for span in spans)
                (connect_point,) = collected_metrics()[CONNECT_TOTAL]
                assert connect_point.value == count
                assert connect_point.attributes == {SERVER: "model"}
    finally:
        provider.shutdown()


async def test_server_client_labels_samples_with_the_destination_server(collected_metrics, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    queued = asyncio.Event()
    original = connection_pool._connector_queue_constraint

    def recording(connector):
        queued.set()
        return original(connector)

    monkeypatch.setattr(connection_pool, "_connector_queue_constraint", recording)

    async def blocked(_request):
        entered.set()
        await release.wait()
        return web.json_response({"ok": True})

    async with _lazy_client(monkeypatch, aggregate_limit=1), _serve(blocked) as url:
        server_client = server_utils.ServerClient(
            head_server_config=server_utils.BaseServerConfig(host="127.0.0.1", port=1),
            global_config_dict=DictConfig(
                {"my_judge": {"resources_servers": {"test": {"host": "127.0.0.1", "port": urlsplit(url).port}}}}
            ),
        )

        async def get_named() -> None:
            response = await server_client.get(server_name="my_judge", url_path="/work")
            assert response.status == 200
            assert await response.json() == {"ok": True}
            assert connection_pool._SERVER_NAME.get() == "external"

        tasks = []
        try:
            tasks.append(asyncio.create_task(get_named()))
            await asyncio.wait_for(entered.wait(), timeout=2)
            tasks.append(asyncio.create_task(get_named()))
            await asyncio.wait_for(queued.wait(), timeout=2)
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)

    (point,) = _points(collected_metrics)
    assert point.count == 1
    assert point.sum > 0
    assert point.attributes == {
        "nemo.gym.http.destination.server.name": "my_judge",
        "nemo.gym.http.connection_pool.queue_constraint": "total",
        "nemo.gym.http.connection_pool.queue_outcome": "ok",
    }
    (connect_point,) = collected_metrics()[CONNECT_TOTAL]
    assert connect_point.value == 2
    assert connect_point.attributes == {"nemo.gym.http.destination.server.name": "my_judge"}


@pytest.mark.parametrize("metrics_enabled", [False, True])
@pytest.mark.parametrize("tracing_enabled", [False, True])
@pytest.mark.parametrize("server_name", ["policy_model", "remote_agent_service", None])
@pytest.mark.parametrize("workers", [1, 4])
async def test_lazy_client_labels_first_and_subsequent_attempts(
    collected_metrics, monkeypatch, metrics_enabled: bool, tracing_enabled: bool, server_name: str | None, workers: int
):
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _group: tracing_enabled)

    async def immediate(_request):
        return web.json_response({"ok": True})

    async with (
        _lazy_client(monkeypatch, metrics_enabled=metrics_enabled, workers=workers),
        _serve(immediate) as url,
    ):
        for count in (1, 2):
            response = await server_utils.request("GET", url, _server_name=server_name)
            assert response.status == 200
            await response.read()
            points = collected_metrics().get(CONNECT_TOTAL, [])
            if metrics_enabled:
                (point,) = points
                assert point.value == count
                assert point.attributes == {"nemo.gym.http.destination.server.name": server_name or "external"}
            else:
                assert points == []
                assert type(server_utils._GLOBAL_AIOHTTP_CLIENT.connector) is TCPConnector
            assert _points(collected_metrics) == []
            assert connection_pool._SERVER_NAME.get() == "external"


async def test_request_restores_the_callers_destination_label(collected_metrics, monkeypatch):
    async def immediate(_request):
        return web.json_response({"ok": True})

    async with _lazy_client(monkeypatch), _serve(immediate) as url:
        token = connection_pool.set_server_name("outer")
        try:
            for name in ("policy_model", None):
                response = await server_utils.request("GET", url, _server_name=name)
                assert response.status == 200
                await response.read()
                assert connection_pool._SERVER_NAME.get() == "outer"
        finally:
            connection_pool.reset_server_name(token)

    points = collected_metrics()[CONNECT_TOTAL]
    assert {point.attributes[SERVER]: point.value for point in points} == {"policy_model": 1, "external": 1}


async def test_lazy_client_preserves_concurrent_destination_labels(collected_metrics, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    queued = asyncio.Event()
    original = connection_pool._connector_queue_constraint

    def recording(connector):
        queued.set()
        return original(connector)

    monkeypatch.setattr(connection_pool, "_connector_queue_constraint", recording)

    async def blocked(_request):
        entered.set()
        await release.wait()
        return web.json_response({"ok": True})

    async def get_named(url: str, server_name: str) -> None:
        response = await server_utils.request("GET", url, _server_name=server_name)
        assert response.status == 200
        await response.read()
        assert connection_pool._SERVER_NAME.get() == "external"

    async with _lazy_client(monkeypatch, aggregate_limit=1), _serve(blocked) as url:
        tasks = []
        try:
            tasks.append(asyncio.create_task(get_named(url, "policy_model")))
            await asyncio.wait_for(entered.wait(), timeout=1)
            tasks.append(asyncio.create_task(get_named(url, "resources")))
            await asyncio.wait_for(queued.wait(), timeout=1)
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)

    points = collected_metrics()[CONNECT_TOTAL]
    assert {point.attributes[SERVER]: point.value for point in points} == {"policy_model": 1, "resources": 1}
    (queue_point,) = _points(collected_metrics)
    assert queue_point.attributes[SERVER] == "resources"
    assert queue_point.count == 1


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
        tasks = []
        try:
            tasks.append(asyncio.create_task(_get(url_a)))
            await asyncio.wait_for(entered.get(), timeout=2)
            tasks.append(asyncio.create_task(_get(url_a)))
            assert await asyncio.wait_for(constraints.get(), timeout=2) == "per_host"
            tasks.append(asyncio.create_task(_get(url_b)))
            await asyncio.wait_for(entered.get(), timeout=2)
            tasks.append(asyncio.create_task(_get(url_b)))
            assert await asyncio.wait_for(constraints.get(), timeout=2) == "total"
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)

    assert _expanded_attribute(_points(collected_metrics), CONSTRAINT) == ["per_host", "total"]


async def test_each_queued_redirect_hop_records_its_own_sample(collected_metrics, monkeypatch):
    source_entered = asyncio.Event()
    target_entered = asyncio.Event()
    release_source = asyncio.Event()
    release_target = asyncio.Event()
    constraints = asyncio.Queue()
    original = connection_pool._connector_queue_constraint

    def recording(connector):
        constraint = original(connector)
        constraints.put_nowait(constraint)
        return constraint

    monkeypatch.setattr(connection_pool, "_connector_queue_constraint", recording)

    async def target(_request):
        target_entered.set()
        await release_target.wait()
        return web.json_response({"ok": True})

    async with _serve(target) as target_url:

        async def source(request):
            if request.query.get("redirect"):
                raise web.HTTPFound(location=target_url)
            source_entered.set()
            await release_source.wait()
            return web.json_response({"ok": True})

        async with _serve(source) as source_url, _client(monkeypatch, limit=2, limit_per_host=1):
            tasks = []
            try:
                tasks.append(asyncio.create_task(_get(source_url)))
                await asyncio.wait_for(source_entered.wait(), timeout=2)
                tasks.append(asyncio.create_task(_get(target_url)))
                await asyncio.wait_for(target_entered.wait(), timeout=2)
                tasks.append(asyncio.create_task(_get(f"{source_url}?redirect=1")))
                assert await asyncio.wait_for(constraints.get(), timeout=2) == "total"
                release_source.set()
                assert await asyncio.wait_for(constraints.get(), timeout=2) == "per_host"
                release_target.set()
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            finally:
                release_source.set()
                release_target.set()
                for task in tasks:
                    task.cancel()
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)

    points = _points(collected_metrics)
    assert _expanded_attribute(points, CONSTRAINT) == ["per_host", "total"]
    assert _expanded_attribute(points, OUTCOME) == ["ok", "ok"]
    assert _expanded_attribute(points, SERVER) == ["model", "model"]
    (connect_point,) = collected_metrics()[CONNECT_TOTAL]
    assert connect_point.value == 4
    assert connect_point.attributes == {SERVER: "model"}


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
        tasks = []
        try:
            occupying = asyncio.create_task(_get(url))
            tasks.append(occupying)
            await asyncio.wait_for(entered.wait(), timeout=1)

            waiting = asyncio.create_task(_get(url, timeout=ClientTimeout(connect=0.1), _max_connection_retries=2))
            tasks.append(waiting)
            assert await asyncio.wait_for(queued.get(), timeout=1) == "total"
            assert await asyncio.wait_for(queued.get(), timeout=1) == "total"
            release.set()
            await asyncio.wait_for(waiting, timeout=5)
            await asyncio.wait_for(occupying, timeout=5)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)

    points = _points(collected_metrics)
    assert _expanded_attribute(points, OUTCOME) == ["abandoned", "ok"]
    assert all(point.attributes[SERVER] == "model" for point in points)
    (connect_point,) = collected_metrics()[CONNECT_TOTAL]
    assert connect_point.value == 3
    assert connect_point.attributes == {SERVER: "model"}
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
        tasks = []
        try:
            occupying = asyncio.create_task(_get(url))
            tasks.append(occupying)
            await asyncio.wait_for(entered.wait(), timeout=1)
            waiting = asyncio.create_task(_get(url))
            tasks.append(waiting)
            await asyncio.wait_for(queued.wait(), timeout=1)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(waiting, timeout=5)
            release.set()
            await asyncio.wait_for(occupying, timeout=5)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)

    cancelled = next(point for point in _points(collected_metrics) if point.attributes[OUTCOME] == "abandoned")
    assert cancelled.sum > 0


async def test_metric_failure_does_not_change_response(collected_metrics, monkeypatch):
    recorder_calls = []

    async def delayed(_request):
        await asyncio.sleep(0.03)
        return web.json_response({"ok": True})

    def failing_recorder(*_args, **kwargs):
        recorder_calls.append(kwargs["queue_outcome"])
        raise RuntimeError("metrics failed")

    monkeypatch.setattr(connection_pool, "record_http_connection_pool_queue_duration", failing_recorder)
    async with _serve(delayed) as url, _client(monkeypatch, limit=1, limit_per_host=1):
        # One attempt per request, so an escaped recorder error fails the request instead of being retried.
        await asyncio.wait_for(
            asyncio.gather(_get(url, _max_connection_retries=1), _get(url, _max_connection_retries=1)), timeout=5
        )

    assert recorder_calls == ["ok"]
    (connect_point,) = collected_metrics()[CONNECT_TOTAL]
    assert connect_point.value == 2


async def test_connect_override_forwards_new_aiohttp_arguments(collected_metrics, monkeypatch):
    received = {}

    async def base_connect(_self, req, *args, **kwargs):
        received.update(req=req, args=args, kwargs=kwargs)
        return "connection"

    monkeypatch.setattr(TCPConnector, "connect", base_connect)
    connector = connection_pool.QueueTimedTCPConnector()
    try:
        assert await connector.connect("request", [], timeout="timeout", added_in_future=True) == "connection"
    finally:
        await connector.close()

    assert received == {
        "req": "request",
        "args": ([],),
        "kwargs": {"timeout": "timeout", "added_in_future": True},
    }
    (connect_point,) = collected_metrics()[CONNECT_TOTAL]
    assert connect_point.value == 1


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
