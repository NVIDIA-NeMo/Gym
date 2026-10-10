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
"""Runtime observer: event-loop, garbage-collection, process, and in-flight request metrics.

These run against a real in-memory metric reader wherever the OTel SDK is importable, without nemo-lens.
"""

import asyncio
import gc
import os
import sys

import pytest


pytest.importorskip("opentelemetry.sdk.metrics")

from nemo_gym.telemetry import runtime  # noqa: E402
from nemo_gym.telemetry import setup as telemetry_setup  # noqa: E402


@pytest.fixture
def collected(monkeypatch):
    """A live meter installed as the process handle, with metrics exporting and runtime metrics on."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", True)
    monkeypatch.setenv("NEMO_GYM_OTEL_RUNTIME_METRICS_ENABLED", "1")
    monkeypatch.setenv("NEMO_GYM_OTEL_RUNTIME_METRICS_INTERVAL_SECONDS", "0.01")

    def collect():
        data = reader.get_metrics_data()
        out = {}
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    out[metric.name] = list(metric.data.data_points)
        return out

    yield collect
    assert runtime._PROCESS_RUNTIME is None, "an observer outlived its event loop"
    assert runtime._ACTIVE_LOOPS == 0


class TestGcRecorder:
    def test_a_paired_collection_records_one_pause_and_its_totals(self):
        recorder = runtime._GcRecorder()
        recorder("start", {"generation": 1})
        recorder("stop", {"generation": 1, "collected": 5, "uncollectable": 1})

        [(generation, seconds)] = recorder.drain_pauses()
        assert generation == 1 and seconds >= 0
        assert recorder.collections == [0, 1, 0]
        assert recorder.collected == [0, 5, 0]
        assert recorder.uncollectable == [0, 1, 0]
        assert recorder.drain_pauses() == []

    def test_a_stop_without_a_start_counts_the_collection_but_records_no_pause(self):
        recorder = runtime._GcRecorder()
        recorder("stop", {"generation": 0, "collected": 2, "uncollectable": 0})

        assert recorder.collections == [1, 0, 0]
        assert recorder.drain_pauses() == []

    def test_buffered_pauses_are_bounded(self, monkeypatch):
        monkeypatch.setattr(runtime, "_MAX_BUFFERED_GC_PAUSES", 3)
        recorder = runtime._GcRecorder()
        for _ in range(10):
            recorder("start", {"generation": 0})
            recorder("stop", {"generation": 0, "collected": 0, "uncollectable": 0})

        assert len(recorder.drain_pauses()) == 3
        assert recorder.collections == [10, 0, 0]


async def _observe_for(seconds, collected):
    """Run the observer for a while and read the metrics before it stops, since observable ones stop with it."""
    async with runtime.observe_event_loop("server"):
        gc.collect(0)
        await asyncio.sleep(seconds)
        return collected()


def test_the_observer_reports_loop_gc_and_process_metrics(collected):
    metrics = asyncio.run(_observe_for(0.1, collected))

    (lag,) = metrics[runtime.EVENT_LOOP_LAG_INSTRUMENT]
    assert lag.count >= 1 and lag.attributes == {"nemo.gym.event_loop": "server"}
    (tasks,) = metrics[runtime.EVENT_LOOP_TASKS_INSTRUMENT]
    assert tasks.value >= 2  # the sampler and the coroutine under test
    assert any(point.attributes == {"generation": 0} for point in metrics[runtime.GC_PAUSE_INSTRUMENT])
    assert {p.attributes["cpu.mode"] for p in metrics[runtime.PROCESS_CPU_TIME_INSTRUMENT]} == {"user", "system"}
    assert metrics[runtime.PROCESS_MEMORY_USAGE_INSTRUMENT][0].value > 0
    assert metrics[runtime.PROCESS_THREAD_COUNT_INSTRUMENT][0].value >= 1
    assert {p.attributes["generation"] for p in metrics[runtime.GC_COUNT_INSTRUMENT]} == {0, 1, 2}
    if sys.platform.startswith("linux"):
        assert metrics[runtime.PROCESS_OPEN_FDS_INSTRUMENT][0].value > 0


def test_a_blocked_event_loop_shows_up_as_lag(collected):
    import time

    async def block():
        async with runtime.observe_event_loop("server"):
            await asyncio.sleep(0.02)
            time.sleep(0.2)  # synchronous work holding the event loop
            await asyncio.sleep(0.05)

    asyncio.run(block())

    (lag,) = collected()[runtime.EVENT_LOOP_LAG_INSTRUMENT]
    assert lag.max >= 0.1


def test_the_gc_callback_is_installed_once_and_removed_with_the_last_loop(collected):
    async def nested():
        async with runtime.observe_event_loop("server"):
            async with runtime.observe_event_loop("driver"):
                assert sum(callback is runtime._PROCESS_RUNTIME.gc for callback in gc.callbacks) == 1
            assert runtime._PROCESS_RUNTIME is not None
        assert runtime._PROCESS_RUNTIME is None

    before = list(gc.callbacks)
    asyncio.run(nested())

    assert gc.callbacks == before


def test_nothing_starts_while_runtime_metrics_are_off(collected, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_OTEL_RUNTIME_METRICS_ENABLED", "0")
    before = list(gc.callbacks)

    async def check():
        async with runtime.observe_event_loop("server"):
            assert runtime.request_counts() is None
            assert gc.callbacks == before

    asyncio.run(check())
    assert runtime.EVENT_LOOP_LAG_INSTRUMENT not in collected()


def test_a_failed_start_never_stops_the_server(collected, monkeypatch):
    def broken(_meter):
        raise RuntimeError("meter is gone")

    monkeypatch.setattr(runtime, "_ProcessRuntime", broken)
    ran = []

    async def serve():
        async with runtime.observe_event_loop("server"):
            ran.append(True)

    asyncio.run(serve())
    assert ran == [True]


def test_the_app_lifespan_keeps_running_inside_the_observer(collected):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    events = []

    @asynccontextmanager
    async def app_lifespan(_app):
        events.append("start")
        assert runtime.request_counts() is not None
        yield
        events.append("stop")

    app = FastAPI(lifespan=app_lifespan)
    runtime.observe_app_event_loop(app, "server")
    with TestClient(app):
        pass

    assert events == ["start", "stop"]


class TestRequestCounters:
    @staticmethod
    def _serve(app, receive, collected=None):
        """Run one request through the outermost middleware and return the totals, read before the observer stops."""
        from nemo_gym.server_utils import ClientDisconnectCancellationMiddleware

        async def send(_message):
            pass

        async def run():
            async with runtime.observe_event_loop("server"):
                try:
                    await ClientDisconnectCancellationMiddleware(app)(
                        {"type": "http", "method": "POST", "path": "/x"}, receive, send
                    )
                except Exception:
                    pass
                counts = runtime.request_counts()
                exported = collected() if collected is not None else None
                return {
                    "in_flight": counts.in_flight,
                    "cancelled": counts.cancelled,
                    "failed": counts.failed,
                }, exported

        return asyncio.run(run())

    @staticmethod
    def _request_then(after):
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return await after()

        return receive

    def test_a_completed_request_leaves_nothing_in_flight(self, collected):
        async def app(_scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def disconnect_after_the_response():
            await asyncio.sleep(0.01)
            return {"type": "http.disconnect"}

        totals, exported = self._serve(app, self._request_then(disconnect_after_the_response), collected)

        assert totals == {"in_flight": 0, "cancelled": 0, "failed": 0}
        (in_flight,) = exported[runtime.REQUESTS_IN_FLIGHT_INSTRUMENT]
        assert in_flight.value == 0

    def test_a_raising_handler_counts_as_failed(self, collected):
        async def app(_scope, receive, _send):
            await receive()
            raise RuntimeError("handler broke")

        async def wait_forever():
            await asyncio.Event().wait()

        totals, exported = self._serve(app, self._request_then(wait_forever), collected)

        assert totals == {"in_flight": 0, "cancelled": 0, "failed": 1}
        (failed,) = exported[runtime.REQUESTS_FAILED_INSTRUMENT]
        assert failed.value == 1

    def test_a_client_disconnect_counts_as_cancelled(self, collected):
        async def app(_scope, receive, _send):
            await receive()
            await asyncio.Event().wait()  # never responds

        async def disconnect():
            await asyncio.sleep(0.01)
            return {"type": "http.disconnect"}

        totals, _ = self._serve(app, self._request_then(disconnect))

        assert totals == {"in_flight": 0, "cancelled": 1, "failed": 0}

    def test_totals_stop_reporting_once_the_observer_stops(self, collected):
        async def app(_scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def disconnect():
            return {"type": "http.disconnect"}

        self._serve(app, self._request_then(disconnect))

        assert runtime.REQUESTS_IN_FLIGHT_INSTRUMENT not in collected()


def test_the_supervisor_hands_runtime_settings_to_every_server(monkeypatch):
    from nemo_gym.telemetry.config import TelemetryConfig

    for name in ("NEMO_GYM_OTEL_RUNTIME_METRICS_ENABLED", "NEMO_GYM_OTEL_RUNTIME_METRICS_INTERVAL_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    telemetry_setup.configure_telemetry_env(
        TelemetryConfig(enabled=True, runtime_metrics={"enabled": True, "interval_seconds": 0.5})
    )

    resolved = telemetry_setup.runtime_metrics_config_from_env()
    assert resolved.enabled is True and resolved.interval_seconds == 0.5


def test_every_process_reports_its_pid_as_a_resource_attribute():
    assert telemetry_setup._build_resource_attributes("srv", "resources_servers")["process.pid"] == os.getpid()


def test_outbound_calls_are_counted_while_they_wait_for_a_response(collected, monkeypatch):
    from nemo_gym import server_utils

    seen = []

    async def fake_send(*_args, **_kwargs):
        seen.append(runtime.request_counts().outbound_in_flight)
        return "response"

    monkeypatch.setattr(server_utils, "_request_with_retries", fake_send)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _group: False)

    async def call():
        async with runtime.observe_event_loop("server"):
            assert await server_utils.request("GET", "http://example.invalid/x") == "response"
            return runtime.request_counts().outbound_in_flight

    assert asyncio.run(call()) == 0
    assert seen == [1]
