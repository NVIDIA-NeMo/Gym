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

from nemo_gym.telemetry import observer  # noqa: E402
from nemo_gym.telemetry import setup as telemetry_setup  # noqa: E402


@pytest.fixture
def collected(monkeypatch):
    """A live meter installed as the process handle, with metrics exporting."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", True)
    monkeypatch.setenv("NEMO_GYM_OTEL_EVENT_LOOP_SAMPLE_INTERVAL_SECONDS", "0.01")
    monkeypatch.delenv("NEMO_GYM_OTEL_EVENT_LOOP_TASK_COUNT", raising=False)

    def collect():
        data = reader.get_metrics_data()
        out = {}
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    out[metric.name] = list(metric.data.data_points)
        return out

    yield collect
    assert observer._PROCESS_OBSERVER is None, "an observer outlived its event loop"
    assert observer._ACTIVE_LOOPS == 0


class TestGcRecorder:
    def test_a_paired_collection_records_one_pause_and_its_totals(self):
        recorder = observer._GcRecorder()
        recorder("start", {"generation": 1})
        recorder("stop", {"generation": 1, "collected": 5, "uncollectable": 1})

        [(generation, seconds)] = recorder.drain_pauses()
        assert generation == 1 and seconds >= 0
        assert recorder.collections == [0, 1, 0]
        assert recorder.collected == [0, 5, 0]
        assert recorder.uncollectable == [0, 1, 0]
        assert recorder.drain_pauses() == []

    def test_a_stop_without_a_start_counts_the_collection_but_records_no_pause(self):
        recorder = observer._GcRecorder()
        recorder("stop", {"generation": 0, "collected": 2, "uncollectable": 0})

        assert recorder.collections == [1, 0, 0]
        assert recorder.drain_pauses() == []

    def test_buffered_pauses_are_bounded(self, monkeypatch):
        monkeypatch.setattr(observer, "_MAX_BUFFERED_GC_PAUSES", 3)
        recorder = observer._GcRecorder()
        for _ in range(10):
            recorder("start", {"generation": 0})
            recorder("stop", {"generation": 0, "collected": 0, "uncollectable": 0})

        assert len(recorder.drain_pauses()) == 3
        assert recorder.collections == [10, 0, 0]


async def _observe_for(seconds, collected):
    """Run the observer for a while and read the metrics before it stops, since observable ones stop with it."""
    async with observer.observe_event_loop("server"):
        gc.collect(0)
        await asyncio.sleep(seconds)
        return collected()


def test_the_observer_reports_loop_gc_and_process_metrics(collected):
    metrics = asyncio.run(_observe_for(0.1, collected))

    (lag,) = metrics[observer.EVENT_LOOP_LAG_INSTRUMENT]
    assert lag.count >= 1 and lag.attributes == {"nemo.gym.event_loop": "server"}
    assert observer.EVENT_LOOP_TASKS_INSTRUMENT not in metrics, "counting tasks is opt-in"
    assert any(point.attributes == {"generation": 0} for point in metrics[observer.GC_PAUSE_INSTRUMENT])
    assert {p.attributes["cpu.mode"] for p in metrics[observer.PROCESS_CPU_TIME_INSTRUMENT]} == {"user", "system"}
    assert metrics[observer.PROCESS_MEMORY_USAGE_INSTRUMENT][0].value > 0
    assert metrics[observer.PROCESS_THREAD_COUNT_INSTRUMENT][0].value >= 1
    assert {p.attributes["generation"] for p in metrics[observer.GC_COUNT_INSTRUMENT]} == {0, 1, 2}
    if sys.platform.startswith("linux"):
        assert metrics[observer.PROCESS_OPEN_FDS_INSTRUMENT][0].value > 0


def test_a_blocked_event_loop_shows_up_as_lag(collected):
    import time

    async def block():
        async with observer.observe_event_loop("server"):
            await asyncio.sleep(0.02)
            time.sleep(0.2)  # synchronous work holding the event loop
            await asyncio.sleep(0.05)

    asyncio.run(block())

    (lag,) = collected()[observer.EVENT_LOOP_LAG_INSTRUMENT]
    assert lag.max >= 0.1


def test_the_gc_callback_is_installed_once_and_removed_with_the_last_loop(collected):
    async def nested():
        async with observer.observe_event_loop("server"):
            async with observer.observe_event_loop("driver"):
                assert sum(callback is observer._PROCESS_OBSERVER.gc for callback in gc.callbacks) == 1
            assert observer._PROCESS_OBSERVER is not None
        assert observer._PROCESS_OBSERVER is None

    before = list(gc.callbacks)
    asyncio.run(nested())

    assert gc.callbacks == before


def test_the_task_count_is_reported_when_opted_in(collected, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_OTEL_EVENT_LOOP_TASK_COUNT", "1")
    metrics = asyncio.run(_observe_for(0.1, collected))

    (tasks,) = metrics[observer.EVENT_LOOP_TASKS_INSTRUMENT]
    assert tasks.value >= 2  # the sampler and the coroutine under test


def test_nothing_starts_while_metrics_are_off(collected, monkeypatch):
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", False)
    before = list(gc.callbacks)

    async def check():
        async with observer.observe_event_loop("server"):
            assert observer.request_counts() is None
            assert gc.callbacks == before

    asyncio.run(check())
    assert observer.EVENT_LOOP_LAG_INSTRUMENT not in collected()


def test_a_failed_start_never_stops_the_server(collected, monkeypatch):
    def broken(_meter):
        raise RuntimeError("meter is gone")

    monkeypatch.setattr(observer, "_ProcessObserver", broken)
    ran = []

    async def serve():
        async with observer.observe_event_loop("server"):
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
        assert observer.request_counts() is not None
        yield
        events.append("stop")

    app = FastAPI(lifespan=app_lifespan)
    observer.observe_app_event_loop(app, "server")
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
            async with observer.observe_event_loop("server"):
                try:
                    await ClientDisconnectCancellationMiddleware(app)(
                        {"type": "http", "method": "POST", "path": "/x"}, receive, send
                    )
                except Exception:
                    pass
                counts = observer.request_counts()
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
        (in_flight,) = exported[observer.REQUESTS_IN_FLIGHT_INSTRUMENT]
        assert in_flight.value == 0

    def test_a_raising_handler_counts_as_failed(self, collected):
        async def app(_scope, receive, _send):
            await receive()
            raise RuntimeError("handler broke")

        async def wait_forever():
            await asyncio.Event().wait()

        totals, exported = self._serve(app, self._request_then(wait_forever), collected)

        assert totals == {"in_flight": 0, "cancelled": 0, "failed": 1}
        (failed,) = exported[observer.REQUESTS_FAILED_INSTRUMENT]
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

        assert observer.REQUESTS_IN_FLIGHT_INSTRUMENT not in collected()


def test_the_supervisor_hands_event_loop_settings_to_every_server(monkeypatch):
    from nemo_gym.telemetry.config import TelemetryConfig

    for name in ("NEMO_GYM_OTEL_EVENT_LOOP_SAMPLE_INTERVAL_SECONDS", "NEMO_GYM_OTEL_EVENT_LOOP_TASK_COUNT"):
        monkeypatch.delenv(name, raising=False)
    telemetry_setup.configure_telemetry_env(
        TelemetryConfig(enabled=True, event_loop_sample_interval_seconds=0.5, event_loop_task_count=True)
    )

    assert telemetry_setup.event_loop_settings_from_env() == (0.5, True)


def test_event_loop_settings_default_to_one_second_without_task_counts(monkeypatch):
    for name in ("NEMO_GYM_OTEL_EVENT_LOOP_SAMPLE_INTERVAL_SECONDS", "NEMO_GYM_OTEL_EVENT_LOOP_TASK_COUNT"):
        monkeypatch.delenv(name, raising=False)
    assert telemetry_setup.event_loop_settings_from_env() == (1.0, False)


def test_every_process_reports_its_pid_as_a_resource_attribute():
    assert telemetry_setup._build_resource_attributes("srv", "resources_servers")["process.pid"] == os.getpid()


def test_outbound_calls_are_counted_while_they_wait_for_a_response(collected, monkeypatch):
    from nemo_gym import server_utils

    seen = []

    async def fake_send(*_args, **_kwargs):
        seen.append(observer.request_counts().outbound_in_flight)
        return "response"

    monkeypatch.setattr(server_utils, "_request_with_retries", fake_send)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _group: False)

    async def call():
        async with observer.observe_event_loop("server"):
            assert await server_utils.request("GET", "http://example.invalid/x") == "response"
            return observer.request_counts().outbound_in_flight

    assert asyncio.run(call()) == 0
    assert seen == [1]


class TestDriverMetrics:
    def test_admission_records_the_wait_and_counts_active_rollouts(self, collected):
        async def run():
            async with observer.observe_event_loop("driver"):
                metrics = observer.driver_metrics()
                admission = observer.MeteredAdmission(asyncio.Semaphore(1), metrics)
                active_seen = []

                async def rollout():
                    async with admission:
                        active_seen.append(metrics.driver.active)
                        await asyncio.sleep(0.05)

                await asyncio.gather(*(observer.count_resident(rollout(), metrics) for _ in range(3)))
                return active_seen, metrics.driver.active, metrics.driver.resident, collected()

        active_seen, active_after, resident_after, exported = asyncio.run(run())

        assert active_seen == [1, 1, 1], "the limit of one admits one rollout at a time"
        assert active_after == 0 and resident_after == 0
        (admission,) = exported[observer.DRIVER_ADMISSION_INSTRUMENT]
        assert admission.count == 3
        assert admission.max >= 0.09, "the third rollout waited for the two ahead of it"

    def test_resident_counts_rollouts_waiting_for_admission(self, collected):
        async def run():
            async with observer.observe_event_loop("driver"):
                metrics = observer.driver_metrics()
                admission = observer.MeteredAdmission(asyncio.Semaphore(1), metrics)
                release = asyncio.Event()

                async def rollout():
                    async with admission:
                        await release.wait()

                tasks = [asyncio.ensure_future(observer.count_resident(rollout(), metrics)) for _ in range(4)]
                await asyncio.sleep(0.01)
                snapshot = (metrics.driver.resident, metrics.driver.active)
                release.set()
                await asyncio.gather(*tasks)
                return snapshot

        assert asyncio.run(run()) == (4, 1)

    def test_only_a_process_that_offered_rollouts_reports_driver_totals(self, collected):
        async def run():
            async with observer.observe_event_loop("server"):
                before = collected()
                observer.driver_metrics().driver.offered += 2
                return before, collected()

        before, after = asyncio.run(run())

        totals = (
            observer.DRIVER_OFFERED_INSTRUMENT,
            observer.DRIVER_RESIDENT_INSTRUMENT,
            observer.DRIVER_ACTIVE_INSTRUMENT,
        )
        assert not any(name in before for name in totals), "a server dispatches no rollouts"
        assert [after[name][0].value for name in totals] == [2, 0, 0]

    def test_output_writes_record_their_duration_and_size(self, collected):
        import time

        from nemo_gym.rollout_collection import _record_output_write

        async def run():
            async with observer.observe_event_loop("driver"):
                _record_output_write(time.perf_counter(), b"x" * 999)
                return collected()

        exported = asyncio.run(run())
        (size,) = exported[observer.DRIVER_WRITE_BYTES_INSTRUMENT]
        assert size.sum == 1000  # the row plus its newline
        assert exported[observer.DRIVER_WRITE_DURATION_INSTRUMENT][0].count == 1

    def test_nothing_is_recorded_without_an_observer(self):
        import time

        from nemo_gym.rollout_collection import _record_output_write

        assert observer.driver_metrics() is None
        _record_output_write(time.perf_counter(), b"row")


class TestRolloutShape:
    def _record(self, collected, row, result):
        from nemo_gym.rollout_collection import _record_rollout_shape

        async def run():
            async with observer.observe_event_loop("driver"):
                _record_rollout_shape(row, result)
                return collected()

        return asyncio.run(run())

    def test_turns_and_tool_calls_come_from_ng_perf(self, collected):
        exported = self._record(
            collected,
            {"responses_create_params": {"input": [{"role": "user", "content": "hi"}] * 3}},
            {"response": {"output": [{"type": "message"}] * 5}, "ng_perf": {"num_turns": 4, "num_tool_calls": 3}},
        )
        assert exported[observer.ROLLOUT_INPUT_ITEMS_INSTRUMENT][0].sum == 3
        assert exported[observer.ROLLOUT_OUTPUT_ITEMS_INSTRUMENT][0].sum == 5
        assert exported[observer.ROLLOUT_TURNS_INSTRUMENT][0].sum == 4
        assert exported[observer.ROLLOUT_TOOL_CALLS_INSTRUMENT][0].sum == 3

    def test_without_ng_perf_tool_calls_are_counted_from_the_output(self, collected):
        exported = self._record(
            collected,
            {"responses_create_params": {"input": "a plain string prompt"}},
            {
                "response": {
                    "output": [{"type": "function_call"}, {"type": "function_call_output"}, {"type": "message"}]
                }
            },
        )
        assert exported[observer.ROLLOUT_TOOL_CALLS_INSTRUMENT][0].sum == 1
        assert observer.ROLLOUT_TURNS_INSTRUMENT not in exported
        assert observer.ROLLOUT_INPUT_ITEMS_INSTRUMENT not in exported  # a string prompt has no item list


def test_the_last_observer_to_stop_exports_the_final_totals(monkeypatch):
    """A short run ends before any periodic export, so stopping must export the counts while they still exist."""
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult, PeriodicExportingMetricReader

    class _CapturingExporter(MetricExporter):
        def __init__(self):
            super().__init__()
            self.batches = []

        def export(self, metrics_data, timeout_millis=10_000, **kwargs):
            self.batches.append(metrics_data)
            return MetricExportResult.SUCCESS

        def force_flush(self, timeout_millis=10_000):
            return True

        def shutdown(self, timeout_millis=30_000, **kwargs):
            pass

    exporter = _CapturingExporter()
    # An interval far longer than the run, as with a real short-lived driver.
    provider = MeterProvider(metric_readers=[PeriodicExportingMetricReader(exporter, export_interval_millis=10**9)])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", True)
    monkeypatch.setattr(metrics, "get_meter_provider", lambda: provider)

    async def short_run():
        async with observer.observe_event_loop("driver"):
            observer.driver_metrics().driver.offered += 7

    asyncio.run(short_run())

    exported = {
        metric.name: list(metric.data.data_points)
        for data in exporter.batches
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    (offered,) = exported[observer.DRIVER_OFFERED_INSTRUMENT]
    assert offered.value == 7
    provider.shutdown()
