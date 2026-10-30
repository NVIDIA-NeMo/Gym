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

import nemo_gym.client.http_client
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
    assert any(
        point.attributes == {observer.GC_GENERATION_ATTRIBUTE: 0} for point in metrics[observer.GC_PAUSE_INSTRUMENT]
    )
    assert {p.attributes["cpu.mode"] for p in metrics[observer.PROCESS_CPU_TIME_INSTRUMENT]} == {"user", "system"}
    assert metrics[observer.PROCESS_MEMORY_USAGE_INSTRUMENT][0].value > 0
    assert metrics[observer.PROCESS_THREAD_COUNT_INSTRUMENT][0].value >= 1
    assert {p.attributes[observer.GC_GENERATION_ATTRIBUTE] for p in metrics[observer.GC_COUNT_INSTRUMENT]} == {0, 1, 2}
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
    assert lag.max >= 100  # milliseconds


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
                    "unhandled_exceptions": counts.unhandled_exceptions,
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

        assert totals == {"in_flight": 0, "cancelled": 0, "unhandled_exceptions": 0}
        (in_flight,) = exported[observer.REQUESTS_IN_FLIGHT_INSTRUMENT]
        assert in_flight.value == 0

    def test_a_raising_handler_counts_as_an_unhandled_exception(self, collected):
        async def app(_scope, receive, _send):
            await receive()
            raise RuntimeError("handler broke")

        async def wait_forever():
            await asyncio.Event().wait()

        totals, exported = self._serve(app, self._request_then(wait_forever), collected)

        assert totals == {"in_flight": 0, "cancelled": 0, "unhandled_exceptions": 1}
        (unhandled,) = exported[observer.REQUESTS_UNHANDLED_EXCEPTIONS_INSTRUMENT]
        assert unhandled.value == 1

    def test_a_client_disconnect_counts_as_cancelled(self, collected):
        async def app(_scope, receive, _send):
            await receive()
            await asyncio.Event().wait()  # never responds

        async def disconnect():
            await asyncio.sleep(0.01)
            return {"type": "http.disconnect"}

        totals, _ = self._serve(app, self._request_then(disconnect))

        assert totals == {"in_flight": 0, "cancelled": 1, "unhandled_exceptions": 0}

    def test_totals_stop_reporting_once_the_observer_stops(self, collected):
        async def app(_scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def disconnect():
            return {"type": "http.disconnect"}

        self._serve(app, self._request_then(disconnect))

        assert observer.REQUESTS_IN_FLIGHT_INSTRUMENT not in collected()


@pytest.mark.parametrize("prefix_middleware", ["none", "model", "model_training", "resources"])
def test_cancellation_metrics_use_bounded_routes_and_response_start_phases(collected, monkeypatch, prefix_middleware):
    from fastapi import FastAPI, Response

    from nemo_gym import server_utils
    from nemo_gym.base_responses_api_model import _CaptureMiddleware
    from nemo_gym.rollout_correlation import RolloutContextMiddleware

    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)

    async def run():
        ready = asyncio.Event()
        response_kind = "none"

        class WaitingResponse(Response):
            async def __call__(self, scope, receive, send):
                if response_kind != "none":
                    await send({"type": "http.response.start", "status": 200, "headers": []})
                if response_kind == "keepalive":
                    await send({"type": "http.response.body", "body": b": keep-alive\n\n", "more_body": True})
                ready.set()
                await asyncio.Event().wait()

        app = FastAPI()
        template = "/v1/chat/completions" if prefix_middleware.startswith("model") else "/work/{item_id}"

        @app.post(template)
        async def work():
            return WaitingResponse()

        if prefix_middleware.startswith("model"):
            app.add_middleware(_CaptureMiddleware, store=None, model_server_name="test")
        elif prefix_middleware == "resources":
            app.add_middleware(RolloutContextMiddleware)
        app.add_middleware(server_utils.ClientDisconnectCancellationMiddleware)

        async def send(_message):
            pass

        async with observer.observe_event_loop("server"):
            for response_kind in ("none", "headers", "keepalive"):
                for index in range(2):
                    path = template.replace("{item_id}", f"item-{index}")
                    if prefix_middleware == "model_training":
                        path = f"/training-token-capture{path}"
                    if prefix_middleware != "none":
                        path = f"/ng-rollout/task-{index}-a0{path}"
                    ready.clear()
                    incoming = asyncio.Queue()
                    await incoming.put({"type": "http.request", "body": b"", "more_body": False})
                    scope = {
                        "type": "http",
                        "method": "POST",
                        "path": path,
                        "raw_path": path.encode(),
                        "headers": [],
                        "query_string": b"",
                        "scheme": "http",
                        "http_version": "1.1",
                    }
                    task = asyncio.create_task(app(scope, incoming.get, send))
                    await asyncio.wait_for(ready.wait(), timeout=1)
                    await incoming.put({"type": "http.disconnect"})
                    await asyncio.wait_for(task, timeout=1)

            counts = observer.request_counts()
            assert counts.cancelled == 6
            assert counts.in_flight == counts.unhandled_exceptions == 0
            assert counts.responses == {200: 4}
            # Repeated collection reports the same cumulative totals, without an extra aggregate series.
            for _ in range(2):
                points = collected()[observer.REQUESTS_CANCELLED_INSTRUMENT]
                assert len(points) == 2
                assert {tuple(sorted(point.attributes.items())): point.value for point in points} == {
                    (("http.route", template), ("nemo.gym.request.phase", "before_response_start")): 2,
                    (("http.route", template), ("nemo.gym.request.phase", "response_started")): 4,
                }
                assert sum(point.value for point in points) == counts.cancelled

    asyncio.run(run())


@pytest.mark.parametrize(
    ("prefix_middleware", "request_path", "status", "route"),
    [
        ("none", "/ng-rollout/task-{index}/work/item-{index}", 404, "<unmatched>"),
        ("none", "/ng-rollout/task-{index}/training-token-capture/work/item-{index}", 404, "<unmatched>"),
        ("resources", "/ng-rollout/invalid!{index}/work/item-{index}", 404, "<unmatched>"),
        ("resources", "/ng-rollout//work/item-{index}", 404, "<unmatched>"),
        ("model", "/ng-rollout//work/item-{index}", 404, "<unmatched>"),
        (
            "resources",
            "/ng-rollout/task-{index}/training-token-capture/work/item-{index}",
            200,
            "/training-token-capture/work/{item_id}",
        ),
    ],
)
def test_cancellation_route_labels_follow_server_prefix_rules(
    collected, monkeypatch, prefix_middleware, request_path, status, route
):
    from fastapi import FastAPI, Response

    from nemo_gym import server_utils
    from nemo_gym.base_responses_api_model import _CaptureMiddleware
    from nemo_gym.rollout_correlation import RolloutContextMiddleware

    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)

    async def run():
        app = FastAPI()

        @app.post("/work/{item_id}")
        @app.post("/training-token-capture/work/{item_id}")
        async def work():
            return Response("ok")

        if prefix_middleware == "model":
            app.add_middleware(_CaptureMiddleware, store=None, model_server_name="test")
        elif prefix_middleware == "resources":
            app.add_middleware(RolloutContextMiddleware)
        app.add_middleware(server_utils.ClientDisconnectCancellationMiddleware)

        ready = asyncio.Event()
        statuses = []

        async def send(message):
            if message["type"] == "http.response.start":
                statuses.append(message["status"])
            elif message["type"] == "http.response.body":
                # Keep both successful and 404 responses in flight until the client disconnects.
                ready.set()
                await asyncio.Event().wait()

        async with observer.observe_event_loop("server"):
            for index in range(2):
                path = request_path.format(index=index)
                scope = {
                    "type": "http",
                    "method": "POST",
                    "path": path,
                    "raw_path": path.encode(),
                    "headers": [],
                    "query_string": b"",
                    "scheme": "http",
                    "http_version": "1.1",
                    "state": {"existing": "preserved"},
                }
                ready.clear()
                incoming = asyncio.Queue()
                await incoming.put({"type": "http.request", "body": b"", "more_body": False})
                task = asyncio.create_task(app(scope, incoming.get, send))
                await asyncio.wait_for(ready.wait(), timeout=1)
                await incoming.put({"type": "http.disconnect"})
                await asyncio.wait_for(task, timeout=1)
                assert scope["path"] == path
                assert scope["raw_path"] == path.encode()
                assert scope["state"]["existing"] == "preserved"

            assert statuses == [status, status]
            counts = observer.request_counts()
            assert counts.cancelled == 2
            assert counts.in_flight == counts.unhandled_exceptions == 0
            assert counts.responses == {status: 2}
            (point,) = collected()[observer.REQUESTS_CANCELLED_INSTRUMENT]
            assert point.value == counts.cancelled
            assert point.attributes == {"http.route": route, "nemo.gym.request.phase": "response_started"}

    asyncio.run(run())


def test_cancellation_before_routing_uses_one_unmatched_bucket(collected, monkeypatch):
    from nemo_gym import server_utils

    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)

    async def run():
        ready = asyncio.Event()

        async def unrouted(scope, receive, send):
            ready.set()
            await asyncio.Event().wait()

        async def send(_message):
            pass

        middleware = server_utils.ClientDisconnectCancellationMiddleware(unrouted)
        async with observer.observe_event_loop("server"):
            for index in range(3):
                ready.clear()
                incoming = asyncio.Queue()
                task = asyncio.create_task(
                    middleware({"type": "http", "method": "POST", "path": f"/unknown/{index}"}, incoming.get, send)
                )
                await asyncio.wait_for(ready.wait(), timeout=1)
                await incoming.put({"type": "http.disconnect"})
                await asyncio.wait_for(task, timeout=1)
            (point,) = collected()[observer.REQUESTS_CANCELLED_INSTRUMENT]
            assert point.value == 3
            assert point.attributes == {
                "http.route": "<unmatched>",
                "nemo.gym.request.phase": "before_response_start",
            }

    asyncio.run(run())


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
    seen = []

    async def fake_send(*_args, **_kwargs):
        seen.append(observer.request_counts().outbound_in_flight)
        return "response"

    monkeypatch.setattr("nemo_gym.client.http_client._request_with_retries", fake_send)
    monkeypatch.setattr(nemo_gym.client.http_client, "is_span_group_enabled", lambda _group: False)

    async def call():
        async with observer.observe_event_loop("server"):
            assert await nemo_gym.client.http_client.request("GET", "http://example.invalid/x") == "response"
            return observer.request_counts().outbound_in_flight

    assert asyncio.run(call()) == 0
    assert seen == [1]


def test_responses_are_counted_by_status_code_and_unhandled_errors_separately(collected):
    """A handled HTTPException is a response with its status; an unhandled exception never becomes one inside."""
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    from nemo_gym.server_utils import ClientDisconnectCancellationMiddleware

    app = FastAPI()

    @app.get("/ok")
    async def ok() -> dict:
        return {}

    @app.get("/missing")
    async def missing() -> dict:
        raise HTTPException(status_code=404)

    @app.get("/broken")
    async def broken() -> dict:
        raise RuntimeError("handler broke")

    app.add_middleware(ClientDisconnectCancellationMiddleware)
    observer.observe_app_event_loop(app, "server")

    with TestClient(app, raise_server_exceptions=False) as client:
        statuses = [client.get(path).status_code for path in ("/ok", "/ok", "/missing", "/broken")]
        counts = observer.request_counts()
        responses, unhandled = dict(counts.responses), counts.unhandled_exceptions
        exported = collected()

    assert statuses == [200, 200, 404, 500]
    assert responses == {200: 2, 404: 1}, "the 500 for /broken is sent outside the middleware, after the exception"
    assert unhandled == 1
    by_status = {
        point.attributes[observer.HTTP_STATUS_CODE_ATTRIBUTE]: point.value
        for point in exported[observer.RESPONSES_INSTRUMENT]
    }
    assert by_status == {200: 2, 404: 1}


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
        assert admission.max >= 90, "the third rollout waited for the two ahead of it, in milliseconds"

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
