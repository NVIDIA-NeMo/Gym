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
"""Server startup spans and histograms, asserted against in-memory OTel exporters.

These run wherever the OTel SDK is importable, without nemo-lens.
The spans are created against the OTel API directly, and the tests patch the span-group gate.
"""

import os
from time import time_ns

import pytest


pytest.importorskip("opentelemetry.sdk.trace")
pytest.importorskip("opentelemetry.sdk.metrics")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from nemo_gym.telemetry import (
    gym_metrics,  # noqa: E402
    startup,  # noqa: E402
)
from nemo_gym.telemetry import setup as telemetry_setup  # noqa: E402
from nemo_gym.telemetry.startup import (  # noqa: E402
    STARTUP_SERVE_NS_ENV,
    STARTUP_SETUP_DONE_NS_ENV,
    STARTUP_SPAWN_NS_ENV,
    STARTUP_TRACEPARENT_ENV,
    ServerStartupTimeline,
    StageTimeline,
    SupervisorStartup,
)


@pytest.fixture
def otel(monkeypatch):
    """A live tracer and meter installed as the process handle, with the startup group on."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        tracer = tracer_provider.get_tracer("test")
        meter = meter_provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    monkeypatch.setattr(startup, "is_span_group_enabled", lambda group: True)
    gym_metrics._INSTRUMENTS.clear()
    for name in (STARTUP_SPAWN_NS_ENV, STARTUP_SETUP_DONE_NS_ENV, STARTUP_SERVE_NS_ENV, STARTUP_TRACEPARENT_ENV):
        monkeypatch.delenv(name, raising=False)

    class _Collected:
        def spans(self):
            return list(exporter.get_finished_spans())

        def metrics(self):
            data = reader.get_metrics_data()
            out = {}
            for resource_metric in data.resource_metrics if data is not None else ():
                for scope_metric in resource_metric.scope_metrics:
                    for metric in scope_metric.metrics:
                        out[metric.name] = list(metric.data.data_points)
            return out

    return _Collected()


def test_marks_are_contiguous_and_cover_the_whole_span_of_time():
    timeline = StageTimeline(start_ns=1_000)
    timeline.mark("a")
    timeline.mark("b")

    first, second = timeline.stages
    assert (first.name, second.name) == ("a", "b")
    assert first.start_ns == 1_000
    assert first.end_ns == second.start_ns
    assert timeline.end_ns == second.end_ns


def test_disabled_server_timeline_records_and_reports_nothing(otel, monkeypatch):
    monkeypatch.setenv(STARTUP_SPAWN_NS_ENV, str(time_ns() - 1_000_000))
    timeline = ServerStartupTimeline("srv", "resources_servers", enabled=False)
    timeline.mark("load_config")
    timeline.report()

    assert timeline.stages == []
    assert otel.spans() == []
    assert otel.metrics() == {}


def test_server_timeline_reconstructs_the_stages_before_its_own_code(otel, monkeypatch):
    now = time_ns()
    monkeypatch.setenv(STARTUP_SPAWN_NS_ENV, str(now - 5_000_000_000))
    monkeypatch.setenv(STARTUP_SETUP_DONE_NS_ENV, str(now - 1_000_000_000))

    timeline = ServerStartupTimeline("srv", "resources_servers")

    venv_setup, interpreter_start = timeline.stages
    assert venv_setup.name == "venv_setup" and venv_setup.duration_ms == pytest.approx(4000)
    assert interpreter_start.name == "interpreter_start" and interpreter_start.duration_ms >= 1000


def test_server_timeline_without_a_launch_stamp_still_times_its_own_stages(otel, monkeypatch):
    now = time_ns()
    monkeypatch.setenv(STARTUP_SPAWN_NS_ENV, str(now - 2_000_000_000))

    timeline = ServerStartupTimeline("srv", "resources_servers")

    (interpreter_start,) = timeline.stages
    assert interpreter_start.name == "interpreter_start" and interpreter_start.duration_ms >= 2000


@pytest.mark.parametrize("stamp", ["not-a-number", "", "-1"])
def test_garbage_stamps_are_ignored(otel, monkeypatch, stamp):
    monkeypatch.setenv(STARTUP_SPAWN_NS_ENV, stamp)
    monkeypatch.setenv(STARTUP_SETUP_DONE_NS_ENV, stamp)

    assert ServerStartupTimeline("srv", None).stages == []


def test_server_report_emits_one_span_per_stage_under_a_server_span(otel):
    timeline = ServerStartupTimeline("srv", "resources_servers")
    timeline.mark("load_config")
    timeline.mark("init_server")
    timeline.report()

    spans = {span.name: span for span in otel.spans()}
    root = spans["gym.server.startup"]
    for stage in ("load_config", "init_server"):
        child = spans[f"gym.server.startup.{stage}"]
        assert child.parent.span_id == root.context.span_id
        assert child.attributes["nemo.gym.startup.stage"] == stage
        assert child.attributes["nemo.gym.server.name"] == "srv"
    # The parent covers its children exactly: stages are contiguous.
    assert root.start_time == spans["gym.server.startup.load_config"].start_time
    assert root.end_time == spans["gym.server.startup.init_server"].end_time


def test_server_report_records_one_stage_histogram_sample_per_stage(otel):
    timeline = ServerStartupTimeline("srv", "resources_servers")
    timeline.mark("load_config")
    timeline.mark("init_server")
    timeline.report()

    points = otel.metrics()[gym_metrics.SERVER_STARTUP_STAGE_INSTRUMENT]
    assert {p.attributes["nemo.gym.startup.stage"] for p in points} == {"load_config", "init_server"}
    assert all(p.count == 1 for p in points)
    assert all(p.attributes["nemo.gym.server.type"] == "resources_servers" for p in points)
    assert list(points[0].explicit_bounds) == list(gym_metrics.SERVER_STARTUP_BOUNDARIES_MS)


def test_server_report_skips_spans_when_the_startup_group_is_off(otel, monkeypatch):
    monkeypatch.setattr(startup, "is_span_group_enabled", lambda group: False)
    timeline = ServerStartupTimeline("srv", None)
    timeline.mark("load_config")
    timeline.report()

    assert otel.spans() == []
    assert gym_metrics.SERVER_STARTUP_STAGE_INSTRUMENT in otel.metrics()


def test_report_without_telemetry_is_a_no_op(monkeypatch):
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    timeline = ServerStartupTimeline("srv", None)
    timeline.mark("load_config")
    timeline.report()


def test_a_telemetry_failure_never_reaches_the_server(otel, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("exporter is gone")

    monkeypatch.setattr(startup, "_start_span", boom)
    timeline = ServerStartupTimeline("srv", None)
    timeline.mark("load_config")
    timeline.report()


def test_server_spans_join_the_supervisors_trace_through_the_environment(otel, monkeypatch):
    supervisor = SupervisorStartup()
    supervisor.begin_trace()
    env = supervisor.server_env("srv", "resources_servers")
    monkeypatch.setenv(STARTUP_TRACEPARENT_ENV, env[STARTUP_TRACEPARENT_ENV])

    timeline = ServerStartupTimeline("srv", "resources_servers")
    timeline.mark("load_config")
    timeline.report()
    supervisor.server_ready("srv")
    supervisor.finish()

    spans = {span.name: span for span in otel.spans()}
    root, per_server, server_root = spans["gym.startup"], spans["gym.startup.server"], spans["gym.server.startup"]
    assert len({span.context.trace_id for span in spans.values()}) == 1
    assert per_server.parent.span_id == root.context.span_id
    assert server_root.parent.span_id == per_server.context.span_id


def test_supervisor_spans_its_stages_and_each_servers_spawn_to_ready(otel):
    supervisor = SupervisorStartup()
    supervisor.begin_trace()
    supervisor.mark("load_config")
    env = supervisor.server_env("srv", "responses_api_models")
    supervisor.mark("spawn_servers")
    supervisor.server_ready("srv")
    ready_ns = supervisor._servers["srv"].ready_ns
    supervisor.server_ready("srv")  # a later poll must not move the ready time
    supervisor.finish()

    assert int(env[STARTUP_SPAWN_NS_ENV]) > 0
    spans = otel.spans()
    names = {span.name for span in spans}
    assert {"gym.startup", "gym.startup.server", "gym.startup.load_config", "gym.startup.spawn_servers"} <= names
    (per_server,) = [span for span in spans if span.name == "gym.startup.server"]
    assert per_server.end_time == ready_ns
    assert per_server.attributes["nemo.gym.server.type"] == "responses_api_models"
    # The supervisor records no histogram: its readiness is only as precise as its health poll.
    assert otel.metrics() == {}


def test_supervisor_with_the_group_off_opens_no_spans_and_sends_no_traceparent(otel, monkeypatch):
    monkeypatch.setattr(startup, "is_span_group_enabled", lambda group: False)
    supervisor = SupervisorStartup()
    supervisor.begin_trace()
    env = supervisor.server_env("srv", "resources_servers")
    supervisor.server_ready("srv")
    supervisor.finish()

    assert STARTUP_TRACEPARENT_ENV not in env
    assert otel.spans() == []


def test_an_unknown_server_becoming_ready_is_ignored(otel):
    supervisor = SupervisorStartup()
    supervisor.begin_trace()
    supervisor.server_ready("never-spawned")
    supervisor.finish()

    assert [span.name for span in otel.spans()] == ["gym.startup"]


def test_worker_timeline_starts_at_the_hand_off_to_uvicorn_not_at_spawn(otel, monkeypatch):
    now = time_ns()
    monkeypatch.setenv(STARTUP_SPAWN_NS_ENV, str(now - 9_000_000_000))
    monkeypatch.setenv(STARTUP_SETUP_DONE_NS_ENV, str(now - 8_000_000_000))
    monkeypatch.setenv(STARTUP_SERVE_NS_ENV, str(now - 2_000_000_000))

    timeline = ServerStartupTimeline("srv", "resources_servers", worker=True)

    (worker_start,) = timeline.stages
    assert worker_start.name == "worker_start" and worker_start.duration_ms >= 2000
    assert worker_start.duration_ms < 8000, "a worker must not count the venv setup its main process did"


def test_worker_without_a_hand_off_stamp_times_only_its_own_stages(otel):
    assert ServerStartupTimeline("srv", None, worker=True).stages == []


def test_worker_report_is_its_own_span_and_metric_series_under_the_supervisors_span(otel, monkeypatch):
    supervisor = SupervisorStartup()
    supervisor.begin_trace()
    env = supervisor.server_env("srv", "resources_servers")
    monkeypatch.setenv(STARTUP_TRACEPARENT_ENV, env[STARTUP_TRACEPARENT_ENV])
    monkeypatch.setenv(STARTUP_SERVE_NS_ENV, str(time_ns() - 1_000_000))

    main = ServerStartupTimeline("srv", "resources_servers")
    main.mark("load_config")
    main.report()
    worker = ServerStartupTimeline("srv", "resources_servers", worker=True)
    worker.mark("load_config")
    worker.report()
    supervisor.server_ready("srv")
    supervisor.finish()

    spans = {span.name: span for span in otel.spans()}
    per_server, worker_root = spans["gym.startup.server"], spans["gym.server.startup.worker"]
    assert worker_root.parent.span_id == per_server.context.span_id
    assert worker_root.attributes["nemo.gym.server.worker"] is True
    assert worker_root.attributes["process.pid"] > 0
    assert spans["gym.server.startup"].attributes["nemo.gym.server.worker"] is False
    assert spans["gym.server.startup.worker.load_config"].parent.span_id == worker_root.context.span_id
    assert len({span.context.trace_id for span in spans.values()}) == 1

    points = otel.metrics()[gym_metrics.SERVER_STARTUP_STAGE_INSTRUMENT]
    workers = {
        p.attributes["nemo.gym.server.worker"]
        for p in points
        if p.attributes["nemo.gym.startup.stage"] == "load_config"
    }
    assert workers == {True, False}


def test_announce_serve_stamps_the_environment_for_the_workers(otel):
    ServerStartupTimeline("srv", None, enabled=False).announce_serve()
    assert STARTUP_SERVE_NS_ENV not in os.environ

    before = time_ns()
    ServerStartupTimeline("srv", None).announce_serve()
    assert before <= int(os.environ[STARTUP_SERVE_NS_ENV]) <= time_ns()
    os.environ.pop(STARTUP_SERVE_NS_ENV)


def test_reporting_twice_exports_once(otel):
    timeline = ServerStartupTimeline("srv", None)
    timeline.mark("load_config")
    timeline.report()
    timeline.report()

    assert [s.name for s in otel.spans()].count("gym.server.startup") == 1


def test_report_when_serving_adds_the_uvicorn_stage_at_app_startup_and_keeps_the_apps_lifespan(otel):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    events = []

    @asynccontextmanager
    async def app_lifespan(app):
        events.append("app-startup")
        yield {}

    app = FastAPI(lifespan=app_lifespan)
    timeline = ServerStartupTimeline("srv", "resources_servers")
    timeline.mark("configure_app")
    timeline.report_when_serving(app)
    assert otel.spans() == [], "nothing may be exported before the app is accepting connections"

    with TestClient(app):
        assert events == ["app-startup"]
        names = {span.name for span in otel.spans()}
        assert {"gym.server.startup", "gym.server.startup.uvicorn_startup"} <= names

    assert [stage.name for stage in timeline.stages] == ["configure_app", "uvicorn_startup"]


def test_report_when_serving_on_a_disabled_timeline_leaves_the_app_alone(otel):
    from fastapi import FastAPI

    app = FastAPI()
    lifespan = app.router.lifespan_context
    ServerStartupTimeline("srv", None, enabled=False).report_when_serving(app)

    assert app.router.lifespan_context is lifespan
