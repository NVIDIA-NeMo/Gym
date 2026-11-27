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
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.request import Request, urlopen

import pytest
import yaml
from omegaconf import OmegaConf

import nemo_gym.server_runtime.supervisor.collector as collector_module
import nemo_gym.server_runtime.supervisor.server_supervisor as server_supervisor
from nemo_gym.config.global_config import DRY_RUN_KEY_NAME
from nemo_gym.server_runtime.supervisor.collector import (
    CollectorError,
    LocalCollector,
    collector_instance,
    collector_run_dir,
    point_exporters_at_collector,
    start_local_collector,
)
from nemo_gym.telemetry.config import CollectorConfig, TelemetryConfig


# Stands in for otelcol-contrib: answers the health check, takes OTLP over HTTP, and like the real file exporter
# truncates its metrics file when it starts. It records its start, each OTLP request, and on SIGTERM its final flush.
# FAKE_COLLECTOR_MODE makes it exit at once (`exit`), never become healthy (`no_health`), or, after the first run
# (FAKE_COLLECTOR_MARKER), leave a line unfinished (`partial`, the first run only) or never become healthy
# (`unhealthy_again`).
FAKE_COLLECTOR = r"""
import os, signal, sys, threading, time

out = None

def stop(signum, frame):
    if out is not None:
        out.write('{"pid": %d, "event": "final"}\n' % os.getpid())
        out.close()
    os._exit(0)

signal.signal(signal.SIGTERM, stop)

import yaml
from http.server import BaseHTTPRequestHandler, HTTPServer

doc = yaml.safe_load(open(sys.argv[sys.argv.index("--config") + 1]))
mode = os.environ.get("FAKE_COLLECTOR_MODE", "")
marker = os.environ.get("FAKE_COLLECTOR_MARKER")
first_run = not (marker and os.path.exists(marker))
if marker and first_run:
    open(marker, "w").close()
if mode == "exit":
    print("cannot start: boom", flush=True)
    sys.exit(3)
out = open(doc["exporters"]["file/metrics"]["path"], "w")
out.write('{"pid": %d, "event": "start"}\n' % os.getpid())
if mode == "partial" and first_run:
    out.write('{"partial')
out.flush()
lock = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with lock:
            out.write('{"pid": %d, "event": "otlp", "path": "%s"}\n' % (os.getpid(), self.path))
            out.flush()
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass

def serve(endpoint):
    host, port = endpoint.rsplit(":", 1)
    server = HTTPServer((host, int(port)), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

if mode != "no_health" and not (mode == "unhealthy_again" and not first_run):
    serve(doc["extensions"]["health_check"]["endpoint"])
    serve(doc["receivers"]["otlp"]["protocols"]["http"]["endpoint"])
while True:
    time.sleep(1)
"""


def _free_ports(n):
    sockets = [socket.socket() for _ in range(n)]
    for s in sockets:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets:
        s.close()
    return ports


@pytest.fixture
def fake_binary(tmp_path):
    path = tmp_path / "bin" / "otelcol-contrib"
    path.parent.mkdir()
    path.write_text(f"#!{sys.executable}\n{FAKE_COLLECTOR}")
    path.chmod(0o755)
    return path


@pytest.fixture
def config(tmp_path, fake_binary):
    grpc, http, health = _free_ports(3)
    return CollectorConfig(
        binary=str(fake_binary),
        output_dir=str(tmp_path / "out"),
        otlp_grpc_port=grpc,
        otlp_http_port=http,
        health_port=health,
        health_check_timeout_seconds=10,
        shutdown_timeout_seconds=5,
    )


_ENVS = (
    ["OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_PROTOCOL", "NEMO_GYM_OTEL_EXPORTER", "NEMO_LENS_EXPORTER"]
    + [
        f"OTEL_EXPORTER_OTLP_{signal}_{setting}"
        for signal in ("TRACES", "METRICS", "LOGS")
        for setting in ("ENDPOINT", "PROTOCOL")
    ]
    + [
        name
        for proxy in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GRPC_PROXY", "NO_PROXY")
        for name in (proxy, proxy.lower())
    ]
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Set first so the test's own writes to os.environ are undone too.
    for name in _ENVS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _short_restart_delays(monkeypatch):
    monkeypatch.setattr(collector_module, "_FIRST_RESTART_DELAY_SEC", 0.05)
    monkeypatch.setattr(collector_module, "_MAX_RESTART_DELAY_SEC", 0.2)


def _collector(config, tmp_path):
    run_dir = tmp_path / "out" / "otel" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    return LocalCollector(config, binary=config.binary, run_dir=run_dir, run_id="run-1", remote_export=False)


def _metrics(collector):
    return [json.loads(line) for line in (collector.run_dir / "metrics.jsonl").read_text().splitlines()]


def _wait_until(condition, timeout_s=15):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def _healthy(config):
    try:
        with urlopen(f"http://127.0.0.1:{config.health_port}/", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def _restarted(collector, config, first_pid, restarts=1):
    return collector.restarts == restarts and collector.pid != first_pid and _healthy(config)


def _post_otlp(endpoint):
    request = Request(f"{endpoint}/v1/metrics", data=b"{}", headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        assert response.status == 200


class TestLocalCollector:
    def test_starts_and_on_stop_keeps_what_it_flushed(self, config, tmp_path):
        collector = _collector(config, tmp_path)
        collector.start()
        pid = collector.pid
        scratch = collector._scratch
        assert yaml.safe_load(collector.config_path.read_text())["exporters"]["file/metrics"]["path"] == str(
            scratch / "0001" / "metrics.jsonl"
        )
        collector.stop()
        assert _metrics(collector) == [{"pid": pid, "event": "start"}, {"pid": pid, "event": "final"}]
        assert not scratch.exists()

    def test_a_killed_collector_is_started_again_and_both_runs_are_kept(self, config, tmp_path):
        collector = _collector(config, tmp_path)
        collector.start()
        first = collector.pid
        os.kill(first, signal.SIGKILL)
        _wait_until(lambda: _restarted(collector, config, first))
        second = collector.pid
        assert yaml.safe_load(collector.config_path.read_text())["exporters"]["file/metrics"]["path"].endswith(
            "/0002/metrics.jsonl"
        )
        collector.stop()
        assert _metrics(collector) == [
            {"pid": first, "event": "start"},
            {"pid": second, "event": "start"},
            {"pid": second, "event": "final"},
        ]

    def test_an_unfinished_record_is_kept_apart(self, config, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("FAKE_COLLECTOR_MODE", "partial")
        monkeypatch.setenv("FAKE_COLLECTOR_MARKER", str(tmp_path / "first-run"))
        collector = _collector(config, tmp_path)
        collector.start()
        first = collector.pid
        os.kill(first, signal.SIGKILL)
        _wait_until(lambda: _restarted(collector, config, first))
        second = collector.pid
        collector.stop()
        assert _metrics(collector) == [
            {"pid": first, "event": "start"},
            {"pid": second, "event": "start"},
            {"pid": second, "event": "final"},
        ]
        assert (collector.run_dir / "metrics.jsonl.truncated").read_text() == '{"partial\n'
        assert "left an unfinished metrics record (9 bytes)" in capsys.readouterr().out

    def test_files_that_could_not_be_appended_are_appended_later_in_order(self, config, tmp_path, monkeypatch):
        append = LocalCollector._append
        failures = []

        def fail_once(self, source_path, name):
            if not failures:
                failures.append(source_path)
                raise OSError("disk full")
            append(self, source_path, name)

        monkeypatch.setattr(LocalCollector, "_append", fail_once)
        collector = _collector(config, tmp_path)
        collector.start()
        first = collector.pid
        os.kill(first, signal.SIGKILL)
        _wait_until(lambda: _restarted(collector, config, first))
        second = collector.pid
        assert failures and failures[0].parent.exists()
        collector.stop()
        assert _metrics(collector) == [
            {"pid": first, "event": "start"},
            {"pid": second, "event": "start"},
            {"pid": second, "event": "final"},
        ]
        assert not collector._scratch.exists()

    def test_a_failed_append_leaves_the_run_file_as_it_was(self, config, tmp_path, monkeypatch):
        collector = _collector(config, tmp_path)
        (collector.run_dir / "metrics.jsonl").write_text('{"before": 1}\n')
        source = collector._scratch / "0001" / "metrics.jsonl"
        source.parent.mkdir()
        source.write_text('{"a": 1}\n{"b": 2}\n')

        def half_then_fail(source, destination, length):
            destination.write(b'{"a": 1}\n')
            raise OSError("disk full")

        monkeypatch.setattr(collector_module, "_copy_prefix", half_then_fail)
        assert collector._collect() is False
        assert (collector.run_dir / "metrics.jsonl").read_text() == '{"before": 1}\n'
        assert source.exists()
        monkeypatch.undo()
        assert collector._collect() is True
        assert _metrics(collector) == [{"before": 1}, {"a": 1}, {"b": 2}]

    def test_a_failed_fragment_write_undoes_the_whole_append(self, config, tmp_path, monkeypatch):
        collector = _collector(config, tmp_path)
        source = collector._scratch / "0001" / "metrics.jsonl"
        source.parent.mkdir()
        source.write_text('{"a": 1}\n{"unfinished')
        write_all = collector_module._write_all

        def fail_on_the_fragment(destination, data):
            if destination.name.endswith(collector_module.TRUNCATED_SUFFIX):
                write_all(destination, data[:3])
                raise OSError("disk full")
            write_all(destination, data)

        monkeypatch.setattr(collector_module, "_write_all", fail_on_the_fragment)
        assert collector._collect() is False
        assert not (collector.run_dir / "metrics.jsonl").exists()
        assert not (collector.run_dir / "metrics.jsonl.truncated").exists()
        assert source.exists()
        monkeypatch.undo()
        assert collector._collect() is True
        assert _metrics(collector) == [{"a": 1}]
        assert (collector.run_dir / "metrics.jsonl.truncated").read_text() == '{"unfinished\n'

    def test_stops_restarting_after_max_restarts(self, config, tmp_path, capsys):
        collector = _collector(config.model_copy(update={"max_restarts": 0}), tmp_path)
        collector.start()
        first = collector.pid
        os.kill(first, signal.SIGKILL)
        _wait_until(lambda: not collector._watcher.is_alive())
        assert (collector.restarts, collector.pid) == (0, first)
        assert "the most `telemetry.collector.max_restarts` allows" in capsys.readouterr().out
        collector.stop()
        assert _metrics(collector) == [{"pid": first, "event": "start"}]

    def test_a_restart_that_cannot_be_spawned_is_retried(self, config, tmp_path, fake_binary, capsys):
        collector = _collector(config, tmp_path)
        collector.start()
        first = collector.pid
        fake_binary.chmod(0o644)
        os.kill(first, signal.SIGKILL)
        _wait_until(lambda: "could not be started again" in capsys.readouterr().out)
        fake_binary.chmod(0o755)
        _wait_until(lambda: collector.pid not in (None, first) and _healthy(config))
        assert collector.restarts >= 2
        collector.stop()
        assert _metrics(collector)[-1] == {"pid": collector.pid, "event": "final"}

    def test_a_restart_that_never_becomes_healthy_is_stopped_and_counted(self, config, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("FAKE_COLLECTOR_MODE", "unhealthy_again")
        monkeypatch.setenv("FAKE_COLLECTOR_MARKER", str(tmp_path / "first-run"))
        config = config.model_copy(update={"max_restarts": 2, "health_check_timeout_seconds": 0.5})
        collector = _collector(config, tmp_path)
        processes = []
        spawn = collector._spawn

        def record_spawn():
            spawn()
            processes.append(collector._process)

        collector._spawn = record_spawn
        collector.start()
        os.kill(collector.pid, signal.SIGKILL)
        _wait_until(lambda: not collector._watcher.is_alive())
        out = capsys.readouterr().out
        assert collector.restarts == 2
        assert out.count("The restarted OpenTelemetry Collector did not answer its health check") == 2
        assert len(processes) == 3 and all(process.poll() is not None for process in processes)
        collector.stop()

    def test_a_collector_that_exits_at_start_fails_with_its_log(self, config, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_COLLECTOR_MODE", "exit")
        collector = _collector(config, tmp_path)
        with pytest.raises(CollectorError, match=r"exited with code 3 before it was ready(.|\n)*cannot start: boom"):
            collector.start()

    def test_the_health_check_does_not_go_through_a_proxy(self, config, tmp_path, monkeypatch):
        (dead_proxy,) = _free_ports(1)
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{dead_proxy}")
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{dead_proxy}")
        collector = _collector(config.model_copy(update={"health_check_timeout_seconds": 3}), tmp_path)
        collector.start()
        collector.stop()

    def test_a_collector_that_never_answers_fails_after_the_timeout(self, config, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_COLLECTOR_MODE", "no_health")
        collector = _collector(config.model_copy(update={"health_check_timeout_seconds": 0.5}), tmp_path)
        with pytest.raises(CollectorError, match="did not answer its health check"):
            collector.start()
        assert collector._process.poll() is not None


class TestPointExportersAtCollector:
    def _endpoints(self):
        return {name: os.environ[name] for name in _ENVS if name.startswith("OTEL_") and name in os.environ}

    def test_without_an_endpoint_the_servers_use_the_collectors_http_port(self, config, capsys):
        point_exporters_at_collector(config)
        assert self._endpoints() == {
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{config.otlp_http_port}",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        }
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("protocol", ["grpc", " GRPC "])
    def test_a_grpc_protocol_gets_the_grpc_port(self, config, monkeypatch, protocol):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", protocol)
        point_exporters_at_collector(config)
        assert self._endpoints() == {
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{config.otlp_grpc_port}",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        }

    def test_a_signal_on_the_other_transport_gets_its_own_endpoint(self, config, monkeypatch, capsys):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/json")
        point_exporters_at_collector(config)
        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://127.0.0.1:{config.otlp_http_port}"
        assert os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == f"http://127.0.0.1:{config.otlp_grpc_port}"
        assert "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT" not in os.environ
        assert capsys.readouterr().out == ""

    def test_an_http_signal_beside_grpc_gets_its_signal_path(self, config, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "http/protobuf")
        point_exporters_at_collector(config)
        assert os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"] == (
            f"http://127.0.0.1:{config.otlp_http_port}/v1/metrics"
        )

    def test_the_collectors_own_endpoint_is_kept_quietly(self, config, monkeypatch, capsys):
        endpoint = f"http://localhost:{config.otlp_http_port}"
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        point_exporters_at_collector(config)
        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == endpoint
        assert capsys.readouterr().out == ""

    def test_an_endpoint_without_a_protocol_is_taken_as_grpc(self, config, monkeypatch, capsys):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{config.otlp_http_port}")
        point_exporters_at_collector(config)
        assert "OTEL_EXPORTER_OTLP_PROTOCOL" not in os.environ
        assert f"traces, metrics, logs to http://127.0.0.1:{config.otlp_http_port} over grpc" in (
            capsys.readouterr().out
        )

    def test_an_endpoint_elsewhere_is_kept_with_a_warning(self, config, monkeypatch, capsys):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otlp.example.com")
        point_exporters_at_collector(config)
        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://otlp.example.com"
        assert "so they do not reach it" in capsys.readouterr().out

    def test_a_signal_sent_elsewhere_is_named(self, config, monkeypatch, capsys):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://otlp.example.com/v1/traces")
        point_exporters_at_collector(config)
        out = capsys.readouterr().out
        assert "Gym exports its traces to https://otlp.example.com/v1/traces over http/protobuf" in out
        assert "metrics" not in out

    def test_the_console_exporter_is_left_alone_with_a_warning(self, config, monkeypatch, capsys):
        monkeypatch.setenv("NEMO_LENS_EXPORTER", "console")
        point_exporters_at_collector(config)
        assert self._endpoints() == {}
        assert "The telemetry exporter is 'console'" in capsys.readouterr().out

    def test_loopback_bypasses_a_proxy(self, config, monkeypatch):
        monkeypatch.setenv("https_proxy", "http://proxy.example.com:3128")
        monkeypatch.setenv("NO_PROXY", "internal.example.com")
        point_exporters_at_collector(config)
        assert os.environ["NO_PROXY"] == os.environ["no_proxy"] == "internal.example.com,127.0.0.1,localhost"

    def test_without_a_proxy_no_proxy_is_not_set(self, config):
        point_exporters_at_collector(config)
        assert "NO_PROXY" not in os.environ and "no_proxy" not in os.environ


class TestStartLocalCollector:
    def test_without_the_block_nothing_starts(self):
        assert start_local_collector(TelemetryConfig(enabled=True), "run-1", {}) is None
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ

    def test_with_telemetry_off_nothing_starts(self, config, capsys):
        assert start_local_collector(TelemetryConfig(collector=config), None, {}) is None
        assert "telemetry is off" in capsys.readouterr().out
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ

    def test_a_missing_binary_fails(self, config, tmp_path):
        config = config.model_copy(update={"binary": str(tmp_path / "missing")})
        with pytest.raises(CollectorError, match="is not an executable file or a command on PATH"):
            start_local_collector(TelemetryConfig(enabled=True, collector=config), "run-1", {})

    def test_a_port_in_use_fails_before_anything_starts(self, config):
        with socket.socket() as holder:
            holder.bind(("127.0.0.1", config.otlp_http_port))
            holder.listen()
            with pytest.raises(CollectorError, match=f"Port\\(s\\) {config.otlp_http_port} on 127.0.0.1 are in use"):
                start_local_collector(TelemetryConfig(enabled=True, collector=config), "run-1", {})

    def test_a_previous_runs_closed_connections_do_not_hold_the_ports(self, config):
        telemetry = TelemetryConfig(enabled=True, collector=config)
        first = start_local_collector(telemetry, "run-1", {})
        _post_otlp(os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])
        first.stop()
        second = start_local_collector(telemetry, "run-2", {})
        second.stop()

    def test_writes_under_the_log_dir_by_default_one_directory_per_supervisor(self, config, tmp_path):
        config = config.model_copy(update={"output_dir": None})
        assert collector_run_dir(config, "run-1", {"nemo_gym_log_dir": str(tmp_path / "logs")}, "h-1-ab") == (
            tmp_path / "logs" / "otel" / "run-1" / "h-1-ab"
        )
        instance = collector_instance()
        assert instance.startswith(f"{socket.gethostname()}-{os.getpid()}-") and instance != collector_instance()

    @pytest.mark.parametrize(
        "run_id, component", [("/other", "%2Fother"), ("..", "_.."), (".", "_."), ("a/../b", "a%2F..%2Fb")]
    )
    def test_a_run_id_stays_one_directory(self, config, tmp_path, run_id, component):
        assert collector_run_dir(config, run_id, {}, "h-1-ab") == tmp_path / "out" / "otel" / component / "h-1-ab"

    def test_what_the_servers_export_reaches_the_run_directory(self, config, tmp_path):
        collector = start_local_collector(TelemetryConfig(enabled=True, collector=config), "run-1", {})
        try:
            _post_otlp(os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])
        finally:
            collector.stop()
        assert {"pid": collector.pid, "event": "otlp", "path": "/v1/metrics"} in _metrics(collector)
        assert collector.run_dir.parent == tmp_path / "out" / "otel" / "run-1"

    def test_a_claimed_directory_is_not_shared(self, config, monkeypatch):
        monkeypatch.setattr(collector_module, "collector_instance", lambda: "h-1-ab")
        telemetry = TelemetryConfig(enabled=True, collector=config)
        collector = start_local_collector(telemetry, "run-1", {})
        collector.stop()
        with pytest.raises(CollectorError, match="already exists; another collector claimed it"):
            start_local_collector(telemetry, "run-1", {})

    def test_a_missing_token_keeps_collection_local(self, config, monkeypatch, capsys):
        monkeypatch.delenv("OTEL_TOKEN", raising=False)
        config = config.model_copy(
            update={"remote_export": True, "endpoint": "https://otlp.example.com", "service_name": "svc"}
        )
        collector = start_local_collector(TelemetryConfig(enabled=True, collector=config), "run-1", {})
        try:
            assert "otlp_http/managed" not in yaml.safe_load(collector.config_path.read_text())["exporters"]
            assert "OTEL_TOKEN is missing or empty" in capsys.readouterr().out
        finally:
            collector.stop()


@pytest.mark.skipif(shutil.which("otelcol-contrib") is None, reason="otelcol-contrib is not on PATH")
def test_with_the_real_collector_exported_metrics_are_kept_across_a_restart(config, tmp_path):
    config = config.model_copy(update={"binary": "otelcol-contrib", "health_check_timeout_seconds": 30})
    collector = start_local_collector(TelemetryConfig(enabled=True, collector=config), "run-1", {})
    endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]

    def post(value):
        body = {
            "resourceMetrics": [
                {
                    "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "probe"}}]},
                    "scopeMetrics": [
                        {
                            "metrics": [
                                {
                                    "name": "probe_value",
                                    "gauge": {
                                        "dataPoints": [{"asInt": str(value), "timeUnixNano": str(time.time_ns())}]
                                    },
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        request = Request(
            f"{endpoint}/v1/metrics", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urlopen(request, timeout=5) as response:
            assert response.status == 200

    try:
        post(1)
        time.sleep(1)
        first = collector.pid
        os.kill(first, signal.SIGTERM)
        _wait_until(lambda: _restarted(collector, config, first), timeout_s=60)
        post(2)
    finally:
        collector.stop()
    records = _metrics(collector)
    values = [
        int(point["asInt"])
        for record in records
        for resource in record["resourceMetrics"]
        if {"key": "run_id", "value": {"stringValue": "run-1"}} in resource["resource"]["attributes"]
        for scope in resource["scopeMetrics"]
        for metric in scope["metrics"]
        if metric["name"] == "probe_value"
        for point in metric["gauge"]["dataPoints"]
    ]
    assert values == [1, 2]


def test_the_supervisor_stops_the_collector_after_the_servers():
    events = []
    server = MagicMock()
    server.pid = 2**22 + 1
    server.wait.side_effect = lambda timeout=None: events.append("server exited")
    collector = MagicMock()
    collector.stop.side_effect = lambda: events.append("collector stopped")
    supervisor = server_supervisor.ServerSupervisor()
    supervisor._processes = {"server": server}
    supervisor._head_server = None
    supervisor._collector = collector
    supervisor._server_shutdown_timeout_seconds = 0.1
    supervisor.shutdown()
    assert events == ["server exited", "collector stopped"]
    assert supervisor._collector is None


# Run by a server process: it inherits the supervisor's environment, as `run_command` gives it.
_SERVER_EXPORT = r"""
import os
from urllib.request import Request, urlopen

endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] + "/v1/metrics"
urlopen(Request(endpoint, data=b"{}", headers={"Content-Type": "application/json"}), timeout=5).read()
"""


@pytest.fixture
def _restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_the_supervisor_starts_the_collector_for_its_servers_and_keeps_their_export(
    config, tmp_path, monkeypatch, _restore_environ
):
    """`ServerSupervisor.start` with a `telemetry.collector` block: the collector answers before the supervisor's
    telemetry starts, a server process gets its endpoint and exports to it, and the shutdown after a failed start
    stops the collector and keeps what the server sent."""
    monkeypatch.setenv("NEMO_GYM_OTEL_RUN_ID", "run-sup")
    global_config_dict = OmegaConf.create(
        {DRY_RUN_KEY_NAME: False, "telemetry": {"enabled": True, "collector": config.model_dump(mode="json")}}
    )
    events = []

    def init_telemetry(**kwargs):
        events.append(("init_telemetry", _healthy(config)))

    def start_head_server():
        subprocess.run([sys.executable, "-c", _SERVER_EXPORT], check=True)
        events.append(("server exported", None))
        raise RuntimeError("stop here")

    monkeypatch.setattr(server_supervisor, "get_global_config_dict", lambda **kwargs: global_config_dict)
    monkeypatch.setattr(
        server_supervisor.GlobalConfigDictParser, "raise_on_no_server_instances", lambda self, config: None
    )
    monkeypatch.setattr(server_supervisor, "_configured_servers", lambda config: iter(()))
    monkeypatch.setattr(server_supervisor, "start_h2_ping_sidecar", lambda config: None)
    monkeypatch.setattr(server_supervisor, "init_telemetry", init_telemetry)
    monkeypatch.setattr(server_supervisor.ServerSupervisor, "_config_dict_env", lambda self, *args: {})
    monkeypatch.setattr(server_supervisor.HeadServer, "run_webserver", start_head_server)

    supervisor = server_supervisor.ServerSupervisor()
    with pytest.raises(RuntimeError, match="stop here"):
        supervisor.start(None)

    assert events == [("init_telemetry", True), ("server exported", None)]
    assert not _healthy(config)
    (run_dir,) = (tmp_path / "out" / "otel" / "run-sup").iterdir()
    assert [record["event"] for record in _metrics(SimpleNamespace(run_dir=run_dir))] == ["start", "otlp", "final"]
