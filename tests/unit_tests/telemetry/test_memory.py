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
"""Process-tree memory collection and Lens metric recording."""

import builtins
import io
import os
import signal
import subprocess
import sys
import time

import pytest

from nemo_gym.telemetry import memory as memory_telemetry
from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.telemetry.memory import (
    HostMemory,
    MemoryProfiler,
    ProcessTreeMemory,
    ServerMemoryTarget,
    collect_host_memory,
    collect_process_tree_memory,
    process_tree_memory_supported,
    record_host_memory,
    record_server_memory,
)
from tests.unit_tests.telemetry.conftest import requires_lens


def test_collects_current_process_memory():
    memory = collect_process_tree_memory(os.getpid())

    assert memory.process_count >= 1
    assert memory.rss_bytes > 0
    assert memory.pss_bytes is not None
    assert memory.pss_bytes > 0
    assert memory.pss_bytes <= memory.rss_bytes


def test_collects_host_memory():
    memory = collect_host_memory()

    assert memory.total_bytes > 0
    assert 0 <= memory.available_bytes <= memory.total_bytes
    assert memory.used_bytes == memory.total_bytes - memory.available_bytes


def test_memory_profiling_requires_linux_procfs(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert process_tree_memory_supported() is False


def test_missing_process_returns_zero_memory():
    memory = collect_process_tree_memory(999_999_999)
    assert memory == ProcessTreeMemory(rss_bytes=0, pss_bytes=0, process_count=0)


def test_process_memory_falls_back_to_statm(monkeypatch):
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path).endswith("/smaps_rollup"):
            raise PermissionError
        if str(path).endswith("/statm"):
            return io.StringIO("100 7 0 0 0 0 0")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert memory_telemetry._read_process_memory(123) == (7 * os.sysconf("SC_PAGE_SIZE"), None)


def test_process_tree_omits_incomplete_pss(monkeypatch):
    monkeypatch.setattr(memory_telemetry, "_descendant_pids", lambda _pid: {1, 2})
    values = {1: (10, 8), 2: (20, None)}
    monkeypatch.setattr(memory_telemetry, "_read_process_memory", values.__getitem__)

    assert collect_process_tree_memory(1) == ProcessTreeMemory(rss_bytes=30, pss_bytes=None, process_count=2)


def test_descendant_walk_tolerates_unreadable_children(monkeypatch):
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path).endswith("/children"):
            raise PermissionError
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert memory_telemetry._descendant_pids(os.getpid()) == {os.getpid()}


def test_collects_descendant_processes():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess, sys, time; "
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                "print(child.pid, flush=True); time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = None
    try:
        assert process.stdout is not None
        child_pid = int(process.stdout.readline())
        memory = collect_process_tree_memory(process.pid)
        assert memory.process_count >= 2
    finally:
        process.terminate()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture
def collected_metrics(monkeypatch):
    pytest.importorskip("nemo.lens")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test-memory")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())

    def collect():
        data = reader.get_metrics_data()
        return {
            metric.name: list(metric.data.data_points)
            for resource_metric in data.resource_metrics
            for scope_metric in resource_metric.scope_metrics
            for metric in scope_metric.metrics
        }

    return collect


@requires_lens
def test_records_current_and_sampled_peak_memory(collected_metrics):
    target = ServerMemoryTarget(name="weather", server_type="resources_servers", pid=123)
    peaks: dict[tuple[str, str], int] = {}

    record_server_memory(
        target,
        ProcessTreeMemory(rss_bytes=120, pss_bytes=100, process_count=3),
        peaks,
    )
    record_server_memory(
        target,
        ProcessTreeMemory(rss_bytes=90, pss_bytes=80, process_count=2),
        peaks,
    )

    metrics = collected_metrics()
    usage = {point.attributes["memory.type"]: point.value for point in metrics["gym.server.memory.usage"]}
    sampled_peak = {
        point.attributes["memory.type"]: point.value for point in metrics["gym.server.memory.sampled_peak"]
    }
    processes = metrics["gym.server.processes"][0]

    assert usage == {"rss": 90, "pss": 80}
    assert sampled_peak == {"rss": 120, "pss": 100}
    assert processes.value == 2
    assert processes.attributes["nemo.gym.server.name"] == "weather"
    assert processes.attributes["nemo.gym.server.type"] == "resources_servers"
    assert processes.attributes["process.pid"] == 123


@requires_lens
def test_records_host_capacity_and_sampled_peak(collected_metrics):
    peaks: dict[str, int] = {}

    record_host_memory(HostMemory(used_bytes=700, available_bytes=300, total_bytes=1000), peaks)
    record_host_memory(HostMemory(used_bytes=600, available_bytes=400, total_bytes=1000), peaks)

    metrics = collected_metrics()
    assert metrics["gym.host.memory.usage"][0].value == 600
    assert metrics["gym.host.memory.available"][0].value == 400
    assert metrics["gym.host.memory.total"][0].value == 1000
    assert metrics["gym.host.memory.sampled_peak"][0].value == 700


def test_recording_without_telemetry_is_a_no_op():
    peaks = {}
    record_server_memory(
        ServerMemoryTarget(name="weather", server_type="resources_servers", pid=123),
        ProcessTreeMemory(rss_bytes=100, pss_bytes=80, process_count=1),
        peaks,
    )
    assert peaks == {}


def test_recording_failure_does_not_reach_server(monkeypatch):
    class BrokenMeter:
        def create_gauge(self, *args, **kwargs):
            raise RuntimeError("broken meter")

    monkeypatch.setattr(
        telemetry_setup,
        "_TELEMETRY_HANDLE",
        type("Handle", (), {"is_exporting": True, "meter": BrokenMeter()})(),
    )
    record_server_memory(
        ServerMemoryTarget(name="weather", server_type="resources_servers", pid=123),
        ProcessTreeMemory(rss_bytes=100, pss_bytes=80, process_count=1),
        {},
    )


def test_profiler_samples_every_target(monkeypatch):
    samples = {
        10: ProcessTreeMemory(rss_bytes=100, pss_bytes=80, process_count=1),
        20: ProcessTreeMemory(rss_bytes=200, pss_bytes=160, process_count=2),
    }
    recorded = []
    host_recorded = []
    monkeypatch.setattr(
        memory_telemetry, "record_server_memory", lambda target, memory, peaks: recorded.append((target, memory))
    )
    monkeypatch.setattr(
        memory_telemetry,
        "record_host_memory",
        lambda memory, peaks: host_recorded.append(memory),
    )
    profiler = MemoryProfiler(
        [
            ServerMemoryTarget(name="a", server_type="resources_servers", pid=10),
            ServerMemoryTarget(name="b", server_type="responses_api_agents", pid=20),
        ],
        interval_seconds=1,
        collector=samples.__getitem__,
        host_collector=lambda: HostMemory(used_bytes=70, available_bytes=30, total_bytes=100),
    )

    profiler.sample_once()

    assert [(target.name, memory.rss_bytes) for target, memory in recorded] == [("a", 100), ("b", 200)]
    assert host_recorded == [HostMemory(used_bytes=70, available_bytes=30, total_bytes=100)]


def test_profiler_lifecycle_is_idempotent(monkeypatch):
    calls = 0

    def collect(_pid):
        nonlocal calls
        calls += 1
        return ProcessTreeMemory(rss_bytes=100, pss_bytes=80, process_count=1)

    monkeypatch.setattr(memory_telemetry, "record_server_memory", lambda *args: None)
    profiler = MemoryProfiler(
        [ServerMemoryTarget(name="a", server_type="resources_servers", pid=10)],
        interval_seconds=0.001,
        collector=collect,
        host_collector=lambda: HostMemory(used_bytes=70, available_bytes=30, total_bytes=100),
    )

    profiler.start()
    thread = profiler._thread
    profiler.start()
    assert profiler._thread is thread
    time.sleep(0.01)
    profiler.stop()
    profiler.stop()

    assert calls >= 3
    assert profiler._thread is None


def test_profiler_suppresses_collection_errors():
    profiler = MemoryProfiler(
        [ServerMemoryTarget(name="a", server_type="resources_servers", pid=10)],
        interval_seconds=1,
        collector=lambda _pid: (_ for _ in ()).throw(RuntimeError("sample failed")),
        host_collector=lambda: (_ for _ in ()).throw(RuntimeError("host sample failed")),
    )
    profiler.sample_once()
