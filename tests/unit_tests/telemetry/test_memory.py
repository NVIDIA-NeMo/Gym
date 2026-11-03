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


def test_cpu_time_counts_new_processes_in_full_and_live_ones_by_their_increase():
    from nemo_gym.telemetry.memory import ProcessCpuTimes, cpu_time_since

    previous = {(10, 1.0): ProcessCpuTimes(5.0, 1.0), (11, 1.0): ProcessCpuTimes(2.0, 0.5)}
    current = {
        (10, 1.0): ProcessCpuTimes(7.0, 1.5),  # live: +2 user, +0.5 system
        (12, 3.0): ProcessCpuTimes(1.0, 0.25),  # new since the previous sample: all of it
        (11, 9.0): ProcessCpuTimes(0.5, 0.0),  # pid 11 reused by a new process: counted from zero
    }

    used = cpu_time_since(previous, current)

    assert used == ProcessCpuTimes(user_seconds=3.5, system_seconds=0.75)


def test_a_process_missing_from_one_walk_is_not_counted_again_when_it_reappears():
    from nemo_gym.telemetry.memory import ProcessCpuTimes, cpu_time_since, next_cpu_baseline

    still_running = lambda key: False  # noqa: E731
    samples = [
        {(10, 5): ProcessCpuTimes(10.0, 0.0)},
        {},  # the walk missed it once
        {(10, 5): ProcessCpuTimes(12.0, 0.0)},
    ]
    baseline: dict = {}
    total = 0.0
    for current in samples:
        total += cpu_time_since(baseline, current).user_seconds
        baseline = next_cpu_baseline(baseline, current, has_exited=still_running)

    assert total == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("read_error", "exited"),
    [(FileNotFoundError(), True), (PermissionError(), False), (OSError("I/O"), False)],
    ids=["pid gone", "unreadable", "I/O error"],
)
def test_only_a_missing_proc_entry_confirms_an_exit(monkeypatch, read_error, exited):
    from nemo_gym.telemetry.memory import process_has_exited

    def unreadable(pid):
        raise read_error

    monkeypatch.setattr(memory_telemetry, "_stat_fields", unreadable)

    assert process_has_exited((10, 5)) is exited


def test_an_exited_process_is_forgotten():
    from nemo_gym.telemetry.memory import ProcessCpuTimes, next_cpu_baseline

    baseline = next_cpu_baseline({(10, 5): ProcessCpuTimes(10.0, 0.0)}, {}, has_exited=lambda key: True)

    assert baseline == {}


def test_collects_the_cpu_time_of_a_child_in_a_real_process_tree():
    if not memory_telemetry.process_tree_memory_supported():
        pytest.skip("needs Linux procfs")
    from nemo_gym.telemetry.memory import collect_process_tree_cpu

    # Burns CPU for a moment, then says so and waits, so it is alive and has used CPU when the tree is read.
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time\nwhile time.process_time() < 0.2: pass\nprint(1, flush=True)\ntime.sleep(30)",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        child.stdout.readline()
        times = collect_process_tree_cpu(os.getpid())
    finally:
        child.kill()
        child.wait()

    child_times = [cpu for (pid, _started), cpu in times.items() if pid == child.pid]
    assert child_times and child_times[0].user_seconds + child_times[0].system_seconds >= 0.1


@requires_lens
def test_the_profiler_adds_each_servers_cpu_time_since_its_previous_sample(collected_metrics):
    from nemo_gym.telemetry.memory import MemoryProfiler, ProcessCpuTimes

    samples = iter(
        [
            {(10, 1.0): ProcessCpuTimes(2.0, 1.0)},
            {(10, 1.0): ProcessCpuTimes(5.0, 1.5), (11, 2.0): ProcessCpuTimes(1.0, 0.0)},
        ]
    )
    profiler = MemoryProfiler(
        [ServerMemoryTarget(name="judge", server_type="resources_servers", pid=10)],
        interval_seconds=60,
        collector=lambda pid: ProcessTreeMemory(rss_bytes=1, pss_bytes=1, process_count=1),
        host_collector=lambda: HostMemory(used_bytes=1, available_bytes=1, total_bytes=2),
        cpu_collector=lambda pid: next(samples),
    )

    profiler.sample_once()
    profiler.sample_once()

    points = {point.attributes["cpu.mode"]: point for point in collected_metrics()["gym.server.cpu.time"]}
    assert points["user"].value == pytest.approx(6.0)  # 2, then +3 and the new child's 1
    assert points["system"].value == pytest.approx(1.5)
    assert points["user"].attributes["nemo.gym.server.name"] == "judge"


@requires_lens
def test_records_host_cpu_capacity(collected_metrics):
    from nemo_gym.telemetry.memory import record_host_cpu_capacity

    record_host_cpu_capacity()

    metrics = collected_metrics()
    assert metrics["gym.host.cpu.logical"][0].value >= 1
    if hasattr(os, "sched_getaffinity"):
        assert 1 <= metrics["gym.host.cpu.allowed"][0].value <= metrics["gym.host.cpu.logical"][0].value
