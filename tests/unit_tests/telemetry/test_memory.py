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
from dataclasses import dataclass

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
    assert memory_telemetry._descendant_pids(os.getpid()) == [os.getpid()]


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


def _counted_user_seconds(samples, exited=()):
    """Feed CPU samples through the profiler's baseline and return the user seconds the counter would add."""
    from nemo_gym.telemetry.memory import advance_cpu_baseline

    baseline: dict = {}
    total = 0.0
    for current in samples:
        used, baseline = advance_cpu_baseline(
            baseline, current, has_exited=lambda key: key in exited, parent_pid_of=lambda key: None
        )
        assert used.user_seconds >= 0
        total += used.user_seconds
    return total


def _cpu(user_seconds, parent=None):
    from nemo_gym.telemetry.memory import ProcessCpuTimes

    return ProcessCpuTimes(user_seconds, 0.0, parent)


ROOT, CHILD, GRANDCHILD = (10, 1), (11, 2), (12, 3)


def test_a_process_missing_from_one_walk_is_not_counted_again_when_it_reappears():
    samples = [
        {(10, 5): _cpu(10.0)},
        {},  # the walk missed it once
        {(10, 5): _cpu(12.0)},
    ]

    assert _counted_user_seconds(samples) == pytest.approx(12.0)


def test_children_that_start_and_exit_between_samples_are_counted_through_the_parent_that_waited():
    # 20 children of 0.3 s each, all started, waited for and gone between the two samples.
    samples = [{ROOT: _cpu(0.1)}, {ROOT: _cpu(0.1 + 20 * 0.3)}]

    assert _counted_user_seconds(samples) == pytest.approx(6.1)


def test_a_child_seen_before_its_parent_waited_for_it_is_not_counted_twice():
    samples = [
        {ROOT: _cpu(1.0), CHILD: _cpu(2.0, parent=ROOT)},
        {ROOT: _cpu(1.5 + 2.5)},  # the child used 0.5 s more, exited, and the root waited for it
    ]

    assert _counted_user_seconds(samples, exited={CHILD}) == pytest.approx(4.0)


def test_a_child_waited_for_by_a_parent_that_exited_in_the_same_interval_is_not_counted_twice():
    samples = [
        {ROOT: _cpu(0.0), CHILD: _cpu(1.0, parent=ROOT), GRANDCHILD: _cpu(2.0, parent=CHILD)},
        # The child waited for the grandchild (2.5 s in all), used 1.2 s itself, exited, and the root waited for it.
        {ROOT: _cpu(1.2 + 2.5)},
    ]

    assert _counted_user_seconds(samples, exited={CHILD, GRANDCHILD}) == pytest.approx(3.7)


def test_a_child_reparented_out_of_the_tree_keeps_what_was_counted_and_adds_nothing_after():
    from nemo_gym.telemetry.memory import advance_cpu_baseline

    samples_and_exits = [
        ({ROOT: _cpu(0.0), CHILD: _cpu(1.0, parent=ROOT), GRANDCHILD: _cpu(2.0, parent=CHILD)}, set()),
        # The child exited without waiting; the grandchild now runs outside the tree.
        ({ROOT: _cpu(1.2)}, {CHILD}),
        # The grandchild exited and was reaped outside the tree.
        ({ROOT: _cpu(1.2)}, {CHILD, GRANDCHILD}),
    ]
    baseline: dict = {}
    counted = []
    for current, exited in samples_and_exits:
        reparented_to_init = lambda key: 1  # noqa: E731
        used, baseline = advance_cpu_baseline(
            baseline, current, has_exited=exited.__contains__, parent_pid_of=reparented_to_init
        )
        counted.append(used.user_seconds)

    assert counted == pytest.approx([3.0, 0.2, 0.0])


def test_time_moved_onto_a_parent_before_its_times_show_it_is_held_back_until_they_do():
    samples = [
        {ROOT: _cpu(0.0), CHILD: _cpu(2.0, parent=ROOT)},
        {ROOT: _cpu(0.1)},  # the root was read just before it waited for the exited child
        {ROOT: _cpu(0.1 + 2.3)},
    ]

    assert _counted_user_seconds(samples, exited={CHILD}) == pytest.approx(2.4)


def test_a_parent_that_does_not_wait_for_its_children_is_undercounted_not_overcounted():
    samples = [
        {ROOT: _cpu(1.0), CHILD: _cpu(2.0, parent=ROOT)},
        {ROOT: _cpu(1.5)},  # SIGCHLD ignored: the exited child's time never reaches the root
        {ROOT: _cpu(4.0)},
    ]

    assert _counted_user_seconds(samples, exited={CHILD}) == pytest.approx(4.0)  # used 4.0 + at least 2.0


def test_a_reused_child_pid_is_counted_from_zero_after_the_old_child_is_folded_into_its_parent():
    samples = [
        {ROOT: _cpu(0.0), (11, 2): _cpu(2.0, parent=ROOT)},
        {ROOT: _cpu(2.0), (11, 9): _cpu(0.5, parent=ROOT)},
    ]

    assert _counted_user_seconds(samples, exited={(11, 2)}) == pytest.approx(2.5)


def test_reads_own_and_waited_for_children_times_and_the_parent_from_proc_stat(monkeypatch):
    from nemo_gym.telemetry.memory import ProcessCpuTimes, collect_process_tree_cpu

    clock_ticks = memory_telemetry._CLOCK_TICKS
    # pid: (ppid, utime, stime, cutime, cstime, starttime) in clock ticks.
    stats = {100: (1, 10, 20, 30, 40, 7), 101: (100, 1, 2, 3, 4, 8), 102: (999, 0, 0, 0, 0, 9)}

    def stat_fields(pid):
        ppid, utime, stime, cutime, cstime, start = stats[pid]
        return ["S", str(ppid), *["0"] * 9, str(utime), str(stime), str(cutime), str(cstime), *["0"] * 4, str(start)]

    monkeypatch.setattr(memory_telemetry, "_descendant_pids", lambda _pid: [100, 101, 102])
    monkeypatch.setattr(memory_telemetry, "_stat_fields", stat_fields)

    assert collect_process_tree_cpu(100) == {
        (100, 7): ProcessCpuTimes(40 / clock_ticks, 60 / clock_ticks, parent=None, parent_pid=1),
        (101, 8): ProcessCpuTimes(4 / clock_ticks, 6 / clock_ticks, parent=(100, 7), parent_pid=100),
        (102, 9): ProcessCpuTimes(0.0, 0.0, parent=None, parent_pid=999),
    }


@dataclass
class _FakeProcess:
    ppid: int
    start: int
    own: float = 0.0
    children: float = 0.0
    state: str = "S"
    threads: int = 1
    unreadable: bool = False
    missed_by_walk: bool = False


class _FakeProc:
    """A procfs stand-in for the CPU walk: processes by pid, and the kernel steps that change them."""

    def __init__(self, monkeypatch):
        self.processes: dict[int, _FakeProcess] = {}
        self.after_next_read: list = []
        self.baseline: dict = {}
        self.counted = 0.0
        monkeypatch.setattr(memory_telemetry, "_child_pids", self.child_pids)
        monkeypatch.setattr(memory_telemetry, "_stat_fields", self.stat_fields)

    def spawn(self, pid, ppid, own=0.0):
        self.processes[pid] = _FakeProcess(ppid=ppid, start=pid, own=own)

    def child_pids(self, pid):
        return [child for child, p in self.processes.items() if p.ppid == pid and not p.missed_by_walk]

    def stat_fields(self, pid):
        process = self.processes.get(pid)
        steps, self.after_next_read = self.after_next_read, []
        try:
            if process is None:
                raise FileNotFoundError(pid)
            if process.unreadable:
                raise PermissionError(pid)
            ticks = memory_telemetry._CLOCK_TICKS
            return [
                process.state,
                str(process.ppid),
                *["0"] * 9,
                str(round(process.own * ticks)),
                "0",
                str(round(process.children * ticks)),
                "0",
                "0",
                "0",
                str(process.threads),
                "0",
                str(process.start),
            ]
        finally:
            for step in steps:
                step()

    def begin_reap(self, pid):
        """``wait_task_zombie`` up to ``release_task``: state ``X``, the total already in the parent's children times."""
        process = self.processes[pid]
        process.state = "X"
        self.processes[process.ppid].children += process.own + process.children

    def reap(self, pid):
        if self.processes[pid].state != "X":
            self.begin_reap(pid)
        del self.processes[pid]

    def sample(self, root):
        from nemo_gym.telemetry.memory import advance_cpu_baseline, collect_process_tree_cpu

        used, self.baseline = advance_cpu_baseline(self.baseline, collect_process_tree_cpu(root))
        self.counted += used.user_seconds
        return self.counted


@pytest.mark.parametrize("state", ["X", "Z"], ids=["being reaped", "zombie"])
def test_a_child_seen_dead_after_its_parent_absorbed_it_is_not_counted_twice(monkeypatch, state):
    proc = _FakeProc(monkeypatch)
    proc.spawn(100, ppid=1)
    proc.spawn(101, ppid=100, own=2.0)
    assert proc.sample(100) == pytest.approx(2.0)

    proc.processes[101].own = 2.5
    if state == "X":
        proc.begin_reap(101)
    else:
        proc.processes[101].state = "Z"
    proc.sample(100)
    proc.reap(101)

    assert proc.sample(100) == pytest.approx(2.5)


def test_a_process_whose_main_thread_exited_while_others_run_is_still_counted(monkeypatch):
    proc = _FakeProc(monkeypatch)
    proc.spawn(100, ppid=1)
    proc.spawn(101, ppid=100, own=1.0)
    proc.sample(100)

    proc.processes[101].state, proc.processes[101].threads, proc.processes[101].own = "Z", 2, 3.0

    assert proc.sample(100) == pytest.approx(3.0)


def test_a_parent_that_could_not_be_read_once_keeps_its_child_link(monkeypatch):
    proc = _FakeProc(monkeypatch)
    proc.spawn(100, ppid=1)
    proc.spawn(101, ppid=100, own=1.0)
    proc.spawn(102, ppid=101, own=2.0)
    proc.sample(100)

    proc.processes[101].unreadable = True
    proc.sample(100)
    proc.processes[101].unreadable = False
    proc.processes[102].own = 2.5
    proc.reap(102)

    assert proc.sample(100) == pytest.approx(3.5)


def test_a_child_reaped_between_two_stat_reads_of_one_walk_is_counted_once(monkeypatch):
    proc = _FakeProc(monkeypatch)
    proc.spawn(100, ppid=1, own=1.0)
    proc.spawn(101, ppid=100, own=2.0)
    proc.sample(100)

    proc.processes[101].own = 2.5
    proc.after_next_read.append(lambda: proc.reap(101))
    proc.sample(100)

    assert proc.sample(100) == pytest.approx(3.5)


def test_an_orphan_adopted_by_a_subreaper_and_missed_by_the_walk_is_counted_once(monkeypatch):
    proc = _FakeProc(monkeypatch)
    proc.spawn(100, ppid=1)  # a child subreaper
    proc.spawn(101, ppid=100, own=1.0)
    proc.spawn(102, ppid=101, own=2.0)
    proc.sample(100)

    proc.processes[101].own = 1.2
    proc.processes[102].ppid = 100  # its parent exits without waiting; the subreaper adopts it
    proc.reap(101)
    proc.processes[102].missed_by_walk = True
    proc.sample(100)
    proc.processes[102].own = 2.5
    proc.reap(102)

    assert proc.sample(100) == pytest.approx(3.7)


def test_an_orphan_that_exits_with_its_parent_is_counted_twice_until_its_time_reaches_the_subreaper(monkeypatch):
    proc = _FakeProc(monkeypatch)
    proc.spawn(100, ppid=1)  # a child subreaper
    proc.spawn(101, ppid=100, own=1.0)
    proc.spawn(102, ppid=101, own=0.5)
    proc.spawn(103, ppid=102, own=2.0)
    assert proc.sample(100) == pytest.approx(3.5)

    proc.processes[103].ppid = 100  # 102 exits without waiting for 103, which the subreaper adopts and reaps
    proc.reap(102)
    proc.processes[103].own = 2.5
    proc.reap(103)
    assert proc.sample(100) == pytest.approx(4.0 + 2.0)  # 103's 2.0 already counted is counted again

    proc.processes[101].own = 1.2
    proc.reap(101)
    assert proc.sample(100) == pytest.approx(6.0)  # 101 hands the 2.0 on to the subreaper

    proc.processes[100].own = 2.0
    assert proc.sample(100) == pytest.approx(6.2)  # taken back from the subreaper's own time: 1.2 + 0.5 + 2.5 + 2.0


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
    from nemo_gym.telemetry.memory import ProcessCpuTimes, advance_cpu_baseline

    _used, baseline = advance_cpu_baseline({(10, 5): ProcessCpuTimes(10.0, 0.0)}, {}, has_exited=lambda key: True)

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


def test_counts_real_children_that_start_and_exit_between_two_samples():
    if not memory_telemetry.process_tree_memory_supported():
        pytest.skip("needs Linux procfs")
    from nemo_gym.telemetry.memory import advance_cpu_baseline, collect_process_tree_cpu

    # After a line on stdin, runs and waits for five children that each burn 0.1 s of CPU, then says so and waits.
    script = (
        "import subprocess, sys, time\n"
        "sys.stdin.readline()\n"
        "burn = 'import time\\nwhile time.process_time() < 0.1: pass'\n"
        "for _ in range(5):\n"
        "    subprocess.run([sys.executable, '-c', burn], check=True)\n"
        "print(1, flush=True)\n"
        "time.sleep(30)"
    )
    parent = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        _used, baseline = advance_cpu_baseline({}, collect_process_tree_cpu(parent.pid))
        parent.stdin.write("\n")
        parent.stdin.flush()
        parent.stdout.readline()
        used, _baseline = advance_cpu_baseline(baseline, collect_process_tree_cpu(parent.pid))
    finally:
        parent.kill()
        parent.wait()

    assert used.user_seconds + used.system_seconds >= 0.4


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

    points = {point.attributes["cpu.mode"]: point for point in collected_metrics()["gym.server.cpu.time_total"]}
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
