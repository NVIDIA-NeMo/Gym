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
"""Linux process-tree memory sampling for Gym-managed servers."""

import logging
import os
import sys
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import psutil


logger = logging.getLogger(__name__)
_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class ServerMemoryTarget:
    """One configured Gym server and the root process Gym spawned for it."""

    name: str
    server_type: str
    pid: int


@dataclass(frozen=True)
class ProcessTreeMemory:
    """One process tree's current memory."""

    rss_bytes: int
    pss_bytes: int | None
    process_count: int


@dataclass(frozen=True)
class HostMemory:
    """Current host memory capacity and pressure."""

    used_bytes: int
    available_bytes: int
    total_bytes: int


# A process is identified by its pid and start time (kernel clock ticks since boot, unaffected by wall-clock changes),
# so a reused pid does not inherit an exited process's CPU time.
ProcessKey = tuple[int, int]


@dataclass(frozen=True)
class ProcessCpuTimes:
    """CPU time one process and the children it waited for have used since it started.

    ``parent_pid`` is the parent pid the process reported when it was read. ``parent`` is that parent's identity, or
    ``None`` when the parent is not a known process of the tree.
    """

    user_seconds: float
    system_seconds: float
    parent: ProcessKey | None = None
    parent_pid: int | None = None


_CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def process_tree_memory_supported() -> bool:
    """Return whether this host exposes Linux procfs memory accounting."""
    return sys.platform.startswith("linux") and Path("/proc/self").is_dir()


def collect_host_memory() -> HostMemory:
    """Measure current host memory using psutil's cross-platform accounting."""
    memory = psutil.virtual_memory()
    return HostMemory(
        used_bytes=memory.total - memory.available,
        available_bytes=memory.available,
        total_bytes=memory.total,
    )


def collect_process_tree_memory(root_pid: int) -> ProcessTreeMemory:
    """Measure current RSS and PSS for a root process and its descendants."""
    rss_bytes = 0
    pss_bytes = 0
    process_count = 0
    complete_pss = True
    for pid in _descendant_pids(root_pid):
        memory = _read_process_memory(pid)
        if memory is None:
            continue
        rss, pss = memory
        rss_bytes += rss
        process_count += 1
        if pss is None:
            complete_pss = False
        else:
            pss_bytes += pss
    return ProcessTreeMemory(
        rss_bytes=rss_bytes,
        pss_bytes=pss_bytes if complete_pss else None,
        process_count=process_count,
    )


def _stat_fields(pid: int) -> list[str]:
    """The fields of ``/proc/<pid>/stat`` after the command name; raises ``FileNotFoundError`` once the pid is gone."""
    with open(f"/proc/{pid}/stat") as stat_file:
        stat = stat_file.read()
    # The command name is parenthesized and may hold spaces or parentheses; the fields follow the last ")".
    return stat[stat.rindex(")") + 2 :].split()


def _has_ended(fields: list[str]) -> bool:
    """Whether ``/proc/<pid>/stat`` fields describe a process that has exited and has no thread still running.

    A child being reaped shows ``X`` once its parent's children times already include it, so it is not counted as
    well. A zombie main thread whose other threads still run is a running process.
    """
    return fields[0] in ("Z", "X") and int(fields[17]) <= 1


def _read_process_cpu(pid: int) -> tuple[ProcessKey, ProcessCpuTimes] | None:
    """One process's identity, CPU times and parent pid from one ``/proc/<pid>/stat`` read.

    ``None`` if it cannot be read or has exited. The times include ``cutime``/``cstime``: the kernel adds a child's own
    and children times to its parent's when the parent waits for it, so a child that starts and exits between two
    samples is still counted.
    """
    try:
        fields = _stat_fields(pid)
        if _has_ended(fields):
            return None
        parent_pid, start_ticks = int(fields[1]), int(fields[19])
        user_ticks = int(fields[11]) + int(fields[13])
        system_ticks = int(fields[12]) + int(fields[14])
    except (OSError, ValueError, IndexError):
        return None
    times = ProcessCpuTimes(user_ticks / _CLOCK_TICKS, system_ticks / _CLOCK_TICKS, parent_pid=parent_pid)
    return (pid, start_ticks), times


def collect_process_tree_cpu(root_pid: int) -> dict[ProcessKey, ProcessCpuTimes]:
    """CPU time used so far by a root process and each of its live descendants (Linux procfs)."""
    # Parents are read before their children, so a child reaped during the walk is never in both its own times and
    # its parent's.
    processes = [process for pid in _descendant_pids(root_pid) if (process := _read_process_cpu(pid)) is not None]
    keys = {key[0]: key for key, _times in processes}
    return {key: replace(times, parent=keys.get(times.parent_pid)) for key, times in processes}


def process_has_exited(key: ProcessKey) -> bool:
    """Whether the process ``key`` names is confirmed gone: its pid is free, belongs to a later process, or is dead.

    An entry that cannot be read for another reason (permissions, I/O) is not proof of an exit, so it counts as running.
    """
    try:
        fields = _stat_fields(key[0])
        return int(fields[19]) != key[1] or _has_ended(fields)
    except FileNotFoundError:
        return True
    except (OSError, ValueError, IndexError):
        return False


def current_parent_pid(key: ProcessKey) -> int | None:
    """The parent pid the process ``key`` names reports now, or ``None`` if it cannot be read or its pid was reused."""
    try:
        fields = _stat_fields(key[0])
        if int(fields[19]) == key[1]:
            return int(fields[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def fold_exited_processes(
    previous: dict[ProcessKey, ProcessCpuTimes],
    current: dict[ProcessKey, ProcessCpuTimes],
    has_exited: Callable[[ProcessKey], bool] = process_has_exited,
) -> dict[ProcessKey, ProcessCpuTimes]:
    """``previous`` without the processes that have exited, each one's counted time moved onto an ancestor's baseline.

    A process can drop out of one walk (an unreadable ``children`` file, a race with a fork); keeping its baseline
    means it is not counted again from zero when it reappears. Only a process that has exited is forgotten.

    When a parent waits for an exited child, the kernel adds the child's whole total to the parent's children times.
    The part already counted is added to the baseline of the child's nearest ancestor that has not exited (its parent,
    or further up when the parent exited in the same interval after waiting for it), so it is not counted twice. An
    ancestor that never received it (one that ignores ``SIGCHLD``, or a child orphaned and reaped outside the tree)
    then counts less than the tree used. A process whose ancestors all left the tree is not moved.
    """
    exited = {key for key in previous if key not in current and has_exited(key)}
    folded = {key: times for key, times in previous.items() if key not in exited}
    for key in exited:
        ancestor, seen = previous[key].parent, {key}
        while ancestor in exited and ancestor not in seen:
            seen.add(ancestor)
            ancestor = previous[ancestor].parent
        if ancestor in folded:
            base = folded[ancestor]
            folded[ancestor] = replace(
                base,
                user_seconds=base.user_seconds + previous[key].user_seconds,
                system_seconds=base.system_seconds + previous[key].system_seconds,
            )
    return folded


def cpu_time_since(
    previous: dict[ProcessKey, ProcessCpuTimes], current: dict[ProcessKey, ProcessCpuTimes]
) -> ProcessCpuTimes:
    """CPU time the tree used between two samples: each live process's increase, all of it for a new process."""
    user = system = 0.0
    for key, times in current.items():
        before = previous.get(key, ProcessCpuTimes(0.0, 0.0))
        user += max(0.0, times.user_seconds - before.user_seconds)
        system += max(0.0, times.system_seconds - before.system_seconds)
    return ProcessCpuTimes(user_seconds=user, system_seconds=system)


def advance_cpu_baseline(
    previous: dict[ProcessKey, ProcessCpuTimes],
    current: dict[ProcessKey, ProcessCpuTimes],
    has_exited: Callable[[ProcessKey], bool] = process_has_exited,
    parent_pid_of: Callable[[ProcessKey], int | None] = current_parent_pid,
) -> tuple[ProcessCpuTimes, dict[ProcessKey, ProcessCpuTimes]]:
    """The CPU time a tree used since ``previous``, and the baseline for the next sample.

    A baseline never goes down, so time moved onto a parent before the parent's own times show it (the parent was
    read just before it waited) is held back until they do, and the total never goes down either.

    A parent that could not be read in this walk but still runs keeps its baseline, so a child naming it is linked to
    it: a running process's reported parent pid always names its live parent. A process missing from this walk that
    still runs is linked to the parent it reports now, such as a child subreaper that adopted it, or to none when it
    was reparented outside the tree.
    """
    baseline = fold_exited_processes(previous, current, has_exited)
    used = cpu_time_since(baseline, current)
    running = {key[0]: key for key in [*baseline, *current]}
    for key, times in list(baseline.items()):
        if key not in current and (parent_pid := parent_pid_of(key)) not in (None, times.parent_pid):
            baseline[key] = replace(times, parent=running.get(parent_pid), parent_pid=parent_pid)
    for key, times in current.items():
        if times.parent is None and times.parent_pid in running:
            times = replace(times, parent=running[times.parent_pid])
        before = baseline.get(key)
        if before is not None:
            times = replace(
                times,
                user_seconds=max(times.user_seconds, before.user_seconds),
                system_seconds=max(times.system_seconds, before.system_seconds),
            )
        baseline[key] = times
    return used, baseline


def record_server_cpu(target: ServerMemoryTarget, used: ProcessCpuTimes) -> None:
    """Add a server process tree's CPU time since the previous sample to ``gym.server.cpu.time_total``."""
    from nemo_gym.telemetry.setup import get_telemetry

    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return
    try:
        counter = _get_instruments(telemetry.meter)["cpu_time"]
        base_attributes = {"nemo.gym.server.name": target.name, "nemo.gym.server.type": target.server_type}
        for mode, seconds in (("user", used.user_seconds), ("system", used.system_seconds)):
            if seconds > 0:
                counter.add(seconds, {**base_attributes, "cpu.mode": mode})
    except Exception:
        logger.debug("Failed to record Gym server CPU time", exc_info=True)


def record_host_cpu_capacity() -> None:
    """Record the host's logical and physical CPU counts and the CPUs this process may run on."""
    from nemo_gym.telemetry.setup import get_telemetry

    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return
    try:
        instruments = _get_instruments(telemetry.meter)
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        if logical:
            instruments["cpu_logical"].set(logical)
        if physical:
            instruments["cpu_physical"].set(physical)
        # The supervisor's main-thread affinity; under Slurm or a container cpuset that is usually the allocation.
        if hasattr(os, "sched_getaffinity"):
            instruments["cpu_allowed"].set(len(os.sched_getaffinity(os.getpid())))
    except Exception:
        logger.debug("Failed to record host CPU capacity", exc_info=True)


def record_server_memory(
    target: ServerMemoryTarget,
    memory: ProcessTreeMemory,
    peaks: dict[tuple[str, str], int],
) -> None:
    """Record current and sampled-peak process-tree memory through Lens."""
    from nemo_gym.telemetry.setup import get_telemetry

    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return
    try:
        instruments = _get_instruments(telemetry.meter)
        base_attributes = {
            "nemo.gym.server.name": target.name,
            "nemo.gym.server.type": target.server_type,
            "process.pid": target.pid,
        }
        instruments["processes"].set(memory.process_count, base_attributes)
        values = {"rss": memory.rss_bytes}
        if memory.pss_bytes is not None:
            values["pss"] = memory.pss_bytes
        for memory_type, value in values.items():
            attributes = {**base_attributes, "memory.type": memory_type}
            peak_key = (target.name, memory_type)
            peaks[peak_key] = max(peaks.get(peak_key, 0), value)
            instruments["usage"].set(value, attributes)
            instruments["sampled_peak"].set(peaks[peak_key], attributes)
    except Exception:
        logger.debug("Failed to record Gym server memory", exc_info=True)


def record_host_memory(memory: HostMemory, peaks: dict[str, int]) -> None:
    """Record current host memory and its sampled peak through Lens."""
    from nemo_gym.telemetry.setup import get_telemetry

    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return
    try:
        instruments = _get_instruments(telemetry.meter)
        peaks["used"] = max(peaks.get("used", 0), memory.used_bytes)
        instruments["host_usage"].set(memory.used_bytes)
        instruments["host_available"].set(memory.available_bytes)
        instruments["host_total"].set(memory.total_bytes)
        instruments["host_sampled_peak"].set(peaks["used"])
    except Exception:
        logger.debug("Failed to record Gym host memory", exc_info=True)


class MemoryProfiler:
    """Periodically report host memory and Gym-managed server process trees."""

    def __init__(
        self,
        targets: list[ServerMemoryTarget],
        interval_seconds: float,
        *,
        collector: Callable[[int], ProcessTreeMemory] = collect_process_tree_memory,
        host_collector: Callable[[], HostMemory] = collect_host_memory,
        cpu_collector: Callable[[int], dict[ProcessKey, ProcessCpuTimes]] = collect_process_tree_cpu,
    ) -> None:
        self._targets = targets
        self._interval_seconds = interval_seconds
        self._collector = collector
        self._host_collector = host_collector
        self._cpu_collector = cpu_collector
        self._cpu_seen: dict[str, dict[ProcessKey, ProcessCpuTimes]] = {}
        self._peaks: dict[tuple[str, str], int] = {}
        self._host_peaks: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        """Record one host sample and one sample for every configured server."""
        try:
            record_host_memory(self._host_collector(), self._host_peaks)
        except Exception:
            logger.debug("Failed to sample host memory", exc_info=True)
        record_host_cpu_capacity()
        for target in self._targets:
            try:
                memory = self._collector(target.pid)
                record_server_memory(target, memory, self._peaks)
            except Exception:
                logger.debug(f"Failed to sample memory for {target.name}", exc_info=True)
            try:
                current = self._cpu_collector(target.pid)
                used, self._cpu_seen[target.name] = advance_cpu_baseline(self._cpu_seen.get(target.name, {}), current)
                record_server_cpu(target, used)
            except Exception:
                logger.debug(f"Failed to sample CPU time for {target.name}", exc_info=True)

    def start(self) -> None:
        """Start background sampling after recording an initial sample."""
        if self._thread is not None:
            return
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="nemo-gym-memory-profiler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and record final live host and process-tree values."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self.sample_once()
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.sample_once()


def _descendant_pids(root_pid: int) -> list[int]:
    """Return a snapshot of a Linux process tree, including its root, each process before its children."""
    pids: dict[int, None] = {}
    to_visit = [root_pid]
    while to_visit:
        pid = to_visit.pop()
        if pid in pids:
            continue
        pids[pid] = None
        to_visit.extend(_child_pids(pid))
    return list(pids)


def _child_pids(pid: int) -> list[int]:
    """The children of every thread of ``pid`` that can be read; a process that has gone has none."""
    child_pids: list[int] = []
    try:
        with os.scandir(f"/proc/{pid}/task") as task_dirs:
            for task_dir in task_dirs:
                try:
                    with open(f"{task_dir.path}/children") as children_file:
                        children = children_file.read().split()
                except (OSError, PermissionError):
                    continue
                child_pids.extend(int(child) for child in children if child.isdigit())
    except (OSError, PermissionError):
        pass
    return child_pids


def _read_process_memory(pid: int) -> tuple[int, int | None] | None:
    """Return ``(rss, pss)`` bytes for one process, or ``None`` after an exit race."""
    try:
        rss_kib = 0
        pss_kib = 0
        saw_rss = False
        saw_pss = False
        with open(f"/proc/{pid}/smaps_rollup") as smaps:
            for line in smaps:
                if line.startswith("Rss:"):
                    rss_kib = int(line.split()[1])
                    saw_rss = True
                elif line.startswith("Pss:"):
                    pss_kib = int(line.split()[1])
                    saw_pss = True
        if saw_rss:
            return rss_kib * 1024, pss_kib * 1024 if saw_pss else None
    except (OSError, PermissionError, ValueError, IndexError):
        pass

    try:
        with open(f"/proc/{pid}/statm") as statm:
            rss_pages = int(statm.read().split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE"), None
    except (OSError, PermissionError, ValueError, IndexError):
        return None


def _get_instruments(meter) -> dict:
    instruments = _INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "usage": meter.create_gauge(
                name="gym.server.memory.usage",
                unit="By",
                description="Current memory used by a Gym server and its descendants.",
            ),
            "sampled_peak": meter.create_gauge(
                name="gym.server.memory.sampled_peak",
                unit="By",
                description="Largest sampled memory usage for a Gym server and its descendants.",
            ),
            "processes": meter.create_gauge(
                name="gym.server.processes",
                unit="{process}",
                description="Processes currently attributed to a Gym server.",
            ),
            "host_usage": meter.create_gauge(
                name="gym.host.memory.usage",
                unit="By",
                description="Current host memory in use.",
            ),
            "host_available": meter.create_gauge(
                name="gym.host.memory.available",
                unit="By",
                description="Current host memory available without swapping.",
            ),
            "host_total": meter.create_gauge(
                name="gym.host.memory.total",
                unit="By",
                description="Total host memory.",
            ),
            "cpu_time": meter.create_counter(
                name="gym.server.cpu.time_total",
                unit="s",
                description="CPU time used by a Gym server and its descendants, by cpu.mode.",
            ),
            "cpu_logical": meter.create_gauge(
                name="gym.host.cpu.logical",
                unit="{cpu}",
                description="Logical CPUs on the host.",
            ),
            "cpu_physical": meter.create_gauge(
                name="gym.host.cpu.physical",
                unit="{cpu}",
                description="Physical CPU cores on the host.",
            ),
            "cpu_allowed": meter.create_gauge(
                name="gym.host.cpu.allowed",
                unit="{cpu}",
                description="CPUs the Gym supervisor's main thread may run on (its CPU affinity).",
            ),
            "host_sampled_peak": meter.create_gauge(
                name="gym.host.memory.sampled_peak",
                unit="By",
                description="Largest sampled host memory usage.",
            ),
        }
        _INSTRUMENTS[meter] = instruments
    return instruments
