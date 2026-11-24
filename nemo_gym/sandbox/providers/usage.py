# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readers of a sandbox's measured CPU and memory for providers that implement ``SupportsSandboxUsage``.

``cgroup_v2_sample`` reads a cgroup the sandbox owns or shares. ``ProcfsUsageReader`` reads the live processes in a
sandbox's namespaces when the runtime gives it no cgroup of its own (Enroot, Apptainer without cgroups).
``IN_SANDBOX_CGROUP_COMMAND`` and ``parse_in_sandbox_cgroup`` read the container's cgroup from inside it, for
runtimes whose host is not this one. A reader that cannot measure returns None, or raises
``SandboxUsageUnavailable`` naming why, rather than a guess.
"""

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from nemo_gym.sandbox.providers.base import SandboxUsageSample, SandboxUsageScope, SandboxUsageUnavailable


def sampling_enabled() -> bool:
    """Whether sandbox usage is measured at all; ``NEMO_GYM_SANDBOX_USAGE=0`` turns it off."""
    return os.getenv("NEMO_GYM_SANDBOX_USAGE", "1").strip().lower() not in ("0", "false", "off", "no")


CGROUP_FILES = ("cpu.stat", "cpu.pressure", "memory.current", "memory.peak", "memory.stat")
# How long one scan of /proc serves every sandbox in this process; bounds scans to one per second however many
# sandboxes sample.
NAMESPACE_SCAN_TTL_S = 1.0


def _int_or_none(text: str | None) -> int | None:
    try:
        return int(text.strip()) if text is not None else None
    except ValueError:
        return None


def _keyed(text: str | None) -> dict[str, int]:
    """Parse ``key value`` lines (cpu.stat, memory.stat)."""
    values = {}
    for line in (text or "").splitlines():
        key, _, value = line.partition(" ")
        parsed = _int_or_none(value)
        if parsed is not None:
            values[key] = parsed
    return values


def _pressure_totals(text: str | None) -> dict[str, int]:
    """The cumulative stall time in microseconds of each PSI line (``some``, ``full``), from its ``total=``."""
    totals = {}
    for line in (text or "").splitlines():
        kind, _, fields = line.partition(" ")
        for field_text in fields.split():
            name, _, value = field_text.partition("=")
            parsed = _int_or_none(value) if name == "total" else None
            if parsed is not None:
                totals[kind] = parsed
    return totals


def parse_cgroup_v2(
    texts: Mapping[str, str | None], *, scope: SandboxUsageScope, sampled_at: float
) -> SandboxUsageSample | None:
    """A sample from the contents of a cgroup v2 directory's files, or None when it has no CPU or memory counter."""
    cpu = _keyed(texts.get("cpu.stat")).get("usage_usec")
    current = _int_or_none(texts.get("memory.current"))
    peak = _int_or_none(texts.get("memory.peak"))
    if cpu is None and current is None and peak is None:
        return None
    stat = _keyed(texts.get("memory.stat"))
    cpu_stat = _keyed(texts.get("cpu.stat"))
    pressure = _pressure_totals(texts.get("cpu.pressure"))
    return SandboxUsageSample(
        source="cgroup_v2",
        scope=scope,
        sampled_at=sampled_at,
        cpu_usage_usec_total=cpu,
        cpu_kind="counter" if cpu is not None else None,
        memory_current_bytes=current,
        memory_peak_bytes=peak,
        peak_kind="counter" if peak is not None else None,
        memory_anon_bytes=stat.get("anon"),
        memory_file_bytes=stat.get("file"),
        includes_page_cache=True if current is not None or peak is not None else None,
        cpu_throttled_usec_total=cpu_stat.get("throttled_usec"),
        cpu_throttled_periods_total=cpu_stat.get("nr_throttled"),
        cpu_pressure_some_usec_total=pressure.get("some"),
        cpu_pressure_full_usec_total=pressure.get("full"),
    )


def cgroup_v2_sample(directory: Path, *, scope: SandboxUsageScope) -> SandboxUsageSample | None:
    """Read a cgroup v2 directory on this host."""
    texts = {}
    for name in CGROUP_FILES:
        try:
            texts[name] = (directory / name).read_text()
        except OSError:
            texts[name] = None
    return parse_cgroup_v2(texts, scope=scope, sampled_at=time.time())


def cgroup_path(pid: int | str, *, proc_root: Path = Path("/proc")) -> str | None:
    """The unified (v2) cgroup path of a process as this process's cgroup namespace sees it."""
    try:
        lines = (proc_root / str(pid) / "cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("0::"):
            return line[3:]
    return None


def process_start_time(pid: int, *, proc_root: Path = Path("/proc")) -> int | None:
    """A process's start time in clock ticks since boot (``/proc/<pid>/stat`` field 22), or None if it is gone.

    With the PID it identifies one process: a PID reused by another process has a different start time.
    """
    try:
        return int((proc_root / str(pid) / "stat").read_text().rpartition(")")[2].split()[19])
    except (OSError, IndexError, ValueError):
        return None


# A command printing the container's cgroup files from inside it, each after a ``==<name>`` line. Run where the
# sandbox has its own cgroup namespace (Docker and Kubernetes on cgroup v2), whose root is the container's cgroup.
IN_SANDBOX_CGROUP_COMMAND = (
    "for f in " + " ".join(CGROUP_FILES) + "; do printf '==%s\\n' \"$f\"; cat /sys/fs/cgroup/$f 2>/dev/null; done; "
    "printf '==cgroup\\n'; cat /proc/self/cgroup"
)


def parse_in_sandbox_cgroup(stdout: str, *, sampled_at: float) -> SandboxUsageSample | None:
    """A container-scope sample from ``IN_SANDBOX_CGROUP_COMMAND`` output.

    Only a reader at its cgroup namespace's root (``0::/``) sees the container's own cgroup. Without that
    namespace it sees the host hierarchy root, whose totals are not the sandbox's, so it raises
    ``SandboxUsageUnavailable("not_cgroup_root")``. Output without the cgroup section is no sample (None).
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in stdout.splitlines():
        if line.startswith("=="):
            current = sections.setdefault(line[2:], [])
        elif current is not None:
            current.append(line)
    if not sections.get("cgroup"):
        return None
    if "0::/" not in sections["cgroup"]:
        raise SandboxUsageUnavailable("not_cgroup_root")
    texts = {name: "\n".join(sections[name]) if sections.get(name) else None for name in CGROUP_FILES}
    return parse_cgroup_v2(texts, scope="container", sampled_at=sampled_at)


ProcessKey = tuple[int, int]


class _ProcStat(NamedTuple):
    # utime + stime + cutime + cstime: the process's own CPU and that of the children it waited for.
    ticks: int
    start: int
    parent_pid: int
    # Exited with no thread still running: once reaped, its CPU is in its parent's children's CPU.
    ended: bool


@dataclass
class ProcessTreeState:
    """Per-sandbox state of process-tree sampling.

    A process is keyed by PID and start time, so a reused PID is a new process. ``cpu_ticks`` holds the most CPU each
    live process was seen to have used, its own and that of the children it waited for (``cutime``/``cstime``), so a
    child that starts and exits between two samples is counted once its parent in the sandbox waits for it.
    ``parents`` holds each process's parent at its last read.

    When a process exits, what it was counted moves onto its nearest ancestor that still runs: that ancestor's
    children's CPU takes in the whole of it once it is reaped, so it is not counted twice. With no such ancestor
    it moves to ``exited_ticks``. The CPU a process used after its last sample is missing unless a process in the
    sandbox waited for it, so the total is a lower bound.
    """

    cpu_ticks: dict[ProcessKey, int] = field(default_factory=dict)
    parents: dict[ProcessKey, ProcessKey | None] = field(default_factory=dict)
    exited_ticks: int = 0


def _parents_first(stats: Mapping[int, _ProcStat]) -> list[int]:
    """The PIDs of ``stats``, each after its parent when that parent is among them."""
    depth: dict[int, int] = {}
    for pid in stats:
        chain: list[int] = []
        while pid in stats and pid not in depth and pid not in chain:
            chain.append(pid)
            pid = stats[pid].parent_pid
        level = depth.get(pid, -1)
        for member in reversed(chain):
            level += 1
            depth[member] = level
    return sorted(stats, key=lambda pid: (depth[pid], pid))


@dataclass
class _NamespaceScan:
    taken_at: float
    members: dict[tuple[str, str], list[int]]


class ProcfsUsageReader:
    """Read a sandbox's processes: those in its init's PID namespace or its mount namespace.

    A namespace this process shares is never matched: it would take in Gym and every other process on the host.
    Scans are shared by every sandbox in this process for ``NAMESPACE_SCAN_TTL_S``.
    """

    _scans: dict[Path, _NamespaceScan] = {}
    _lock = threading.Lock()

    def __init__(self, proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")) -> None:
        self.proc_root = proc_root
        self.cgroup_root = cgroup_root
        self.clock_ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

    def _namespace(self, pid: int | str, kind: str) -> str | None:
        try:
            return os.readlink(self.proc_root / str(pid) / "ns" / kind)
        except OSError:
            return None

    def _scan(self, now: float) -> dict[tuple[str, str], list[int]]:
        with self._lock:
            scan = self._scans.get(self.proc_root)
            if scan is not None and now - scan.taken_at < NAMESPACE_SCAN_TTL_S:
                return scan.members
            members: dict[tuple[str, str], list[int]] = {}
            for entry in os.scandir(self.proc_root):
                if not entry.name.isdigit():
                    continue
                for kind in ("pid", "mnt"):
                    namespace = self._namespace(entry.name, kind)
                    if namespace is not None:
                        members.setdefault((kind, namespace), []).append(int(entry.name))
            self._scans[self.proc_root] = _NamespaceScan(now, members)
            return members

    def members(self, init_pid: int) -> tuple[list[int], tuple[tuple[str, str], ...]] | None:
        """PIDs in the sandbox's namespaces and the ``(kind, namespace)`` pairs that selected them (kinds ``pid``
        and ``mnt``), or None when no namespace separates the sandbox from this process. Enroot nested in another
        container keeps this process's PID namespace, so only its mount namespace identifies it."""
        keys = []
        for kind in ("pid", "mnt"):
            namespace = self._namespace(init_pid, kind)
            if namespace is not None and namespace != self._namespace("self", kind):
                keys.append((kind, namespace))
        if not keys:
            return None
        scan = self._scan(time.monotonic())
        # The scan may be a second old: a PID in it counts only while it is still in the namespace, so a host
        # process that reused the PID of an exited sandbox process is never charged to the sandbox.
        pids = sorted(
            {
                pid
                for kind, namespace in keys
                for pid in scan.get((kind, namespace), [])
                if self._namespace(pid, kind) == namespace
            }
            | {init_pid}
        )
        return pids, tuple(keys)

    def _read_stat(self, pid: int) -> _ProcStat:
        """/proc/<pid>/stat, whose command name may hold spaces or parentheses; raises ``OSError`` once it is gone."""
        fields = (self.proc_root / str(pid) / "stat").read_text().rpartition(")")[2].split()
        try:
            return _ProcStat(
                ticks=int(fields[11]) + int(fields[12]) + int(fields[13]) + int(fields[14]),
                start=int(fields[19]),
                parent_pid=int(fields[1]),
                ended=fields[0] in ("Z", "X") and int(fields[17]) <= 1,
            )
        except (IndexError, ValueError) as error:
            raise OSError(f"unparsable /proc/{pid}/stat") from error

    def _stat(self, pid: int) -> _ProcStat | None:
        try:
            return self._read_stat(pid)
        except OSError:
            return None

    def _has_exited(self, key: ProcessKey) -> bool:
        """Whether the process ``key`` names is gone: its PID is free, belongs to a later process, or has ended.

        An entry that cannot be read for another reason (permissions, I/O) is no proof of an exit.
        """
        try:
            stat = self._read_stat(key[0])
        except (FileNotFoundError, ProcessLookupError):
            return True
        except OSError:
            return False
        return stat.start != key[1] or stat.ended

    def _memory(self, pid: int) -> dict[str, int]:
        """Resident kB from /proc/<pid>/status: VmRSS, RssAnon and RssFile."""
        values = {}
        try:
            lines = (self.proc_root / str(pid) / "status").read_text().splitlines()
        except OSError:
            return values
        for line in lines:
            key, _, value = line.partition(":")
            if key in ("VmRSS", "RssAnon", "RssFile"):
                parsed = _int_or_none(value.strip().removesuffix("kB"))
                if parsed is not None:
                    values[key] = parsed * 1024
        return values

    def process_tree_sample(self, init_pid: int, state: ProcessTreeState) -> SandboxUsageSample | None:
        """CPU and summed resident memory of the sandbox's live processes.

        The CPU, including that of the children the sandbox's processes waited for, is a lower bound of the
        container's. The memory is the sum of per-process RSS: it leaves out page
        cache and kernel memory but counts a page shared by several processes once per process, so it can be above
        or below the container's charged memory; its sampled peak bounds only this sum's own peak.
        """
        matched = self.members(init_pid)
        if matched is None:
            return None
        members, keys = matched
        sampled_at = time.time()
        first = {pid: stat for pid in members if (stat := self._stat(pid)) is not None}
        observed: dict[ProcessKey, _ProcStat] = {}
        rss = anon = file = 0
        # Parents are read before their children: a child reaped between its own read and its parent's would be
        # counted in both. The CPU is read after the namespace check and must be the same process (start time) as
        # the first read: a PID that exited and was reused by a host process after ``members()`` is never charged.
        for pid in _parents_first(first):
            memory = self._memory(pid)
            if not any(self._namespace(pid, kind) == namespace for kind, namespace in keys):
                continue
            stat = self._stat(pid)
            if stat is None or stat.start != first[pid].start or stat.ended:
                continue
            observed[(pid, stat.start)] = stat
            rss += memory.get("VmRSS", 0)
            anon += memory.get("RssAnon", 0)
            file += memory.get("RssFile", 0)
        if not observed:
            return None
        self._advance(state, observed)
        return SandboxUsageSample(
            # procfs_pidns, procfs_mntns or procfs_pidns_mntns: the namespaces that selected the processes.
            source="procfs_" + "_".join(kind + "ns" for kind, _ in keys),
            scope="process_tree",
            sampled_at=sampled_at,
            cpu_usage_usec_total=(state.exited_ticks + sum(state.cpu_ticks.values())) * 1_000_000 // self.clock_ticks,
            cpu_kind="sampled_lower_bound",
            memory_current_bytes=rss,
            memory_peak_bytes=rss,
            peak_kind="sampled_lower_bound",
            memory_anon_bytes=anon,
            memory_file_bytes=file,
            includes_page_cache=False,
            process_count=len(observed),
        )

    def _advance(self, state: ProcessTreeState, observed: Mapping[ProcessKey, _ProcStat]) -> None:
        """Move each exited process's counted CPU onto its nearest running ancestor, then take in this sample.

        Each process's count only grows, so CPU moved onto a parent that has not waited for the child yet is held
        until the parent's own reading shows it, and the total never goes down.
        """
        exited = {key for key in state.cpu_ticks if key not in observed and self._has_exited(key)}
        for key in exited:
            ancestor, visited = state.parents.get(key), {key}
            while ancestor in exited and ancestor not in visited:
                visited.add(ancestor)
                ancestor = state.parents.get(ancestor)
            if ancestor in state.cpu_ticks and ancestor not in exited:
                state.cpu_ticks[ancestor] += state.cpu_ticks[key]
            else:
                state.exited_ticks += state.cpu_ticks[key]
        for key in exited:
            del state.cpu_ticks[key]
            state.parents.pop(key, None)
        running = {key[0]: key for key in [*state.cpu_ticks, *observed]}
        for key, stat in observed.items():
            state.cpu_ticks[key] = max(stat.ticks, state.cpu_ticks.get(key, 0))
            state.parents[key] = running.get(stat.parent_pid)

    def enclosing_cgroup_sample(self, pid: int, *, own_cgroup_scope: SandboxUsageScope) -> SandboxUsageSample | None:
        """Read the cgroup ``pid`` runs in.

        A cgroup this process also runs in is ``shared_step``. A different one is labelled ``own_cgroup_scope``:
        a runtime that creates a cgroup per sandbox passes ``container``, one that never does ``shared_step``.
        """
        path = cgroup_path(pid, proc_root=self.proc_root)
        if path is None or path.endswith(" (deleted)"):
            return None
        scope = "shared_step" if path == cgroup_path("self", proc_root=self.proc_root) else own_cgroup_scope
        return cgroup_v2_sample(self.cgroup_root / path.lstrip("/"), scope=scope)

    def host_process_usage(
        self, init_pid: int, state: ProcessTreeState, *, own_cgroup_scope: SandboxUsageScope
    ) -> tuple[SandboxUsageSample, ...]:
        """Every measurement of a sandbox whose init runs on this host: its own cgroup if it has one, otherwise
        its process tree; plus the enclosing cgroup when that is shared."""
        cgroup = self.enclosing_cgroup_sample(init_pid, own_cgroup_scope=own_cgroup_scope)
        if cgroup is not None and cgroup.scope == "container":
            return (cgroup,)
        tree = self.process_tree_sample(init_pid, state)
        return tuple(sample for sample in (tree, cgroup) if sample is not None)
