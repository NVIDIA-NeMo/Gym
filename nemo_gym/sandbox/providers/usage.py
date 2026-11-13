# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readers of a sandbox's measured CPU and memory for providers that implement ``SupportsSandboxUsage``.

``cgroup_v2_sample`` reads a cgroup the sandbox owns or shares. ``ProcfsUsageReader`` reads the live processes in a
sandbox's namespaces when the runtime gives it no cgroup of its own (Enroot, Apptainer without cgroups).
``in_sandbox_cgroup_command`` and ``parse_in_sandbox_cgroup`` read the container's cgroup from inside it, for
runtimes whose host is not this one. Every reader returns None rather than a guess when it cannot measure.
"""

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from nemo_gym.sandbox.providers.base import SandboxUsageSample, SandboxUsageScope


CGROUP_FILES = ("cpu.stat", "memory.current", "memory.peak", "memory.stat")
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
    namespace it sees the host hierarchy root, whose totals are not the sandbox's, so it returns None.
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in stdout.splitlines():
        if line.startswith("=="):
            current = sections.setdefault(line[2:], [])
        elif current is not None:
            current.append(line)
    if "0::/" not in sections.get("cgroup", []):
        return None
    texts = {name: "\n".join(sections[name]) if sections.get(name) else None for name in CGROUP_FILES}
    return parse_cgroup_v2(texts, scope="container", sampled_at=sampled_at)


@dataclass
class ProcessTreeState:
    """Per-sandbox state of process-tree sampling: the most CPU each process was seen to have used.

    A process is keyed by PID and start time, so a reused PID is a new process. A process that exited keeps the
    CPU it had at its last sample; the CPU it used after that sample, and processes that started and exited
    between two samples, are missing, so the total is a lower bound.
    """

    cpu_ticks: dict[tuple[int, int], int] = field(default_factory=dict)


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
        self.page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

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

    def members(self, init_pid: int) -> tuple[list[int], tuple[str, ...]] | None:
        """PIDs in the sandbox's namespaces and the namespace kinds that matched them (``pid``, ``mnt``), or None
        when no namespace separates the sandbox from this process. Enroot nested in another container keeps this
        process's PID namespace, so only its mount namespace identifies it."""
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
        return pids, tuple(kind for kind, _ in keys)

    def _stat(self, pid: int) -> tuple[int, int] | None:
        """(utime + stime ticks, start time) from /proc/<pid>/stat; the command name may hold spaces."""
        try:
            fields = (self.proc_root / str(pid) / "stat").read_text().rpartition(")")[2].split()
            return int(fields[11]) + int(fields[12]), int(fields[19])
        except (OSError, IndexError, ValueError):
            return None

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

        The CPU is a lower bound of the container's. The memory is the sum of per-process RSS: it leaves out page
        cache and kernel memory but counts a page shared by several processes once per process, so it can be above
        or below the container's charged memory; its sampled peak bounds only this sum's own peak.
        """
        matched = self.members(init_pid)
        if matched is None:
            return None
        members, kinds = matched
        sampled_at = time.time()
        rss = anon = file = 0
        seen = 0
        for pid in members:
            stat = self._stat(pid)
            if stat is None:
                continue
            seen += 1
            ticks, started = stat
            key = (pid, started)
            state.cpu_ticks[key] = max(ticks, state.cpu_ticks.get(key, 0))
            memory = self._memory(pid)
            rss += memory.get("VmRSS", 0)
            anon += memory.get("RssAnon", 0)
            file += memory.get("RssFile", 0)
        if not seen:
            return None
        return SandboxUsageSample(
            # procfs_pidns, procfs_mntns or procfs_pidns_mntns: the namespaces that selected the processes.
            source="procfs_" + "_".join(kind + "ns" for kind in kinds),
            scope="process_tree",
            sampled_at=sampled_at,
            cpu_usage_usec_total=sum(state.cpu_ticks.values()) * 1_000_000 // self.clock_ticks,
            cpu_kind="sampled_lower_bound",
            memory_current_bytes=rss,
            memory_peak_bytes=rss,
            peak_kind="sampled_lower_bound",
            memory_anon_bytes=anon,
            memory_file_bytes=file,
            includes_page_cache=False,
            process_count=seen,
        )

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
