# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured sandbox CPU and memory: the readers, their labels, and owner-side sampling."""

import asyncio
import json
from pathlib import Path

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import usage as usage_readers
from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec, SandboxUsageSample, SandboxUsageUnavailable
from nemo_gym.sandbox.providers.usage import (
    IN_SANDBOX_CGROUP_COMMAND,
    ProcessTreeState,
    ProcfsUsageReader,
    cgroup_v2_sample,
    parse_cgroup_v2,
    parse_in_sandbox_cgroup,
)
from nemo_gym.sandbox.usage import SandboxUsageTracker, summarize, usage_sample_interval


CPU_STAT = "usage_usec 2500000\nuser_usec 2000000\nsystem_usec 500000\n"
MEMORY_STAT = "anon 104857600\nfile 52428800\nkernel 1024\n"


CONTENDED_CPU_STAT = CPU_STAT + "nr_periods 400\nnr_throttled 37\nthrottled_usec 912000\n"
CPU_PRESSURE = (
    "some avg10=1.50 avg60=0.80 avg300=0.20 total=3400000\nfull avg10=0.40 avg60=0.10 avg300=0.00 total=700000\n"
)


def _write_cgroup(directory: Path, *, peak: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cpu.stat").write_text(CPU_STAT)
    (directory / "memory.current").write_text("157286400\n")
    if peak:
        (directory / "memory.peak").write_text("314572800\n")
    (directory / "memory.stat").write_text(MEMORY_STAT)
    return directory


def test_cgroup_v2_counters_are_container_totals_with_page_cache(tmp_path: Path) -> None:
    sample = cgroup_v2_sample(_write_cgroup(tmp_path / "cg"), scope="container")

    assert sample is not None
    assert (sample.source, sample.scope, sample.clock) == ("cgroup_v2", "container", "gym_host")
    assert sample.cpu_usage_usec_total == 2_500_000 and sample.cpu_kind == "counter"
    assert sample.memory_peak_bytes == 314_572_800 and sample.peak_kind == "counter"
    assert (sample.memory_current_bytes, sample.memory_anon_bytes, sample.memory_file_bytes) == (
        157_286_400,
        104_857_600,
        52_428_800,
    )
    assert sample.includes_page_cache is True


def test_a_kernel_without_memory_peak_reports_no_peak_rather_than_the_current_value(tmp_path: Path) -> None:
    sample = cgroup_v2_sample(_write_cgroup(tmp_path / "cg", peak=False), scope="container")

    assert sample is not None and sample.memory_peak_bytes is None and sample.peak_kind is None
    assert sample.memory_current_bytes == 157_286_400


def test_a_directory_without_counters_is_no_sample(tmp_path: Path) -> None:
    assert cgroup_v2_sample(tmp_path, scope="container") is None
    assert (
        parse_cgroup_v2({"cpu.stat": "nr_periods 0\n", "memory.current": "max"}, scope="container", sampled_at=1)
        is None
    )


def test_a_cgroup_read_reports_throttling_and_cpu_pressure_as_counters(tmp_path: Path) -> None:
    directory = _write_cgroup(tmp_path / "cg")
    (directory / "cpu.stat").write_text(CONTENDED_CPU_STAT)
    (directory / "cpu.pressure").write_text(CPU_PRESSURE)

    sample = cgroup_v2_sample(directory, scope="container")

    assert sample is not None
    assert (sample.cpu_throttled_usec_total, sample.cpu_throttled_periods_total) == (912_000, 37)
    assert (sample.cpu_pressure_some_usec_total, sample.cpu_pressure_full_usec_total) == (3_400_000, 700_000)
    # Without the quota counters or pressure accounting, those values are unknown, not zero.
    plain = cgroup_v2_sample(_write_cgroup(tmp_path / "plain"), scope="container")
    assert plain is not None and plain.cpu_throttled_usec_total is None and plain.cpu_pressure_some_usec_total is None


def _in_sandbox_output(cgroup_line: str) -> str:
    return (
        f"==cpu.stat\n{CPU_STAT}==memory.current\n157286400\n==memory.peak\n314572800\n"
        f"==memory.stat\n{MEMORY_STAT}==cgroup\n{cgroup_line}\n"
    )


def test_an_in_sandbox_read_counts_only_at_the_cgroup_namespace_root() -> None:
    sample = parse_in_sandbox_cgroup(_in_sandbox_output("0::/"), sampled_at=5.0)
    assert sample is not None and sample.scope == "container" and sample.cpu_usage_usec_total == 2_500_000
    assert sample.sampled_at == 5.0

    # Without a private cgroup namespace the files are the host hierarchy's root, not the sandbox's.
    with pytest.raises(SandboxUsageUnavailable, match="not_cgroup_root"):
        parse_in_sandbox_cgroup(_in_sandbox_output("0::/system.slice/job.scope"), sampled_at=5.0)
    assert parse_in_sandbox_cgroup("", sampled_at=5.0) is None
    assert "/sys/fs/cgroup/$f" in IN_SANDBOX_CGROUP_COMMAND and "/proc/self/cgroup" in IN_SANDBOX_CGROUP_COMMAND


class FakeProc:
    """A /proc tree: each process has namespace links, stat, status and cgroup."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.add("self", pidns="A", mntns="M0", cgroup="/job/step")

    def add(
        self,
        pid: int | str,
        *,
        pidns: str,
        mntns: str,
        ticks: tuple[int, int] = (0, 0),
        children_ticks: tuple[int, int] = (0, 0),
        start: int = 1,
        ppid: int = 0,
        state: str = "S",
        threads: int = 1,
        rss_kb: tuple[int, int] = (0, 0),
        cgroup: str = "/job/step",
        comm: str = "sh",
    ) -> None:
        directory = self.root / str(pid)
        (directory / "ns").mkdir(parents=True, exist_ok=True)
        for kind, name in (("pid", f"pid:[{pidns}]"), ("mnt", f"mnt:[{mntns}]")):
            link = directory / "ns" / kind
            link.unlink(missing_ok=True)
            link.symlink_to(name)
        # Fields 3..27 of /proc/<pid>/stat: state is field 3, ppid 4, utime 14, stime 15, cutime 16, cstime 17,
        # num_threads 20 and starttime 22.
        fields = (
            [state, str(ppid)]
            + ["0"] * 9
            + [str(ticks[0]), str(ticks[1]), str(children_ticks[0]), str(children_ticks[1])]
            + ["0"] * 2
            + [str(threads), "0", str(start)]
            + ["0"] * 5
        )
        (directory / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")
        anon, file = rss_kb
        (directory / "status").write_text(
            f"Name:\t{comm}\nVmRSS:\t{anon + file} kB\nRssAnon:\t{anon} kB\nRssFile:\t{file} kB\n"
        )
        (directory / "cgroup").write_text(f"0::{cgroup}\n")

    def remove(self, pid: int) -> None:
        for path in sorted((self.root / str(pid)).rglob("*"), reverse=True):
            path.unlink() if path.is_file() or path.is_symlink() else path.rmdir()
        (self.root / str(pid)).rmdir()


@pytest.fixture
def proc(tmp_path: Path) -> FakeProc:
    ProcfsUsageReader._scans.clear()
    yield FakeProc(tmp_path / "proc")
    ProcfsUsageReader._scans.clear()


def _reader(proc: FakeProc, tmp_path: Path) -> ProcfsUsageReader:
    reader = ProcfsUsageReader(proc_root=proc.root, cgroup_root=tmp_path / "cgroup")
    reader.clock_ticks = 100
    return reader


def test_the_process_tree_is_the_sandbox_namespaces_and_a_labelled_lower_bound(proc: FakeProc, tmp_path: Path) -> None:
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 5), rss_kb=(1000, 500))  # container init
    proc.add(101, pidns="B", mntns="M1", ticks=(200, 100), start=7, rss_kb=(30000, 2000), comm="python app.py")
    proc.add(150, pidns="A", mntns="M1", ticks=(1, 1), rss_kb=(100, 0))  # runtime helper in the mount namespace
    proc.add(300, pidns="A", mntns="M0", ticks=(9999, 9999), rss_kb=(900000, 0))  # Gym and the rest of the host
    reader = _reader(proc, tmp_path)
    state = ProcessTreeState()

    sample = reader.process_tree_sample(100, state)

    assert sample is not None
    assert (sample.source, sample.scope) == ("procfs_pidns_mntns", "process_tree")
    assert sample.cpu_kind == "sampled_lower_bound" and sample.peak_kind == "sampled_lower_bound"
    assert sample.includes_page_cache is False
    assert sample.process_count == 3
    assert sample.cpu_usage_usec_total == (15 + 300 + 2) * 10_000
    assert sample.memory_current_bytes == sample.memory_peak_bytes == (1500 + 32000 + 100) * 1024
    assert sample.memory_anon_bytes == 31100 * 1024 and sample.memory_file_bytes == 2500 * 1024

    # An exited process keeps the CPU last seen; a reused PID with a new start time is a new process.
    proc.remove(101)
    proc.add(101, pidns="B", mntns="M1", ticks=(3, 0), start=99, rss_kb=(10, 0))
    ProcfsUsageReader._scans.clear()
    later = reader.process_tree_sample(100, state)
    assert later is not None and later.cpu_usage_usec_total == (15 + 300 + 2 + 3) * 10_000
    assert later.memory_current_bytes == (1500 + 10 + 100) * 1024


def test_a_host_process_reusing_a_cached_sandbox_pid_is_not_charged(proc: FakeProc, tmp_path: Path) -> None:
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 5), rss_kb=(1000, 0))
    proc.add(101, pidns="B", mntns="M1", ticks=(200, 100), start=7, rss_kb=(30000, 0))
    reader = _reader(proc, tmp_path)
    state = ProcessTreeState()
    assert reader.process_tree_sample(100, state).process_count == 2

    # Within the scan's TTL, PID 101 exits and the host reuses it in Gym's own namespaces.
    proc.remove(101)
    proc.add(101, pidns="A", mntns="M0", ticks=(99999, 0), start=50, rss_kb=(900000, 0))
    later = reader.process_tree_sample(100, state)

    assert later.process_count == 1 and later.memory_current_bytes == 1000 * 1024
    assert later.cpu_usage_usec_total == (15 + 300) * 10_000  # the exited process keeps what it was seen to use


def test_a_pid_replaced_after_selection_or_between_reads_is_not_charged(proc: FakeProc, tmp_path: Path) -> None:
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 5), rss_kb=(1000, 0))
    proc.add(101, pidns="B", mntns="M1", ticks=(200, 100), start=7, rss_kb=(30000, 0))
    reader = _reader(proc, tmp_path)
    state = ProcessTreeState()
    assert reader.process_tree_sample(100, state).process_count == 2
    selected = ProcfsUsageReader.members

    def select_then_replace(self: ProcfsUsageReader, init_pid: int) -> tuple:
        # 101 exits after selection and a host process takes its PID before it is read.
        matched = selected(self, init_pid)
        proc.remove(101)
        proc.add(101, pidns="A", mntns="M0", ticks=(99999, 0), start=50, rss_kb=(900000, 0))
        return matched

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ProcfsUsageReader, "members", select_then_replace)
        later = reader.process_tree_sample(100, state)
    assert later.process_count == 1 and later.memory_current_bytes == 1000 * 1024
    assert later.cpu_usage_usec_total == (15 + 300) * 10_000

    # A PID replaced between its stat and memory reads, even by a process in the sandbox, is a mixed read: skipped.
    proc.remove(101)
    proc.add(101, pidns="B", mntns="M1", ticks=(1, 0), start=60, rss_kb=(10, 0))
    read_memory = ProcfsUsageReader._memory

    def replace_then_read(self: ProcfsUsageReader, pid: int) -> dict[str, int]:
        if pid == 101:
            proc.remove(101)
            proc.add(101, pidns="B", mntns="M1", ticks=(7, 0), start=61, rss_kb=(800000, 0))
        return read_memory(self, pid)

    ProcfsUsageReader._scans.clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ProcfsUsageReader, "_memory", replace_then_read)
        mixed = reader.process_tree_sample(100, state)
    assert mixed.process_count == 1 and mixed.memory_current_bytes == 1000 * 1024
    assert mixed.cpu_usage_usec_total == (15 + 300) * 10_000


def test_a_container_in_gyms_pid_namespace_is_found_by_its_mount_namespace(proc: FakeProc, tmp_path: Path) -> None:
    # Enroot nested in another container cannot unshare the PID namespace: only the mount namespace is its own.
    proc.add(100, pidns="A", mntns="M1", ticks=(10, 5), rss_kb=(1000, 0))
    proc.add(101, pidns="A", mntns="M1", ticks=(20, 0), rss_kb=(500, 0))
    proc.add(300, pidns="A", mntns="M0", ticks=(9999, 0), rss_kb=(900000, 0))  # Gym and the host

    sample = _reader(proc, tmp_path).process_tree_sample(100, ProcessTreeState())

    assert sample is not None and sample.source == "procfs_mntns" and sample.process_count == 2
    assert sample.cpu_usage_usec_total == 35 * 10_000


def _sample_cpu_ticks(reader: ProcfsUsageReader, state: ProcessTreeState) -> int:
    ProcfsUsageReader._scans.clear()
    sample = reader.process_tree_sample(100, state)
    assert sample is not None
    return sample.cpu_usage_usec_total // 10_000


def test_children_reaped_between_samples_are_counted_once(proc: FakeProc, tmp_path: Path) -> None:
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 5))  # container init
    proc.add(101, pidns="B", mntns="M1", ticks=(20, 0), start=5, ppid=100)  # a shell
    reader = _reader(proc, tmp_path)
    state = ProcessTreeState()
    assert _sample_cpu_ticks(reader, state) == 35

    # Short commands start and exit between two samples; the shell waited for them.
    proc.add(101, pidns="B", mntns="M1", ticks=(20, 0), children_ticks=(30, 10), start=5, ppid=100)
    assert _sample_cpu_ticks(reader, state) == 35 + 40

    # A long command is seen running, then the shell reaps it: its CPU is counted once, not again in the shell's.
    proc.add(102, pidns="B", mntns="M1", ticks=(50, 10), start=9, ppid=101)
    assert _sample_cpu_ticks(reader, state) == 35 + 40 + 60
    proc.remove(102)
    proc.add(101, pidns="B", mntns="M1", ticks=(20, 0), children_ticks=(110, 30), start=5, ppid=100)
    assert _sample_cpu_ticks(reader, state) == 35 + 40 + 100  # 102 used 40 more after its last sample

    # Seen exited before the shell has waited for it: what it was counted is kept until the shell's reading shows it.
    proc.add(103, pidns="B", mntns="M1", ticks=(25, 0), start=12, ppid=101)
    assert _sample_cpu_ticks(reader, state) == 175 + 25
    proc.add(103, pidns="B", mntns="M1", ticks=(30, 0), start=12, ppid=101, state="Z")
    assert _sample_cpu_ticks(reader, state) == 200
    proc.remove(103)
    proc.add(101, pidns="B", mntns="M1", ticks=(20, 0), children_ticks=(140, 30), start=5, ppid=100)
    assert _sample_cpu_ticks(reader, state) == 205


def test_an_exited_chain_moves_to_the_nearest_running_ancestor(proc: FakeProc, tmp_path: Path) -> None:
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 0))
    proc.add(101, pidns="B", mntns="M1", ticks=(5, 0), start=5, ppid=100)
    proc.add(102, pidns="B", mntns="M1", ticks=(40, 0), start=6, ppid=101)
    reader = _reader(proc, tmp_path)
    state = ProcessTreeState()
    assert _sample_cpu_ticks(reader, state) == 55

    # 101 reaps 102, then exits and init reaps it, all between two samples.
    proc.remove(101)
    proc.remove(102)
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 0), children_ticks=(50, 0))
    assert _sample_cpu_ticks(reader, state) == 60

    # With no running ancestor in the sandbox, an exited process keeps what it was counted.
    proc.add(104, pidns="B", mntns="M1", ticks=(7, 0), start=20, ppid=4242)
    assert _sample_cpu_ticks(reader, state) == 67
    proc.remove(104)
    assert _sample_cpu_ticks(reader, state) == 67
    assert state.exited_ticks == 7 and set(state.cpu_ticks) == {(100, 1)}


def test_a_zombie_is_not_counted_until_reaped_but_a_zombie_leader_with_threads_runs(
    proc: FakeProc, tmp_path: Path
) -> None:
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 0))
    proc.add(101, pidns="B", mntns="M1", ticks=(30, 0), start=5, ppid=100, state="Z")
    proc.add(102, pidns="B", mntns="M1", ticks=(20, 0), start=6, ppid=100, state="Z", threads=3)
    reader = _reader(proc, tmp_path)

    sample = reader.process_tree_sample(100, ProcessTreeState())

    assert sample is not None and sample.process_count == 2
    assert sample.cpu_usage_usec_total == 30 * 10_000


def test_a_parent_is_read_before_its_child(proc: FakeProc, tmp_path: Path) -> None:
    # The child has the lower PID (PIDs wrapped). Reaped between the two reads, it would be in both if read first.
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 0))
    proc.add(300, pidns="B", mntns="M1", ticks=(5, 0), start=5, ppid=100)
    proc.add(120, pidns="B", mntns="M1", ticks=(40, 0), start=6, ppid=300)
    reader = _reader(proc, tmp_path)
    state = ProcessTreeState()
    assert _sample_cpu_ticks(reader, state) == 55

    proc.add(120, pidns="B", mntns="M1", ticks=(60, 0), start=6, ppid=300)
    read_stat = ProcfsUsageReader._read_stat
    reads: dict[int, int] = {}

    def reap_when_the_parent_is_read_again(self: ProcfsUsageReader, pid: int) -> object:
        reads[pid] = reads.get(pid, 0) + 1
        if pid == 300 and reads[pid] == 2 and (self.proc_root / "120").exists():
            proc.remove(120)
            proc.add(300, pidns="B", mntns="M1", ticks=(5, 0), children_ticks=(60, 0), start=5, ppid=100)
        return read_stat(self, pid)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ProcfsUsageReader, "_read_stat", reap_when_the_parent_is_read_again)
        assert _sample_cpu_ticks(reader, state) == 75
    assert _sample_cpu_ticks(reader, state) == 75


def test_a_sandbox_sharing_every_namespace_with_gym_is_not_measured(proc: FakeProc, tmp_path: Path) -> None:
    proc.add(100, pidns="A", mntns="M0", ticks=(10, 5))
    reader = _reader(proc, tmp_path)

    assert reader.members(100) is None
    assert reader.process_tree_sample(100, ProcessTreeState()) is None
    assert reader.process_tree_sample(424242, ProcessTreeState()) is None


def test_the_cgroup_gym_shares_is_labelled_shared_step(proc: FakeProc, tmp_path: Path) -> None:
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 5), rss_kb=(1000, 0))
    _write_cgroup(tmp_path / "cgroup" / "job" / "step")
    reader = _reader(proc, tmp_path)

    samples = reader.host_process_usage(100, ProcessTreeState(), own_cgroup_scope="container")

    assert [(sample.source, sample.scope) for sample in samples] == [
        ("procfs_pidns_mntns", "process_tree"),
        ("cgroup_v2", "shared_step"),
    ]


def test_a_cgroup_of_its_own_replaces_the_process_tree_only_when_the_runtime_creates_one(
    proc: FakeProc, tmp_path: Path
) -> None:
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 5), cgroup="/job/step/instance-1")
    _write_cgroup(tmp_path / "cgroup" / "job" / "step" / "instance-1")
    reader = _reader(proc, tmp_path)

    own = reader.host_process_usage(100, ProcessTreeState(), own_cgroup_scope="container")
    assert [(sample.source, sample.scope) for sample in own] == [("cgroup_v2", "container")]

    # A runtime that never creates a cgroup per sandbox (Enroot) never calls another cgroup the container's.
    shared = reader.host_process_usage(100, ProcessTreeState(), own_cgroup_scope="shared_step")
    assert [sample.scope for sample in shared] == ["process_tree", "shared_step"]


def test_unreadable_processes_and_cgroups_are_skipped_not_guessed(proc: FakeProc, tmp_path: Path) -> None:
    proc.add(100, pidns="B", mntns="M1", ticks=(10, 5), rss_kb=(1000, 0), cgroup="/gone (deleted)")
    proc.add(101, pidns="B", mntns="M1", ticks=(1, 1))
    (proc.root / "101" / "stat").write_text("101 (exiting")  # truncated as the process exited
    (proc.root / "100" / "status").unlink()
    reader = _reader(proc, tmp_path)

    (tree,) = reader.host_process_usage(100, ProcessTreeState(), own_cgroup_scope="container")
    assert tree.scope == "process_tree" and tree.process_count == 1 and tree.memory_current_bytes == 0
    # The scan is reused within its TTL, so a process that appears meanwhile waits for the next scan.
    proc.add(102, pidns="B", mntns="M1", ticks=(50, 0))
    assert 102 not in reader.members(100)[0]

    (proc.root / "100" / "stat").unlink()
    (proc.root / "101" / "stat").unlink()
    (proc.root / "102" / "stat").unlink()
    assert reader.process_tree_sample(100, ProcessTreeState()) is None


def test_a_process_without_a_unified_cgroup_has_no_cgroup_path(proc: FakeProc) -> None:
    proc.add(100, pidns="B", mntns="M1")
    (proc.root / "100" / "cgroup").write_text("12:memory:/job\n")
    assert usage_readers.cgroup_path(100, proc_root=proc.root) is None
    assert usage_readers.cgroup_path(424242, proc_root=proc.root) is None


@pytest.mark.parametrize(("value", "expected"), [("2.5", 2.5), ("0", 0.0), ("inf", 0.0), ("soon", 5.0)])
def test_the_sample_interval_setting(monkeypatch: pytest.MonkeyPatch, value: str, expected: float) -> None:
    monkeypatch.setenv("NEMO_GYM_SANDBOX_USAGE_SAMPLE_INTERVAL_SECONDS", value)
    assert usage_sample_interval() == expected


def _sample(scope: str, at: float, cpu: int, peak: int, kind: str = "counter") -> SandboxUsageSample:
    return SandboxUsageSample(
        source="cgroup_v2" if kind == "counter" else "procfs_pidns",
        scope=scope,
        sampled_at=at,
        cpu_usage_usec_total=cpu,
        cpu_kind=kind,
        memory_current_bytes=peak,
        memory_peak_bytes=peak,
        peak_kind=kind,
    )


def test_a_summary_keeps_the_last_counter_the_largest_peak_and_the_window_delta() -> None:
    summary = summarize(
        [
            _sample("shared_step", 1.0, 1000, 50),
            _sample("shared_step", 2.0, 4000, 80),
            _sample("shared_step", 3.0, 9000, 60),
        ]
    )

    assert (summary.first_sampled_at, summary.last_sampled_at, summary.sample_count) == (1.0, 3.0, 3)
    assert summary.cpu_usage_usec_total == 9000 and summary.cpu_usage_usec_delta == 8000
    assert summary.memory_peak_bytes == 80 and summary.memory_current_bytes == 60
    assert summarize([_sample("container", 1.0, 1000, 50)]).cpu_usage_usec_delta is None


def test_a_summary_reports_contention_counters_as_lifetime_totals_and_window_deltas() -> None:
    def contended(at: float, throttled: int, some: int) -> SandboxUsageSample:
        return SandboxUsageSample(
            source="cgroup_v2",
            scope="container",
            sampled_at=at,
            cpu_usage_usec_total=1000,
            cpu_kind="counter",
            cpu_throttled_usec_total=throttled,
            cpu_throttled_periods_total=throttled // 100,
            cpu_pressure_some_usec_total=some,
        )

    samples = [contended(1.0, 1000, 5000), contended(2.0, 4000, 9000)]
    summary = summarize(samples)

    assert (summary.cpu_throttled_usec_total, summary.cpu_throttled_usec_delta) == (4000, 3000)
    assert (summary.cpu_throttled_periods_total, summary.cpu_throttled_periods_delta) == (40, 30)
    assert (summary.cpu_pressure_some_usec_total, summary.cpu_pressure_some_usec_delta) == (9000, 4000)
    assert summary.cpu_pressure_full_usec_total is None and summary.cpu_pressure_full_usec_delta is None
    missed = summarize(samples, final_read=False, unavailable_reason="final_read_failed")
    assert missed.cpu_throttled_usec_total is None and missed.cpu_pressure_some_usec_delta is None


class UsageProvider:
    """A provider whose sandbox reports a container counter and a sampled process tree."""

    name = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.reads = 0
        self.fail = fail
        self.closed = False

    async def usage(self, handle: SandboxHandle) -> tuple[SandboxUsageSample, ...]:
        self.reads += 1
        if self.fail:
            raise OSError("unreadable")
        if self.closed:
            raise AssertionError("sampled after close")
        return (
            _sample("container", float(self.reads), 1000 * self.reads, 100 * self.reads),
            _sample("process_tree", float(self.reads), 10 * self.reads, 7 - self.reads, kind="sampled_lower_bound"),
        )

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return SandboxHandle(sandbox_id="box-1", provider_name=self.name, raw=None)

    async def close(self, handle: SandboxHandle) -> None:
        self.closed = True

    async def aclose(self) -> None:
        return None


async def test_the_owner_samples_after_create_periodically_for_lower_bounds_and_before_close() -> None:
    provider = UsageProvider()
    tracker = SandboxUsageTracker(provider, SandboxHandle("box-1", "fake", None), interval=0.01)
    tracker.start()
    await asyncio.sleep(0.05)
    await tracker.stop()

    summaries = {summary.scope: summary for summary in tracker.summaries()}
    assert provider.reads >= 3 and tracker.task is None
    assert summaries["container"].sample_count == provider.reads
    assert summaries["container"].cpu_usage_usec_total == 1000 * provider.reads
    assert summaries["process_tree"].memory_peak_bytes == 6  # the first sample's, the largest seen
    assert all(summary.final_read and summary.unavailable_reason is None for summary in summaries.values())


async def test_without_a_read_at_close_the_totals_are_unknown_not_the_baseline() -> None:
    class GoneAtClose(UsageProvider):
        async def usage(self, handle: SandboxHandle) -> tuple[SandboxUsageSample, ...]:
            if self.reads >= 1:
                self.reads += 1
                raise SandboxUsageUnavailable("sandbox_gone")
            return await super().usage(handle)

    tracker = SandboxUsageTracker(GoneAtClose(), SandboxHandle("box-1", "fake", None), interval=0)
    tracker.start()
    await asyncio.sleep(0)
    assert all(not summary.final_read and summary.unavailable_reason is None for summary in tracker.summaries())
    await tracker.stop()

    container = next(summary for summary in tracker.summaries() if summary.scope == "container")
    assert (container.final_read, container.unavailable_reason) == (False, "sandbox_gone")
    assert container.cpu_usage_usec_total is None and container.cpu_usage_usec_delta is None
    assert container.memory_peak_bytes is None and container.first_sampled_at == 1.0


async def test_a_partial_read_at_close_marks_only_the_missing_measurement() -> None:
    class ContainerOnlyAtClose(UsageProvider):
        async def usage(self, handle: SandboxHandle) -> tuple[SandboxUsageSample, ...]:
            taken = await super().usage(handle)
            return taken if self.reads == 1 else taken[:1]

    tracker = SandboxUsageTracker(ContainerOnlyAtClose(), SandboxHandle("box-1", "fake", None), interval=0)
    tracker.start()
    await tracker.stop()

    summaries = {summary.scope: summary for summary in tracker.summaries()}
    assert summaries["container"].final_read and summaries["container"].cpu_usage_usec_total == 2000
    assert (summaries["process_tree"].final_read, summaries["process_tree"].unavailable_reason) == (
        False,
        "final_read_failed",
    )


async def test_a_hung_create_read_is_not_followed_by_a_second_read_at_close(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_gym.sandbox import usage as usage_module

    monkeypatch.setattr(usage_module, "SAMPLE_TIMEOUT_S", 0.05)

    class Hung(UsageProvider):
        async def usage(self, handle: SandboxHandle) -> tuple[SandboxUsageSample, ...]:
            self.reads += 1
            await asyncio.sleep(10)
            return ()

    provider = Hung()
    tracker = SandboxUsageTracker(provider, SandboxHandle("box-1", "fake", None), interval=0)
    tracker.start()
    await tracker.stop()

    assert provider.reads == 1 and tracker.final_failure == "final_read_timeout"


async def test_an_unavailable_first_read_is_retried_before_deciding_to_poll() -> None:
    class FlakyFirstRead(UsageProvider):
        async def usage(self, handle: SandboxHandle) -> tuple[SandboxUsageSample, ...]:
            self.reads += 1
            if self.reads == 1:
                raise OSError("not ready")
            peak = 50 if self.reads == 3 else 10  # the peak happens between create and stop
            return (_sample("process_tree", float(self.reads), self.reads, peak, kind="sampled_lower_bound"),)

    provider = FlakyFirstRead()
    tracker = SandboxUsageTracker(provider, SandboxHandle("box-1", "fake", None), interval=0.01)
    tracker.start()
    await asyncio.sleep(0.08)
    await tracker.stop()

    (summary,) = tracker.summaries()
    assert provider.reads > 3 and summary.sample_count == provider.reads - 1 and summary.memory_peak_bytes == 50


async def test_counter_only_measurements_are_not_polled() -> None:
    class CounterProvider(UsageProvider):
        async def usage(self, handle: SandboxHandle) -> tuple[SandboxUsageSample, ...]:
            self.reads += 1
            return (_sample("container", float(self.reads), 1000, 100),)

    provider = CounterProvider()
    tracker = SandboxUsageTracker(provider, SandboxHandle("box-1", "fake", None), interval=0.01)
    tracker.start()
    await asyncio.sleep(0.05)
    await tracker.stop()

    assert provider.reads == 2


async def test_a_failed_read_loses_the_sample_not_the_sandbox() -> None:
    provider = UsageProvider(fail=True)
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(image="img"))
    await sandbox.stop()

    assert provider.reads == 2 and provider.closed and sandbox.resource_usage == ()


async def test_the_creating_sandbox_records_usage_and_its_final_sample_precedes_close() -> None:
    provider = UsageProvider()
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(image="img"))
    await sandbox.stop()

    summaries = {summary.scope: summary for summary in sandbox.resource_usage}
    assert set(summaries) == {"container", "process_tree"}
    assert summaries["container"].sample_count == provider.reads >= 2


async def test_a_retried_stop_keeps_the_final_read_taken_before_the_failed_close() -> None:
    class CloseFailsOnce(UsageProvider):
        close_calls = 0

        async def usage(self, handle: SandboxHandle) -> tuple[SandboxUsageSample, ...]:
            if self.close_calls:
                # The failed close removed the container after all: any later read finds it gone.
                self.reads += 1
                raise SandboxUsageUnavailable("sandbox_gone")
            return await super().usage(handle)

        async def close(self, handle: SandboxHandle) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise TimeoutError("docker rm timed out")
            await super().close(handle)

    provider = CloseFailsOnce()
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(image="img"))
    with pytest.raises(TimeoutError):
        await sandbox.stop()
    reads_before_retry = provider.reads
    first = {summary.scope: summary for summary in sandbox.resource_usage}
    await sandbox.stop()

    assert provider.closed and provider.reads == reads_before_retry
    assert {summary.scope: summary for summary in sandbox.resource_usage} == first
    assert all(summary.final_read and summary.unavailable_reason is None for summary in first.values())
    assert first["container"].cpu_usage_usec_total == 1000 * reads_before_retry


async def test_usage_sampling_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_GYM_SANDBOX_USAGE", "0")
    provider = UsageProvider()
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(image="img"))
    await sandbox.stop()

    assert provider.reads == 0 and sandbox.resource_usage == ()


async def test_a_connected_sandbox_leaves_measurement_to_its_creator() -> None:
    class ConnectableUsageProvider(UsageProvider):
        async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict:
            return {"sandbox_id": handle.sandbox_id}

        async def connect(self, descriptor) -> SandboxHandle:
            return SandboxHandle(sandbox_id=descriptor["sandbox_id"], provider_name=self.name, raw=None)

    provider = ConnectableUsageProvider()
    borrowed = await AsyncSandbox.connect({"sandbox_id": "box-1"}, provider=provider)
    await borrowed.stop()

    assert provider.reads == 0 and borrowed.resource_usage == ()


# Provider wiring --------------------------------------------------------------------------------------------- #


@pytest.mark.sandbox
async def test_enroot_reads_its_init_process_tree_and_labels_the_shared_cgroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_gym.sandbox.providers.enroot import provider as enroot_provider

    monkeypatch.setattr(enroot_provider, "_require_enroot", lambda: "/usr/bin/enroot")
    provider = enroot_provider.EnrootProvider(create={"base_dir": "/tmp/nemo-gym-enroot-usage-test"})
    calls = []

    def host_process_usage(pid, state, *, own_cgroup_scope):
        calls.append((pid, own_cgroup_scope))
        return ()

    monkeypatch.setattr(provider._usage_reader, "host_process_usage", host_process_usage)
    monkeypatch.setattr(
        enroot_provider.EnrootProvider, "_pid_has_container_marker", staticmethod(lambda pid, name: True)
    )
    instance = enroot_provider._EnrootInstance(
        name="box", sqsh_path=Path("/x.sqsh"), staging_dir=Path("/s"), mount_point="/sandbox", image="img"
    )
    handle = SandboxHandle("box", "enroot", instance)

    assert await provider.usage(handle) == ()  # no init PID yet
    instance.container_pid = 4242
    await provider.usage(handle)
    assert calls == [(4242, "shared_step")]


@pytest.mark.sandbox
async def test_docker_reads_the_container_cgroup_on_this_host_or_from_inside(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nemo_gym.sandbox.providers.docker import provider as docker_provider

    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    provider = docker_provider.DockerProvider()
    container_id = "c0ffee" * 10
    runs = []

    async def run(argv, *, timeout_s, stdin=None):
        runs.append(argv)
        if argv[1] == "inspect":
            return 0, f"{container_id} 4242\n", ""
        return 0, _in_sandbox_output("0::/"), ""

    monkeypatch.setattr(provider, "_run", run)
    handle = SandboxHandle("box", "docker", docker_provider._DockerContainer(name="box", image="img"))

    _write_cgroup(tmp_path / "system.slice" / f"docker-{container_id}.scope")
    monkeypatch.setattr(docker_provider, "cgroup_path", lambda pid: f"/system.slice/docker-{container_id}.scope")
    monkeypatch.setattr(
        docker_provider,
        "cgroup_v2_sample",
        lambda directory, scope: cgroup_v2_sample(tmp_path / directory.relative_to("/sys/fs/cgroup"), scope=scope),
    )
    (host,) = await provider.usage(handle)
    assert host.scope == "container" and host.cpu_usage_usec_total == 2_500_000
    assert [argv[1] for argv in runs] == ["inspect"]

    # A remote daemon's PID is not this host's process, so the read falls back to the container itself.
    monkeypatch.setattr(docker_provider, "cgroup_path", lambda pid: "/user.slice/something-else")
    handle.raw.cgroup_dir = None
    (inside,) = await provider.usage(handle)
    assert inside.scope == "container" and runs[-1][1] == "exec" and runs[-1][-1] == IN_SANDBOX_CGROUP_COMMAND
    assert runs[-1][runs[-1].index("-w") + 1] == "/" and "--user" in runs[-1]


@pytest.mark.sandbox
@pytest.mark.parametrize(("exec_reply", "cgroup"), [((0, "", ""), "/x"), ((1, "", "no sh"), None)])
async def test_docker_reports_nothing_without_a_readable_container_cgroup(
    monkeypatch: pytest.MonkeyPatch, exec_reply, cgroup
) -> None:
    from nemo_gym.sandbox.providers.docker import provider as docker_provider

    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    provider = docker_provider.DockerProvider()

    async def run(argv, *, timeout_s, stdin=None):
        return (0, "abc 0\n", "") if argv[1] == "inspect" else exec_reply

    monkeypatch.setattr(provider, "_run", run)
    monkeypatch.setattr(docker_provider, "cgroup_path", lambda pid: cgroup)
    handle = SandboxHandle("box", "docker", docker_provider._DockerContainer(name="box", image="img"))
    assert await provider.usage(handle) == ()


@pytest.mark.sandbox
async def test_docker_reports_nothing_when_the_container_cannot_be_inspected(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_gym.sandbox.providers.docker import provider as docker_provider

    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    provider = docker_provider.DockerProvider()

    async def run(argv, *, timeout_s, stdin=None):
        return 1, "", "permission denied"

    monkeypatch.setattr(provider, "_run", run)
    handle = SandboxHandle("box", "docker", docker_provider._DockerContainer(name="box", image="img"))
    assert await provider.usage(handle) == ()


@pytest.mark.sandbox
async def test_docker_names_why_a_container_cannot_be_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_gym.sandbox.providers.docker import provider as docker_provider

    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    provider = docker_provider.DockerProvider()
    replies = {"inspect": (1, "", "Error: No such object: box")}

    async def run(argv, *, timeout_s, stdin=None):
        return replies[argv[1]]

    monkeypatch.setattr(provider, "_run", run)
    handle = SandboxHandle("box", "docker", docker_provider._DockerContainer(name="box", image="img"))
    with pytest.raises(SandboxUsageUnavailable, match="sandbox_gone"):
        await provider.usage(handle)

    # Without a private cgroup namespace the in-container read sees the host root, never the container's cgroup.
    replies.update(inspect=(0, "abc 0\n", ""), exec=(0, _in_sandbox_output("0::/docker/abc"), ""))
    with pytest.raises(SandboxUsageUnavailable, match="not_cgroup_root"):
        await provider.usage(handle)


@pytest.mark.sandbox
async def test_docker_keeps_reading_a_local_container_on_this_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from nemo_gym.sandbox.providers.docker import provider as docker_provider

    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    provider = docker_provider.DockerProvider()
    container_id = "c0ffee" * 10
    runs = []

    async def run(argv, *, timeout_s, stdin=None):
        runs.append(argv)
        return 0, f"{container_id} 4242\n", ""

    monkeypatch.setattr(provider, "_run", run)
    directory = _write_cgroup(tmp_path / f"docker-{container_id}.scope")
    monkeypatch.setattr(docker_provider, "cgroup_path", lambda pid: f"/docker-{container_id}.scope")
    monkeypatch.setattr(
        docker_provider,
        "cgroup_v2_sample",
        lambda d, scope: cgroup_v2_sample(tmp_path / d.relative_to("/sys/fs/cgroup"), scope=scope),
    )
    handle = SandboxHandle("box", "docker", docker_provider._DockerContainer(name="box", image="img"))
    assert len(await provider.usage(handle)) == 1

    # The container exited (say OOM-killed) before teardown: its cgroup is gone, and no `docker exec` is tried.
    for path in directory.iterdir():
        path.unlink()
    with pytest.raises(SandboxUsageUnavailable, match="sandbox_gone"):
        await provider.usage(handle)
    assert [argv[1] for argv in runs] == ["inspect"]


@pytest.mark.sandbox
async def test_apptainer_measures_its_instance_only_while_the_pid_is_still_that_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_gym.sandbox.providers.apptainer import provider as apptainer_provider

    monkeypatch.setattr(apptainer_provider, "_require_apptainer", lambda _bin_path=None: "/usr/bin/apptainer")
    provider = apptainer_provider.ApptainerProvider()
    listings = []
    listed = {"pid": 777}

    async def run(argv, *, timeout_s, stdin=None):
        listings.append(argv)
        entries = [{"instance": "box", "pid": listed["pid"]}, {"instance": "other", "pid": 1}] if listed["pid"] else []
        return 0, json.dumps({"instances": entries}), ""

    monkeypatch.setattr(provider, "_run", run)
    start_times = {777: 5000}
    monkeypatch.setattr(apptainer_provider, "process_start_time", lambda pid: start_times.get(pid))
    seen = []
    monkeypatch.setattr(
        usage_readers.ProcfsUsageReader,
        "host_process_usage",
        lambda self, pid, state, *, own_cgroup_scope: seen.append((pid, own_cgroup_scope))
        or (_sample("container", 1, 1, 1),),
    )
    instance = apptainer_provider._ApptainerInstance(name="box", staging_dir=Path("/s"), mount_point="/m", image="i")
    handle = SandboxHandle("box", "apptainer", instance)

    assert len(await provider.usage(handle)) == 1
    assert len(await provider.usage(handle)) == 1
    assert seen == [(777, "container"), (777, "container")] and len(listings) == 1  # a live instance is listed once

    # The instance exits and another process reuses its PID: that process is never measured as the instance.
    start_times[777] = 9999
    listed["pid"] = None
    assert await provider.usage(handle) == ()
    assert len(seen) == 2 and len(listings) == 2


async def test_stop_puts_the_usage_on_its_span_and_tells_listeners(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    from nemo_gym.sandbox import api, observability

    spans = {}

    @contextmanager
    def managed_span(group, name, **attributes):
        spans[name] = attributes
        yield None

    monkeypatch.setattr(api, "is_span_group_enabled", lambda group: True)
    monkeypatch.setattr(api, "managed_span", managed_span)
    monkeypatch.setattr(observability, "_stopped_sandbox_listeners", [])
    told = []
    observability.add_stopped_sandbox_listener(told.append)
    observability.add_stopped_sandbox_listener(told.append)  # registered once
    observability.add_stopped_sandbox_listener(lambda sandbox: 1 / 0)  # a failing listener never fails the stop

    provider = UsageProvider()
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(image="img"))
    await sandbox.stop()

    attributes = spans["gym.sandbox.stop"]
    assert attributes["nemo.gym.sandbox.id"] == "box-1"
    assert attributes["nemo.gym.sandbox.usage.container.cgroup_v2.cpu_kind"] == "counter"
    assert attributes["nemo.gym.sandbox.usage.process_tree.procfs_pidns.peak_kind"] == "sampled_lower_bound"
    assert attributes["nemo.gym.sandbox.usage.container.cgroup_v2.sample_count"] == provider.reads
    assert told == [sandbox]


async def test_an_unmeasured_or_failed_stop_tells_no_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_gym.sandbox import observability

    monkeypatch.setattr(observability, "_stopped_sandbox_listeners", [])
    told = []
    observability.add_stopped_sandbox_listener(told.append)

    unmeasured = await AsyncSandbox(UsageProvider(fail=True)).start(SandboxSpec(image="img"))
    await unmeasured.stop()

    class FailingClose(UsageProvider):
        async def close(self, handle: SandboxHandle) -> None:
            raise RuntimeError("close failed")

    failing = await AsyncSandbox(FailingClose()).start(SandboxSpec(image="img"))
    with pytest.raises(RuntimeError):
        await failing.stop()
    assert told == [] and failing.resource_usage


@pytest.mark.sandbox
@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="no /bin/sh")
@pytest.mark.parametrize("runtime", ["docker", "apptainer", "enroot"])
async def test_a_cancelled_cli_call_kills_its_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime: str
) -> None:
    import os

    if runtime == "docker":
        from nemo_gym.sandbox.providers.docker import provider as module

        monkeypatch.setattr(module, "_require_docker", lambda: "/usr/bin/docker")
        provider = module.DockerProvider()
    elif runtime == "apptainer":
        from nemo_gym.sandbox.providers.apptainer import provider as module

        monkeypatch.setattr(module, "_require_apptainer", lambda _bin_path=None: "/usr/bin/apptainer")
        provider = module.ApptainerProvider()
    else:
        from nemo_gym.sandbox.providers.enroot import provider as module

        monkeypatch.setattr(module, "_require_enroot", lambda *args, **kwargs: "/usr/bin/enroot")
        provider = module.EnrootProvider(create={"base_dir": str(tmp_path / "home")})
    pid_file = tmp_path / "pid"
    # A stalled CLI (such as `docker inspect`) whose child would outlive a leaked parent.
    command = ["/bin/sh", "-c", f"sleep 30 & echo $! > {pid_file}; wait"]

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(1):
            await provider._run(command, timeout_s=30)

    child = int(pid_file.read_text())
    for _ in range(100):  # the killed child is reaped by init shortly after
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.02)
    pytest.fail(f"the {runtime} CLI's child {child} outlived the cancelled call")


async def test_opensandbox_reads_its_own_cgroup_from_inside_and_stops_reading_where_it_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")
    from nemo_gym.sandbox.providers.base import SandboxExecResult
    from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider

    provider = opensandbox_provider.OpenSandboxProvider(connection={"request_timeout_s": 1})
    contended = (
        f"==cpu.stat\n{CONTENDED_CPU_STAT}==cpu.pressure\n{CPU_PRESSURE}==memory.current\n157286400\n"
        f"==memory.peak\n314572800\n==memory.stat\n{MEMORY_STAT}==cgroup\n0::/\n"
    )
    outputs = {
        "own": SandboxExecResult(stdout=contended, stderr=None, return_code=0),
        "host": SandboxExecResult(stdout=_in_sandbox_output("0::/kubepods/pod1/abc"), stderr=None, return_code=0),
        "failed": SandboxExecResult(stdout=None, stderr="unreachable", return_code=-1, error_type="timeout"),
    }
    calls: list[tuple[str, str, dict]] = []

    async def fake_exec(handle: SandboxHandle, command: str, **kwargs: object) -> SandboxExecResult:
        calls.append((handle.sandbox_id, command, kwargs))
        return outputs[handle.sandbox_id]

    monkeypatch.setattr(provider, "_exec", fake_exec)

    (sample,) = await provider.usage(SandboxHandle("own", "opensandbox", None))
    assert (sample.source, sample.scope, sample.cpu_kind, sample.peak_kind) == (
        "cgroup_v2",
        "container",
        "counter",
        "counter",
    )
    assert (sample.cpu_usage_usec_total, sample.memory_peak_bytes, sample.includes_page_cache) == (
        2_500_000,
        314_572_800,
        True,
    )
    assert (sample.cpu_throttled_usec_total, sample.cpu_pressure_full_usec_total) == (912_000, 700_000)
    # The provider's own exec from the root, as the image's user, without its retries: no exec span, no tool call.
    assert calls[0][1] == IN_SANDBOX_CGROUP_COMMAND
    assert calls[0][2]["cwd"] == "/" and calls[0][2]["retries"] == 0 and "user" not in calls[0][2]

    for _ in range(2):
        with pytest.raises(SandboxUsageUnavailable, match="not_cgroup_root"):
            await provider.usage(SandboxHandle("host", "opensandbox", None))
    assert [call[0] for call in calls].count("host") == 1  # a sandbox it cannot measure is not read again

    assert await provider.usage(SandboxHandle("failed", "opensandbox", None)) == ()
