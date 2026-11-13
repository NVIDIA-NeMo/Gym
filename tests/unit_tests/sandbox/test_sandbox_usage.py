# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured sandbox CPU and memory: the readers, their labels, and owner-side sampling."""

import asyncio
from pathlib import Path

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import usage as usage_readers
from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec, SandboxUsageSample
from nemo_gym.sandbox.providers.usage import (
    IN_SANDBOX_CGROUP_COMMAND,
    cgroup_v2_sample,
    parse_cgroup_v2,
    parse_in_sandbox_cgroup,
)
from nemo_gym.sandbox.usage import SandboxUsageTracker, sample_interval, summarize


CPU_STAT = "usage_usec 2500000\nuser_usec 2000000\nsystem_usec 500000\n"
MEMORY_STAT = "anon 104857600\nfile 52428800\nkernel 1024\n"


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
    assert parse_in_sandbox_cgroup(_in_sandbox_output("0::/system.slice/job.scope"), sampled_at=5.0) is None
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
        start: int = 1,
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
        # Fields 3..22 of /proc/<pid>/stat: utime is field 14, stime 15, starttime 22.
        fields = ["S"] + ["0"] * 10 + [str(ticks[0]), str(ticks[1])] + ["0"] * 6 + [str(start)] + ["0"] * 5
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
    return FakeProc(tmp_path / "proc")


def test_a_process_without_a_unified_cgroup_has_no_cgroup_path(proc: FakeProc) -> None:
    proc.add(100, pidns="B", mntns="M1")
    (proc.root / "100" / "cgroup").write_text("12:memory:/job\n")
    assert usage_readers.cgroup_path(100, proc_root=proc.root) is None
    assert usage_readers.cgroup_path(424242, proc_root=proc.root) is None


@pytest.mark.parametrize(("value", "expected"), [("2.5", 2.5), ("0", 0.0), ("inf", 0.0), ("soon", 5.0)])
def test_the_sample_interval_setting(monkeypatch: pytest.MonkeyPatch, value: str, expected: float) -> None:
    monkeypatch.setenv("NEMO_GYM_SANDBOX_USAGE_SAMPLE_INTERVAL_SECONDS", value)
    assert sample_interval() == expected


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
    (inside,) = await provider.usage(handle)
    assert inside.scope == "container" and runs[-1][1] == "exec" and runs[-1][-1] == IN_SANDBOX_CGROUP_COMMAND


@pytest.mark.sandbox
@pytest.mark.parametrize(
    ("exec_reply", "cgroup"), [((0, _in_sandbox_output("0::/docker/abc"), ""), "/x"), ((1, "", "no sh"), None)]
)
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
        return 1, "", "No such object"

    monkeypatch.setattr(provider, "_run", run)
    handle = SandboxHandle("box", "docker", docker_provider._DockerContainer(name="box", image="img"))
    assert await provider.usage(handle) == ()


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
@pytest.mark.parametrize("runtime", ["docker", "apptainer"])
async def test_a_cancelled_cli_call_kills_its_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime: str
) -> None:
    import os

    if runtime == "docker":
        from nemo_gym.sandbox.providers.docker import provider as module

        monkeypatch.setattr(module, "_require_docker", lambda: "/usr/bin/docker")
        provider = module.DockerProvider()
    else:
        from nemo_gym.sandbox.providers.apptainer import provider as module

        monkeypatch.setattr(module, "_require_apptainer", lambda _bin_path=None: "/usr/bin/apptainer")
        provider = module.ApptainerProvider()
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
