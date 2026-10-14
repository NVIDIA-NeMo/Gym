# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nemo_gym._checkpoint.harness`` against real processes: real bash, real /proc, real signals.

Each launch runs through a local stand-in for a sandbox's exec daemon (bash in a process group of its own, stderr
returned inside stdout, as OpenSandbox's background executions do). The tests signal only processes their own
launches started, so they are safe on any Linux host.
"""

import asyncio
import os
import re
import shutil
import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shlex import quote
from typing import Optional

import pytest

from nemo_gym._checkpoint import harness
from nemo_gym._checkpoint.harness import (
    LAUNCH_TAG,
    HarnessInterrupted,
    HarnessLaunch,
    run_until_park_requested,
    stdout_only,
)
from nemo_gym.sandbox import SandboxExecResult


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None, reason="the stop walks /proc from bash"
)


class LocalShell:
    """Runs each command with bash in a process group of its own and returns stderr inside stdout."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, *, timeout_s: Optional[float] = None) -> SandboxExecResult:
        self.commands.append(command)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(stdout=stdout.decode(errors="replace"), stderr=None, return_code=process.returncode)


class ParkRequest:
    """What ``run_until_park_requested`` awaits of a ``LegacyRun``."""

    def __init__(self) -> None:
        self.requested = asyncio.Event()

    async def wait_park_requested(self) -> None:
        await self.requested.wait()


def alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(") ", 1)[1].split()[0] != "Z"


def launch_id(shell: LocalShell) -> str:
    match = re.search(rf"{LAUNCH_TAG}=([0-9a-f]+)", shell.commands[0])
    assert match, "the launch carries its tag"
    return match.group(1)


async def until(condition, timeout_s: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.05)


@pytest.fixture(autouse=True)
def short_graces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "_GRACEFUL_S", 2.0)
    monkeypatch.setattr(harness, "_KILL_S", 10.0)
    monkeypatch.setattr(harness, "_SURVIVOR_CHECK_INTERVAL_S", 0.2)


# What a launch is doing when the checkpoint asks the run to park, and what must be gone afterwards.
PHASES = {
    # The install is still running: nothing after it may start.
    "installing": "sleep 300 && touch {started}",
    # The harness runs a command in a session of its own, out of reach of the launch's process group.
    "running_a_command": (
        "touch {started} && {python} -c \"import subprocess, time; c = subprocess.Popen(['sleep', '300'], "
        "start_new_session=True); open('{pids}', 'w').write(str(c.pid)); time.sleep(300)\""
    ),
    # A background process detached at once: PID 1's child, found only by its tag.
    "with_a_detached_process": (
        "touch {started} && bash -c 'nohup sleep 300 >/dev/null 2>&1 & echo $! > {pids}' && sleep 300"
    ),
    # A child that cleared its environment and ignores SIGINT; the SIGINT pass orphans it by killing its parent,
    # and only the pass's record names it for the SIGKILL pass.
    "with_an_untagged_child": 'touch {started}; env -i "$(command -v sleep)" 300 & echo $! > {pids}; wait',
    # The harness ignores SIGINT: the SIGKILL pass ends it.
    "ignoring_sigint": "trap '' INT; touch {started} && sleep 300",
}


@pytest.mark.parametrize("phase", sorted(PHASES))
async def test_a_park_request_stops_the_launch_and_everything_it_started(tmp_path: Path, phase: str) -> None:
    started, pids = tmp_path / "started", tmp_path / "pids"
    shell, park = LocalShell(), ParkRequest()
    run = asyncio.create_task(
        run_until_park_requested(
            shell, PHASES[phase].format(started=started, pids=pids, python=quote(sys.executable)), park, timeout_s=600
        )  # type: ignore[arg-type]
    )
    if phase == "installing":
        await asyncio.sleep(0.5)
    else:
        await until(lambda: started.exists() and (phase == "ignoring_sigint" or (pids.exists() and pids.read_text())))

    park.requested.set()
    with pytest.raises(HarnessInterrupted, match="SIG(INT|KILL) ended it after"):
        await asyncio.wait_for(run, timeout=30)

    if phase == "installing":
        await asyncio.sleep(0.5)
        assert not started.exists(), "nothing after the install started"
    if pids.exists():
        assert not alive(int(pids.read_text())), "the process the launch started is gone"
    found = await shell.exec(f"grep -l {LAUNCH_TAG}={launch_id(shell)} /proc/[0-9]*/environ 2>/dev/null; true")
    assert not [line for line in (found.stdout or "").split() if alive(int(line.split("/")[2]))], "no process is left"
    assert not list(Path("/tmp").glob(f"nemo-gym-harness-{launch_id(shell)}.*")), "the stop leaves no file"


async def test_a_stop_that_arrives_before_the_launch_starts_nothing(tmp_path: Path) -> None:
    shell, launch, started = LocalShell(), HarnessLaunch(), tmp_path / "started"
    await shell.exec(launch.stop_command("INT"))

    result = await shell.exec(launch.command(f"touch {started}"), timeout_s=30)

    assert result.return_code == 130 and not started.exists()
    await shell.exec(launch.cleanup_command())
    assert not list(Path("/tmp").glob(f"nemo-gym-harness-{launch.id}.*"))


async def test_a_stop_leaves_other_launches_alone() -> None:
    shell, stopped, other = LocalShell(), HarnessLaunch(), HarnessLaunch()
    runs = [asyncio.create_task(shell.exec(launch.command("sleep 300"))) for launch in (stopped, other)]
    try:
        await asyncio.sleep(0.5)
        for signal_name in ("INT", "KILL"):
            await shell.exec(stopped.stop_command(signal_name))
        await asyncio.wait_for(runs[0], timeout=10)

        assert not runs[1].done(), "the other launch still runs"
        assert (await shell.exec(other.survivors_command())).stdout, "and its process is alive"
    finally:
        await shell.exec(other.stop_command("KILL"))
        await asyncio.gather(*runs, return_exceptions=True)
        for launch in (stopped, other):
            await shell.exec(launch.cleanup_command())


async def test_a_launch_that_ends_first_returns_its_result_and_writes_no_file() -> None:
    shell = LocalShell()

    result = await run_until_park_requested(shell, "echo done", ParkRequest(), timeout_s=30)  # type: ignore[arg-type]

    assert (result.return_code, result.stdout) == (0, "done\n")
    assert not list(Path("/tmp").glob(f"nemo-gym-harness-{launch_id(shell)}.*"))
    unchecked = await run_until_park_requested(shell, "echo plain", None, timeout_s=30)  # type: ignore[arg-type]
    assert unchecked.stdout == "plain\n" and shell.commands[-1] == "echo plain", "without checkpointing: as given"


async def test_stdout_only_keeps_progress_on_stderr_out_of_parsed_output() -> None:
    result = await LocalShell().exec(stdout_only("echo 'Exporting session: ses_1' >&2; echo '{\"id\": 1}'"))

    assert result.stdout == '{"id": 1}\n'


class SlowDaemon(LocalShell):
    """An exec daemon that is slow around the launch: it starts the launch late, or reports its result late, as
    OpenSandbox can by up to about 30 s. Other commands run at once."""

    def __init__(self, *, start_delay_s: float = 0.0, result_delay_s: float = 0.0) -> None:
        super().__init__()
        self.start_delay_s = start_delay_s
        self.result_delay_s = result_delay_s
        self.launch_returned = asyncio.Event()

    async def exec(self, command: str, *, timeout_s: Optional[float] = None) -> SandboxExecResult:
        if not command.startswith(f"env {LAUNCH_TAG}="):
            return await super().exec(command, timeout_s=timeout_s)
        await asyncio.sleep(self.start_delay_s)
        result = await super().exec(command, timeout_s=timeout_s)
        self.launch_returned.set()
        await asyncio.sleep(self.result_delay_s)
        return result


async def test_a_launch_that_ended_before_the_park_request_keeps_its_late_result() -> None:
    shell, park = SlowDaemon(result_delay_s=2.0), ParkRequest()
    run = asyncio.create_task(run_until_park_requested(shell, "echo done", park, timeout_s=30))  # type: ignore[arg-type]
    await asyncio.wait_for(shell.launch_returned.wait(), timeout=10)

    park.requested.set()
    result = await asyncio.wait_for(run, timeout=30)

    assert (result.return_code, result.stdout) == (0, "done\n"), "the finished launch is not relaunched"
    assert not [command for command in shell.commands if "kill -KILL" in command], "nothing was killed"
    assert not list(Path("/tmp").glob(f"nemo-gym-harness-{launch_id(shell)}.*"))


async def test_a_park_request_before_the_launch_starts_interrupts_it_and_nothing_runs(tmp_path: Path) -> None:
    """A launch the daemon has not started yet has no record either; it must not count as one that ended."""
    started = tmp_path / "started"
    shell, park = SlowDaemon(start_delay_s=1.0), ParkRequest()
    park.requested.set()

    with pytest.raises(HarnessInterrupted):
        await asyncio.wait_for(
            run_until_park_requested(shell, f"touch {started}", park, timeout_s=30),  # type: ignore[arg-type]
            timeout=30,
        )
    await asyncio.sleep(1.5)

    assert not started.exists(), "the stop that came first kept the launch from starting"


async def test_cancelling_the_run_kills_the_launch_and_everything_it_started(tmp_path: Path) -> None:
    started, pids = tmp_path / "started", tmp_path / "pids"
    shell, park = LocalShell(), ParkRequest()
    command = PHASES["with_a_detached_process"].format(started=started, pids=pids, python=quote(sys.executable))
    run = asyncio.create_task(run_until_park_requested(shell, command, park, timeout_s=600))  # type: ignore[arg-type]
    await until(lambda: started.exists() and pids.exists() and pids.read_text())

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, timeout=60)

    assert not alive(int(pids.read_text())), "the detached process the launch started is gone"
    found = await shell.exec(f"grep -l {LAUNCH_TAG}={launch_id(shell)} /proc/[0-9]*/environ 2>/dev/null; true")
    assert not [line for line in (found.stdout or "").split() if alive(int(line.split("/")[2]))], "no process is left"


# -- pausing instead of stopping -------------------------------------------------------------------------------


class Parker:
    """The run's ``park``: holds until the test opens the checkpoint, which withdraws the park request."""

    def __init__(self, park: ParkRequest, *, error: Optional[Exception] = None) -> None:
        self.park = park
        self.error = error
        self.calls = 0
        self.parked = asyncio.Event()
        self.opened = asyncio.Event()

    async def __call__(self) -> None:
        self.calls += 1
        self.parked.set()
        if self.error is not None:
            raise self.error
        await self.opened.wait()
        self.opened.clear()
        self.parked.clear()
        self.park.requested.clear()


def run_pausing(
    shell: LocalShell, command: str, park: ParkRequest, parker: Parker, *, timeout_s: float = 600
) -> "asyncio.Task[SandboxExecResult]":
    """``run_until_park_requested`` with ``parker`` as the run's park callback, as a task."""
    run = run_until_park_requested(shell, command, park, timeout_s=timeout_s, park=parker)  # type: ignore[arg-type]
    return asyncio.create_task(run)


def launch_states(tag: str) -> dict[int, str]:
    """State of every live process of the launch tagged ``tag``, and of their descendants."""
    states: dict[int, str] = {}
    parents: dict[int, int] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
            environ = (proc / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        parents[int(proc.name)] = int(fields[1])
        if f"{LAUNCH_TAG}={tag}".encode() in environ:
            states[int(proc.name)] = fields[0]
    for pid, parent in parents.items():
        if parent in states and pid not in states:
            try:
                states[pid] = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
            except OSError:
                pass
    return {pid: state for pid, state in states.items() if state not in "ZX"}


def kill_launch(shell: LocalShell) -> None:
    """Kill what a test's launch left running, such as a detached writer (not through ``shell``, whose commands
    the tests inspect)."""
    for pid in launch_states(launch_id(shell)):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


async def test_a_paused_launch_changes_nothing_while_parked_and_continues_in_place(tmp_path: Path) -> None:
    counter, steps, detached = tmp_path / "counter", tmp_path / "steps", tmp_path / "detached"
    command = (
        f"bash -c 'nohup bash -c \"while :; do date +%s%N > {detached}; sleep 0.05; done\" >/dev/null 2>&1 &'\n"
        f"i=0; while [ $i -lt 60 ]; do i=$((i+1)); echo $i > {counter}; echo $i >> {steps}; sleep 0.05; done; "
        "echo counted $i"
    )
    shell, park = LocalShell(), ParkRequest()
    parker = Parker(park)
    run = run_pausing(shell, command, park, parker, timeout_s=60)
    try:
        for checkpoint in range(2):
            await until(
                lambda: counter.exists() and detached.exists() and int(counter.read_text()) >= 10 * (checkpoint + 1)
            )
            park.requested.set()
            await asyncio.wait_for(parker.parked.wait(), 10)

            states = launch_states(launch_id(shell))
            assert states and set(states.values()) <= {"T", "t"}, f"every process is stopped: {states}"
            frozen = (counter.read_text(), detached.read_text())
            await asyncio.sleep(0.5)
            assert (counter.read_text(), detached.read_text()) == frozen, "nothing changes while paused"
            assert launch_states(launch_id(shell)) == states, "the same processes stay stopped"

            parker.opened.set()
            await until(lambda: counter.read_text() != frozen[0])

        result = await asyncio.wait_for(run, 30)
    finally:
        run.cancel()
        kill_launch(shell)

    assert (result.return_code, result.stdout) == (0, "counted 60\n"), "the same launch finished"
    assert steps.read_text().split() == [str(i) for i in range(1, 61)], "it continued where it was: no step lost"
    assert parker.calls == 2
    assert sum(command.startswith(f"env {LAUNCH_TAG}=") for command in shell.commands) == 1, "never relaunched"
    assert not [command for command in shell.commands if "kill -INT" in command or "kill -KILL" in command]
    assert not list(Path("/tmp").glob(f"nemo-gym-harness-{launch_id(shell)}.*")), "no file is left"


async def test_a_pause_before_the_launch_starts_holds_it_until_the_checkpoint_opens(tmp_path: Path) -> None:
    started = tmp_path / "started"
    shell, park = SlowDaemon(start_delay_s=1.0), ParkRequest()
    parker = Parker(park)
    park.requested.set()
    run = run_pausing(shell, f"touch {started} && echo ran", park, parker, timeout_s=30)
    await asyncio.wait_for(parker.parked.wait(), 10)
    await asyncio.sleep(1.5)

    assert not started.exists(), "the launch waits while the checkpoint is open"
    parker.opened.set()
    result = await asyncio.wait_for(run, 30)

    assert (result.return_code, result.stdout) == (0, "ran\n") and started.exists()
    assert not list(Path("/tmp").glob(f"nemo-gym-harness-{launch_id(shell)}.*"))


async def test_a_launch_that_ended_before_the_pause_keeps_its_late_result_and_does_not_park() -> None:
    shell, park = SlowDaemon(result_delay_s=2.0), ParkRequest()
    parker = Parker(park)
    run = run_pausing(shell, "echo done", park, parker, timeout_s=30)
    await asyncio.wait_for(shell.launch_returned.wait(), timeout=10)

    park.requested.set()
    result = await asyncio.wait_for(run, timeout=30)

    assert (result.return_code, result.stdout) == (0, "done\n")
    assert parker.calls == 0, "a finished launch returns its result; the run parks after it"


async def test_a_pause_that_cannot_be_confirmed_stops_the_launch_instead(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HarnessLaunch, "pause_command", lambda self: "echo PENDING 1:D")
    shell, park = LocalShell(), ParkRequest()
    parker = Parker(park)
    run = run_pausing(shell, "sleep 300", park, parker, timeout_s=600)
    await asyncio.sleep(0.5)

    park.requested.set()
    with pytest.raises(HarnessInterrupted):
        await asyncio.wait_for(run, timeout=30)

    assert parker.calls == 0
    assert not launch_states(launch_id(shell)), "the stop ended the launch"


async def test_a_park_that_fails_kills_the_paused_launch(tmp_path: Path) -> None:
    shell, park = LocalShell(), ParkRequest()
    parker = Parker(park, error=RuntimeError("the bookmark could not be recorded"))
    run = run_pausing(shell, "sleep 300", park, parker, timeout_s=600)
    await asyncio.sleep(0.5)

    park.requested.set()
    with pytest.raises(RuntimeError, match="could not be recorded"):
        await asyncio.wait_for(run, timeout=60)

    assert not launch_states(launch_id(shell)), "no paused process is left behind"


async def test_cancelling_a_paused_run_kills_the_launch_and_everything_it_started(tmp_path: Path) -> None:
    started, pids = tmp_path / "started", tmp_path / "pids"
    shell, park = LocalShell(), ParkRequest()
    parker = Parker(park)
    command = PHASES["with_a_detached_process"].format(started=started, pids=pids, python=quote(sys.executable))
    run = run_pausing(shell, command, park, parker, timeout_s=600)
    await until(lambda: started.exists() and pids.exists() and pids.read_text())
    park.requested.set()
    await asyncio.wait_for(parker.parked.wait(), 10)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, timeout=60)

    assert not alive(int(pids.read_text())), "the paused detached process is gone"
    assert not launch_states(launch_id(shell)), "no process is left"


class PollingShell(LocalShell):
    """A sandbox with a checkpoint polling scope, which counts how often the scope was entered and left."""

    def __init__(self) -> None:
        super().__init__()
        self.scopes: list[str] = []

    @contextmanager
    def checkpoint_exec_polling(self) -> Iterator[None]:
        self.scopes.append("enter")
        try:
            yield
        finally:
            self.scopes.append("exit")


async def test_a_stop_runs_inside_the_sandboxs_checkpoint_polling_scope() -> None:
    shell, park = PollingShell(), ParkRequest()
    run = asyncio.create_task(run_until_park_requested(shell, "sleep 300", park, timeout_s=600))  # type: ignore[arg-type]
    await asyncio.sleep(0.5)

    park.requested.set()
    with pytest.raises(HarnessInterrupted):
        await asyncio.wait_for(run, timeout=30)

    assert shell.scopes == ["enter", "exit"], "the whole stop ran inside one polling scope"


async def test_a_pause_does_not_enter_the_polling_scope() -> None:
    shell, park = PollingShell(), ParkRequest()
    parker = Parker(park)
    run = run_pausing(shell, "sleep 1 && echo done", park, parker, timeout_s=60)
    await asyncio.sleep(0.3)
    park.requested.set()
    await asyncio.wait_for(parker.parked.wait(), 10)
    parker.opened.set()

    result = await asyncio.wait_for(run, 30)

    assert result.stdout == "done\n" and shell.scopes == [], "a paused launch is not ending: no fast polling"
