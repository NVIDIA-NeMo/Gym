# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pausing a supervised harness in place for a checkpoint, from the bash supervisor up to ``run_until_parked``."""

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents.utils.run_until_parked import CONTINUE_ATTEMPTS, Finished, Stopped, run_until_parked
from nemo_gym.agents.utils.sandbox_session import HarnessPauseResult, SandboxCommand, SandboxSession
from nemo_gym.agents.utils.supervisor_client import (
    CLEANUP_RECEIPT_FILE,
    CONTINUE_REQUEST_FILE,
    CONTINUED_FILE,
    EXPIRE_REQUEST_FILE,
    PAUSE_REQUEST_FILE,
    PAUSED_FILE,
    STOP_REQUEST_FILE,
    SUPERVISOR_SCRIPT,
    exec_timeout,
    parse_cleanup_receipt,
    request_continue,
    request_pause,
)
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_runtime._checkpoint.errors import StaleAttemptError


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")


# -- the bash supervisor -------------------------------------------------------------------------------------------


def _start(tmp_path: Path, *command: str, timeout: str = "30") -> subprocess.Popen:
    return subprocess.Popen(
        # Cleanup phases end early when the harness does; the limit only matters on a busy node.
        ["bash", str(SUPERVISOR_SCRIPT), "--timeout", timeout, "--cleanup-timeout", "5"]
        + ["--receipt", str(tmp_path / CLEANUP_RECEIPT_FILE), "--stop-file", str(tmp_path / "stop.request")]
        + ["--expire-file", str(tmp_path / EXPIRE_REQUEST_FILE)]
        + ["--pause-file", str(tmp_path / PAUSE_REQUEST_FILE), "--paused", str(tmp_path / PAUSED_FILE)]
        + ["--continue-file", str(tmp_path / CONTINUE_REQUEST_FILE), "--continued", str(tmp_path / CONTINUED_FILE)]
        + ["--", *command],
        cwd=tmp_path,
        # Not pipes: a service the harness left running would hold them open.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _request(path: Path, seq: int) -> None:
    path.with_suffix(".tmp").write_text(str(seq))
    os.replace(path.with_suffix(".tmp"), path)


def _wait_for(predicate, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


def _acknowledgement(path: Path, seq: int) -> dict:
    def answered() -> bool:
        return path.exists() and json.loads(path.read_text())["seq"] == seq

    _wait_for(answered)
    return json.loads(path.read_text())


def _state(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return "gone"


def _finish(process: subprocess.Popen) -> None:
    """Stop the harness of a test that failed half way, so its processes do not outlive the test."""
    if process.poll() is None:
        stop = Path(process.args[process.args.index("--stop-file") + 1])
        stop.touch()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


# The worker ticks, starts a ticking child and a service that leaves the harness: its parent exits, so only the
# supervisor's environment marker reaches it.
_SERVICE = (
    "import os, sys, time\nos.setsid()\nwhile True:\n    open('service.ticks', 'a').write('.')\n    time.sleep(0.02)\n"
)
_WORKER = f"""
(while :; do printf . >> child.ticks; sleep 0.02; done) &
( "{sys.executable}" service.py & echo $! > service.pid )
while [ ! -s service.pid ]; do sleep 0.01; done
touch ready
while :; do printf . >> worker.ticks; sleep 0.02; done
"""


def _ticks(tmp_path: Path) -> dict[str, int]:
    return {
        name: len((tmp_path / f"{name}.ticks").read_text()) if (tmp_path / f"{name}.ticks").exists() else 0
        for name in ("worker", "child", "service")
    }


@linux_only
def test_pause_freezes_the_harness_and_a_detached_service_and_continue_runs_them_again(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(_SERVICE)
    process = _start(tmp_path, "bash", "-c", _WORKER)
    service = None
    try:
        _wait_for(lambda: (tmp_path / "ready").exists())
        service = int((tmp_path / "service.pid").read_text())
        _wait_for(lambda: all(_ticks(tmp_path).values()))
        # Re-armable: every checkpoint pauses and continues with a later number.
        for seq in (1, 2):
            _request(tmp_path / PAUSE_REQUEST_FILE, seq)
            paused = _acknowledgement(tmp_path / PAUSED_FILE, seq)
            assert paused["confirmed"] is True and paused["error"] is None, paused
            assert service in paused["pids"]
            assert _state(service) in ("T", "t")
            frozen = _ticks(tmp_path)
            time.sleep(0.3)
            assert _ticks(tmp_path) == frozen
            _request(tmp_path / CONTINUE_REQUEST_FILE, seq)
            continued = _acknowledgement(tmp_path / CONTINUED_FILE, seq)
            assert continued["confirmed"] is True, continued
            _wait_for(lambda: all(after > before for after, before in zip(_ticks(tmp_path).values(), frozen.values())))
        (tmp_path / "stop.request").touch()
        process.wait(timeout=15)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt["cleanup_confirmed"] is True and receipt["stopped"] is True, receipt
        # A stop leaves services that left the harness before it began to the sandbox owner, as before.
        assert _state(service) not in ("T", "t", "gone")
    finally:
        _finish(process)
        if service is not None:
            try:
                os.kill(service, signal.SIGKILL)
            except ProcessLookupError:
                pass


@linux_only
def test_a_pause_before_the_launch_holds_it_until_continued(tmp_path: Path) -> None:
    _request(tmp_path / PAUSE_REQUEST_FILE, 1)
    process = _start(tmp_path, "touch", "started")
    try:
        paused = _acknowledgement(tmp_path / PAUSED_FILE, 1)
        assert paused == {"seq": 1, "confirmed": True, "pids": [], "error": None}
        time.sleep(0.3)
        assert not (tmp_path / "started").exists()
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        process.wait(timeout=15)
        assert (tmp_path / "started").exists()
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt == {
            "cleanup_confirmed": True,
            "return_code": 0,
            "error": None,
            "timed_out": False,
            "stopped": False,
        }
    finally:
        _finish(process)


@linux_only
def test_a_continued_pause_written_before_the_launch_is_acknowledged_and_not_served(tmp_path: Path) -> None:
    _request(tmp_path / PAUSE_REQUEST_FILE, 3)
    _request(tmp_path / CONTINUE_REQUEST_FILE, 3)
    process = _start(tmp_path, "touch", "started")
    try:
        process.wait(timeout=15)
        assert (tmp_path / "started").exists()
        assert not (tmp_path / PAUSED_FILE).exists()
        assert json.loads((tmp_path / CONTINUED_FILE).read_text())["seq"] == 3
    finally:
        _finish(process)


@linux_only
def test_time_spent_paused_does_not_count_against_the_deadline(tmp_path: Path) -> None:
    # The harness needs a fraction of its two second budget; it spends more than the whole budget paused.
    process = _start(tmp_path, "bash", "-c", "touch ready; sleep 0.2; touch finished", timeout="2")
    try:
        _wait_for(lambda: (tmp_path / "ready").exists())
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["confirmed"] is True
        time.sleep(2.5)
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        process.wait(timeout=15)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt["timed_out"] is False and receipt["return_code"] == 0, receipt
        assert (tmp_path / "finished").exists()
    finally:
        _finish(process)


@linux_only
def test_a_hold_before_the_launch_is_not_added_to_the_deadline_again_by_a_later_pause(tmp_path: Path) -> None:
    _request(tmp_path / PAUSE_REQUEST_FILE, 1)
    process = _start(tmp_path, "bash", "-c", "touch ready; sleep 60", timeout="2")
    try:
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["confirmed"] is True
        # Held for longer than the whole budget, which starts at the launch.
        time.sleep(3)
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        _wait_for(lambda: (tmp_path / "ready").exists())
        launched = time.monotonic()
        _request(tmp_path / PAUSE_REQUEST_FILE, 2)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 2)["confirmed"] is True
        _request(tmp_path / CONTINUE_REQUEST_FILE, 2)
        process.wait(timeout=30)
        elapsed = time.monotonic() - launched
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt["timed_out"] is True and receipt["stopped"] is False, receipt
        # Two seconds of budget plus the short pause; the three seconds of hold would make it five.
        assert elapsed < 4.5, elapsed
    finally:
        _finish(process)


@linux_only
def test_an_expire_during_a_hold_before_the_launch_launches_nothing(tmp_path: Path) -> None:
    _request(tmp_path / PAUSE_REQUEST_FILE, 1)
    process = _start(tmp_path, "touch", "started")
    try:
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["confirmed"] is True
        (tmp_path / EXPIRE_REQUEST_FILE).touch()
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        process.wait(timeout=15)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt == {
            "cleanup_confirmed": True,
            "return_code": None,
            "error": None,
            "timed_out": True,
            "stopped": False,
        }
        assert not (tmp_path / "started").exists()
    finally:
        _finish(process)


_TICKER = "touch ready; while :; do printf . >> ticks; sleep 0.02; done"


def _grown(tmp_path: Path, by: int = 5) -> None:
    before = len((tmp_path / "ticks").read_text())
    _wait_for(lambda: len((tmp_path / "ticks").read_text()) >= before + by)


@linux_only
def test_a_malformed_or_earlier_pause_number_is_not_served(tmp_path: Path) -> None:
    process = _start(tmp_path, "bash", "-c", _TICKER)
    try:
        _wait_for(lambda: (tmp_path / "ticks").exists())
        (tmp_path / PAUSE_REQUEST_FILE).write_text("not-a-number")
        _grown(tmp_path)
        assert not (tmp_path / PAUSED_FILE).exists()
        _request(tmp_path / PAUSE_REQUEST_FILE, 2)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 2)["confirmed"] is True
        _request(tmp_path / CONTINUE_REQUEST_FILE, 2)
        assert _acknowledgement(tmp_path / CONTINUED_FILE, 2)["confirmed"] is True
        # An earlier number, for example a request delivered late, pauses nothing.
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        _grown(tmp_path)
        assert json.loads((tmp_path / PAUSED_FILE).read_text())["seq"] == 2
    finally:
        _finish(process)


@linux_only
def test_an_expire_while_paused_takes_effect_once_the_pause_is_continued(tmp_path: Path) -> None:
    worker = "trap 'echo saved > checkpoint; exit 0' TERM; touch ready; while :; do sleep 0.05; done"
    process = _start(tmp_path, "bash", "-c", worker)
    try:
        _wait_for(lambda: (tmp_path / "ready").exists())
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["confirmed"] is True
        (tmp_path / EXPIRE_REQUEST_FILE).touch()
        time.sleep(0.3)
        assert process.poll() is None and not (tmp_path / CLEANUP_RECEIPT_FILE).exists()
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        process.wait(timeout=15)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt["timed_out"] is True and receipt["stopped"] is False and receipt["return_code"] == 0, receipt
        assert (tmp_path / "checkpoint").read_text() == "saved\n"
    finally:
        _finish(process)


@linux_only
def test_a_stop_while_paused_continues_the_harness_so_it_can_save_on_term(tmp_path: Path) -> None:
    worker = "trap 'echo saved > checkpoint; exit 0' TERM; touch ready; while :; do sleep 0.05; done"
    process = _start(tmp_path, "bash", "-c", worker)
    try:
        _wait_for(lambda: (tmp_path / "ready").exists())
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["confirmed"] is True
        (tmp_path / "stop.request").touch()
        process.wait(timeout=15)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt == {
            "cleanup_confirmed": True,
            "return_code": 0,
            "error": None,
            "timed_out": False,
            "stopped": True,
        }
        assert (tmp_path / "checkpoint").read_text() == "saved\n"
    finally:
        _finish(process)


@linux_only
def test_a_deadline_is_not_a_requested_stop(tmp_path: Path) -> None:
    process = _start(tmp_path, "sleep", "30", timeout="0.3")
    try:
        process.wait(timeout=15)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt["timed_out"] is True and receipt["stopped"] is False, receipt
    finally:
        _finish(process)


def test_pause_options_go_together(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SUPERVISOR_SCRIPT), "--timeout", "1", "--cleanup-timeout", "1", "--receipt", "r"]
        + ["--stop-file", "s", "--pause-file", "p", "--", "true"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2 and "go together" in result.stderr


# -- the controller side, against a local sandbox ------------------------------------------------------------------


class LocalSandbox:
    """Runs sandbox commands with sh on this host, as the supervisor runs in a task sandbox."""

    sandbox_id = "local"

    async def upload(self, source, destination):
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        shutil.copyfile(source, destination)

    async def exec(self, command, *, cwd=None, timeout_s=30, **kwargs):
        process = await asyncio.create_subprocess_exec(
            "sh", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)

    async def disconnect(self, *, timeout_s=None):
        pass


def test_receipts_without_stopped_still_parse_and_stopped_is_kept() -> None:
    old = {"cleanup_confirmed": True, "return_code": 0, "error": None, "timed_out": False}
    assert parse_cleanup_receipt(old) == old
    assert parse_cleanup_receipt(old | {"stopped": True}) == old | {"stopped": True}


def test_a_receipt_with_malformed_diagnostics_keeps_only_a_boolean_stopped() -> None:
    malformed = {"cleanup_confirmed": True, "return_code": "0", "error": None, "timed_out": False}
    assert parse_cleanup_receipt(malformed | {"stopped": True}) == {
        "cleanup_confirmed": True,
        "return_code": None,
        "error": None,
        "timed_out": False,
        "stopped": True,
    }
    assert "stopped" not in parse_cleanup_receipt(malformed | {"stopped": "yes"})
    assert "stopped" not in parse_cleanup_receipt(malformed)


def test_the_provider_deadline_covers_the_pause_allowance() -> None:
    assert (
        exec_timeout(timeout=10, cleanup_timeout=1, pause_allowance=600)
        == exec_timeout(timeout=10, cleanup_timeout=1) + 600
    )


async def test_a_pause_is_answered_by_the_receipt_of_a_harness_that_already_ended(tmp_path: Path) -> None:
    (tmp_path / CLEANUP_RECEIPT_FILE).write_text(json.dumps({"cleanup_confirmed": True}))
    assert await request_pause(LocalSandbox(), session_dir=str(tmp_path), workdir=None, seq=1, timeout=1) == (
        "finished",
        None,
    )
    assert not (tmp_path / PAUSE_REQUEST_FILE).exists()


async def test_a_pause_before_any_launch_claim_is_left_for_the_launch(tmp_path: Path) -> None:
    sandbox = LocalSandbox()
    assert await request_pause(sandbox, session_dir=str(tmp_path), workdir=None, seq=4, timeout=1) == (
        "not_launched",
        None,
    )
    assert (tmp_path / PAUSE_REQUEST_FILE).read_text() == "4"
    assert await request_continue(sandbox, session_dir=str(tmp_path), workdir=None, seq=4, timeout=1) == (
        "not_launched",
        None,
    )
    # A missing session directory is not created.
    missing = tmp_path / "missing"
    assert (await request_pause(sandbox, session_dir=str(missing), workdir=None, seq=1, timeout=1))[
        0
    ] == "not_launched"
    assert not missing.exists()


@linux_only
async def test_a_request_the_supervisor_never_answers_times_out(tmp_path: Path) -> None:
    (tmp_path / "launch.claim").symlink_to("launch")
    outcome, acknowledgement = await request_pause(
        LocalSandbox(), session_dir=str(tmp_path), workdir=None, seq=1, timeout=0.3
    )
    assert (outcome, acknowledgement) == ("timeout", None)


def _ticking_stage(session_dir: Path):
    async def stage() -> SandboxCommand:
        session_dir.mkdir(parents=True, exist_ok=True)
        # Ticks until the test lets it end, so a pause that is slow on a busy node never races the harness's exit.
        worker = "touch ready; until [ -e finish ]; do printf . >> ticks; sleep 0.02; done"
        return SandboxCommand(argv=["bash", "-c", f"cd {session_dir} && {worker}"])

    return stage


async def _ticks_grow(launch_dir: Path, since: str) -> None:
    async with asyncio.timeout(15):
        while (launch_dir / "ticks").read_text() == since:
            await asyncio.sleep(0.02)


async def _end_launch(running: asyncio.Task, launch_dir: Path) -> None:
    """End the launch of a test that failed half way; a stop while paused continues the processes first."""
    if not running.done():
        launch_dir.mkdir(parents=True, exist_ok=True)
        (launch_dir / STOP_REQUEST_FILE).touch()
        (launch_dir / "finish").touch()
        await asyncio.wait({running}, timeout=20)
        running.cancel()
    await asyncio.gather(running, return_exceptions=True)


def _session(tmp_path: Path, **kwargs) -> SandboxSession:
    return SandboxSession(
        sandbox=LocalSandbox(),  # type: ignore[arg-type]
        session_dir=str(tmp_path / "launch-1"),
        workdir=str(tmp_path),
        harness="test",
        pausable=True,
        **kwargs,
    )


@linux_only
async def test_a_session_pauses_and_continues_its_running_harness(tmp_path: Path) -> None:
    session = _session(tmp_path)
    launch_dir = tmp_path / "launch-1"
    running = asyncio.create_task(
        session.execute(
            stage_activation=_ticking_stage(launch_dir),
            collect=AsyncMock(return_value="artifacts"),
            timeout=30,
            close_timeout=3,
        )
    )
    try:
        async with asyncio.timeout(15):
            while not (launch_dir / "ticks").exists():
                await asyncio.sleep(0.02)
        paused = await session.pause_harness(timeout=10)
        assert paused.confirmed and paused.pids, paused
        frozen = (launch_dir / "ticks").read_text()
        await asyncio.sleep(0.3)
        assert (launch_dir / "ticks").read_text() == frozen
        assert (await session.continue_harness(timeout=10)).confirmed
        # Continuing twice is answered at once.
        assert (await session.continue_harness(timeout=10)) == HarnessPauseResult(status="confirmed")
        await _ticks_grow(launch_dir, frozen)
        (launch_dir / "finish").touch()
        assert await asyncio.wait_for(running, 30) == "artifacts"
        assert session.cleanup is not None and session.cleanup["stopped"] is False
        # The receipt answers a pause, and a continue, of a harness that has ended.
        assert (await session.pause_harness(timeout=1)).status == "finished"
        assert (await session.continue_harness(timeout=1)).status == "finished"
    finally:
        await _end_launch(running, launch_dir)


@linux_only
async def test_a_session_paused_before_its_launch_holds_the_launch_until_continued(tmp_path: Path) -> None:
    session = _session(tmp_path)
    launch_dir = tmp_path / "launch-1"
    assert (await session.pause_harness(timeout=1)).confirmed
    running = asyncio.create_task(
        session.execute(
            stage_activation=_ticking_stage(launch_dir),
            collect=AsyncMock(return_value="artifacts"),
            timeout=30,
            close_timeout=3,
        )
    )
    try:
        async with asyncio.timeout(15):
            while not (launch_dir / PAUSED_FILE).exists():
                await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        assert not (launch_dir / "ready").exists()
        assert (await session.continue_harness(timeout=10)).confirmed
        (launch_dir / "finish").touch()
        assert await asyncio.wait_for(running, 30) == "artifacts"
        assert (launch_dir / "ready").exists()
    finally:
        await _end_launch(running, launch_dir)


async def test_a_session_that_was_not_launched_pausable_refuses_to_pause(tmp_path: Path) -> None:
    session = SandboxSession(sandbox=LocalSandbox(), session_dir=str(tmp_path), workdir=None, harness="test")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="pausable"):
        await session.pause_harness()


async def test_execute_adds_the_pause_allowance_to_the_provider_deadline() -> None:
    timeouts = []
    for allowance in (0.0, 900.0):
        sandbox = AsyncMock(spec=AsyncSandbox)
        sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
        session = SandboxSession(sandbox=sandbox, session_dir="/s", workdir="/w", harness="test", pausable=True)
        session.stop_harness = AsyncMock()  # type: ignore[method-assign]
        await session.execute(
            stage_activation=AsyncMock(return_value=SandboxCommand(argv=["worker"])),
            collect=AsyncMock(return_value=None),
            timeout=10,
            close_timeout=3,
            pause_allowance_s=allowance,
        )
        [launch] = [call for call in sandbox.exec.await_args_list if "process_supervisor.sh" in call.args[0]]
        assert "--pause-file" in launch.args[0]
        timeouts.append(launch.kwargs["timeout_s"])
    assert timeouts[1] - timeouts[0] == 900.0


async def test_reattach_rebinds_a_restored_sandbox_without_the_dead_launch(tmp_path: Path) -> None:
    session = _session(tmp_path)
    await session.begin_activation(session_dir=str(tmp_path / "launch-2"), timeout=1)
    session.launch_started = True
    state = json.loads(json.dumps(session.checkpoint_state()))
    assert state == {
        "session_root": str(tmp_path / "launch-1"),
        "session_dir": str(tmp_path / "launch-2"),
        "launch_started": True,
    }
    restored_sandbox = LocalSandbox()
    reattached = SandboxSession.reattach(restored_sandbox, state, workdir=str(tmp_path), harness="test", pausable=True)  # type: ignore[arg-type]
    assert reattached.sandbox is restored_sandbox
    assert reattached.root_dir == state["session_root"] and reattached.session_dir == state["session_dir"]
    assert reattached.launch_started is False and reattached.pausable
    with pytest.raises(ValueError):
        SandboxSession.reattach(restored_sandbox, {"session_dir": "x"}, workdir=None, harness="test")  # type: ignore[arg-type]
    # Nothing ran in the restored sandbox, so its close only removes the session's files.
    (tmp_path / "launch-1").mkdir()
    await reattached.close(timeout=5)
    assert reattached.closed and not (tmp_path / "launch-1").exists()


# -- run_until_parked -----------------------------------------------------------------------------------------------


class FakeActivation:
    def __init__(self) -> None:
        self.park = asyncio.Event()
        self.resume = asyncio.Event()
        self.parked = asyncio.Event()
        self.boundaries: list[dict] = []
        self.retired = False

    @property
    def park_requested(self) -> bool:
        return self.park.is_set()

    async def wait_park_requested(self) -> None:
        await self.park.wait()
        if self.retired:
            raise StaleAttemptError("retired")

    async def boundary(self, snapshot) -> None:
        self.boundaries.append(snapshot())
        self.parked.set()
        await self.resume.wait()
        self.park.clear()
        self.resume.clear()
        self.parked.clear()

    def queued(self):  # pragma: no cover - not used by run_until_parked
        raise NotImplementedError


class FakeSession:
    harness = "fake"
    # Its harness is running: a stop ends it.
    launch_started = True

    def __init__(self, *, pauses=(), continues=(), stopped_receipt: bool = True) -> None:
        self.events: list[str] = []
        self.cleanup = None
        self.done = asyncio.Event()
        self._pauses = list(pauses)
        self._continues = list(continues)
        self._stopped_receipt = stopped_receipt

    async def launch(self) -> str:
        try:
            await self.done.wait()
        except asyncio.CancelledError:
            self.events.append("launch_cancelled")
            raise
        return "result"

    async def pause_harness(self) -> HarnessPauseResult:
        self.events.append("pause")
        result = self._pauses.pop(0) if self._pauses else HarnessPauseResult(status="confirmed")
        if isinstance(result, Exception):
            raise result
        return result

    async def continue_harness(self) -> HarnessPauseResult:
        self.events.append("continue")
        result = self._continues.pop(0) if self._continues else HarnessPauseResult(status="confirmed")
        if isinstance(result, Exception):
            raise result
        return result

    async def stop_harness(self, *, timeout: float) -> None:
        self.events.append("stop")
        self.cleanup = {"cleanup_confirmed": True, "return_code": 143, "error": None, "timed_out": False}
        self.cleanup["stopped"] = self._stopped_receipt
        self.done.set()


def _run(session: FakeSession, activation: FakeActivation) -> asyncio.Task:
    return asyncio.create_task(
        run_until_parked(
            session,  # type: ignore[arg-type]
            activation,
            launch=session.launch,
            on_parked=lambda: {"bookmark": len(session.events)},
            stop_timeout=5,
        )
    )


async def test_parks_in_place_at_each_checkpoint_and_finishes_the_same_launch() -> None:
    session, activation = FakeSession(), FakeActivation()
    running = _run(session, activation)
    for _ in range(2):
        activation.park.set()
        await asyncio.wait_for(activation.parked.wait(), 5)
        assert session.events[-1] == "pause"
        activation.resume.set()
        async with asyncio.timeout(5):
            while session.events[-1] != "continue":
                await asyncio.sleep(0)
    session.done.set()
    assert await asyncio.wait_for(running, 5) == Finished("result")
    assert session.events == ["pause", "continue", "pause", "continue"]
    assert activation.boundaries == [{"bookmark": 1}, {"bookmark": 3}]


async def test_an_unconfirmed_pause_stops_the_harness_parks_and_asks_for_a_relaunch() -> None:
    session = FakeSession(pauses=[HarnessPauseResult(status="unconfirmed", error="D state")])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    # The bookmark is taken after the stop, so it describes the stopped launch.
    assert activation.boundaries == [{"bookmark": 2}]
    activation.resume.set()
    result = await asyncio.wait_for(running, 5)
    assert isinstance(result, Stopped) and result.receipt["stopped"] is True
    assert session.events == ["pause", "stop"]


async def test_a_pause_request_that_fails_in_transport_falls_back_to_a_stop() -> None:
    session = FakeSession(pauses=[RuntimeError("sandbox exec failed")])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    assert isinstance(await asyncio.wait_for(running, 5), Stopped)
    assert session.events == ["pause", "stop"]


async def test_a_harness_that_ends_on_its_own_as_the_stop_arrives_is_finished() -> None:
    session = FakeSession(pauses=[HarnessPauseResult(status="unconfirmed")], stopped_receipt=False)
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    assert await asyncio.wait_for(running, 5) == Finished("result")
    assert activation.boundaries == []


async def test_a_continue_that_keeps_failing_stops_the_harness_after_the_checkpoint() -> None:
    session = FakeSession(continues=[HarnessPauseResult(status="unconfirmed")] * CONTINUE_ATTEMPTS)
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    result = await asyncio.wait_for(running, 5)
    assert isinstance(result, Stopped)
    assert session.events == ["pause", *["continue"] * CONTINUE_ATTEMPTS, "stop"]
    assert len(activation.boundaries) == 1


async def test_a_harness_that_finished_as_the_park_arrived_keeps_its_result() -> None:
    session = FakeSession(pauses=[HarnessPauseResult(status="finished")])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    async with asyncio.timeout(5):
        while session.events != ["pause"]:
            await asyncio.sleep(0)
    session.done.set()
    assert await asyncio.wait_for(running, 5) == Finished("result")
    assert session.events == ["pause"] and activation.boundaries == []


async def test_a_launch_that_ends_first_never_pauses() -> None:
    session, activation = FakeSession(), FakeActivation()
    session.done.set()
    assert await asyncio.wait_for(_run(session, activation), 5) == Finished("result")
    assert session.events == []


async def test_cancelling_a_parked_activation_stops_its_launch() -> None:
    session, activation = FakeSession(), FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 10)
    assert session.events == ["pause", "launch_cancelled"]


async def test_a_retired_session_stops_its_launch_and_raises() -> None:
    session, activation = FakeSession(), FakeActivation()
    activation.retired = True
    running = _run(session, activation)
    activation.park.set()
    with pytest.raises(StaleAttemptError):
        await asyncio.wait_for(running, 5)
    assert session.events == ["launch_cancelled"]


class StagingSession(FakeSession):
    """Its launch has not started the harness yet: a stop has nothing to end."""

    launch_started = False

    async def stop_harness(self, *, timeout: float) -> None:
        self.events.append("stop")


async def test_a_failed_pause_of_a_launch_still_staging_cancels_it_and_asks_for_a_relaunch() -> None:
    session, activation = StagingSession(pauses=[RuntimeError("sandbox exec failed")]), FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    # Left alone, the launch would run its harness through the checkpoint, or wait for a continue nobody sends.
    assert await asyncio.wait_for(running, 5) == Stopped(receipt=None)
    assert session.events == ["pause", "stop", "launch_cancelled"]


class DeadSandboxSession(FakeSession):
    """Its sandbox died: the stop finds nothing to ask and leaves no receipt, and the launch fails."""

    async def launch(self) -> str:
        await self.done.wait()
        raise RuntimeError("sandbox OOM-killed")

    async def stop_harness(self, *, timeout: float) -> None:
        self.events.append("stop")
        self.done.set()


async def test_a_launch_whose_sandbox_died_raises_its_own_error_and_does_not_park() -> None:
    session, activation = DeadSandboxSession(pauses=[RuntimeError("exec failed")]), FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    with pytest.raises(RuntimeError, match="OOM-killed"):
        await asyncio.wait_for(running, 5)
    assert activation.boundaries == []


async def test_a_continue_request_that_fails_in_transport_is_tried_again() -> None:
    session = FakeSession(continues=[RuntimeError("sandbox exec failed"), HarnessPauseResult(status="finished")])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    async with asyncio.timeout(5):
        while session.events != ["pause", "continue", "continue"]:
            await asyncio.sleep(0)
    session.done.set()
    assert await asyncio.wait_for(running, 5) == Finished("result")
    assert "stop" not in session.events


class FailingLaunchSession(FakeSession):
    """Its launch raises once it ends, as an execute whose capture failed does."""

    async def launch(self) -> str:
        await self.done.wait()
        raise RuntimeError("capture failed")


async def test_a_stopped_launch_that_raised_parks_and_reports_its_error() -> None:
    session = FailingLaunchSession(pauses=[HarnessPauseResult(status="unconfirmed", error="D state")])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    result = await asyncio.wait_for(running, 5)
    assert isinstance(result, Stopped) and result.receipt["stopped"] is True
    assert isinstance(result.error, RuntimeError) and str(result.error) == "capture failed"


async def test_an_error_of_a_launch_that_ended_on_its_own_as_the_stop_arrived_propagates() -> None:
    session = FailingLaunchSession(pauses=[HarnessPauseResult(status="unconfirmed")], stopped_receipt=False)
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    with pytest.raises(RuntimeError, match="capture failed"):
        await asyncio.wait_for(running, 5)
    assert activation.boundaries == [] and session.events == ["pause", "stop"]


async def test_a_park_that_ends_in_a_retired_session_stops_the_paused_launch() -> None:
    class Retiring(FakeActivation):
        async def boundary(self, snapshot) -> None:
            raise StaleAttemptError("retired while parked")

    session, activation = FakeSession(), Retiring()
    running = _run(session, activation)
    activation.park.set()
    with pytest.raises(StaleAttemptError):
        await asyncio.wait_for(running, 5)
    assert session.events == ["pause", "launch_cancelled"]


async def test_a_pause_keeps_its_number_until_a_continue_is_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusal = "A process the pause reached did not stop"
    answers = {
        "pause": [
            ("timeout", None),
            ("acknowledged", {"seq": 1, "confirmed": False, "pids": [7], "error": refusal}),
            ("acknowledged", {"seq": 1, "confirmed": True, "pids": [7], "error": None}),
            ("acknowledged", {"seq": 2, "confirmed": True, "pids": [7], "error": None}),
        ],
        "continue": [("timeout", None), ("acknowledged", {"seq": 1, "confirmed": True, "pids": [], "error": None})],
    }
    requests: list[tuple[str, int]] = []

    def answering(kind: str):
        async def request(sandbox, *, session_dir, workdir, seq, timeout):
            requests.append((kind, seq))
            return answers[kind].pop(0)

        return request

    # sandbox_session imports both by name and calls them.
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_pause", answering("pause"))
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_continue", answering("continue"))
    session = _session(tmp_path)

    assert (await session.pause_harness()).status == "unconfirmed"
    refused = await session.pause_harness()
    assert refused.status == "unconfirmed" and refused.error == refusal
    assert (await session.pause_harness()).confirmed
    assert (await session.continue_harness()).status == "unconfirmed"
    assert (await session.continue_harness()).confirmed
    assert (await session.pause_harness()).confirmed
    assert requests == [("pause", 1)] * 3 + [("continue", 1)] * 2 + [("pause", 2)]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (SandboxExecResult(stdout="", stderr="disk full", return_code=1), "disk full"),
        (SandboxExecResult(stdout=None, stderr=None, return_code=-1, error_type="timeout"), "timeout"),
        (SandboxExecResult(stdout="acknowledged\n", stderr="", return_code=0), "malformed supervisor acknowledgement"),
        (
            SandboxExecResult(stdout='acknowledged\n{"seq": "1", "confirmed": true}\n', stderr="", return_code=0),
            "malformed supervisor acknowledgement",
        ),
        (SandboxExecResult(stdout="paused\n", stderr="", return_code=0), "unexpected supervisor request outcome"),
    ],
    ids=["exit", "provider_error", "no_acknowledgement", "wrong_types", "unknown"],
)
async def test_a_request_the_sandbox_answered_wrongly_is_an_error(result: SandboxExecResult, message: str) -> None:
    sandbox = AsyncMock(spec=AsyncSandbox)
    sandbox.exec.return_value = result
    with pytest.raises(RuntimeError, match=re.escape(message)):
        await request_pause(sandbox, session_dir="/s", workdir=None, seq=1, timeout=1)


@linux_only
async def test_run_until_parked_pauses_a_real_supervised_harness(tmp_path: Path) -> None:
    session, activation = _session(tmp_path), FakeActivation()
    launch_dir = tmp_path / "launch-1"
    frozen: list[str] = []

    def bookmark() -> dict:
        frozen.append((launch_dir / "ticks").read_text())
        return {"ticks": len(frozen[-1])}

    running = asyncio.create_task(
        run_until_parked(
            session,
            activation,
            launch=lambda: session.execute(
                stage_activation=_ticking_stage(launch_dir),
                collect=AsyncMock(return_value="artifacts"),
                timeout=30,
                close_timeout=3,
                pause_allowance_s=60,
            ),
            on_parked=bookmark,
            stop_timeout=10,
        )
    )
    try:
        async with asyncio.timeout(15):
            while not (launch_dir / "ticks").exists():
                await asyncio.sleep(0.02)
        activation.park.set()
        await asyncio.wait_for(activation.parked.wait(), 15)
        await asyncio.sleep(0.3)
        assert (launch_dir / "ticks").read_text() == frozen[0]
        activation.resume.set()
        await _ticks_grow(launch_dir, frozen[0])
        (launch_dir / "finish").touch()
        assert await asyncio.wait_for(running, 30) == Finished("artifacts")
    finally:
        await _end_launch(running, launch_dir)


class ScopedLocalSandbox(LocalSandbox):
    """A local sandbox that records the provider polling scopes a session opens around its commands."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def _scope(self, name: str):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            self.events.append(f"enter {name}")
            try:
                yield
            finally:
                self.events.append(f"exit {name}")

        return scope()

    def parked_exec_polling(self):
        return self._scope("parked")

    def checkpoint_exec_polling(self):
        return self._scope("checkpoint")

    async def exec(self, command, **kwargs):
        # The request writes a temporary file it renames; the launch only names continue.request.
        if "continue.request." in command and ".tmp" in command:
            self.events.append("continue request")
        return await super().exec(command, **kwargs)


@linux_only
async def test_a_paused_harness_s_command_polls_are_held_until_continue_and_a_stop_polls_promptly(
    tmp_path: Path,
) -> None:
    sandbox = ScopedLocalSandbox()
    session = SandboxSession(
        sandbox=sandbox, session_dir=str(tmp_path / "launch-1"), workdir=str(tmp_path), harness="test", pausable=True
    )
    launch_dir = tmp_path / "launch-1"
    running = asyncio.create_task(
        session.execute(
            stage_activation=_ticking_stage(launch_dir),
            collect=AsyncMock(return_value=None),
            timeout=30,
            close_timeout=3,
        )
    )
    try:
        async with asyncio.timeout(15):
            while not (launch_dir / "ticks").exists():
                await asyncio.sleep(0.02)
        assert (await session.pause_harness(timeout=10)).confirmed
        held = list(sandbox.events)
        assert (await session.continue_harness(timeout=10)).confirmed
        continued = list(sandbox.events)
        assert (await session.pause_harness(timeout=10)).confirmed
        before_stop = len(sandbox.events)
        await session.stop_harness(timeout=10)
        stopping = sandbox.events[before_stop:]
        await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 30)
    finally:
        await _end_launch(running, launch_dir)

    # The polls are held from the confirmed pause, and asked again before the continue request runs in the sandbox.
    assert held == ["enter parked"]
    assert continued[:3] == ["enter parked", "exit parked", "continue request"]
    # A stop releases a held park first and polls promptly while it waits for the launch to end. The launch's own
    # finish can stop it at the same time; the provider's scopes nest.
    assert stopping[:2] == ["exit parked", "enter checkpoint"], sandbox.events
    assert set(stopping[2:]) <= {"enter checkpoint", "exit checkpoint"}, sandbox.events
    assert stopping.count("enter checkpoint") == stopping.count("exit checkpoint"), sandbox.events
