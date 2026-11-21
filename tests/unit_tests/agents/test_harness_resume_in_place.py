# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Taking over, or abandoning, a supervised launch that a restore of a process-keeping snapshot brought back.

A restore of such a snapshot is simulated on this host: a supervisor launched outside any session plays the launch
the crashed agent server started, and a fresh session reattaches to its files.
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents.utils.run_with_checkpoint_pauses import LaunchFinished, run_with_checkpoint_pauses
from nemo_gym.agents.utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.agents.utils.supervisor_client import (
    CLEANUP_RECEIPT_FILE,
    CONTINUE_REQUEST_FILE,
    CONTINUED_FILE,
    PAUSED_FILE,
    SUPERVISOR_FILE,
    SUPERVISOR_SCRIPT,
    NotResumable,
    RestoredLaunch,
    abandon_launch,
    check_restored,
    is_harness_termination_failure,
    request_continue,
    request_pause,
    supervised_launch_command,
)
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.providers.base import SandboxExecResult
from tests.unit_tests.agents.test_harness_pause import FakeActivation, LocalSandbox, _wait_for


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
pytestmark = linux_only


def _start_time(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state not in ("Z", "X")


def _dead_pid() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


# A worker that ticks until a file named finish exists, starts a ticking child, and a service that leaves the harness (only the
# supervisor's marker reaches it); it records the pids, and that it was continued if a CONT ever reaches it.
_SERVICE = (
    "import os, time\nos.setsid()\nwhile True:\n    open('service.ticks', 'a').write('.')\n    time.sleep(0.02)\n"
)
_WORKER = f"""
trap 'touch continued' CONT
(while :; do printf . >> child.ticks; sleep 0.02; done) & echo $! > child.pid
( "{sys.executable}" service.py & echo $! > service.pid )
echo $$ > worker.pid
touch ready
until [ -e finish ]; do printf . >> worker.ticks; sleep 0.02; done
touch finished
"""


class Launch:
    """A supervised launch started the way an agent server starts one, then left alone, as a crash leaves it."""

    def __init__(self, session_dir: Path, *, pause_mode: str = "hold") -> None:
        session_dir.mkdir(parents=True)
        (session_dir / "service.py").write_text(_SERVICE)
        shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
        command = supervised_launch_command(
            session_dir=str(session_dir),
            command=["bash", "-c", f"cd {session_dir} && {_WORKER}"],
            timeout=60,
            cleanup_timeout=1,
            pausable=True,
            pause_mode=pause_mode,  # type: ignore[arg-type]
        )
        self.session_dir = session_dir
        self.process = subprocess.Popen(
            ["sh", "-c", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
        )
        try:
            _wait_for(lambda: (session_dir / "ready").exists() and (session_dir / "service.pid").exists())
        except BaseException:
            # The caller never got the launch, so its fixture cannot clean it up.
            self.cleanup()
            raise

    def finish(self) -> None:
        """Let the worker end by itself."""
        (self.session_dir / "finish").touch()

    def pids(self) -> dict[str, int]:
        return {name: int((self.session_dir / f"{name}.pid").read_text()) for name in ("worker", "child", "service")}

    def ticks(self, name: str) -> int:
        path = self.session_dir / f"{name}.ticks"
        return len(path.read_text()) if path.exists() else 0

    async def pause(self, seq: int = 1) -> dict:
        outcome, acknowledgement = await request_pause(
            LocalSandbox(), session_dir=str(self.session_dir), workdir=None, seq=seq, timeout=15
        )
        assert outcome == "acknowledged" and acknowledgement is not None and acknowledgement["confirmed"]
        return dict(acknowledgement)

    def cleanup(self) -> None:
        """Kill whatever a failed test left running."""
        pids = []
        try:
            pids = list(self.pids().values())
        except (FileNotFoundError, ValueError):
            pass
        for pid in [self.process.pid, *pids]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait(timeout=10)


@pytest.fixture
def launches():
    started: list[Launch] = []
    yield started
    for launch in started:
        launch.cleanup()


async def _end(task: asyncio.Task) -> None:
    """Cancel what a failed test left waiting; a cancelled adopt stops its harness."""
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _continued(launch: Launch, seq: int) -> None:
    """Wait until the launch's supervisor acknowledged continue ``seq``."""
    path = launch.session_dir / CONTINUED_FILE
    async with asyncio.timeout(15):
        while not path.exists() or json.loads(path.read_text())["seq"] != seq:
            await asyncio.sleep(0.02)


def _abandon(identity: str, receipt: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUPERVISOR_SCRIPT), "--abandon", identity, "--receipt", str(receipt), "--cleanup-timeout", "5"],
        capture_output=True,
        text=True,
        timeout=30,
    )


ABANDONED = {
    "cleanup_confirmed": True,
    "return_code": None,
    "error": None,
    "timed_out": False,
    "stopped": True,
    "abandoned": True,
}


# -- abandon ---------------------------------------------------------------------------------------------------------


async def test_abandon_kills_a_held_launch_and_the_service_it_marked(tmp_path: Path, launches: list[Launch]) -> None:
    launch = Launch(tmp_path / "launch-1")
    launches.append(launch)
    identity = (await launch.pause())["identity"]
    pids = launch.pids()

    result = _abandon(identity, tmp_path / "launch-1" / CLEANUP_RECEIPT_FILE)

    assert result.returncode == 0, result.stderr
    assert not any(_alive(pid) for pid in [*pids.values(), int(identity.split(".")[0])])
    launch.process.wait(timeout=10)
    assert json.loads((tmp_path / "launch-1" / CLEANUP_RECEIPT_FILE).read_text()) == ABANDONED


async def test_abandon_kills_stopped_processes_without_continuing_them(tmp_path: Path, launches: list[Launch]) -> None:
    launch = Launch(tmp_path / "launch-1", pause_mode="stop")
    launches.append(launch)
    paused = await launch.pause()
    assert set(launch.pids().values()) <= set(paused["pids"])
    frozen = launch.ticks("worker")

    result = _abandon(paused["identity"], tmp_path / "launch-1" / CLEANUP_RECEIPT_FILE)

    assert result.returncode == 0, result.stderr
    assert not any(_alive(pid) for pid in launch.pids().values())
    # A CONT would have run the worker's trap, and let it tick again before the KILL.
    assert not (tmp_path / "launch-1" / "continued").exists()
    assert launch.ticks("worker") == frozen


async def test_abandon_after_the_supervisor_died_still_kills_what_it_marked(
    tmp_path: Path, launches: list[Launch]
) -> None:
    launch = Launch(tmp_path / "launch-1")
    launches.append(launch)
    identity = (await launch.pause())["identity"]
    os.kill(int(identity.split(".")[0]), signal.SIGKILL)
    launch.process.wait(timeout=10)
    pids = launch.pids()
    assert _alive(pids["service"])

    result = _abandon(identity, tmp_path / "launch-1" / CLEANUP_RECEIPT_FILE)

    assert result.returncode == 0, result.stderr
    assert not any(_alive(pid) for pid in pids.values())
    assert json.loads((tmp_path / "launch-1" / CLEANUP_RECEIPT_FILE).read_text()) == ABANDONED


async def test_abandon_keeps_the_receipt_of_a_launch_that_had_finished(tmp_path: Path) -> None:
    receipt = tmp_path / CLEANUP_RECEIPT_FILE
    finished = {"cleanup_confirmed": True, "return_code": 0, "error": None, "timed_out": False, "stopped": False}
    receipt.write_text(json.dumps(finished))

    result = _abandon(f"{_dead_pid()}.1", receipt)

    assert result.returncode == 0, result.stderr
    assert json.loads(receipt.read_text()) == finished


def test_abandon_needs_an_identity(tmp_path: Path) -> None:
    result = _abandon("not-an-identity", tmp_path / CLEANUP_RECEIPT_FILE)
    assert result.returncode == 2 and "PID.START" in result.stderr
    assert not (tmp_path / CLEANUP_RECEIPT_FILE).exists()


async def test_abandon_launch_returns_the_abandoned_receipt(tmp_path: Path, launches: list[Launch]) -> None:
    launch = Launch(tmp_path / "launch-1")
    launches.append(launch)
    identity = (await launch.pause())["identity"]

    receipt = await abandon_launch(
        LocalSandbox(),  # type: ignore[arg-type]
        str(tmp_path / "launch-1"),
        identity=identity,
        timeout=10,
        harness="test",
    )

    assert receipt == ABANDONED
    assert not any(_alive(pid) for pid in launch.pids().values())


async def test_abandon_launch_raises_when_a_process_it_killed_did_not_exit(tmp_path: Path) -> None:
    receipt = tmp_path / CLEANUP_RECEIPT_FILE
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=1)

    async def download(source: str, destination: Path) -> None:
        destination.write_text(json.dumps({**ABANDONED, "cleanup_confirmed": False, "error": "did not exit"}))

    sandbox.download.side_effect = download
    with pytest.raises(NgFailureEvent, match="did not exit") as raised:
        await abandon_launch(sandbox, str(tmp_path), identity="12.34", timeout=1, harness="test")
    assert is_harness_termination_failure(raised.value)
    assert not receipt.exists()


# -- check_restored --------------------------------------------------------------------------------------------------


def _held(session_dir: Path, identity: str, *, seq: int = 1, mode: str = "hold") -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / PAUSED_FILE).write_text(
        json.dumps({"seq": seq, "confirmed": True, "pids": [], "error": None, "mode": mode, "identity": identity})
    )


def _relay_status(directory: Path, pid: int, start: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    status = directory / "status.json"
    status.write_text(json.dumps({"pid": pid, "start": int(start), "gen": 1, "held": 0, "inflight": 0, "resent": 0}))
    return status


async def _check(session_dir: Path, identity: str, *, relay: Path | None = None, seq: int = 1):
    return await check_restored(
        LocalSandbox(),  # type: ignore[arg-type]
        str(session_dir),
        pause_seq=seq,
        identity=identity,
        relay_status_path=str(relay) if relay else None,
        timeout=15,
    )


async def test_a_held_launch_with_a_live_relay_can_be_taken_over(tmp_path: Path) -> None:
    me = f"{os.getpid()}.{_start_time(os.getpid())}"
    relay = subprocess.Popen(["sleep", "30"])
    try:
        _held(tmp_path, me)
        status = _relay_status(tmp_path / "relay", relay.pid, _start_time(relay.pid))
        assert await _check(tmp_path, me, relay=status) == RestoredLaunch(pause_seq=1, identity=me)
        assert await _check(tmp_path, me) == RestoredLaunch(pause_seq=1, identity=me)
    finally:
        relay.kill()
        relay.wait()


async def test_a_real_supervisor_holding_its_pause_can_be_taken_over(tmp_path: Path, launches: list[Launch]) -> None:
    launch = Launch(tmp_path / "launch-1")
    launches.append(launch)
    identity = (await launch.pause())["identity"]
    assert await _check(tmp_path / "launch-1", identity) == RestoredLaunch(pause_seq=1, identity=identity)


@pytest.mark.parametrize(
    "setup, reason",
    [
        ("receipt", "launch_finished"),
        ("dead supervisor", "supervisor_gone"),
        ("reused pid", "identity_mismatch"),
        ("other pause", "pause_mismatch"),
        ("stop mode", "pause_mismatch"),
        ("continued", "pause_mismatch"),
        ("no relay", "relay_unavailable"),
        ("dead relay", "relay_gone"),
    ],
)
async def test_a_restored_launch_that_cannot_be_taken_over_says_why(tmp_path: Path, setup: str, reason: str) -> None:
    me = f"{os.getpid()}.{_start_time(os.getpid())}"
    identity = {"dead supervisor": f"{_dead_pid()}.1", "reused pid": f"{os.getpid()}.1"}.get(setup, me)
    _held(tmp_path, identity, seq=2 if setup == "other pause" else 1, mode="stop" if setup == "stop mode" else "hold")
    relay = None
    if setup == "receipt":
        (tmp_path / CLEANUP_RECEIPT_FILE).write_text("{}")
    if setup == "continued":
        (tmp_path / CONTINUE_REQUEST_FILE).write_text("1")
    if setup == "no relay":
        (tmp_path / "relay").mkdir()
        (tmp_path / "relay" / "unavailable").touch()
        relay = tmp_path / "relay" / "status.json"
    if setup == "dead relay":
        relay = _relay_status(tmp_path / "relay", _dead_pid(), "1")

    verdict = await _check(tmp_path, identity, relay=relay)

    assert verdict == NotResumable(reason)  # type: ignore[arg-type]


async def test_a_check_that_cannot_run_fails_as_such() -> None:
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(stdout=None, stderr="unreachable", return_code=-1, error_type="502")
    verdict = await check_restored(sandbox, "/s", pause_seq=1, identity="1.2", relay_status_path=None, timeout=1)
    assert verdict == NotResumable("check_failed", "502")
    assert (
        await check_restored(sandbox, "/s", pause_seq=1, identity="x", relay_status_path=None, timeout=1)
    ).reason == ("identity_mismatch")


async def test_a_check_the_sandbox_keeps_refusing_fails_as_such_and_one_it_refuses_once_is_repeated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nemo_gym.agents.utils.setup_retry.asyncio", SimpleNamespace(sleep=AsyncMock()))
    # The timeout is the budget of every attempt and the waits between them: it must cover the retries' waits.
    sandbox = AsyncMock()
    sandbox.exec.side_effect = ConnectionError("502: the sandbox could not be reached")
    verdict = await check_restored(sandbox, "/s", pause_seq=1, identity="1.2", relay_status_path=None, timeout=30)
    # Not a verdict on the launch: the check never ran.
    assert isinstance(verdict, NotResumable) and verdict.reason == "check_failed" and "502" in (verdict.detail or "")
    assert sandbox.exec.await_count == 3

    sandbox.exec.reset_mock()
    sandbox.exec.side_effect = [
        ConnectionError("502: the sandbox could not be reached"),
        SandboxExecResult(stdout="ok\n", stderr="", return_code=0),
    ]
    verdict = await check_restored(sandbox, "/s", pause_seq=1, identity="1.2", relay_status_path=None, timeout=30)
    assert verdict == RestoredLaunch(pause_seq=1, identity="1.2") and sandbox.exec.await_count == 2


# -- adopt -----------------------------------------------------------------------------------------------------------


def _rebound(tmp_path: Path, acknowledgement: dict) -> tuple[SandboxSession, dict]:
    """The session a restored agent server rebinds to the launch's files, from the bookmark its checkpoint saved."""
    state = {
        "session_root": str(tmp_path / "launch-1"),
        "session_dir": str(tmp_path / "launch-1"),
        "launch_started": True,
        "pause_seq": acknowledgement["seq"],
        "supervisor_identity": acknowledgement["identity"],
    }
    session = SandboxSession.rebind(
        LocalSandbox(),  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        workdir=str(tmp_path),
        harness="test",
        pausable=True,
        pause_mode="hold",
    )
    return session, state


async def test_adopt_continues_the_held_launch_and_finishes_it_as_execute_would(
    tmp_path: Path, launches: list[Launch]
) -> None:
    launch = Launch(tmp_path / "launch-1")
    launches.append(launch)
    session, state = _rebound(tmp_path, await launch.pause())
    assert await _check(tmp_path / "launch-1", state["supervisor_identity"]) == RestoredLaunch(
        pause_seq=1, identity=state["supervisor_identity"]
    )
    collect = AsyncMock(return_value="artifacts")
    adopting = asyncio.create_task(
        session.adopt(pause_seq=1, collect=collect, timeout=60, close_timeout=3, poll_window_s=1)
    )
    try:
        await _continued(launch, 1)
        assert not adopting.done(), "the adopted launch runs until its harness ends"
        launch.finish()
        result = await asyncio.wait_for(adopting, 30)
    finally:
        await _end(adopting)

    assert result == "artifacts" and collect.await_count == 1
    assert (tmp_path / "launch-1" / "finished").exists()
    assert session.cleanup is not None and session.cleanup["return_code"] == 0 and session.cleanup_confirmed


async def test_an_adopted_launch_pauses_again_at_the_next_checkpoint(tmp_path: Path, launches: list[Launch]) -> None:
    launch = Launch(tmp_path / "launch-1")
    launches.append(launch)
    held = await launch.pause()
    session, _ = _rebound(tmp_path, held)
    activation = FakeActivation()
    running = asyncio.create_task(
        run_with_checkpoint_pauses(
            session,
            activation,
            launch=lambda: session.adopt(
                pause_seq=1, collect=AsyncMock(return_value="artifacts"), timeout=60, close_timeout=3, poll_window_s=1
            ),
            on_parked=lambda: dict(session.checkpoint_state()),
            stop_timeout=10,
        )
    )
    try:
        await _continued(launch, 1)
        activation.park.set()
        await asyncio.wait_for(activation.parked.wait(), 15)
        assert activation.boundaries[0]["pause_seq"] == 2
        assert activation.boundaries[0]["supervisor_identity"] == held["identity"]
        ticking = launch.ticks("worker")
        async with asyncio.timeout(15):  # A hold pause leaves the harness running.
            while launch.ticks("worker") == ticking:
                await asyncio.sleep(0.02)
        activation.resume.set()
        await _continued(launch, 2)
        launch.finish()
        assert await asyncio.wait_for(running, 30) == LaunchFinished("artifacts")
    finally:
        launch.finish()
        await _end(running)


async def test_a_launch_restored_a_second_time_is_taken_over_at_its_later_pause(
    tmp_path: Path, launches: list[Launch]
) -> None:
    launch = Launch(tmp_path / "launch-1")
    launches.append(launch)
    first = await launch.pause(1)
    # What the first restoring job did: continued pause 1, then held pause 2 at its own checkpoint, then crashed.
    outcome, _ = await request_continue(
        LocalSandbox(), session_dir=str(launch.session_dir), workdir=None, seq=1, timeout=15
    )
    assert outcome == "acknowledged"
    second = await launch.pause(2)
    assert second["identity"] == first["identity"]
    session, state = _rebound(tmp_path, second)
    identity = state["supervisor_identity"]

    assert await _check(launch.session_dir, identity, seq=1) == NotResumable("pause_mismatch")
    assert await _check(launch.session_dir, identity, seq=2) == RestoredLaunch(pause_seq=2, identity=identity)
    adopting = asyncio.create_task(
        session.adopt(
            pause_seq=2, collect=AsyncMock(return_value="artifacts"), timeout=60, close_timeout=3, poll_window_s=1
        )
    )
    try:
        await _continued(launch, 2)
        launch.finish()
        assert await asyncio.wait_for(adopting, 30) == "artifacts"
    finally:
        await _end(adopting)
    assert session.cleanup is not None and session.cleanup["return_code"] == 0


async def test_cancelling_an_adopted_launch_stops_the_restored_harness(tmp_path: Path, launches: list[Launch]) -> None:
    launch = Launch(tmp_path / "launch-1")
    launches.append(launch)
    session, _ = _rebound(tmp_path, await launch.pause())
    pids = launch.pids()
    adopting = asyncio.create_task(
        session.adopt(
            pause_seq=1, collect=AsyncMock(return_value="artifacts"), timeout=60, close_timeout=10, poll_window_s=1
        )
    )
    try:
        await _continued(launch, 1)
        adopting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(adopting, 30)
    finally:
        await _end(adopting)
    assert session.cleanup is not None and session.cleanup["stopped"] is True and session.cleanup_confirmed
    _wait_for(lambda: not _alive(pids["worker"]) and not _alive(pids["child"]))
    assert not (tmp_path / "launch-1" / "finished").exists()
    launch.process.wait(timeout=10)


async def test_adopt_repeats_a_continue_the_sandbox_did_not_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = [ConnectionError("502"), ("timeout", None), ("finished", None)]

    async def request(sandbox, *, session_dir, workdir, seq, timeout):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_continue", request)
    session, _ = _rebound(tmp_path, {"seq": 1, "identity": "1.2"})
    reached: list[str] = []

    async def receipt(*, deadline: float, window_s: float) -> SandboxExecResult:
        reached.append("receipt wait")
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    monkeypatch.setattr(session, "_wait_for_receipt", receipt)
    monkeypatch.setattr(session, "_finish", AsyncMock())
    await asyncio.wait_for(
        session.adopt(pause_seq=1, collect=AsyncMock(return_value=None), timeout=5, close_timeout=3), 30
    )
    # The continue is the same request each time, so repeating it is safe; the third answer ended the retries.
    assert answers == [] and reached == ["receipt wait"]


async def test_adopt_fails_when_the_restored_pause_never_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    async def request(sandbox, *, session_dir, workdir, seq, timeout):
        calls.append(seq)
        return ("timeout", None)

    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_continue", request)
    session, _ = _rebound(tmp_path, {"seq": 1, "identity": "1.2"})
    monkeypatch.setattr(session, "_finish", AsyncMock())
    with pytest.raises(RuntimeError, match="restored launch did not continue"):
        await asyncio.wait_for(
            session.adopt(pause_seq=1, collect=AsyncMock(return_value=None), timeout=5, close_timeout=3), 30
        )
    assert calls == [1, 1, 1]


async def test_adopt_writes_an_expiry_that_was_requested_before_the_take_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _rebound(tmp_path, {"seq": 1, "identity": "1.2"})
    events: list[str] = []

    async def expiry(*, timeout: float) -> None:
        events.append("expiry")

    async def request(sandbox, *, session_dir, workdir, seq, timeout):
        events.append("continue")
        return ("finished", None)

    async def receipt(*, deadline: float, window_s: float) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    monkeypatch.setattr(session, "_request_expiry", expiry)
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_continue", request)
    monkeypatch.setattr(session, "_wait_for_receipt", receipt)
    monkeypatch.setattr(session, "_finish", AsyncMock())
    # The session has no launch of its own yet, so the request is only recorded.
    assert await session.expire(timeout=1) is True and events == []

    await asyncio.wait_for(
        session.adopt(pause_seq=1, collect=AsyncMock(return_value=None), timeout=5, close_timeout=3), 30
    )
    # The supervisor honours the marker once the pause is continued, so it is written first.
    assert events == ["expiry", "continue"]


# -- a launch whose connection is lost ---------------------------------------------------------------------------------


class DroppingSandbox(LocalSandbox):
    """Loses the launch command's connection, as a provider stream that drops; the launch itself runs on.

    With ``start=True`` the launch runs in the background and the connection drops once its supervisor wrote its pid;
    with ``start=False`` it drops before the launch ever ran.
    """

    def __init__(self, session_dir: Path, *, start: bool) -> None:
        self.session_dir = session_dir
        self.start = start
        self.launch: asyncio.Task | None = None

    async def exec(self, command, *, cwd=None, timeout_s=30, **kwargs):
        if "--receipt" in command and "--abandon" not in command and self.launch is None:
            if not self.start:
                self.launch = asyncio.get_running_loop().create_future()
                raise TimeoutError("Request timed out")
            self.launch = asyncio.create_task(super().exec(command, cwd=cwd, timeout_s=timeout_s))
            async with asyncio.timeout(15):
                while not (self.session_dir / "supervisor.pid").exists():
                    await asyncio.sleep(0.02)
            raise TimeoutError("Request timed out")
        return await super().exec(command, cwd=cwd, timeout_s=timeout_s, **kwargs)


def _dropping_session(tmp_path: Path, *, start: bool) -> tuple[SandboxSession, DroppingSandbox]:
    sandbox = DroppingSandbox(tmp_path / "launch-1", start=start)
    session = SandboxSession(
        sandbox=sandbox,  # type: ignore[arg-type]
        session_dir=str(tmp_path / "launch-1"),
        workdir=str(tmp_path),
        harness="test",
        pausable=True,
    )
    return session, sandbox


def _worker_stage(session_dir: Path, *, ticks: int = 60):
    async def stage() -> SandboxCommand:
        session_dir.mkdir(parents=True, exist_ok=True)
        worker = f"for _ in $(seq 1 {ticks}); do printf . >> ticks; sleep 0.02; done; exit 4"
        return SandboxCommand(argv=["bash", "-c", f"cd {session_dir} && {worker}"])

    return stage


async def test_a_lost_launch_connection_is_finished_from_the_supervisors_receipt(tmp_path: Path, caplog) -> None:
    session, sandbox = _dropping_session(tmp_path, start=True)
    collect = AsyncMock(return_value="artifacts")

    result = await asyncio.wait_for(
        session.execute(
            stage_activation=_worker_stage(tmp_path / "launch-1"), collect=collect, timeout=60, close_timeout=3
        ),
        30,
    )

    assert result == "artifacts" and collect.await_count == 1
    assert len((tmp_path / "launch-1" / "ticks").read_text()) == 60, "the harness ran to its end"
    assert session.cleanup is not None and session.cleanup["return_code"] == 4
    assert "launch connection lost" in caplog.text
    await sandbox.launch


async def test_a_launch_whose_supervisor_never_started_fails_at_once(tmp_path: Path) -> None:
    session, _ = _dropping_session(tmp_path, start=False)
    with pytest.raises(TimeoutError, match="Request timed out"):
        await asyncio.wait_for(
            session.execute(
                stage_activation=_worker_stage(tmp_path / "launch-1"),
                collect=AsyncMock(return_value="artifacts"),
                timeout=60,
                close_timeout=3,
            ),
            15,
        )


async def test_cancelling_the_wait_for_a_lost_launchs_receipt_stops_the_harness(tmp_path: Path) -> None:
    session, sandbox = _dropping_session(tmp_path, start=True)
    running = asyncio.create_task(
        session.execute(
            stage_activation=_worker_stage(tmp_path / "launch-1", ticks=1500),
            collect=AsyncMock(return_value="artifacts"),
            timeout=60,
            close_timeout=3,
        )
    )
    async with asyncio.timeout(15):
        while sandbox.launch is None or not (tmp_path / "launch-1" / "ticks").exists():
            await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert session.cleanup is not None and session.cleanup["stopped"] is True
    await asyncio.wait_for(sandbox.launch, 15)


async def test_a_raised_receipt_poll_is_one_failed_poll() -> None:
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        ConnectionError("stream lost"),
        SandboxExecResult(
            stdout=json.dumps({"cleanup_confirmed": True, "return_code": 3}), stderr=None, return_code=0
        ),
    ]
    session = SandboxSession(sandbox=sandbox, session_dir="/s", workdir="/w", harness="test", pausable=True)
    result = await session._wait_for_receipt(deadline=monotonic() + 600, window_s=1)
    # As the launch command ends: 0 once the supervisor confirmed the cleanup, whatever the harness returned.
    assert result.return_code == 0 and sandbox.exec.await_count == 2


async def test_a_receipt_that_does_not_confirm_the_cleanup_ends_the_wait_as_a_failed_launch_command() -> None:
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(
        stdout=json.dumps({"cleanup_confirmed": False, "return_code": None}), stderr=None, return_code=0
    )
    session = SandboxSession(sandbox=sandbox, session_dir="/s", workdir="/w", harness="test", pausable=True)
    result = await session._wait_for_receipt(deadline=monotonic() + 600, window_s=1)
    assert (result.return_code, result.error_type) == (1, None)


async def test_failed_receipt_polls_do_not_end_the_wait_before_they_span_the_failure_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.asyncio.sleep", AsyncMock())
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [ConnectionError("gateway restart")] * 8 + [
        SandboxExecResult(stdout=json.dumps({"cleanup_confirmed": True}), stderr=None, return_code=0)
    ]
    session = SandboxSession(sandbox=sandbox, session_dir="/s", workdir="/w", harness="test", pausable=True)
    result = await session._wait_for_receipt(deadline=monotonic() + 600, window_s=1)
    # Eight failures in well under a minute: a sandbox that refuses commands for a while is waited out.
    assert result.return_code == 0 and sandbox.exec.await_count == 9


async def test_failed_receipt_polls_are_not_counted_while_a_checkpoint_holds_the_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session._RECEIPT_POLL_FAILURE_S", 0.0)
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.asyncio.sleep", AsyncMock())
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [ConnectionError("mid-snapshot")] * 8 + [
        SandboxExecResult(stdout=json.dumps({"cleanup_confirmed": True}), stderr=None, return_code=0)
    ]
    session = SandboxSession(sandbox=sandbox, session_dir="/s", workdir="/w", harness="test", pausable=True)
    session._pause_seq, session._continued_seq = 2, 1  # A pause is held: the sandbox may be mid-snapshot.
    result = await session._wait_for_receipt(deadline=monotonic() + 600, window_s=1)
    assert result.return_code == 0 and sandbox.exec.await_count == 9


async def test_five_raised_receipt_polls_give_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session._RECEIPT_POLL_FAILURE_S", 0.0)
    sandbox = AsyncMock()
    sandbox.exec.side_effect = ConnectionError("sandbox gone")
    session = SandboxSession(sandbox=sandbox, session_dir="/s", workdir="/w", harness="test", pausable=True)
    result = await session._wait_for_receipt(deadline=monotonic() + 600, window_s=1)
    assert (result.return_code, result.error_type) == (-1, "poll_failed") and sandbox.exec.await_count == 5
    assert "sandbox gone" in (result.stderr or "")
