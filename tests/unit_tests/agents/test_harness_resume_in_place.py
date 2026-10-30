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
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents.utils.run_until_parked import Finished, run_until_parked
from nemo_gym.agents.utils.sandbox_session import SandboxSession
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


# A worker that ticks for about 3 seconds, starts a ticking child, and a service that leaves the harness (only the
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
for _ in $(seq 1 150); do printf . >> worker.ticks; sleep 0.02; done
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
        _wait_for(lambda: (session_dir / "ready").exists() and (session_dir / "service.pid").exists())

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
    assert not (tmp_path / "launch-1" / "finished").exists()


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


# -- adopt -----------------------------------------------------------------------------------------------------------


def _reattached(tmp_path: Path, acknowledgement: dict) -> tuple[SandboxSession, dict]:
    """The session a restored agent server rebinds to the launch's files, from the bookmark its checkpoint saved."""
    state = {
        "session_root": str(tmp_path / "launch-1"),
        "session_dir": str(tmp_path / "launch-1"),
        "launch_started": True,
        "pause_seq": acknowledgement["seq"],
        "supervisor_identity": acknowledgement["identity"],
    }
    session = SandboxSession.reattach(
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
    session, state = _reattached(tmp_path, await launch.pause())
    assert await _check(tmp_path / "launch-1", state["supervisor_identity"]) == RestoredLaunch(
        pause_seq=1, identity=state["supervisor_identity"]
    )
    collect = AsyncMock(return_value="artifacts")

    result = await asyncio.wait_for(
        session.adopt(pause_seq=1, collect=collect, timeout=60, close_timeout=3, poll_window_s=1), 30
    )

    assert result == "artifacts" and collect.await_count == 1
    assert (tmp_path / "launch-1" / "finished").exists()
    assert json.loads((tmp_path / "launch-1" / CONTINUED_FILE).read_text())["seq"] == 1
    assert session.cleanup is not None and session.cleanup["return_code"] == 0 and session.cleanup_confirmed


async def test_an_adopted_launch_pauses_again_at_the_next_checkpoint(tmp_path: Path, launches: list[Launch]) -> None:
    launch = Launch(tmp_path / "launch-1")
    launches.append(launch)
    held = await launch.pause()
    session, _ = _reattached(tmp_path, held)
    activation = FakeActivation()
    running = asyncio.create_task(
        run_until_parked(
            session,
            activation,
            launch=lambda: session.adopt(
                pause_seq=1, collect=AsyncMock(return_value="artifacts"), timeout=60, close_timeout=3, poll_window_s=1
            ),
            on_parked=lambda: dict(session.checkpoint_state()),
            stop_timeout=10,
        )
    )
    async with asyncio.timeout(15):
        while not (tmp_path / "launch-1" / CONTINUED_FILE).exists():
            await asyncio.sleep(0.02)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 15)
    assert activation.boundaries[0]["pause_seq"] == 2
    assert activation.boundaries[0]["supervisor_identity"] == held["identity"]
    ticking = launch.ticks("worker")
    await asyncio.sleep(0.2)
    assert launch.ticks("worker") > ticking, "a hold pause leaves the harness running"
    activation.resume.set()
    assert await asyncio.wait_for(running, 30) == Finished("artifacts")
