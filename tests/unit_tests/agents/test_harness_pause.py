# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pausing a supervised harness in place for a checkpoint.

From the bash supervisor up to ``run_with_checkpoint_pauses``.
"""

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
from shlex import quote
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents.utils import run_with_checkpoint_pauses as checkpoint_pauses
from nemo_gym.agents.utils import supervisor_client
from nemo_gym.agents.utils.run_with_checkpoint_pauses import (
    CONTINUE_ATTEMPTS,
    PAUSE_ATTEMPTS,
    LaunchFinished,
    LaunchStopped,
    run_with_checkpoint_pauses,
)
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
    supervised_launch_command,
)
from nemo_gym.core.episode_types import EpisodeId
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec
from nemo_gym.sandbox.providers.local import LocalProvider
from nemo_gym.server_runtime._checkpoint.agent import AgentSessionParticipant
from nemo_gym.server_runtime._checkpoint.control import CheckpointRequest, ParticipantControlPlane
from nemo_gym.server_runtime._checkpoint.errors import StaleAttemptError


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")


class AsyncioWith:
    """Stands in for the ``asyncio`` name of one module with some functions replaced; the real module is untouched."""

    def __init__(self, **replaced) -> None:
        self.__dict__.update(replaced)

    def __getattr__(self, name: str):
        return getattr(asyncio, name)


# -- the bash supervisor -------------------------------------------------------------------------------------------


def _start(
    tmp_path: Path,
    *command: str,
    timeout: str = "30",
    options: tuple[str, ...] = (),
    pause_mode: str = "stop",
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    return subprocess.Popen(
        # Cleanup phases end early when the harness does; the limit only matters on a busy node.
        ["bash", str(SUPERVISOR_SCRIPT), "--timeout", timeout, "--cleanup-timeout", "5"]
        + ["--receipt", str(tmp_path / CLEANUP_RECEIPT_FILE), "--stop-file", str(tmp_path / "stop.request")]
        + ["--expire-file", str(tmp_path / EXPIRE_REQUEST_FILE)]
        + ["--pause-file", str(tmp_path / PAUSE_REQUEST_FILE), "--paused", str(tmp_path / PAUSED_FILE)]
        + ["--continue-file", str(tmp_path / CONTINUE_REQUEST_FILE), "--continued", str(tmp_path / CONTINUED_FILE)]
        + ["--pause-mode", pause_mode, *options, "--", *command],
        cwd=tmp_path,
        env=None if env is None else {**os.environ, **env},
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


def _start_time(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def _is_identity_of_a_live_process(identity: str) -> bool:
    pid, start = identity.split(".")
    try:
        return _start_time(int(pid)) == start
    except FileNotFoundError:
        return False


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
        assert paused == {
            "seq": 1,
            "confirmed": True,
            "pids": [],
            "error": None,
            "mode": "stop",
            "identity": paused["identity"],
        }
        assert _is_identity_of_a_live_process(paused["identity"])
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
        # Written before the supervisor has an identity to name.
        assert json.loads((tmp_path / CONTINUED_FILE).read_text()) == {
            "seq": 3,
            "confirmed": True,
            "pids": [],
            "error": None,
            "mode": "stop",
            "identity": None,
        }
    finally:
        _finish(process)


@linux_only
def test_an_identity_variable_in_the_sandbox_environment_is_not_taken_for_the_supervisor_s(tmp_path: Path) -> None:
    _request(tmp_path / PAUSE_REQUEST_FILE, 3)
    _request(tmp_path / CONTINUE_REQUEST_FILE, 3)
    # Named like the supervisor's own variable, with characters that would break the acknowledgement's JSON.
    process = _start(
        tmp_path, "bash", "-c", "touch ready; until [ -e finish ]; do sleep 0.02; done", env={"identity": 'a"b\nc'}
    )
    try:
        _wait_for(lambda: (tmp_path / "ready").exists())
        # The acknowledgement written before the supervisor has an identity names none.
        assert json.loads((tmp_path / CONTINUED_FILE).read_text())["identity"] is None
        # A later one names the supervisor itself.
        _request(tmp_path / PAUSE_REQUEST_FILE, 4)
        paused = _acknowledgement(tmp_path / PAUSED_FILE, 4)
        assert _is_identity_of_a_live_process(paused["identity"])
        _request(tmp_path / CONTINUE_REQUEST_FILE, 4)
        _acknowledgement(tmp_path / CONTINUED_FILE, 4)
        (tmp_path / "finish").touch()
        process.wait(timeout=15)
    finally:
        _finish(process)


@linux_only
def test_time_spent_paused_does_not_count_against_the_deadline(tmp_path: Path) -> None:
    # The harness has work left when it is paused, which it can only do once continued; it is paused for longer than
    # its whole budget, so the work ends after the deadline the launch started with.
    worker = "touch ready; for i in $(seq 15); do sleep 0.1; done; touch finished"
    process = _start(tmp_path, "bash", "-c", worker, timeout="4")
    try:
        _wait_for(lambda: (tmp_path / "ready").exists())
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["confirmed"] is True
        time.sleep(4.5)
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        process.wait(timeout=30)
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
        # Held for ten times the budget, which starts at the launch.
        time.sleep(20)
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        _wait_for(lambda: (tmp_path / "ready").exists())
        launched = time.monotonic()
        _request(tmp_path / PAUSE_REQUEST_FILE, 2)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 2)["confirmed"] is True
        _request(tmp_path / CONTINUE_REQUEST_FILE, 2)
        process.wait(timeout=60)
        elapsed = time.monotonic() - launched
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt["timed_out"] is True and receipt["stopped"] is False, receipt
        # Two seconds of budget plus the short pause; the hold added again would make it more than twenty-two.
        assert elapsed < 12, elapsed
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


# Like _WORKER, but the harness and its child end once the test lets them.
_HOLD_WORKER = f"""
(until [ -e finish ]; do printf . >> child.ticks; sleep 0.02; done) &
( "{sys.executable}" service.py & echo $! > service.pid )
while [ ! -s service.pid ]; do sleep 0.01; done
touch ready
until [ -e finish ]; do printf . >> worker.ticks; sleep 0.02; done
"""


@linux_only
def test_a_hold_pause_signals_nothing_names_the_supervisor_and_credits_the_deadline(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(_SERVICE)
    process = _start(tmp_path, "bash", "-c", _HOLD_WORKER, timeout="3", pause_mode="hold")
    service = None
    try:
        _wait_for(lambda: (tmp_path / "ready").exists())
        service = int((tmp_path / "service.pid").read_text())
        _wait_for(lambda: all(_ticks(tmp_path).values()))
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        paused = _acknowledgement(tmp_path / PAUSED_FILE, 1)
        assert paused == {
            "seq": 1,
            "confirmed": True,
            "pids": [],
            "error": None,
            "mode": "hold",
            "identity": paused["identity"],
        }
        assert _is_identity_of_a_live_process(paused["identity"])
        # Nothing is stopped: the harness, its child and a service that left it keep running through a hold
        # longer than the whole budget.
        before = _ticks(tmp_path)
        time.sleep(3.5)
        assert _state(service) not in ("T", "t", "gone")
        after = _ticks(tmp_path)
        assert all(after[name] > before[name] for name in before), (before, after)
        assert process.poll() is None and not (tmp_path / CLEANUP_RECEIPT_FILE).exists()
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        continued = _acknowledgement(tmp_path / CONTINUED_FILE, 1)
        assert continued["confirmed"] is True and continued["mode"] == "hold", continued
        assert continued["identity"] == paused["identity"]
        # The held time was credited: several supervisor rounds later the deadline has still not ended the harness.
        time.sleep(0.3)
        assert process.poll() is None
        (tmp_path / "finish").touch()
        process.wait(timeout=15)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt["timed_out"] is False and receipt["stopped"] is False and receipt["return_code"] == 0, receipt
    finally:
        _finish(process)
        if service is not None:
            try:
                os.kill(service, signal.SIGKILL)
            except ProcessLookupError:
                pass


@linux_only
def test_a_harness_that_ends_during_a_hold_pause_is_reported_once_the_pause_is_continued(tmp_path: Path) -> None:
    worker = "touch ready; until [ -e finish ]; do sleep 0.02; done; touch ended; exit 7"
    process = _start(tmp_path, "bash", "-c", worker, pause_mode="hold")
    try:
        _wait_for(lambda: (tmp_path / "ready").exists())
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["confirmed"] is True
        (tmp_path / "finish").touch()
        _wait_for(lambda: (tmp_path / "ended").exists())
        time.sleep(0.3)
        # The supervisor serves the hold until its continue: no receipt yet.
        assert process.poll() is None and not (tmp_path / CLEANUP_RECEIPT_FILE).exists()
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        assert _acknowledgement(tmp_path / CONTINUED_FILE, 1)["confirmed"] is True
        process.wait(timeout=15)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt == {
            "cleanup_confirmed": True,
            "return_code": 7,
            "error": None,
            "timed_out": False,
            "stopped": False,
        }
    finally:
        _finish(process)


@linux_only
def test_a_stop_during_a_hold_pause_ends_the_harness_with_term(tmp_path: Path) -> None:
    worker = "trap 'echo saved > checkpoint; exit 0' TERM; touch ready; while :; do sleep 0.05; done"
    process = _start(tmp_path, "bash", "-c", worker, pause_mode="hold")
    try:
        _wait_for(lambda: (tmp_path / "ready").exists())
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["mode"] == "hold"
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
def test_a_hold_pause_before_the_launch_holds_the_launch_until_continued(tmp_path: Path) -> None:
    _request(tmp_path / PAUSE_REQUEST_FILE, 1)
    process = _start(tmp_path, "touch", "started", pause_mode="hold")
    try:
        paused = _acknowledgement(tmp_path / PAUSED_FILE, 1)
        assert paused["confirmed"] is True and paused["pids"] == [] and paused["mode"] == "hold", paused
        assert _is_identity_of_a_live_process(paused["identity"])
        time.sleep(0.3)
        assert not (tmp_path / "started").exists()
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        process.wait(timeout=15)
        assert (tmp_path / "started").exists()
        continued = json.loads((tmp_path / CONTINUED_FILE).read_text())
        assert continued["seq"] == 1 and continued["mode"] == "hold" and continued["identity"] == paused["identity"]
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt["return_code"] == 0 and receipt["timed_out"] is False and receipt["stopped"] is False, receipt
    finally:
        _finish(process)


def test_the_pause_mode_must_be_stop_or_hold(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SUPERVISOR_SCRIPT), "--timeout", "1", "--cleanup-timeout", "1", "--receipt", "r"]
        + ["--stop-file", "s", "--pause-mode", "freeze", "--", "true"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2 and "stop or hold" in result.stderr


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


def test_a_pause_limit_needs_the_pause_files_and_a_positive_number(tmp_path: Path) -> None:
    required = ["bash", str(SUPERVISOR_SCRIPT), "--timeout", "1", "--cleanup-timeout", "1", "--receipt", "r"]
    required += ["--stop-file", "s"]
    pause_files = ["--pause-file", "p", "--paused", "a", "--continue-file", "c", "--continued", "d"]
    for options, message in (
        (["--max-pause", "5"], "--max-pause needs --pause-file"),
        ([*pause_files, "--max-pause", "0"], "--max-pause must be finite and positive"),
    ):
        result = subprocess.run(
            [*required, *options, "--", "touch", "started"], cwd=tmp_path, capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 2 and message in result.stderr, result
        assert not (tmp_path / "started").exists() and not (tmp_path / "r").exists()


@linux_only
def test_a_pause_nobody_continues_ends_the_harness_as_timed_out_after_the_pause_limit(tmp_path: Path) -> None:
    process = _start(tmp_path, "bash", "-c", _TICKER, options=("--max-pause", "1"))
    try:
        _wait_for(lambda: (tmp_path / "ticks").exists())
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        paused = _acknowledgement(tmp_path / PAUSED_FILE, 1)
        assert paused["confirmed"] is True and paused["pids"], paused
        process.wait(timeout=30)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt["cleanup_confirmed"] is True, receipt
        assert receipt["timed_out"] is True and receipt["pause_expired"] is True, receipt
        assert receipt["stopped"] is False, receipt
        assert not (tmp_path / CONTINUE_REQUEST_FILE).exists()
        # Nothing the pause stopped is left stopped: the harness was continued, then terminated.
        assert [pid for pid in paused["pids"] if _state(pid) in ("T", "t")] == []
        ended = (tmp_path / "ticks").read_text()
        time.sleep(0.3)
        assert (tmp_path / "ticks").read_text() == ended
    finally:
        _finish(process)


@linux_only
def test_a_pause_continued_within_the_pause_limit_leaves_no_trace_in_the_receipt(tmp_path: Path) -> None:
    worker = "until [ -e finish ]; do printf . >> ticks; sleep 0.02; done"
    process = _start(tmp_path, "bash", "-c", worker, options=("--max-pause", "60"))
    try:
        _wait_for(lambda: (tmp_path / "ticks").exists())
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["confirmed"] is True
        frozen = (tmp_path / "ticks").read_text()
        time.sleep(0.3)
        assert (tmp_path / "ticks").read_text() == frozen
        _request(tmp_path / CONTINUE_REQUEST_FILE, 1)
        assert _acknowledgement(tmp_path / CONTINUED_FILE, 1)["confirmed"] is True
        _grown(tmp_path)
        (tmp_path / "finish").touch()
        process.wait(timeout=30)
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
def test_a_hold_before_the_launch_nobody_continues_launches_nothing_after_the_pause_limit(tmp_path: Path) -> None:
    _request(tmp_path / PAUSE_REQUEST_FILE, 1)
    process = _start(tmp_path, "touch", "started", options=("--max-pause", "1"))
    try:
        assert _acknowledgement(tmp_path / PAUSED_FILE, 1)["confirmed"] is True
        process.wait(timeout=30)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt == {
            "cleanup_confirmed": True,
            "return_code": None,
            "error": None,
            "timed_out": True,
            "stopped": False,
            "pause_expired": True,
        }
        assert not (tmp_path / "started").exists()
        assert not (tmp_path / CONTINUE_REQUEST_FILE).exists()
    finally:
        _finish(process)


@linux_only
def test_a_harness_killed_while_paused_ends_the_pause_without_a_continue(tmp_path: Path) -> None:
    process = _start(tmp_path, "bash", "-c", f"echo $$ > harness.pid; {_TICKER}")
    try:
        _wait_for(lambda: (tmp_path / "ticks").exists())
        harness = int((tmp_path / "harness.pid").read_text())
        _request(tmp_path / PAUSE_REQUEST_FILE, 1)
        paused = _acknowledgement(tmp_path / PAUSED_FILE, 1)
        assert paused["confirmed"] is True and harness in paused["pids"], paused
        # The supervisor gave the harness its own process group, which its PID names.
        os.killpg(harness, signal.SIGKILL)
        process.wait(timeout=30)
        receipt = json.loads((tmp_path / CLEANUP_RECEIPT_FILE).read_text())
        assert receipt == {
            "cleanup_confirmed": True,
            "return_code": 128 + signal.SIGKILL,
            "error": None,
            "timed_out": False,
            "stopped": False,
        }
        assert not (tmp_path / CONTINUE_REQUEST_FILE).exists()
    finally:
        _finish(process)


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


def test_a_receipt_keeps_an_expired_pause_only_when_it_is_true() -> None:
    receipt = {"cleanup_confirmed": True, "return_code": 143, "error": None, "timed_out": True, "stopped": False}
    assert parse_cleanup_receipt(receipt | {"pause_expired": True}) == receipt | {"pause_expired": True}
    malformed = receipt | {"return_code": "143"}
    assert parse_cleanup_receipt(malformed | {"pause_expired": True}) == receipt | {
        "return_code": None,
        "pause_expired": True,
    }
    for other in (False, "true", 1, None):
        assert "pause_expired" not in parse_cleanup_receipt(malformed | {"pause_expired": other})
    assert "pause_expired" not in parse_cleanup_receipt(malformed)


def test_the_launch_command_passes_the_pause_limit_only_to_a_pausable_supervisor() -> None:
    def launch(**options) -> str:
        return supervised_launch_command(
            session_dir="/s", command=["worker"], timeout=10, cleanup_timeout=1, **options
        )

    assert "--max-pause 90 -- worker" in launch(pausable=True, max_pause=90)
    assert "--pause-file" in launch(pausable=True, max_pause=None)
    assert "--max-pause" not in launch(pausable=True, max_pause=None)
    assert "--max-pause" not in launch(pausable=True)
    assert "--max-pause" not in launch(pausable=False, max_pause=90)


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


async def test_a_pause_of_a_sent_launch_waits_for_its_claim_and_returns_the_acknowledgement(tmp_path: Path) -> None:
    acknowledgement = {"seq": 3, "confirmed": True, "pids": [], "error": None, "mode": "hold", "identity": "7.100"}

    async def start_supervisor() -> None:
        await asyncio.sleep(0.3)
        (tmp_path / "launch.claim").symlink_to("launch")
        await asyncio.sleep(0.1)
        (tmp_path / PAUSED_FILE).write_text(json.dumps(acknowledgement))

    starting = asyncio.create_task(start_supervisor())
    outcome, answer = await request_pause(
        LocalSandbox(), session_dir=str(tmp_path), workdir=None, seq=3, timeout=10, launch_sent=True
    )
    await starting
    assert outcome == "acknowledged" and answer == acknowledgement


async def test_a_pause_of_a_sent_launch_that_never_claims_is_not_launched_only_after_the_bound(tmp_path: Path) -> None:
    started = time.monotonic()
    outcome, answer = await request_pause(
        LocalSandbox(), session_dir=str(tmp_path), workdir=None, seq=1, timeout=0.5, launch_sent=True
    )
    assert (outcome, answer) == ("not_launched", None)
    assert time.monotonic() - started >= 0.4, "the request waited for the launch instead of answering at once"
    assert (tmp_path / PAUSE_REQUEST_FILE).read_text() == "1"


async def test_a_pause_of_a_sent_launch_that_claims_but_never_answers_times_out(tmp_path: Path) -> None:
    class ClaimingSandbox(LocalSandbox):
        async def exec(self, command: str, **kwargs) -> SandboxExecResult:
            # Claim between polling rounds in the same shell, not on a racing Python timer. The request must
            # first observe no claim, then a claim without an acknowledgement, regardless of host load.
            claim = quote(str(tmp_path / "launch.claim"))
            return await super().exec(
                f"sleep() {{ ln -s launch {claim} || exit 1; unset -f sleep; }}; {command}",
                **kwargs,
            )

    outcome, answer = await request_pause(
        ClaimingSandbox(), session_dir=str(tmp_path), workdir=None, seq=1, timeout=0.6, launch_sent=True
    )
    assert (outcome, answer) == ("timeout", None)
    assert (tmp_path / "launch.claim").readlink() == Path("launch")
    assert (tmp_path / PAUSE_REQUEST_FILE).read_text() == "1"


async def test_a_pause_of_a_sent_launch_that_a_stop_fences_meanwhile_is_finished(tmp_path: Path) -> None:
    async def fence() -> None:
        await asyncio.sleep(0.2)
        (tmp_path / "launch.claim").symlink_to("stop")

    fencing = asyncio.create_task(fence())
    outcome, answer = await request_pause(
        LocalSandbox(), session_dir=str(tmp_path), workdir=None, seq=1, timeout=10, launch_sent=True
    )
    await fencing
    assert (outcome, answer) == ("finished", None)


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
        assert (
            paused.mode == "stop" and paused.identity is not None and _is_identity_of_a_live_process(paused.identity)
        )
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
async def test_a_hold_session_keeps_its_harness_running_and_records_the_pause_it_holds(tmp_path: Path) -> None:
    session = _session(tmp_path, pause_mode="hold")
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
        assert paused.confirmed and paused.pids == () and paused.mode == "hold", paused
        assert paused.identity is not None and _is_identity_of_a_live_process(paused.identity)
        # Nothing is stopped: the harness keeps ticking while the pause holds.
        await _ticks_grow(launch_dir, (launch_dir / "ticks").read_text())
        state = json.loads(json.dumps(session.checkpoint_state()))
        assert state["launch_started"] is True
        assert state["pause_seq"] == 1 and state["supervisor_identity"] == paused.identity
        # Pausing again before the continue asks for the same pause; the same supervisor answers.
        again = await session.pause_harness(timeout=10)
        assert again.confirmed and again.mode == "hold" and again.identity == paused.identity, again
        assert session.checkpoint_state()["pause_seq"] == 1
        continued = await session.continue_harness(timeout=10)
        assert continued.confirmed and continued.mode == "hold" and continued.identity == paused.identity, continued
        released = session.checkpoint_state()
        assert "pause_seq" not in released and "supervisor_identity" not in released
        # The next checkpoint holds a later pause under the same supervisor.
        assert (await session.pause_harness(timeout=10)).confirmed
        assert session.checkpoint_state()["pause_seq"] == 2
        assert session.checkpoint_state()["supervisor_identity"] == paused.identity
        assert (await session.continue_harness(timeout=10)).confirmed
        await _ticks_grow(launch_dir, (launch_dir / "ticks").read_text())
        (launch_dir / "finish").touch()
        assert await asyncio.wait_for(running, 30) == "artifacts"
        assert session.cleanup is not None
        assert session.cleanup["stopped"] is False and session.cleanup["timed_out"] is False
    finally:
        await _end_launch(running, launch_dir)


class DelayedLaunchSandbox(LocalSandbox):
    """Holds the launch command back until ``release`` is set, as a provider slow to start a command does."""

    def __init__(self) -> None:
        self.launch_received = asyncio.Event()
        self.release = asyncio.Event()

    async def exec(self, command, **kwargs):
        if "--pause-file" in command:
            self.launch_received.set()
            await self.release.wait()
        return await super().exec(command, **kwargs)


@linux_only
async def test_a_hold_pause_of_a_delayed_launch_is_unconfirmed_and_the_stop_fences_the_launch(tmp_path: Path) -> None:
    sandbox = DelayedLaunchSandbox()
    launch_dir = tmp_path / "launch-1"
    session = SandboxSession(
        sandbox=sandbox,  # type: ignore[arg-type]
        session_dir=str(launch_dir),
        workdir=str(tmp_path),
        harness="test",
        pausable=True,
        pause_mode="hold",
    )
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
            await sandbox.launch_received.wait()
        # The launch command was sent, and nothing has claimed the launch when the pause's bound ends.
        paused = await session.pause_harness(timeout=0.5)
        assert paused.status == "unconfirmed" and not paused.acknowledged, paused
        assert not (launch_dir / "launch.claim").is_symlink()
        state = session.checkpoint_state()
        assert state["launch_started"] is True and "pause_seq" not in state and "supervisor_identity" not in state
        # What a caller does with an unconfirmed pause: the stop wins the claim, so the launch can no longer start.
        await session.stop_harness(timeout=10)
        assert os.readlink(launch_dir / "launch.claim") == "stop"
        sandbox.release.set()
        await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 30)
        assert not (launch_dir / "ready").exists() and not (launch_dir / "ticks").exists()
        assert not (launch_dir / PAUSED_FILE).exists()
    finally:
        sandbox.release.set()
        await _end_launch(running, launch_dir)


@linux_only
async def test_a_stopped_launch_no_longer_records_the_pause_it_held(tmp_path: Path) -> None:
    session = _session(tmp_path, pause_mode="hold")
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
        assert (await session.pause_harness(timeout=10)).confirmed
        assert "pause_seq" in session.checkpoint_state()
        await session.stop_harness(timeout=10)
        # Its supervisor is gone: a restore must not look for one holding this pause.
        assert session.cleanup is not None and "pause_seq" not in session.checkpoint_state()
        await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 30)
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


@linux_only
async def test_a_pause_of_a_launch_still_staging_after_its_wait_is_unconfirmed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    launch_dir = tmp_path / "launch-1"
    staging, release = asyncio.Event(), asyncio.Event()

    async def stage() -> SandboxCommand:
        launch_dir.mkdir(parents=True, exist_ok=True)
        staging.set()
        await release.wait()
        (launch_dir / "staged").write_text("late")
        return SandboxCommand(argv=["true"])

    running = asyncio.create_task(
        session.execute(
            stage_activation=stage, collect=AsyncMock(return_value="artifacts"), timeout=30, close_timeout=3
        )
    )
    try:
        await asyncio.wait_for(staging.wait(), 5)
        paused = await session.pause_harness(timeout=0.01)
        assert paused == HarnessPauseResult(status="unconfirmed", error="the launch is still staging its files")
        # Nothing was asked of a supervisor: the staging the pause could not wait out is still writing.
        assert not (launch_dir / PAUSE_REQUEST_FILE).exists() and not (launch_dir / "staged").exists()
        release.set()
        assert await asyncio.wait_for(running, 30) == "artifacts"
        assert (launch_dir / "staged").read_text() == "late"
    finally:
        release.set()
        await _end_launch(running, launch_dir)


@linux_only
async def test_a_pause_waits_for_staging_that_ends_within_its_bound_and_then_pauses_the_launch(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    launch_dir = tmp_path / "launch-1"
    staging, release = asyncio.Event(), asyncio.Event()
    staged = _ticking_stage(launch_dir)

    async def stage() -> SandboxCommand:
        staging.set()
        await release.wait()
        return await staged()

    running = asyncio.create_task(
        session.execute(
            stage_activation=stage, collect=AsyncMock(return_value="artifacts"), timeout=30, close_timeout=3
        )
    )
    try:
        await asyncio.wait_for(staging.wait(), 5)
        pausing = asyncio.create_task(session.pause_harness(timeout=20))
        await asyncio.sleep(0.1)
        assert not pausing.done()
        release.set()
        assert (await asyncio.wait_for(pausing, 30)).confirmed
        # The supervisor serves the pause whether the request found it started or it read the request on starting.
        async with asyncio.timeout(15):
            while not (launch_dir / PAUSED_FILE).exists():
                await asyncio.sleep(0.02)
        ticks = launch_dir / "ticks"
        frozen = ticks.read_text() if ticks.exists() else ""
        await asyncio.sleep(0.3)
        assert (ticks.read_text() if ticks.exists() else "") == frozen
        assert (await session.continue_harness(timeout=10)).confirmed
        async with asyncio.timeout(15):
            while not ticks.exists():
                await asyncio.sleep(0.02)
        await _ticks_grow(launch_dir, frozen)
        (launch_dir / "finish").touch()
        assert await asyncio.wait_for(running, 30) == "artifacts"
    finally:
        release.set()
        await _end_launch(running, launch_dir)


async def test_a_session_that_was_not_launched_pausable_refuses_to_pause(tmp_path: Path) -> None:
    session = SandboxSession(sandbox=LocalSandbox(), session_dir=str(tmp_path), workdir=None, harness="test")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="pausable"):
        await session.pause_harness()


async def test_execute_adds_the_pause_allowance_to_the_provider_deadline_and_limits_one_pause_to_it() -> None:
    timeouts, limited = [], []
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
        limited.append("--max-pause 900.0 -- worker" in launch.args[0])
        assert limited[-1] or "--max-pause" not in launch.args[0]
    assert timeouts[1] - timeouts[0] == 900.0
    assert limited == [False, True]


async def test_rebind_binds_a_session_to_a_restored_sandbox_without_the_dead_launch(tmp_path: Path) -> None:
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
    rebound = SandboxSession.rebind(
        restored_sandbox,  # type: ignore[arg-type]
        state,
        workdir=str(tmp_path),
        harness="test",
        pausable=True,
    )
    assert rebound.sandbox is restored_sandbox
    assert rebound.root_dir == state["session_root"] and rebound.session_dir == state["session_dir"]
    assert rebound.launch_started is False and rebound.pausable
    with pytest.raises(ValueError):
        SandboxSession.rebind(restored_sandbox, {"session_dir": "x"}, workdir=None, harness="test")  # type: ignore[arg-type]
    # Nothing ran in the restored sandbox, so its close only removes the session's files.
    (tmp_path / "launch-1").mkdir()
    await rebound.close(timeout=5)
    assert rebound.closed and not (tmp_path / "launch-1").exists()


def test_rebind_accepts_a_held_pause_and_keeps_the_pause_mode(tmp_path: Path) -> None:
    state = {
        "session_root": str(tmp_path / "launch-1"),
        "session_dir": str(tmp_path / "launch-2"),
        "launch_started": True,
        "pause_seq": 3,
        "supervisor_identity": "41.9000",
    }
    rebound = SandboxSession.rebind(
        LocalSandbox(),  # type: ignore[arg-type]
        json.loads(json.dumps(state)),
        workdir=str(tmp_path),
        harness="test",
        pausable=True,
        pause_mode="hold",
    )
    assert rebound.pause_mode == "hold" and rebound.pausable
    # The re-bound session has no launch of its own, so it holds no pause either.
    assert rebound.checkpoint_state() == {
        "session_root": state["session_root"],
        "session_dir": state["session_dir"],
        "launch_started": False,
    }
    default = SandboxSession.rebind(LocalSandbox(), state, workdir=None, harness="test")  # type: ignore[arg-type]
    assert default.pause_mode == "stop"
    with pytest.raises(ValueError):
        SandboxSession.rebind(LocalSandbox(), state | {"pause_seq": "3"}, workdir=None, harness="test")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("pausable", "pause_mode", "option"), [(True, "hold", True), (True, "stop", False), (False, "hold", False)]
)
async def test_execute_launches_the_supervisor_in_the_session_s_pause_mode(
    pausable: bool, pause_mode: str, option: bool
) -> None:
    sandbox = AsyncMock(spec=AsyncSandbox)
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    session = SandboxSession(
        sandbox=sandbox,
        session_dir="/s",
        workdir="/w",
        harness="test",
        pausable=pausable,
        pause_mode=pause_mode,  # type: ignore[arg-type]
    )
    session.stop_harness = AsyncMock()  # type: ignore[method-assign]
    await session.execute(
        stage_activation=AsyncMock(return_value=SandboxCommand(argv=["worker"])),
        collect=AsyncMock(return_value=None),
        timeout=10,
        close_timeout=3,
    )
    [launch] = [call for call in sandbox.exec.await_args_list if "process_supervisor.sh" in call.args[0]]
    # stop is the supervisor's default and is not passed; a session that cannot pause passes no pause option.
    assert ("--pause-mode hold" in launch.args[0]) is option
    assert ("--pause-mode" in launch.args[0]) is option


# -- run_with_checkpoint_pauses -------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint_pauses, "RETRY_DELAY_S", 0.0)


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

    def queued(self):  # pragma: no cover - not used by run_with_checkpoint_pauses
        raise NotImplementedError


class FakeSession:
    harness = "fake"
    # Its harness is running: a stop ends it.
    launch_started = True
    staging = False

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


def _run(session: FakeSession, activation: FakeActivation, *, stop_timeout: float = 5) -> asyncio.Task:
    return asyncio.create_task(
        run_with_checkpoint_pauses(
            session,  # type: ignore[arg-type]
            activation,
            launch=session.launch,
            on_parked=lambda: {"bookmark": len(session.events)},
            stop_timeout=stop_timeout,
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
    assert await asyncio.wait_for(running, 5) == LaunchFinished("result")
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
    assert isinstance(result, LaunchStopped) and result.receipt["stopped"] is True
    assert session.events == ["pause", "stop"]


async def test_a_pause_request_that_keeps_failing_in_transport_falls_back_to_a_stop() -> None:
    session = FakeSession(pauses=[RuntimeError("sandbox exec failed")] * PAUSE_ATTEMPTS)
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    assert isinstance(await asyncio.wait_for(running, 5), LaunchStopped)
    assert session.events == [*["pause"] * PAUSE_ATTEMPTS, "stop"]


async def test_a_pause_request_that_fails_in_transport_once_is_sent_again_and_pauses_in_place() -> None:
    session = FakeSession(pauses=[RuntimeError("sandbox exec failed")])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    assert session.events == ["pause", "pause"]
    activation.resume.set()
    async with asyncio.timeout(5):
        while session.events[-1] != "continue":
            await asyncio.sleep(0)
    session.done.set()
    assert await asyncio.wait_for(running, 5) == LaunchFinished("result")
    assert session.events == ["pause", "pause", "continue"]


@pytest.mark.parametrize(
    "answer",
    [TimeoutError("no provider slot"), HarnessPauseResult(status="unconfirmed", error="D state")],
    ids=["timed_out", "answered_unconfirmed"],
)
async def test_a_pause_that_timed_out_or_that_the_supervisor_answered_is_not_sent_again(answer) -> None:
    session = FakeSession(pauses=[answer])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    assert isinstance(await asyncio.wait_for(running, 5), LaunchStopped)
    assert session.events == ["pause", "stop"]


async def test_repeated_pause_and_continue_requests_wait_twice_as_long_before_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(checkpoint_pauses, "RETRY_DELAY_S", 1.0)
    monkeypatch.setattr(checkpoint_pauses, "asyncio", AsyncioWith(sleep=sleep))
    session = FakeSession(
        pauses=[RuntimeError("sandbox exec failed")] * (PAUSE_ATTEMPTS - 1),
        continues=[RuntimeError("sandbox exec failed")] * CONTINUE_ATTEMPTS,
    )
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    assert delays == [2.0**waited for waited in range(PAUSE_ATTEMPTS - 1)]
    del delays[:]
    activation.resume.set()
    assert isinstance(await asyncio.wait_for(running, 5), LaunchStopped)
    # No wait follows the last attempt.
    assert delays == [2.0**waited for waited in range(CONTINUE_ATTEMPTS - 1)]
    assert session.events == [*["pause"] * PAUSE_ATTEMPTS, *["continue"] * CONTINUE_ATTEMPTS, "stop"]


async def test_a_harness_that_ends_on_its_own_as_the_stop_arrives_is_finished() -> None:
    session = FakeSession(pauses=[HarnessPauseResult(status="unconfirmed")], stopped_receipt=False)
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    assert await asyncio.wait_for(running, 5) == LaunchFinished("result")
    assert activation.boundaries == []


async def test_a_continue_that_keeps_failing_stops_the_harness_after_the_checkpoint() -> None:
    session = FakeSession(continues=[HarnessPauseResult(status="unconfirmed")] * CONTINUE_ATTEMPTS)
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    result = await asyncio.wait_for(running, 5)
    assert isinstance(result, LaunchStopped)
    assert session.events == ["pause", *["continue"] * CONTINUE_ATTEMPTS, "stop"]
    assert len(activation.boundaries) == 1


async def test_a_continue_the_supervisor_refused_is_not_sent_again() -> None:
    refused = HarnessPauseResult(status="unconfirmed", error="A paused process did not continue", acknowledged=True)
    session = FakeSession(continues=[refused] * CONTINUE_ATTEMPTS)
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    result = await asyncio.wait_for(running, 5)
    assert isinstance(result, LaunchStopped)
    # The same request would read the same answer from the acknowledgement file: one continue, then the stop.
    assert session.events == ["pause", "continue", "stop"]


async def test_a_harness_that_finished_as_the_park_arrived_keeps_its_result() -> None:
    session = FakeSession(pauses=[HarnessPauseResult(status="finished")])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    async with asyncio.timeout(5):
        while session.events != ["pause"]:
            await asyncio.sleep(0)
    session.done.set()
    assert await asyncio.wait_for(running, 5) == LaunchFinished("result")
    assert session.events == ["pause"] and activation.boundaries == []


async def test_a_launch_that_ends_first_never_pauses() -> None:
    session, activation = FakeSession(), FakeActivation()
    session.done.set()
    assert await asyncio.wait_for(_run(session, activation), 5) == LaunchFinished("result")
    assert session.events == []


async def test_a_park_request_that_fails_as_its_activation_is_cancelled_has_its_exception_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetiredWhenClosed(FakeActivation):
        def __init__(self) -> None:
            super().__init__()
            self.closed = asyncio.Event()

        async def wait_park_requested(self) -> None:
            await self.closed.wait()
            raise StaleAttemptError("agent session is closing")

    created: list[asyncio.Future] = []

    def ensure_future(awaitable):
        created.append(asyncio.ensure_future(awaitable))
        return created[-1]

    monkeypatch.setattr(checkpoint_pauses, "asyncio", AsyncioWith(ensure_future=ensure_future))
    session, activation = FakeSession(), RetiredWhenClosed()
    running = _run(session, activation)
    for _ in range(3):
        await asyncio.sleep(0)
    # A close cancels the activation and retires its session in the same turn of the loop.
    running.cancel()
    activation.closed.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 5)
    parking = created[1]
    assert parking.done() and not parking.cancelled()
    # A finished task nobody cancels or reads is logged as "Task exception was never retrieved" when collected.
    assert not parking._log_traceback
    assert isinstance(parking.exception(), StaleAttemptError)


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


STILL_STAGING = HarnessPauseResult(status="unconfirmed", error="the launch is still staging its files")


class StagingSession(FakeSession):
    """Its launch has not started the harness yet: a stop has nothing to end."""

    launch_started = False

    async def stop_harness(self, *, timeout: float) -> None:
        self.events.append("stop")


async def test_a_failed_pause_of_a_launch_still_staging_cancels_it_and_asks_for_a_relaunch() -> None:
    session, activation = StagingSession(pauses=[STILL_STAGING]), FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    # Left alone, the launch would run its harness through the checkpoint, or wait for a continue nobody sends.
    assert await asyncio.wait_for(running, 5) == LaunchStopped(receipt=None)
    assert session.events == ["pause", "stop", "launch_cancelled"]


class UncancellableStagingSession(StagingSession):
    """Its staging does not end when cancelled, as a stage callback that shields a write to the sandbox."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.release = asyncio.Event()
        self.task: asyncio.Task | None = None

    async def launch(self) -> str:
        self.task = asyncio.current_task()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.events.append("launch_cancelled")
        return "result"


async def test_a_launch_still_staging_that_does_not_end_when_cancelled_fails_only_once_it_has_ended() -> None:
    session, activation = UncancellableStagingSession(pauses=[STILL_STAGING]), FakeActivation()
    running = _run(session, activation, stop_timeout=0.05)
    activation.park.set()
    try:
        # While the staging can still write, the activation neither parks nor fails: either would let the checkpoint
        # take its snapshot.
        done, _ = await asyncio.wait({running}, timeout=0.5)
        assert not done and activation.boundaries == [] and not activation.parked.is_set()
        assert session.events[:3] == ["pause", "stop", "launch_cancelled"]
        session.release.set()
        with pytest.raises(RuntimeError, match="did not end within 0.05s of being cancelled while staging"):
            await asyncio.wait_for(running, 5)
        assert activation.boundaries == []
    finally:
        session.release.set()
        activation.resume.set()
    assert session.task is not None and await asyncio.wait_for(session.task, 5) == "result"


async def test_a_retire_ends_the_wait_for_a_staging_that_does_not_end() -> None:
    session, activation = UncancellableStagingSession(pauses=[STILL_STAGING]), FakeActivation()
    running = _run(session, activation, stop_timeout=0.05)
    activation.park.set()
    try:
        done, _ = await asyncio.wait({running}, timeout=0.3)
        assert not done
        # A retire or a close cancels the activation: its session is then no longer part of the checkpoint.
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 5)
        assert activation.boundaries == []
    finally:
        session.release.set()
    assert session.task is not None and await asyncio.wait_for(session.task, 5) == "result"


class AbandonedStagingSession(StagingSession):
    """Its launch ends when cancelled and leaves its staging running, as ``SandboxSession.execute`` does once it
    has waited its close timeout for a staging that ignores the cancellation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.staged = asyncio.Event()

    @property
    def staging(self) -> bool:
        return not self.staged.is_set()

    async def wait_staged(self, *, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self.staged.wait(), timeout)
        except TimeoutError:
            pass
        return not self.staging


async def test_a_cancelled_launch_whose_staging_still_runs_is_not_reported_stopped() -> None:
    session, activation = AbandonedStagingSession(pauses=[STILL_STAGING]), FakeActivation()
    running = _run(session, activation, stop_timeout=0.05)
    activation.park.set()
    done, _ = await asyncio.wait({running}, timeout=0.5)
    # The launch itself ended, cancelled; reporting it stopped would park the activation beside the staging.
    assert not done and session.events == ["pause", "stop", "launch_cancelled"]
    assert activation.boundaries == [] and not activation.parked.is_set()
    session.staged.set()
    with pytest.raises(RuntimeError, match="was cancelled while staging, and its staging has not ended"):
        await asyncio.wait_for(running, 5)
    assert activation.boundaries == []


class _NoSessions:
    async def export_agent_sessions(self, session_keys: list[str], *, checkpoint_id: str) -> dict[str, dict]:
        return {key: {} for key in session_keys}


@linux_only
async def test_a_checkpoint_waits_for_a_staging_that_does_not_end_when_cancelled(tmp_path: Path) -> None:
    """With the real session and the real checkpoint participant: no prepared checkpoint, and so no export and no
    snapshot, while a staging that ignored its cancellation can still write to the sandbox."""
    session = _session(tmp_path)
    launch_dir = tmp_path / "launch-1"
    staging, release = asyncio.Event(), asyncio.Event()

    async def stage() -> SandboxCommand:
        launch_dir.mkdir(parents=True, exist_ok=True)
        staging.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass  # as a stage callback that shields a write to the sandbox
        (launch_dir / "staged").write_text("late")
        return SandboxCommand(argv=["true"])

    # The pause gives up on the staging quickly; the default waits most of a minute.
    pause = session.pause_harness
    session.pause_harness = lambda: pause(timeout=0.05)  # type: ignore[method-assign]
    participant = AgentSessionParticipant(_NoSessions())
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    boundaries: list[dict] = []

    async def activate() -> None:
        async with participant.activation("s", episode_id) as activation:
            await run_with_checkpoint_pauses(
                session,
                activation,
                launch=lambda: session.execute(
                    stage_activation=stage, collect=AsyncMock(return_value="artifacts"), timeout=30, close_timeout=0.1
                ),
                on_parked=lambda: boundaries.append({"launch": 1}) or boundaries[-1],
                stop_timeout=2,
            )

    running = asyncio.create_task(activate())
    request = CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 60)
    prepare: asyncio.Task | None = None
    try:
        await asyncio.wait_for(staging.wait(), 5)
        prepare = asyncio.create_task(controller.prepare(request))
        # Long enough for the pause to give up, the launch to be cancelled and to stop waiting for its staging.
        done, _ = await asyncio.wait({prepare, running}, timeout=1.5)
        assert not done, "the checkpoint is not prepared and the activation has not failed while staging still runs"
        assert session.staging and boundaries == [] and not (launch_dir / "staged").exists()

        release.set()
        with pytest.raises(RuntimeError, match="cancelled while staging"):
            await asyncio.wait_for(running, 10)
        # Only now is the session ready, and the staging's last write is in the sandbox the snapshot would hold.
        prepared = await asyncio.wait_for(prepare, 10)
        assert prepared["phase"] == "prepared" and not session.staging
        assert (launch_dir / "staged").read_text() == "late" and boundaries == []
        records = await participant.export(None, checkpoint_id="c1")
        assert [(record.session_key, record.boundary) for record in records] == [("s", None)]
    finally:
        release.set()
        for task in (running, prepare):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (running, prepare) if task is not None), return_exceptions=True)
        await session.close(timeout=5)


class DeadSandboxSession(FakeSession):
    """Its sandbox died: the stop finds nothing to ask and leaves no receipt, and the launch fails."""

    async def launch(self) -> str:
        await self.done.wait()
        raise RuntimeError("sandbox OOM-killed")

    async def stop_harness(self, *, timeout: float) -> None:
        self.events.append("stop")
        self.done.set()


async def test_a_launch_whose_sandbox_died_raises_its_own_error_and_does_not_park() -> None:
    session, activation = DeadSandboxSession(pauses=[RuntimeError("exec failed")] * PAUSE_ATTEMPTS), FakeActivation()
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
    assert await asyncio.wait_for(running, 5) == LaunchFinished("result")
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
    assert isinstance(result, LaunchStopped) and result.receipt["stopped"] is True
    assert isinstance(result.error, RuntimeError) and str(result.error) == "capture failed"


async def test_an_error_of_a_launch_that_ended_on_its_own_as_the_stop_arrived_propagates() -> None:
    session = FailingLaunchSession(pauses=[HarnessPauseResult(status="unconfirmed")], stopped_receipt=False)
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    with pytest.raises(RuntimeError, match="capture failed"):
        await asyncio.wait_for(running, 5)
    assert activation.boundaries == [] and session.events == ["pause", "stop"]


class LostSupervisorSession(FakeSession):
    """Its supervisor died during the stop; the provider proved the sandbox empty, which stands in for the receipt."""

    async def stop_harness(self, *, timeout: float) -> None:
        self.events.append("stop")
        self.cleanup = {
            "cleanup_confirmed": True,
            "return_code": None,
            "error": None,
            "timed_out": False,
            "receipt_missing": True,
        }
        self.done.set()


class LostSupervisorFailingLaunchSession(LostSupervisorSession, FailingLaunchSession):
    pass


@pytest.mark.parametrize(
    ("session_type", "error"),
    [(LostSupervisorSession, None), (LostSupervisorFailingLaunchSession, "capture failed")],
    ids=["launch_returned", "launch_raised"],
)
async def test_a_stop_only_the_provider_could_prove_is_a_stopped_launch(
    session_type: type[FakeSession], error: str | None
) -> None:
    session = session_type(pauses=[HarnessPauseResult(status="unconfirmed", error="D state")])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    await asyncio.wait_for(activation.parked.wait(), 5)
    activation.resume.set()
    result = await asyncio.wait_for(running, 5)
    assert isinstance(result, LaunchStopped)
    assert result.receipt == session.cleanup and "stopped" not in result.receipt
    if error is None:
        assert result.error is None
    else:
        assert isinstance(result.error, RuntimeError) and str(result.error) == error
    assert session.events == ["pause", "stop"] and len(activation.boundaries) == 1


async def test_a_receipt_that_records_neither_a_stop_nor_a_lost_supervisor_is_a_finished_launch() -> None:
    class EarlierSupervisorSession(FakeSession):
        async def stop_harness(self, *, timeout: float) -> None:
            self.events.append("stop")
            self.cleanup = {"cleanup_confirmed": True, "return_code": 0, "error": None, "timed_out": False}
            self.done.set()

    session = EarlierSupervisorSession(pauses=[HarnessPauseResult(status="unconfirmed")])
    activation = FakeActivation()
    running = _run(session, activation)
    activation.park.set()
    assert await asyncio.wait_for(running, 5) == LaunchFinished("result")
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
        async def request(sandbox, *, session_dir, workdir, seq, timeout, launch_sent=False):
            requests.append((kind, seq))
            return answers[kind].pop(0)

        return request

    # sandbox_session imports both by name and calls them.
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_pause", answering("pause"))
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_continue", answering("continue"))
    session = _session(tmp_path)

    unanswered = await session.pause_harness()
    assert unanswered.status == "unconfirmed" and not unanswered.acknowledged
    refused = await session.pause_harness()
    # An answer of the supervisor, as opposed to no answer: a caller need not send the same request again.
    assert refused.status == "unconfirmed" and refused.error == refusal and refused.acknowledged
    assert (await session.pause_harness()).confirmed
    unanswered = await session.continue_harness()
    assert unanswered.status == "unconfirmed" and not unanswered.acknowledged
    assert (await session.continue_harness()).confirmed
    assert (await session.pause_harness()).confirmed
    assert requests == [("pause", 1)] * 3 + [("continue", 1)] * 2 + [("pause", 2)]


def _scripted_requests(monkeypatch: pytest.MonkeyPatch, answers: dict[str, list]) -> list[tuple[str, int]]:
    """Answer the session's pause and continue requests from ``answers``; returns the requests made."""
    requests: list[tuple[str, int]] = []

    def answering(kind: str):
        async def request(sandbox, *, session_dir, workdir, seq, timeout, launch_sent=False):
            requests.append((kind, seq))
            return answers[kind].pop(0)

        return request

    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_pause", answering("pause"))
    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_continue", answering("continue"))
    return requests


def _ack(seq: int, confirmed: bool = True, **fields) -> tuple:
    return ("acknowledged", {"seq": seq, "confirmed": confirmed, "pids": [], "error": None, **fields})


async def test_a_pause_after_the_launch_command_was_sent_is_recorded_with_the_supervisor_that_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = {"mode": "hold", "identity": "7.100"}
    requests = _scripted_requests(monkeypatch, {"pause": [_ack(1, **named)], "continue": [_ack(1, **named)]})
    session = _session(tmp_path, pause_mode="hold")
    session.launch_started = True

    # The launch command is on its way: the one request waits for its supervisor, whose answer names it.
    paused = await session.pause_harness()
    assert paused.confirmed and paused.identity == "7.100"
    state = session.checkpoint_state()
    assert state["pause_seq"] == 1 and state["supervisor_identity"] == "7.100"
    assert requests == [("pause", 1)]
    assert (await session.continue_harness()).confirmed
    assert "pause_seq" not in session.checkpoint_state()


async def test_a_hold_pause_of_a_sent_launch_whose_supervisor_did_not_start_is_unconfirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _scripted_requests(monkeypatch, {"pause": [("not_launched", None)], "continue": []})
    session = _session(tmp_path, pause_mode="hold")
    session.launch_started = True

    # The supervisor could still start before a snapshot that keeps processes, holding a pause nothing names.
    paused = await session.pause_harness()
    assert paused.status == "unconfirmed" and not paused.acknowledged
    assert paused.error is not None and "supervisor did not start" in paused.error
    state = session.checkpoint_state()
    assert "pause_seq" not in state and "supervisor_identity" not in state
    assert requests == [("pause", 1)]


async def test_a_stop_pause_of_a_sent_launch_whose_supervisor_did_not_start_is_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _scripted_requests(monkeypatch, {"pause": [("not_launched", None)], "continue": []})
    session = _session(tmp_path)
    session.launch_started = True

    # The snapshot keeps no processes: the launch reads the request when it starts, as before any launch.
    paused = await session.pause_harness()
    assert paused.confirmed and paused.identity is None
    assert requests == [("pause", 1)]


async def test_the_session_tells_the_request_whether_its_launch_command_was_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[bool] = []

    async def request(sandbox, *, session_dir, workdir, seq, timeout, launch_sent=False):
        sent.append(launch_sent)
        return ("not_launched", None)

    monkeypatch.setattr("nemo_gym.agents.utils.sandbox_session.request_pause", request)
    session = _session(tmp_path)
    await session.pause_harness()
    session.launch_started = True
    await session.pause_harness()
    assert sent == [False, True]


async def test_a_pause_before_the_launch_command_is_not_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _scripted_requests(monkeypatch, {"pause": [("not_launched", None)], "continue": []})
    session = _session(tmp_path, pause_mode="hold")

    assert (await session.pause_harness()).confirmed
    assert requests == [("pause", 1)]


async def test_checkpoint_state_records_a_pause_only_while_a_named_supervisor_holds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = {"mode": "hold", "identity": "7.100"}
    answers = {
        "pause": [
            _ack(1, confirmed=False, **named),
            _ack(1),  # A supervisor from before acknowledgements named it.
            _ack(1, **named),
            _ack(2, **named),
        ],
        "continue": [("timeout", None), _ack(1, **named)],
    }
    _scripted_requests(monkeypatch, answers)
    session = _session(tmp_path, pause_mode="hold")
    session.launch_started = True

    def held() -> tuple:
        state = session.checkpoint_state()
        return state.get("pause_seq"), state.get("supervisor_identity")

    # An unconfirmed pause, and a supervisor that does not name itself: nothing to find again.
    for _ in range(2):
        await session.pause_harness()
        assert held() == (None, None)
    paused = await session.pause_harness()
    assert paused.confirmed and paused.identity == "7.100" and paused.mode == "hold"
    assert held() == (1, "7.100")
    session.launch_started = False
    assert held() == (None, None)
    session.launch_started = True
    # A continue that was not confirmed leaves the pause held.
    assert (await session.continue_harness()).status == "unconfirmed"
    assert held() == (1, "7.100")
    assert (await session.continue_harness()).confirmed
    assert held() == (None, None)
    assert (await session.pause_harness()).confirmed
    assert held() == (2, "7.100")
    # The next activation has a fresh supervisor: nothing held carries over.
    session.launch_started = False
    await session.begin_activation(session_dir=str(tmp_path / "launch-2"), timeout=1)
    session.launch_started = True
    assert held() == (None, None)


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


def _without_the_request_margin(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Take the ten seconds a request adds to its timeout off its outer bound; return the bounds it asked for."""
    bounds: list[float] = []

    def timeout(delay: float) -> asyncio.Timeout:
        bounds.append(delay)
        return asyncio.timeout(delay - 10)

    monkeypatch.setattr(supervisor_client, "asyncio", AsyncioWith(timeout=timeout))
    return bounds


async def _outcome(request: asyncio.Task, within: float) -> asyncio.Task:
    """Wait for ``request`` without a timeout of the test's own, which would look like the request's."""
    done, _ = await asyncio.wait({request}, timeout=within)
    if not done:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
    assert done, "the request did not end by itself"
    return request


class StuckSandbox:
    """Its provider never runs the command, as one whose every slot is taken."""

    async def exec(self, command, *, cwd=None, timeout_s=None):
        await asyncio.Event().wait()


@pytest.mark.parametrize("send", [request_pause, request_continue], ids=["pause", "continue"])
async def test_a_request_the_provider_never_runs_ends_at_its_own_bound(send, monkeypatch: pytest.MonkeyPatch) -> None:
    bounds = _without_the_request_margin(monkeypatch)
    request = asyncio.create_task(send(StuckSandbox(), session_dir="/s", workdir=None, seq=1, timeout=0.01))
    with pytest.raises(TimeoutError):
        (await _outcome(request, within=10)).result()
    assert bounds == [pytest.approx(10.01)]


class ProviderSandbox:
    """The commands of one sandbox of a provider, as a request runs them."""

    def __init__(self, provider: LocalProvider, handle) -> None:
        self.provider, self.handle = provider, handle

    async def exec(self, command, *, cwd=None, timeout_s=None):
        return await self.provider.exec(self.handle, command, cwd=cwd, timeout_s=timeout_s)


async def test_requests_behind_a_full_local_provider_end_before_the_command_that_holds_its_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The provider shares one semaphore per concurrency between its instances; this test gets its own.
    monkeypatch.setattr(LocalProvider, "_semaphores", {})
    _without_the_request_margin(monkeypatch)
    provider = LocalProvider(workspace_root=str(tmp_path), concurrency=1)
    handle = await provider.create(SandboxSpec(workdir=str(tmp_path)))
    sandbox = ProviderSandbox(provider, handle)
    session_dir = tmp_path / "session"
    holder = asyncio.create_task(sandbox.exec("touch holding; sleep 120", timeout_s=300))
    try:
        async with asyncio.timeout(15):
            while not (tmp_path / "holding").exists():
                await asyncio.sleep(0.02)
        for send in (request_pause, request_continue):
            request = asyncio.create_task(
                send(sandbox, session_dir=str(session_dir), workdir=None, seq=1, timeout=0.2)
            )
            with pytest.raises(TimeoutError):
                (await _outcome(request, within=30)).result()
            assert not holder.done()
        # The requests never ran: they waited for the slot.
        assert not session_dir.exists()
    finally:
        holder.cancel()
        await asyncio.gather(holder, return_exceptions=True)
        await provider.close(handle)


async def test_a_request_runs_from_the_root_unless_its_session_directory_is_relative() -> None:
    sandbox = AsyncMock(spec=AsyncSandbox)
    sandbox.exec.return_value = SandboxExecResult(stdout="not_launched\n", stderr="", return_code=0)
    for send in (request_pause, request_continue):
        for session_dir in ("/tmp/session", ".gym/session"):
            assert await send(sandbox, session_dir=session_dir, workdir="/app", seq=1, timeout=1) == (
                "not_launched",
                None,
            )
    assert [call.kwargs["cwd"] for call in sandbox.exec.await_args_list] == ["/", "/app", "/", "/app"]


@linux_only
async def test_pause_and_continue_do_not_need_a_workdir_the_task_moved(tmp_path: Path) -> None:
    # A task may run `mv /app /app.bak`; Docker then refuses `exec -w /app`, and so does this local sandbox.
    sandbox = LocalSandbox()
    workdir, launch_dir = tmp_path / "app", tmp_path / "session"
    workdir.mkdir()
    session = SandboxSession(
        sandbox=sandbox,  # type: ignore[arg-type]
        session_dir=str(launch_dir),
        workdir=str(workdir),
        harness="test",
        pausable=True,
    )
    running = asyncio.create_task(
        session.execute(
            stage_activation=_ticking_stage(launch_dir),
            collect=AsyncMock(return_value="artifacts"),
            timeout=60,
            close_timeout=3,
        )
    )
    try:
        async with asyncio.timeout(15):
            while not (launch_dir / "ticks").exists():
                await asyncio.sleep(0.02)
        workdir.rename(tmp_path / "app.bak")
        outcome, paused = await request_pause(
            sandbox, session_dir=str(launch_dir), workdir=str(workdir), seq=1, timeout=20
        )
        assert outcome == "acknowledged" and paused["seq"] == 1 and paused["confirmed"] is True, (outcome, paused)
        frozen = (launch_dir / "ticks").read_text()
        await asyncio.sleep(0.3)
        assert (launch_dir / "ticks").read_text() == frozen
        outcome, continued = await request_continue(
            sandbox, session_dir=str(launch_dir), workdir=str(workdir), seq=1, timeout=20
        )
        assert outcome == "acknowledged" and continued["seq"] == 1 and continued["confirmed"] is True, continued
        # The harness that was paused runs again: nothing was launched in its place.
        await _ticks_grow(launch_dir, frozen)
        assert (launch_dir / "ticks").read_text().startswith(frozen)
        assert any(_state(pid) not in ("gone", "T", "t", "Z") for pid in paused["pids"]), paused
        (launch_dir / "finish").touch()
        assert await asyncio.wait_for(running, 30) == "artifacts"
        assert not workdir.exists()
    finally:
        await _end_launch(running, launch_dir)


@linux_only
async def test_run_with_checkpoint_pauses_pauses_a_real_supervised_harness(tmp_path: Path) -> None:
    session, activation = _session(tmp_path), FakeActivation()
    launch_dir = tmp_path / "launch-1"
    frozen: list[str] = []

    def bookmark() -> dict:
        frozen.append((launch_dir / "ticks").read_text())
        return {"ticks": len(frozen[-1])}

    running = asyncio.create_task(
        run_with_checkpoint_pauses(
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
        assert await asyncio.wait_for(running, 30) == LaunchFinished("artifacts")
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
