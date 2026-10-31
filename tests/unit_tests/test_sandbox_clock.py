# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The clock helpers for a restore that keeps a sandbox's processes: their shell commands run for real here, with a
clock setter that does not touch the host's clock."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents.utils.sandbox_clock import (
    CLOCK_SYNC_SERVICES,
    ClockRewind,
    disable_clock_sync,
    disable_clock_sync_command,
    parse_clock_rewind,
    rewind_clock,
    rewind_clock_command,
)
from nemo_gym.sandbox.providers.base import SandboxExecResult


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")


def _uptime() -> float:
    return float(Path("/proc/uptime").read_text().split()[0])


def _run(command: str) -> ClockRewind:
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return parse_clock_rewind(result.stdout)


@linux_only
def test_a_clock_that_kept_running_from_the_pause_is_left_alone() -> None:
    rewind = _run(
        rewind_clock_command(pause_wall=time.time(), pause_uptime=_uptime(), touch_dirs=[], set_clock="false")
    )

    assert rewind == ClockRewind(applied=False, shift_s=rewind.shift_s, reason="continuous")
    assert abs(rewind.shift_s) < 1


@linux_only
def test_a_clock_that_jumped_is_set_to_the_pause_plus_the_uptime_since(tmp_path: Path) -> None:
    # The sandbox paused 1000 s of wall time ago but has run only 10 s since: it was stopped for 990 s.
    record = tmp_path / "set"
    setter = tmp_path / "set_clock"
    setter.write_text(f'#!/bin/bash\necho "$*" > {record}\n')
    setter.chmod(0o755)
    now, uptime = time.time(), _uptime()

    rewind = _run(
        rewind_clock_command(
            pause_wall=now - 1000, pause_uptime=uptime - 10, touch_dirs=[], set_clock=str(setter), verify=False
        )
    )

    assert rewind.applied and abs(rewind.shift_s - 990) < 2, rewind
    set_to = float(record.read_text().strip().removeprefix("@"))
    assert abs(set_to - (now - 990)) < 2, set_to


@linux_only
def test_files_written_after_the_resume_get_the_rewound_time(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    old = work / "old.txt"
    old.write_text("before the stop")
    os.utime(old, (time.time() - 5000, time.time() - 5000))
    old_mtime = old.stat().st_mtime
    new = work / "new.txt"
    new.write_text("written between the resume and the rewind")
    now, uptime = time.time(), _uptime()

    rewind = _run(
        rewind_clock_command(
            pause_wall=now - 1000, pause_uptime=uptime - 10, touch_dirs=[str(work)], set_clock="true", verify=False
        )
    )

    assert rewind.applied and rewind.touched >= 1, rewind
    assert abs(new.stat().st_mtime - (now - 990)) < 2
    assert old.stat().st_mtime == old_mtime


@linux_only
@pytest.mark.parametrize(
    ("set_clock", "verify", "reason"), [("false", True, "set_failed"), ("true", True, "readback_mismatch")]
)
def test_a_clock_that_cannot_be_set_is_reported_not_applied(set_clock: str, verify: bool, reason: str) -> None:
    rewind = _run(
        rewind_clock_command(
            pause_wall=time.time() - 1000, pause_uptime=_uptime(), touch_dirs=[], set_clock=set_clock, verify=verify
        )
    )

    assert not rewind.applied and rewind.reason == reason and rewind.shift_s > 900, rewind


def test_unreadable_output_or_a_failed_exec_is_a_check_failure() -> None:
    assert parse_clock_rewind("") == ClockRewind(applied=False, reason="check_failed")
    assert parse_clock_rewind("not json") == ClockRewind(applied=False, reason="check_failed")
    assert parse_clock_rewind('{"applied": true}', error="timeout").reason == "check_failed"


@linux_only
def test_disable_clock_sync_ends_a_running_time_daemon(tmp_path: Path) -> None:
    # A copy of sleep under a time daemon's name: no systemd unit has that name, so the /proc scan ends it.
    daemon = tmp_path / "ngclockdaemon"
    shutil.copy(shutil.which("sleep"), daemon)
    process = subprocess.Popen([str(daemon), "60"])
    try:
        time.sleep(0.2)
        result = subprocess.run(
            ["bash", "-c", disable_clock_sync_command(["ngclockdaemon", "ngabsentdaemon"])],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["ngclockdaemon"]
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            process.kill()


async def test_the_helpers_run_one_root_command_each() -> None:
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(stdout="systemd-timesyncd\n", stderr=None, return_code=0)

    stopped = await disable_clock_sync(sandbox, timeout=30)

    assert stopped == ("systemd-timesyncd",)
    command = sandbox.exec.await_args.args[0]
    assert all(service in command for service in CLOCK_SYNC_SERVICES)
    assert sandbox.exec.await_args.kwargs["user"] == "root"

    sandbox.exec.return_value = SandboxExecResult(
        stdout='{"applied": true, "shift_ms": 86400000, "touched": 3, "target": "1.000"}\n', stderr=None, return_code=0
    )
    rewind = await rewind_clock(sandbox, pause_wall=1.0, pause_uptime=2.0, touch_dirs=["/work"], timeout=30)

    assert rewind == ClockRewind(applied=True, shift_s=86400.0, touched=3)
    assert sandbox.exec.await_args.kwargs["user"] == "root" and "/work" in sandbox.exec.await_args.args[0]
