# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from subprocess import TimeoutExpired
from unittest.mock import MagicMock

import psutil
import pytest
from omegaconf import OmegaConf

import nemo_gym.server_runtime.supervisor.server_supervisor as server_supervisor
from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.global_config import SERVER_SHUTDOWN_TIMEOUT_SECONDS_KEY_NAME
from nemo_gym.server_runtime.supervisor.server_venv import run_command


def _runner(processes, grace_s=0.1):
    runner = server_supervisor.RunHelper()
    runner._processes = processes
    runner._head_server = None
    runner._server_shutdown_timeout_seconds = grace_s
    return runner


def _wait_for(path, process, timeout_s=5):
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        assert process.poll() is None and time.monotonic() < deadline
        time.sleep(0.01)


def _launch(tmp_path, script, *, logged=False):
    script_path = tmp_path / "server.py"
    script_path.write_text(script)
    config = OmegaConf.create(
        {"uv_cache_dir": str(tmp_path), "nemo_gym_log_dir": str(tmp_path / "logs") if logged else None}
    )
    return run_command(
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))}",
        tmp_path,
        server_name="server",
        global_config_dict=config,
        start_new_session=True,
    )


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def test_logged_server_has_own_session_and_receives_one_interrupt(tmp_path):
    ready, interrupts = tmp_path / "ready", tmp_path / "interrupts"
    process = _launch(
        tmp_path,
        f"""
import os, signal, time
from pathlib import Path
interrupts = Path({str(interrupts)!r})
def stop(*_):
    interrupts.write_text(interrupts.read_text() + 'SIGINT\\n' if interrupts.exists() else 'SIGINT\\n')
signal.signal(signal.SIGINT, stop)
Path({str(ready)!r}).write_text(str(os.getsid(0)))
while not interrupts.exists():
    time.sleep(0.01)
time.sleep(0.1)
print('cleanup finished', flush=True)
""",
        logged=True,
    )
    try:
        _wait_for(ready, process)
        assert int(ready.read_text()) == process.pid
        assert process.pid != os.getsid(0)
        runner = _runner({"server": process}, grace_s=2)
        runner.shutdown()
        runner.shutdown()
        assert process.returncode == 0
        assert interrupts.read_text() == "SIGINT\n"
        assert "cleanup finished" in (tmp_path / "logs/server.log").read_text()
    finally:
        _kill_group(process)


def test_server_launched_with_the_interrupt_blocked_still_receives_it(tmp_path):
    # Ray blocks SIGINT in every thread of its workers, and a process starts with the mask of the thread that launched
    # it: every server that a Ray worker launches starts with the interrupt blocked.
    # This test covers the blocked mask passing through the launch shell and the log tee. Its stand-in calls the helper
    # itself, so the call in `run_webserver` is covered by the unit test in
    # tests/unit_tests/server_runtime/test_base_server.py (test_uvicorn_starts_with_the_shutdown_signals_unblocked).
    started_blocked, ready, stopped = tmp_path / "started_blocked", tmp_path / "ready", tmp_path / "stopped"
    script = f"""
import signal, sys, time
from pathlib import Path
from nemo_gym.server_runtime.base_server import _unblock_shutdown_signals
blocked = signal.pthread_sigmask(signal.SIG_BLOCK, ())  # reads the mask, adds nothing
Path({str(started_blocked)!r}).write_text(str(signal.SIGINT in blocked))
_unblock_shutdown_signals()  # as run_webserver does first
def stop(*_):
    print('server stopping', flush=True)
    Path({str(stopped)!r}).touch()
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
Path({str(ready)!r}).touch()
while True:
    time.sleep(0.01)
"""

    def launch_from_a_thread_that_blocks_the_interrupt():
        signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGINT,))
        return _launch(tmp_path, script, logged=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        process = pool.submit(launch_from_a_thread_that_blocks_the_interrupt).result()
    try:
        _wait_for(ready, process, timeout_s=30)  # the stand-in imports the server runtime first
        assert started_blocked.read_text() == "True"  # the block reached it through its launch shell and log tee
        _runner({"server": process}, grace_s=10).shutdown()
        assert stopped.exists()  # yet it ran its own shutdown
        assert process.returncode == 0
        assert "server stopping" in (tmp_path / "logs/server.log").read_text()
    finally:
        _kill_group(process)


def test_shutdown_kills_descendant_after_its_wrapper_exited(tmp_path):
    ready = tmp_path / "child-pid"
    child = f"""
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGINT, signal.SIG_IGN)
Path({str(ready)!r}).write_text(str(os.getpid()))
while True:
    time.sleep(0.01)
"""
    process = _launch(
        tmp_path,
        f"""
import subprocess, sys
subprocess.Popen([sys.executable, '-c', {child!r}], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
""",
    )
    try:
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        child_pid = int(ready.read_text())
        assert os.getpgid(child_pid) == process.pid
        _runner({"server": process}).shutdown()
        deadline = time.monotonic() + 2
        while psutil.pid_exists(child_pid) and psutil.Process(child_pid).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        _kill_group(process)


def test_shutdown_tolerates_an_already_exited_group(tmp_path):
    process = _launch(tmp_path, "pass")
    process.wait(timeout=5)
    runner = _runner({"finished": process})
    runner.shutdown()
    assert runner._processes == {}


@pytest.mark.parametrize("value", [True, False, -1, "-1", "nan", "inf", float("-inf"), "later", []])
def test_invalid_shutdown_budget_is_rejected(value):
    with pytest.raises(ConfigError, match="finite nonnegative"):
        server_supervisor._server_shutdown_timeout_seconds(
            OmegaConf.create({SERVER_SHUTDOWN_TIMEOUT_SECONDS_KEY_NAME: value})
        )


@pytest.mark.parametrize(("value", "expected"), [(None, 1), (0, 0), ("0.5", 0.5), (2, 2)])
def test_shutdown_budget_accepts_finite_seconds(value, expected):
    assert (
        server_supervisor._server_shutdown_timeout_seconds(
            OmegaConf.create({SERVER_SHUTDOWN_TIMEOUT_SECONDS_KEY_NAME: value})
        )
        == expected
    )


def test_shutdown_uses_shared_grace_and_reap_deadlines(monkeypatch):
    now = [0.0]
    calls = []
    processes = [MagicMock(name="first"), MagicMock(name="second")]
    for process in processes:

        def wait(timeout, process=process):
            calls.append(("wait", process, timeout))
            now[0] += timeout
            raise TimeoutExpired("server", timeout)

        process.wait.side_effect = wait
    monkeypatch.setattr(server_supervisor, "monotonic", lambda: now[0])
    monkeypatch.setattr(server_supervisor, "_server_group_exists", lambda process: True)
    monkeypatch.setattr(
        server_supervisor, "_signal_server_group", lambda process, sig: calls.append(("signal", process, sig))
    )
    _runner(dict(zip(("first", "second"), processes)), grace_s=0.3).shutdown()
    waits = [call[2] for call in calls if call[0] == "wait"]
    assert waits == pytest.approx([0.3, 0, server_supervisor._FORCE_KILL_REAP_TIMEOUT_SEC, 0])
    kill_indexes = [i for i, call in enumerate(calls) if call[0] == "signal" and call[2] == signal.SIGKILL]
    assert len(kill_indexes) == 2
    assert max(kill_indexes) < next(
        i
        for i, call in enumerate(calls)
        if call[0] == "wait" and call[2] == pytest.approx(server_supervisor._FORCE_KILL_REAP_TIMEOUT_SEC)
    )
