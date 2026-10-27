# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import signal
import sys
import time
from subprocess import TimeoutExpired
from unittest.mock import MagicMock

import psutil
import pytest
from omegaconf import OmegaConf

import nemo_gym.cli.env as env
from nemo_gym.cli.setup_command import run_command
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import SERVER_SHUTDOWN_TIMEOUT_SECONDS_KEY_NAME


def _runner(processes, grace=0.1):
    runner = env.RunHelper()
    runner._processes = processes
    runner._head_server = None
    runner._server_shutdown_timeout_seconds = grace
    return runner


def _wait_for(path, process):
    deadline = time.monotonic() + 5
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
        runner = _runner({"server": process}, grace=2)
        runner.shutdown()
        runner.shutdown()
        assert process.returncode == 0
        assert interrupts.read_text() == "SIGINT\n"
        assert "cleanup finished" in (tmp_path / "logs/server.log").read_text()
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
        env._server_shutdown_timeout_seconds(OmegaConf.create({SERVER_SHUTDOWN_TIMEOUT_SECONDS_KEY_NAME: value}))


@pytest.mark.parametrize(("value", "expected"), [(None, 1), (0, 0), ("0.5", 0.5), (2, 2)])
def test_shutdown_budget_accepts_finite_seconds(value, expected):
    assert (
        env._server_shutdown_timeout_seconds(OmegaConf.create({SERVER_SHUTDOWN_TIMEOUT_SECONDS_KEY_NAME: value}))
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
    monkeypatch.setattr(env, "monotonic", lambda: now[0])
    monkeypatch.setattr(env, "_server_group_exists", lambda process: True)
    monkeypatch.setattr(env, "_signal_server_group", lambda process, sig: calls.append(("signal", process, sig)))
    _runner(dict(zip(("first", "second"), processes)), grace=0.3).shutdown()
    waits = [call[2] for call in calls if call[0] == "wait"]
    assert waits == pytest.approx([0.3, 0, env._FORCE_KILL_REAP_TIMEOUT_SEC, 0])
    kill_indexes = [i for i, call in enumerate(calls) if call[0] == "signal" and call[2] == signal.SIGKILL]
    assert len(kill_indexes) == 2
    assert max(kill_indexes) < next(
        i
        for i, call in enumerate(calls)
        if call[0] == "wait" and call[2] == pytest.approx(env._FORCE_KILL_REAP_TIMEOUT_SEC)
    )
