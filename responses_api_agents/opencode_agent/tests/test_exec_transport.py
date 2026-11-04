# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real supervisor through a provider with no PTY interface."""

import asyncio
import json
import os
import shutil
import signal
import sqlite3
import sys
from pathlib import Path

import pytest

from nemo_gym.agents import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT, SandboxSession
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision required")


class ExecOnlySandbox:
    def __init__(self):
        self.cancelled_launch = False
        self.lost_launch = False
        self.delayed_command = None
        self.disconnected = False
        self.deadline = None
        self.snapshots = 0
        self.worker_pid = None

    async def upload(self, source, destination):
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        shutil.copyfile(source, destination)

    async def disconnect(self):
        self.disconnected = True

    async def exec(self, command, *, cwd=None, timeout_s=30):
        if "opencode.db-wal" in command:
            self.snapshots += 1
            # The CLI writer must finish before capture; task services can remain.
            if self.worker_pid is not None:
                with pytest.raises(ProcessLookupError):
                    os.kill(int(self.worker_pid.read_text()), 0)
        launch = "--receipt" in command and "process_supervisor.sh" in command
        if launch:
            self.deadline = timeout_s
            self.delayed_command = command
            if self.lost_launch:
                raise TimeoutError("lost launch response")
        process = await asyncio.create_subprocess_shell(
            command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        except BaseException:
            self.cancelled_launch |= launch
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)


def make_session(tmp_path):
    directory = tmp_path / "session"
    directory.mkdir()
    workdir = tmp_path / "task"
    workdir.mkdir()
    request = seed()
    request.sandbox_access.workdir = str(workdir)
    provider = ExecOnlySandbox()
    state = OpenCodeSandboxSession(
        request=request,
        session=SandboxSession(
            sandbox=provider, session_dir=str(directory), workdir=request.sandbox_access.workdir, harness="OpenCode"
        ),
        runtime=str(tmp_path / "runtime"),
    )
    shutil.copyfile(HARNESS_RUNNER_SCRIPT, directory / HARNESS_RUNNER_FILE)
    assert not (directory / "process_supervisor.sh").exists()
    database = directory / "data/opencode/opencode.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            create table session(id text, parent_id text, time_created integer);
            create table message(id text, session_id text, data text, time_created integer);
            create table part(id text, message_id text, session_id text, data text, time_created integer);
            insert into session values('root', null, 0);
        """)
    return state, provider, workdir


def payload(state, code):
    directory = state.session.session_dir
    return {
        "prompt": "task",
        "cwd": state.request.sandbox_access.workdir,
        "command": [sys.executable, "-c", code],
        "env": {
            key: f"{directory}/{name}"
            for key, name in (
                ("HOME", "home"),
                ("XDG_DATA_HOME", "data"),
                ("XDG_CONFIG_HOME", "config"),
                ("XDG_CACHE_HOME", "cache"),
                ("XDG_STATE_HOME", "state"),
            )
        },
    }


@pytest.mark.parametrize("ending", ["natural", "crash", "timeout", "cancel", "close"])
async def test_exec_only_supervision_leaves_task_children_for_sandbox_owner(tmp_path, ending):
    state, provider, workdir = make_session(tmp_path)
    provider.worker_pid = workdir / "worker.pid"
    # Complete the tool before interruption so its detached service is already adopted.
    tool = (
        "import pathlib,subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "pathlib.Path('child.pid').write_text(str(p.pid))"
    )
    code = (
        "import os,subprocess,sys,time,pathlib; "
        f"pathlib.Path({state.session.session_dir!r},'task-environment.ready').write_text('ready\\n'); "
        "pathlib.Path('worker.pid').write_text(str(os.getpid())); "
        f"subprocess.run([sys.executable,'-c',{tool!r}],check=True); "
        "pathlib.Path('ready').touch(); "
        + {"natural": "print('{}')", "crash": "raise SystemExit(7)"}.get(ending, "time.sleep(60)")
    )
    deadline = 0.5 if ending == "timeout" else 60
    task = state.task = asyncio.create_task(state.execute(payload(state, code), timeout=deadline, close_timeout=3))
    try:
        async with asyncio.timeout(5):
            while not (workdir / "ready").exists():
                await asyncio.sleep(0.01)
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif ending == "close":
            await state.close(3)
            await asyncio.gather(task, return_exceptions=True)
        else:
            await task
        assert state.session.cleanup["cleanup_confirmed"] is True
        if ending == "timeout":
            assert state.session.cleanup["timed_out"] is True
        child_pid = int((workdir / "child.pid").read_text())
        assert Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        assert provider.cancelled_launch is False
        assert provider.deadline == pytest.approx(deadline + 3 + 30)
        assert provider.snapshots == 1
        assert state.observations is not None
        assert any(getattr(record, "invocation_id", None) == "root" for record in state.observations.records)
        assert "observation_capture_failed" not in {gap.code for gap in state.observations.gaps}
        await state.close(3)
        await state.close(3)
        assert provider.disconnected
        assert not Path(state.session.session_dir).exists()
        # Resources still owns the detached service after every ending.
        assert Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    finally:
        try:
            if not state.session.closed:
                await state.close(3)
        finally:
            if (workdir / "child.pid").exists():
                try:
                    os.kill(int((workdir / "child.pid").read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


async def test_supervisor_and_runner_run_under_neutral_names(tmp_path):
    """A task agent's `pkill -f python3` or `killall python3` must not match the processes above OpenCode."""
    state, provider, workdir = make_session(tmp_path)
    code = (
        "import pathlib,time; "
        f"pathlib.Path({state.session.session_dir!r},'task-environment.ready').write_text('ready\\n'); "
        "pathlib.Path('ready').touch(); time.sleep(60)"
    )
    task = state.task = asyncio.create_task(state.execute(payload(state, code), timeout=60, close_timeout=3))
    try:
        async with asyncio.timeout(5):
            while not (workdir / "ready").exists():
                await asyncio.sleep(0.01)
        supervisor_pid = int((Path(state.session.session_dir) / "supervisor.pid").read_text())
        # The supervisor is bash, and the bash runner execs the CLI, so no process above the CLI is python.
        assert Path(f"/proc/{supervisor_pid}/comm").read_text().strip() == "bash"
        cmdline = Path(f"/proc/{supervisor_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert "python" not in cmdline and ".py" not in cmdline, cmdline
    finally:
        await state.close(3)
        await asyncio.gather(task, return_exceptions=True)
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert provider.disconnected


async def test_lost_launch_is_fenced_even_after_directory_retirement(tmp_path):
    state, provider, workdir = make_session(tmp_path)
    provider.lost_launch = True
    with pytest.raises(TimeoutError, match="lost launch response"):
        await state.execute(payload(state, "open('started','w').close()"), timeout=0.5, close_timeout=3)
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert state.session.cleanup["return_code"] is None
    assert state.session.cleanup["error"] is None
    assert state.runtime_info is None
    await state.close(3)
    provider.lost_launch = False
    await provider.exec(provider.delayed_command, cwd=str(workdir))
    assert not (workdir / "started").exists()
    assert not Path(state.session.session_dir).exists()


async def test_terminal_failed_receipt_releases_borrowed_session(tmp_path):
    state, provider, _ = make_session(tmp_path)
    state.session.launch_started = True
    receipt = {
        "return_code": 1,
        "timed_out": False,
        "cleanup_confirmed": False,
        "error": "harness termination unconfirmed",
    }
    path = Path(state.session.session_dir) / "cleanup.json"
    path.write_text(json.dumps(receipt))
    await state.close(3)
    assert provider.disconnected
    assert state.session.closed
    assert state.session.cleanup is None
    assert state.session.termination_error.receipt == receipt
    assert state.session.capture_error is state.session.termination_error
    assert not path.exists()


async def test_confirmed_cleanup_cancels_stuck_transport(tmp_path):
    state, provider, _ = make_session(tmp_path)
    state.session.launch_started = True
    receipt = {
        "return_code": 0,
        "timed_out": False,
        "cleanup_confirmed": True,
        "error": None,
    }
    (Path(state.session.session_dir) / "cleanup.json").write_text(json.dumps(receipt))
    state.session._exec_task = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    await state.close(0.5)
    assert state.session._exec_task.cancelled()
    assert state.session.closed and provider.disconnected
    assert not Path(state.session.session_dir).exists()
