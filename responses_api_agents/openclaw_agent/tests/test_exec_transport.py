# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real supervisor through a provider with no PTY interface."""

import asyncio
import json
import os
import shutil
import signal
import sys
from pathlib import Path

import pytest

from nemo_gym.agents import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT, SUPERVISOR_FILE, SandboxSession
from nemo_gym.agents.utils.supervisor_client import LAUNCH_CLAIM_FILE, SUPERVISOR_PID_FILE
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.openclaw_agent.sandbox import OpenClawSandboxSession
from responses_api_agents.openclaw_agent.tests.test_sandbox import seed


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision required")

# These integration tests run real shells and scan /proc on shared CI runners.
# Tight cancellation deadlines are covered separately by the SandboxSession unit tests.
CLOSE_TIMEOUT = 10


class ExecOnlySandbox:
    # File copies do not wait for the supervised exec in this local provider.
    uploads_independent_of_execs = True

    def __init__(self):
        self.cancelled_launch = False
        self.lost_launch = False
        self.delayed_command = None
        self.disconnected = False
        self.deadline = None

    async def upload(self, source, destination):
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        shutil.copyfile(source, destination)

    async def disconnect(self):
        self.disconnected = True

    async def exec(self, command, *, cwd=None, timeout_s=30):
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
    # Mirror the directory prepared by the real seed hook before activation.
    (directory / "home/.openclaw").mkdir(parents=True)
    workdir = tmp_path / "task"
    workdir.mkdir()
    request = seed()
    request.sandbox_access.workdir = str(workdir)
    provider = ExecOnlySandbox()
    state = OpenClawSandboxSession(
        request=request,
        session=SandboxSession(
            sandbox=provider, session_dir=str(directory), workdir=request.sandbox_access.workdir, harness="OpenClaw"
        ),
        runtime=str(tmp_path / "runtime"),
    )
    shutil.copyfile(HARNESS_RUNNER_SCRIPT, directory / HARNESS_RUNNER_FILE)
    assert not (directory / "process_supervisor.sh").exists()
    return state, provider, workdir


def payload(state, code):
    return {
        "prompt": "task",
        "config": {},
        "cwd": state.request.sandbox_access.workdir,
        "command": [sys.executable, "-c", code],
        "env": {},
    }


@pytest.mark.parametrize("ending", ["natural", "crash", "timeout", "cancel", "close"])
@pytest.mark.parametrize("startup_delay", [0, 0.6])
async def test_exec_only_supervision_leaves_task_children_for_sandbox_owner(tmp_path, ending, startup_delay):
    state, provider, workdir = make_session(tmp_path)
    # Complete the tool before interruption so its detached service is already adopted.
    tool = (
        "import pathlib,subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "pathlib.Path('child.pid').write_text(str(p.pid))"
    )
    code = (
        "import os,subprocess,sys,time,pathlib; "
        f"time.sleep({startup_delay}); "
        f"subprocess.run([sys.executable,'-c',{tool!r}],check=True); "
        "pathlib.Path('ready').touch(); "
        + {"natural": "print('{}')", "crash": "raise SystemExit(7)"}.get(ending, "time.sleep(60)")
    )
    # Expire only after the detached child is ready; startup speed is not under test.
    deadline = 60
    task = state.task = asyncio.create_task(
        state.execute(payload(state, code), timeout=deadline, close_timeout=CLOSE_TIMEOUT)
    )
    try:
        async with asyncio.timeout(30):
            while not (workdir / "ready").exists():
                if task.done():
                    await task
                    pytest.fail("Harness exited before its detached child was ready")
                await asyncio.sleep(0.01)
        if ending == "timeout":
            assert await state.session.expire(timeout=CLOSE_TIMEOUT)
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif ending == "close":
            await state.close(CLOSE_TIMEOUT)
            await asyncio.gather(task, return_exceptions=True)
        else:
            await task
        assert state.session.cleanup["cleanup_confirmed"] is True
        if ending == "natural":
            assert state.session.artifacts["stdout.log"] == "{}\n"
            assert state.runtime_info is not None
        if ending == "crash":
            assert state.session.cleanup["return_code"] == 7
        if ending == "timeout":
            assert state.session.cleanup["timed_out"] is True
        child_pid = int((workdir / "child.pid").read_text())
        assert Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        assert provider.cancelled_launch is False
        assert provider.deadline == pytest.approx(deadline + CLOSE_TIMEOUT + 30)
        await state.close(CLOSE_TIMEOUT)
        await state.close(CLOSE_TIMEOUT)
        assert provider.disconnected
        assert not Path(state.session.session_dir).exists()
        # Resources still owns the detached service after every ending.
        assert Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    finally:
        try:
            if not state.session.closed:
                await state.close(CLOSE_TIMEOUT)
        finally:
            if (workdir / "child.pid").exists():
                try:
                    os.kill(int((workdir / "child.pid").read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


async def test_supervisor_and_runner_run_under_neutral_names(tmp_path):
    """A task agent's `pkill -f python3` or `killall python3` must not match the processes above OpenClaw."""
    state, provider, workdir = make_session(tmp_path)
    code = "import pathlib,time; pathlib.Path('ready').touch(); time.sleep(60)"
    task = state.task = asyncio.create_task(
        state.execute(payload(state, code), timeout=60, close_timeout=CLOSE_TIMEOUT)
    )
    try:
        async with asyncio.timeout(30):
            while not (workdir / "ready").exists():
                await asyncio.sleep(0.01)
        supervisor_pid = int((Path(state.session.session_dir) / "supervisor.pid").read_text())
        # The supervisor is bash, and the bash runner execs the CLI, so no process above the CLI is python.
        assert Path(f"/proc/{supervisor_pid}/comm").read_text().strip() == "bash"
        cmdline = Path(f"/proc/{supervisor_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert "python" not in cmdline and ".py" not in cmdline, cmdline
    finally:
        await state.close(CLOSE_TIMEOUT)
        await asyncio.gather(task, return_exceptions=True)
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert provider.disconnected


@pytest.mark.parametrize("retired", [False, True], ids=["before_retirement", "after_retirement"])
async def test_lost_launch_is_fenced_before_and_after_directory_retirement(tmp_path, retired):
    state, provider, workdir = make_session(tmp_path)
    provider.lost_launch = True
    with pytest.raises(TimeoutError, match="lost launch response"):
        await state.execute(payload(state, "open('started','w').close()"), timeout=10, close_timeout=CLOSE_TIMEOUT)
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert state.session.cleanup["return_code"] is None
    assert state.session.cleanup["error"] is None
    assert state.runtime_info is None
    provider.lost_launch = False
    session_dir = Path(state.session.session_dir)
    if retired:
        await state.close(CLOSE_TIMEOUT)
    else:
        # The supervisor is still in place, so only the stop's launch claim can fence the delayed command.
        assert (session_dir / SUPERVISOR_FILE).exists()
    await provider.exec(provider.delayed_command, cwd=str(workdir))
    # Under a broken claim the stop file halts the harness before it writes this; the PID file shows the start.
    assert not (workdir / "started").exists()
    if not retired:
        assert not (session_dir / SUPERVISOR_PID_FILE).exists()
        assert (session_dir / LAUNCH_CLAIM_FILE).readlink() == Path("stop")
        await state.close(CLOSE_TIMEOUT)
    assert not session_dir.exists()


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
    await state.close(CLOSE_TIMEOUT)
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
    await state.close(CLOSE_TIMEOUT)
    assert state.session._exec_task.cancelled()
    assert state.session.closed and provider.disconnected
    assert not Path(state.session.session_dir).exists()
