# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real supervisor and Kilo runner through a provider with no PTY interface."""

import asyncio
import json
import os
import shutil
import signal
import sys
from pathlib import Path

import pytest

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.sandbox.providers.base import SandboxExecResult
from responses_api_agents.kilocode_agent import sandbox_runner
from responses_api_agents.kilocode_agent.sandbox import KiloSandboxSession
from responses_api_agents.kilocode_agent.tests.test_sandbox import seed


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux child-subreaper and /proc required")


class ExecOnlySandbox:
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
        launch = "--receipt" in command and "process_supervisor" in command
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
    directory.mkdir()
    workdir = tmp_path / "task"
    workdir.mkdir()
    provider = ExecOnlySandbox()
    state = KiloSandboxSession(
        request=seed(workdir=str(workdir)),
        provider_name="local",
        session=SandboxSession(sandbox=provider, session_dir=str(directory), workdir=str(workdir), harness="KiloCode"),
    )
    shutil.copyfile(sandbox_runner.__file__, directory / "sandbox_runner.py")
    assert not (directory / "process_supervisor").exists()
    return state, provider, workdir


def activate(state, code, timeout=0.5):
    return state.session.execute(
        stage_activation=lambda: state.stage_activation(command=[sys.executable, "-c", code], config="{}"),
        collect=state.collect_artifacts,
        timeout=timeout,
        close_timeout=3,
    )


@pytest.mark.parametrize("ending", ["natural", "timeout", "cancel", "close"])
async def test_exec_only_supervision_reaps_detached_child(tmp_path, ending):
    state, provider, workdir = make_session(tmp_path)
    code = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "pathlib.Path('child.pid').write_text(str(p.pid)); "
        + ("print('{}')" if ending == "natural" else "time.sleep(60)")
    )
    deadline = 0.5 if ending == "timeout" else 60
    task = state.task = asyncio.create_task(activate(state, code, timeout=deadline))
    try:
        async with asyncio.timeout(5):
            while not (workdir / "child.pid").exists():
                await asyncio.sleep(0.01)
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif ending == "close":
            await state.close(timeout=3)
            await asyncio.gather(task, return_exceptions=True)
        else:
            artifacts = await task
            assert artifacts.exit_code == (0 if ending == "natural" else None)
        assert state.session.cleanup["cleanup_confirmed"] is True
        if ending == "timeout":
            assert state.session.cleanup["timed_out"] is True
        child_pid = int((workdir / "child.pid").read_text())
        if ending == "natural":
            # Natural CLI exit has no adoption cutoff; the sandbox owner tears down detached tools.
            os.kill(child_pid, 0)
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
        assert provider.cancelled_launch is False
        assert provider.deadline > deadline + 3 * 1
        await state.close(timeout=3)
        await state.close(timeout=3)
        assert provider.disconnected
        assert not Path(state.session.session_dir).exists()
    finally:
        try:
            if not state.session.closed:
                await state.close(timeout=3)
        finally:
            if (workdir / "child.pid").exists():
                try:
                    os.kill(int((workdir / "child.pid").read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


async def test_lost_launch_is_fenced_even_after_directory_retirement(tmp_path):
    state, provider, workdir = make_session(tmp_path)
    provider.lost_launch = True
    with pytest.raises(TimeoutError, match="lost launch response"):
        await activate(state, "open('started','w').close()")
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert state.session.cleanup["return_code"] is None
    assert state.session.cleanup["error"] is None
    await state.close(timeout=3)
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
    await state.close(timeout=3)
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
    await state.close(timeout=0.5)
    assert state.session._exec_task.cancelled()
    assert state.session.closed and provider.disconnected
    assert not Path(state.session.session_dir).exists()
