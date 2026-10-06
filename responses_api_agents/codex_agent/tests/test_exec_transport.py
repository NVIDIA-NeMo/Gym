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

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.sandbox.providers.base import SandboxExecResult
from responses_api_agents.codex_agent import sandbox_runner
from responses_api_agents.codex_agent.sandbox import CodexSandboxSession
from responses_api_agents.codex_agent.tests.test_native_sessions import seed


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
        launch = "--receipt" in command and "process_supervisor.py" in command
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
    state = CodexSandboxSession(
        request=request,
        session=SandboxSession(
            sandbox=provider, session_dir=str(directory), workdir=request.sandbox_access.workdir, harness="Codex"
        ),
        runtime=str(tmp_path / "runtime"),
    )
    shutil.copyfile(sandbox_runner.__file__, directory / "sandbox_runner.py")
    return state, provider, workdir


def payload(state, code, timeout=0.5):
    return {
        "directory": state.session.session_dir,
        "prompt": "task",
        "cwd": state.request.sandbox_access.workdir,
        "command": [sys.executable, "-c", code],
        "env": {},
        "timeout": timeout,
        "cleanup_timeout": 1,
    }


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
    task = state.task = asyncio.create_task(
        state.execute(payload(state, code, timeout=deadline), timeout=deadline, close_timeout=3)
    )
    try:
        async with asyncio.timeout(5):
            while not (workdir / "child.pid").exists():
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
        with pytest.raises(ProcessLookupError):
            os.kill(int((workdir / "child.pid").read_text()), 0)
        assert provider.cancelled_launch is False
        assert provider.deadline > deadline + 3 * 1
        await state.close(3)
        await state.close(3)
        assert provider.disconnected
        assert not Path(state.session.session_dir).exists()
    finally:
        if not state.session.closed:
            await state.close(3)


async def test_lost_launch_is_fenced_even_after_directory_retirement(tmp_path):
    state, provider, workdir = make_session(tmp_path)
    provider.lost_launch = True
    with pytest.raises(TimeoutError, match="lost launch response"):
        await state.execute(payload(state, "open('started','w').close()"), timeout=0.5, close_timeout=3)
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert state.runtime_info is None  # No worker ran after the failed launch.
    await state.close(3)
    provider.lost_launch = False
    await provider.exec(provider.delayed_command, cwd=str(workdir))
    assert not (workdir / "started").exists()
    assert not Path(state.session.session_dir).exists()


async def test_failed_receipt_keeps_files_and_can_retry(tmp_path):
    state, provider, _ = make_session(tmp_path)
    state.session.launch_started = True
    receipt = {
        "return_code": 1,
        "timed_out": False,
        "cleanup_confirmed": False,
        "error": "descendants remain",
    }
    path = Path(state.session.session_dir) / "cleanup.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await state.close(3)
    assert not provider.disconnected
    assert path.exists()
    receipt.update(cleanup_confirmed=True, error=None)
    path.write_text(json.dumps(receipt))
    await state.close(3)
    assert provider.disconnected


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
