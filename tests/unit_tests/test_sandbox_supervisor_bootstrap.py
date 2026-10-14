# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failed interpreter/supervisor bootstrap must remain safe to close."""

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

from nemo_gym.agent_utils import process_supervisor
from nemo_gym.agent_utils.supervisor_client import (
    CLEANUP_RECEIPT_FILE,
    LAUNCH_CLAIM_FILE,
    SUPERVISOR_FILE,
    SUPERVISOR_PID_FILE,
    remove_session_directory,
    stop_and_confirm_cleanup,
    supervised_launch_command,
)
from nemo_gym.sandbox.providers.base import SandboxExecResult


class LocalSandbox:
    async def download(self, source: str, destination: Path) -> None:
        shutil.copyfile(source, destination)

    async def exec(self, command: str, *, cwd: str | None = None, timeout_s: float = 30) -> SandboxExecResult:
        process = await asyncio.create_subprocess_exec(
            "sh", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)


@pytest.mark.parametrize("failure", ["missing_python", "broken_python", "missing_supervisor", "broken_supervisor"])
async def test_bootstrap_failure_preserves_diagnostics_and_allows_close(tmp_path: Path, failure: str) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    python = sys.executable
    supervisor = session_dir / SUPERVISOR_FILE
    shutil.copyfile(process_supervisor.__file__, supervisor)
    expected_error = ""
    if failure == "missing_python":
        python = str(tmp_path / "missing-python")
        expected_error = "missing-python"
    elif failure == "broken_python":
        interpreter = tmp_path / "incompatible-python"
        interpreter.write_text("#!/bin/sh\nprintf '%s\\n' 'incompatible interpreter' >&2\nexit 17\n")
        interpreter.chmod(0o755)
        python = str(interpreter)
        expected_error = "incompatible interpreter"
    elif failure == "missing_supervisor":
        supervisor.unlink()
        expected_error = "process_supervisor.py"
    else:
        supervisor.write_text("def invalid syntax\n")
        expected_error = "SyntaxError"
    sandbox = LocalSandbox()
    command = supervised_launch_command(
        session_dir=str(session_dir),
        command=["touch", str(session_dir / "worker-started")],
        timeout=1,
        cleanup_timeout=1,
        python=python,
    )
    result = await sandbox.exec(command)
    assert result.return_code != 0
    if failure == "broken_python":
        assert result.return_code == 17
    assert expected_error in result.stderr
    assert not (session_dir / LAUNCH_CLAIM_FILE).is_symlink()
    assert not (session_dir / SUPERVISOR_PID_FILE).exists()
    assert not (session_dir / "worker-started").exists()
    assert not (session_dir / CLEANUP_RECEIPT_FILE).exists()

    receipt = await stop_and_confirm_cleanup(
        sandbox, session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
    )
    assert receipt["cleanup_confirmed"] is True
    assert receipt["return_code"] is None
    assert (session_dir / LAUNCH_CLAIM_FILE).readlink() == Path("stop")


@pytest.mark.parametrize("removed", [False, True])
async def test_fenced_delayed_launch_does_not_require_the_runtime(tmp_path: Path, removed: bool) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / LAUNCH_CLAIM_FILE).symlink_to("stop")
    if removed:
        shutil.rmtree(session_dir)
    result = await LocalSandbox().exec(
        supervised_launch_command(
            session_dir=str(session_dir),
            command=["touch", str(tmp_path / "worker-started")],
            timeout=1,
            cleanup_timeout=1,
            python=str(tmp_path / "missing-python"),
        )
    )
    assert result.return_code == 0
    assert result.stderr == ""
    assert not (tmp_path / "worker-started").exists()


@pytest.mark.parametrize("remove_directory", [False, True])
async def test_close_during_preflight_fences_later_launch(tmp_path: Path, remove_directory: bool) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    preflight_started = tmp_path / "preflight-started"
    continue_preflight = tmp_path / "continue-preflight"
    worker_started = tmp_path / "worker-started"
    supervisor = session_dir / SUPERVISOR_FILE
    supervisor.write_text(
        "import pathlib, sys, time\n"
        "if '--help' in sys.argv:\n"
        f"    pathlib.Path({str(preflight_started)!r}).touch()\n"
        f"    while not pathlib.Path({str(continue_preflight)!r}).exists():\n"
        "        time.sleep(0.01)\n"
        "else:\n"
        f"    pathlib.Path({str(worker_started)!r}).touch()\n"
    )
    sandbox = LocalSandbox()
    launch = asyncio.create_task(
        sandbox.exec(
            supervised_launch_command(
                session_dir=str(session_dir),
                command=["worker"],
                timeout=1,
                cleanup_timeout=1,
                python=sys.executable,
            )
        )
    )
    try:
        async with asyncio.timeout(5):
            while not preflight_started.exists():
                await asyncio.sleep(0.01)
        receipt = await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
        )
        assert receipt["cleanup_confirmed"] is True
        assert receipt["return_code"] is None
        if remove_directory:
            await remove_session_directory(
                sandbox, session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
            )
    finally:
        continue_preflight.touch()
        result = await asyncio.wait_for(launch, timeout=5)
    assert result.return_code == 0
    assert not worker_started.exists()


async def test_bootstrap_quotes_private_paths_and_uses_the_same_interpreter(tmp_path: Path) -> None:
    session_dir = tmp_path / "session's $(touch unintended) files"
    session_dir.mkdir()
    runtime = tmp_path / "runtime's $(touch unintended) files"
    runtime.mkdir()
    python = runtime / "python interpreter"
    python.symlink_to(sys.executable)
    supervisor = session_dir / SUPERVISOR_FILE
    arguments = session_dir / "arguments.json"
    supervisor.write_text(
        "import json, pathlib, sys\n"
        "if '--help' not in sys.argv:\n"
        f"    pathlib.Path({str(arguments)!r}).write_text(json.dumps([sys.executable, *sys.argv[1:]]))\n"
    )
    command = supervised_launch_command(
        session_dir=str(session_dir),
        command=["worker", "$(touch unintended)", "argument's value"],
        timeout=1,
        cleanup_timeout=1,
        python=str(python),
    )
    result = await LocalSandbox().exec(command, cwd=str(tmp_path))
    assert result.return_code == 0, result.stderr
    captured = json.loads(arguments.read_text())
    assert captured[0] == str(python)
    assert captured[-3:] == ["worker", "$(touch unintended)", "argument's value"]
    assert not (tmp_path / "unintended").exists()
    assert (session_dir / LAUNCH_CLAIM_FILE).readlink() == Path("launch")
