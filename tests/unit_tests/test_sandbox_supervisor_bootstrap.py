# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failed supervisor bootstrap must remain safe to close."""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from shlex import quote

import pytest

from nemo_gym.agent_utils.sandbox_session import SandboxCommand
from nemo_gym.agent_utils.supervisor_client import (
    CLEANUP_RECEIPT_FILE,
    LAUNCH_CLAIM_FILE,
    RUNNER_INTERPRETER,
    SUPERVISOR_FILE,
    SUPERVISOR_PID_FILE,
    SUPERVISOR_SCRIPT,
    remove_session_directory,
    stop_and_confirm_cleanup,
    supervised_launch_command,
)
from nemo_gym.sandbox import SandboxExecResult


class LocalSandbox:
    def __init__(self, path: str | None = None) -> None:
        self.env = os.environ | ({"PATH": path} if path is not None else {})

    async def download(self, source: str, destination: Path) -> None:
        shutil.copyfile(source, destination)

    async def exec(self, command: str, *, cwd: str | None = None, timeout_s: float = 30) -> SandboxExecResult:
        process = await asyncio.create_subprocess_exec(
            shutil.which("sh"),
            "-c",
            command,
            cwd=cwd,
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)


def _tools(tmp_path: Path, *names: str) -> str:
    """A PATH holding only the named programs."""
    directory = tmp_path / "bin"
    directory.mkdir()
    for name in names:
        (directory / name).symlink_to(shutil.which(name))
    return str(directory)


@pytest.mark.parametrize("failure", ["missing_bash", "missing_supervisor", "broken_supervisor"])
async def test_bootstrap_failure_preserves_diagnostics_and_allows_close(tmp_path: Path, failure: str) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    supervisor = session_dir / SUPERVISOR_FILE
    shutil.copyfile(SUPERVISOR_SCRIPT, supervisor)
    sandbox = LocalSandbox()
    if failure == "missing_bash":
        sandbox = LocalSandbox(_tools(tmp_path, "touch"))
        expected_error = "bash"
    elif failure == "missing_supervisor":
        supervisor.unlink()
        expected_error = SUPERVISOR_FILE
    else:
        supervisor.write_text("if then fi\n")
        expected_error = "syntax error"
    command = supervised_launch_command(
        session_dir=str(session_dir),
        command=["touch", str(session_dir / "worker-started")],
        timeout=1,
        cleanup_timeout=1,
    )
    result = await sandbox.exec(command)
    assert result.return_code != 0
    assert expected_error in result.stderr
    assert not (session_dir / LAUNCH_CLAIM_FILE).is_symlink()
    assert not (session_dir / SUPERVISOR_PID_FILE).exists()
    assert not (session_dir / "worker-started").exists()
    assert not (session_dir / CLEANUP_RECEIPT_FILE).exists()

    receipt = await stop_and_confirm_cleanup(
        LocalSandbox(), session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
    )
    assert receipt["cleanup_confirmed"] is True
    assert receipt["return_code"] is None
    assert (session_dir / LAUNCH_CLAIM_FILE).readlink() == Path("stop")


@pytest.mark.parametrize("removed", [False, True])
async def test_fenced_delayed_launch_does_not_require_bash(tmp_path: Path, removed: bool) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / LAUNCH_CLAIM_FILE).symlink_to("stop")
    if removed:
        shutil.rmtree(session_dir)
    result = await LocalSandbox(_tools(tmp_path, "touch")).exec(
        supervised_launch_command(
            session_dir=str(session_dir),
            command=["touch", str(tmp_path / "worker-started")],
            timeout=1,
            cleanup_timeout=1,
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
    shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
    # A bash that pauses in the syntax check stands in for a slow sandbox.
    fake = tmp_path / "fake-bin"
    fake.mkdir()
    (fake / "bash").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = -n ]; then touch {str(preflight_started)!r}; '
        f"while [ ! -e {str(continue_preflight)!r} ]; do sleep 0.01; done; exit 0; fi\n"
        f"touch {str(worker_started)!r}\n"
    )
    (fake / "bash").chmod(0o755)
    sandbox = LocalSandbox(f"{fake}:{os.environ['PATH']}")
    launch = asyncio.create_task(
        sandbox.exec(
            supervised_launch_command(session_dir=str(session_dir), command=["worker"], timeout=1, cleanup_timeout=1)
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


async def test_launch_delayed_before_linking_does_not_recreate_a_removed_session(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / SUPERVISOR_FILE).write_text("")
    resolving = tmp_path / "resolving"
    resume = tmp_path / "resume"
    python = tmp_path / "python"
    python.write_text(
        "#!/bin/sh\n"
        f'if [ "$2" = -c ]; then touch {quote(str(resolving))}; '
        f"while [ ! -e {quote(str(resume))} ]; do sleep 0.01; done; fi\n"
        f'exec {quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    sandbox = LocalSandbox()
    launch = asyncio.create_task(
        sandbox.exec(
            supervised_launch_command(
                session_dir=str(session_dir),
                command=[str(python), "worker"],
                timeout=1,
                cleanup_timeout=1,
                stdlib_runner=True,
            )
        )
    )
    try:
        async with asyncio.timeout(5):
            while not resolving.exists():
                await asyncio.sleep(0.01)
        await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
        )
        await remove_session_directory(
            sandbox, session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
        )
    finally:
        resume.touch()
        await asyncio.wait_for(launch, timeout=5)
    assert not session_dir.exists()


@pytest.mark.parametrize("stdlib_runner", [False, True])
@pytest.mark.parametrize("linkable", [True, False])
async def test_bootstrap_quotes_private_paths(tmp_path: Path, linkable: bool, stdlib_runner: bool) -> None:
    # The runner link resolves /proc/self/exe; elsewhere the launch runs the given argv.
    linkable = linkable and sys.platform == "linux"
    session_dir = tmp_path / "session's $(touch unintended) files"
    session_dir.mkdir()
    if not linkable:
        (session_dir / "bin").write_text("not a directory")  # The launch must fall back to the given argv.
    runtime = tmp_path / "runtime's $(touch unintended) files"
    runtime.mkdir()
    python = runtime / "python interpreter"
    python.symlink_to(sys.executable)
    supervisor = session_dir / SUPERVISOR_FILE
    arguments = session_dir / "arguments"
    supervisor.write_text(f'printf "%s\\n" "$@" > {quote(str(arguments))}\n')
    command = supervised_launch_command(
        session_dir=str(session_dir),
        command=[str(python), "worker", "$(touch unintended)", "argument's value"],
        timeout=1,
        cleanup_timeout=1,
        stdlib_runner=stdlib_runner,
    )
    result = await LocalSandbox().exec(command, cwd=str(tmp_path))
    assert result.return_code == 0, result.stderr
    captured = arguments.read_text().splitlines()
    runner = session_dir / RUNNER_INTERPRETER if linkable and stdlib_runner else python
    assert captured[captured.index("--") + 1 :] == [str(runner), "worker", "$(touch unintended)", "argument's value"]
    assert runner.resolve() == python.resolve()
    assert not (tmp_path / "unintended").exists()
    assert (session_dir / LAUNCH_CLAIM_FILE).readlink() == Path("launch")


@pytest.mark.parametrize("argv", [["python3"], []])
def test_stdlib_runner_rejects_an_argv_without_an_interpreter_and_runner(argv: list[str]) -> None:
    with pytest.raises(ValueError, match="stdlib_runner requires"):
        supervised_launch_command(
            session_dir="/session", command=argv, timeout=1, cleanup_timeout=1, stdlib_runner=True
        )
    with pytest.raises(ValueError, match="stdlib_runner requires"):
        SandboxCommand(argv=argv, stdlib_runner=True)


def test_other_harness_commands_keep_their_interpreter() -> None:
    command = supervised_launch_command(
        session_dir="/session",
        command=["/runtime/venv/bin/python", "runner"],
        timeout=1,
        cleanup_timeout=1,
    )
    assert command.endswith("-- /runtime/venv/bin/python runner >/session/output.log 2>&1")
    assert RUNNER_INTERPRETER not in command
    assert SandboxCommand(argv=["/runtime/venv/bin/python", "runner"]).stdlib_runner is False
