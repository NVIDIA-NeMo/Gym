# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shell tool's runner, on the local sandbox provider: real bash processes in a temporary directory."""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider, resolve_provider_config
from responses_api_agents.crewai_agent import sandbox_shell
from responses_api_agents.crewai_agent.sandbox_shell import SandboxShell, ShellRunnerError


# Killing a process group is reliable on Linux; macOS's bash 3.2 sometimes leaves a subshell of a killed group running.
_LINUX_PROCESS_GROUPS = pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group kills")


@pytest.fixture
async def shell(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_shell, "SESSION_ROOT", str(tmp_path / "sessions"))
    workdir = tmp_path / "task"
    workdir.mkdir()
    provider = create_provider(resolve_provider_config("sandbox", {"sandbox": {"local": {}}}))
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(workdir=str(workdir)))
    # Room for a loaded machine; the deadline tests shorten it.
    shell = SandboxShell(
        sandbox, workdir=str(workdir), env={"TASK_LABEL": "owner"}, timeout_seconds=10, max_output_bytes=1000
    )
    await shell.setup()
    try:
        yield shell
    finally:
        await sandbox.stop()


async def _eventually(predicate, *, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.05)


async def test_a_command_runs_in_a_fresh_bash_in_the_task_directory(shell):
    result = await asyncio.wait_for(shell.run('pwd; echo "out $TASK_LABEL"; echo err >&2; exit 3'), 10)

    assert result.output.splitlines() == [shell.workdir, "out owner", "err"]
    assert (result.exit_code, result.timed_out) == (3, False)
    assert result.for_model(1).endswith("[exit code: 3]")


async def test_the_runner_adds_nothing_to_output_from_a_provider_that_merges_stderr(shell, monkeypatch):
    exec_command = shell.sandbox.exec

    # OpenSandbox returns a command's stderr within its output, as this wrapper does.
    async def merged(command, **options):
        result = await exec_command(command, **options)
        return replace(result, stdout=(result.stdout or "") + (result.stderr or ""), stderr="")

    monkeypatch.setattr(shell.sandbox, "exec", merged)

    finished = await asyncio.wait_for(shell.run("echo done"), 10)
    shell.timeout_seconds = 1
    killed = await asyncio.wait_for(shell.run("echo begun; exec sleep 5"), 10)

    # Job control reports the killed watchdog, or the killed command, only on the runner's stderr.
    assert finished.output == "done\n"
    assert killed.timed_out and killed.output == "begun\n"


@_LINUX_PROCESS_GROUPS
async def test_the_deadline_kills_the_commands_whole_process_group(shell):
    shell.timeout_seconds = 1
    workdir = Path(shell.workdir)
    started = asyncio.get_running_loop().time()

    result = await asyncio.wait_for(shell.run("(sleep 2; touch late.txt) & echo begun; sleep 5"), 10)

    # Bash may also report the kill of its child after the command's own output.
    assert result.timed_out and result.output.startswith("begun\n")
    assert asyncio.get_running_loop().time() - started < 4
    assert result.for_model(1).endswith("[command timed out after 1 s and was killed]")
    await asyncio.sleep(2.5)
    assert not (workdir / "late.txt").exists()


async def test_a_background_process_neither_holds_the_exec_open_nor_dies_with_its_command(shell):
    workdir = Path(shell.workdir)

    # The background process keeps the command's stdout open; the exec must still return at once.
    result = await asyncio.wait_for(shell.run("(sleep 1.5; touch service.txt) & echo started"), 5)

    assert (result.output.strip(), result.exit_code, result.timed_out) == ("started", 0, False)
    await _eventually(lambda: (workdir / "service.txt").exists())


async def test_long_output_keeps_its_head_and_tail(shell):
    result = await asyncio.wait_for(shell.run("printf 'head'; head -c 5000 /dev/zero | tr '\\0' x; printf 'tail'"), 10)

    assert result.output.startswith("head") and result.output.endswith("tail")
    assert "[... 4008 bytes of output omitted ...]" in result.output
    assert len(result.output) < 1100


@_LINUX_PROCESS_GROUPS
async def test_release_kills_a_running_command_and_removes_the_session_files(shell):
    shell.timeout_seconds = 60
    session_dir = Path(shell.session_dir)
    running = asyncio.create_task(shell.run("sleep 30; touch late.txt"))
    await _eventually(lambda: any(session_dir.glob("*.pid")))

    await asyncio.wait_for(shell.release(timeout_seconds=10), 10)

    result = await asyncio.wait_for(running, 10)
    assert result.exit_code == 137 and not result.timed_out
    assert not session_dir.exists()
    assert not (Path(shell.workdir) / "late.txt").exists()


async def test_a_runner_that_leaves_no_status_is_a_sandbox_failure(shell, monkeypatch):
    exec_command = shell.sandbox.exec

    async def lost(command, **options):
        result = await exec_command(command, **options)
        # The provider lost the runner's status line, as a dropped stream or a stopped container does.
        return replace(result, stdout="partial output\n")

    monkeypatch.setattr(shell.sandbox, "exec", lost)

    with pytest.raises(ShellRunnerError, match="without a command status"):
        await asyncio.wait_for(shell.run("echo unreachable"), 10)


async def test_a_command_that_deletes_the_session_directory_leaves_the_next_command_working(shell):
    first = await asyncio.wait_for(shell.run(f"rm -rf {Path(shell.session_dir).parent}"), 10)
    second = await asyncio.wait_for(shell.run("echo still here"), 10)

    assert first.exit_code == 0
    assert (second.output, second.exit_code, second.timed_out) == ("still here\n", 0, False)


async def test_a_command_that_exits_on_its_own_is_never_a_timeout(shell):
    # A watchdog that outlives the kill of its group (bash 3.2) can leave the timeout mark after a normal exit.
    Path(shell.session_dir, f"command-{shell._commands + 1}.pid.timeout").touch()

    result = await asyncio.wait_for(shell.run("echo done"), 10)

    assert (result.output, result.exit_code, result.timed_out) == ("done\n", 0, False)
