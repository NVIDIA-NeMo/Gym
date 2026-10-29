# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport contracts shared by sandboxed harness controllers."""

import asyncio
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from shlex import quote
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents import SUPERVISOR_FILE, SUPERVISOR_SCRIPT, exec_timeout
from nemo_gym.agents.utils.supervisor_client import (
    CLEANUP_PHASE_COUNT,
    CLEANUP_RECEIPT_FILE,
    LAUNCH_CLAIM_FILE,
    OUTPUT_LOG_FILE,
    STOP_REQUEST_FILE,
    SUPERVISOR_PID_FILE,
    HarnessTerminationError,
    parse_cleanup_receipt,
    stop_and_confirm_cleanup,
    supervised_launch_command,
    supervision_timeouts,
)
from nemo_gym.sandbox import SandboxExecResult, read_text, upload_text


class LocalSandbox:
    async def upload(self, source, destination):
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        shutil.copyfile(source, destination)

    async def exec(self, command, *, cwd=None, timeout_s=30):
        process = await asyncio.create_subprocess_exec(
            "sh", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)


@pytest.fixture
def session(tmp_path):
    session_dir = tmp_path / "session's files"
    session_dir.mkdir()
    return LocalSandbox(), session_dir


async def test_file_transport_keeps_contents_out_of_shell(session):
    sandbox, session_dir = session
    sandbox.exec = AsyncMock(side_effect=AssertionError("file transfer must not invoke a shell"))
    path = str(session_dir / "input.json")
    payload = '{"prompt": "$(touch unwanted); `echo surprise`"}\n'
    await upload_text(sandbox, path=path, text=payload)
    assert Path(path).read_text() == payload
    assert await read_text(sandbox, path=path) == payload
    Path(path).write_bytes(b"partial output\xff\n")
    assert await read_text(sandbox, path=path) == "partial output\ufffd\n"
    sandbox.exec.assert_not_awaited()


async def test_confirmed_receipt_does_not_signal_stored_pid(session):
    sandbox, session_dir = session
    receipt = {"cleanup_confirmed": True, "error": None, "return_code": None, "timed_out": False}
    (session_dir / CLEANUP_RECEIPT_FILE).write_text(json.dumps(receipt))
    (session_dir / SUPERVISOR_PID_FILE).write_text("12345")
    sandbox.exec = AsyncMock(side_effect=AssertionError("must not signal a possibly reused PID"))
    assert await stop_and_confirm_cleanup(
        sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
    ) == parse_cleanup_receipt(receipt)
    sandbox.exec.assert_not_awaited()


async def test_confirmed_receipt_with_malformed_diagnostics_still_allows_cleanup(session, caplog):
    sandbox, session_dir = session
    receipt = {"cleanup_confirmed": True, "return_code": "0", "error": {"secret": "do not log"}, "version": 2}
    (session_dir / CLEANUP_RECEIPT_FILE).write_text(json.dumps(receipt))
    sandbox.exec = AsyncMock(side_effect=AssertionError("must not signal a possibly reused PID"))
    assert await stop_and_confirm_cleanup(
        sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
    ) == {"cleanup_confirmed": True, "return_code": None, "timed_out": False, "error": None}
    assert "diagnostic fields" in caplog.text
    assert "do not log" not in caplog.text
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("receipt", [[], "stopped", None, 1])
async def test_nonobject_cleanup_receipt_is_unconfirmed_without_signalling_a_stale_pid(session, receipt):
    sandbox, session_dir = session
    path = session_dir / CLEANUP_RECEIPT_FILE
    path.write_text(json.dumps(receipt))
    (session_dir / SUPERVISOR_PID_FILE).write_text("2147483647")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cleanup receipt is not a JSON object"):
            await stop_and_confirm_cleanup(
                sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
            )
    assert json.loads(path.read_text()) == receipt
    assert not (session_dir / STOP_REQUEST_FILE).exists()
    assert not (session_dir / LAUNCH_CLAIM_FILE).exists()


async def test_stop_wins_claim_and_fences_delayed_launch(session, caplog):
    sandbox, session_dir = session
    receipt = await stop_and_confirm_cleanup(
        sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
    )
    assert receipt == {"cleanup_confirmed": True, "error": None, "return_code": None, "timed_out": False}
    assert json.loads((session_dir / CLEANUP_RECEIPT_FILE).read_text()) == receipt
    assert "diagnostic fields" not in caplog.text
    assert (session_dir / LAUNCH_CLAIM_FILE).readlink() == Path("stop")
    assert (session_dir / STOP_REQUEST_FILE).exists()
    command = supervised_launch_command(
        session_dir=str(session_dir),
        command=["touch", str(session_dir / "started")],
        timeout=1,
        cleanup_timeout=1,
    )
    assert (await sandbox.exec(command)).return_code == 0
    assert not (session_dir / SUPERVISOR_PID_FILE).exists()
    assert not (session_dir / OUTPUT_LOG_FILE).exists()
    assert not (session_dir / "started").exists()
    # A fenced launch is safe to close, without claiming a successful harness exit.
    assert receipt["return_code"] is None


@pytest.mark.parametrize("seconds", [5e-05, 0.004, 1e16])
async def test_launch_renders_seconds_the_supervisor_accepts(session, seconds):
    # Python renders tiny and huge floats with exponents, which the supervisor would reject after the claim.
    sandbox, session_dir = session
    shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
    (session_dir / STOP_REQUEST_FILE).touch()
    command = supervised_launch_command(
        session_dir=str(session_dir), command=["true"], timeout=seconds, cleanup_timeout=seconds
    )
    rendered = re.search(r"--timeout (\S+) --cleanup-timeout (\S+) ", command).groups()
    assert all(re.fullmatch(r"[0-9]+\.[0-9]+", value) for value in rendered), rendered
    assert (await sandbox.exec(command)).return_code == 0
    assert json.loads((session_dir / CLEANUP_RECEIPT_FILE).read_text())["cleanup_confirmed"] is True


async def test_missing_receipt_after_launch_is_not_cleanup_confirmation(session):
    sandbox, session_dir = session
    (session_dir / LAUNCH_CLAIM_FILE).symlink_to("launch")
    with pytest.raises(RuntimeError, match="test launch outcome is unknown"):
        await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
        )
    assert (session_dir / LAUNCH_CLAIM_FILE).readlink() == Path("launch")
    assert not (session_dir / CLEANUP_RECEIPT_FILE).exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
async def test_launch_supervises_harness_and_shares_its_output_log(session, tmp_path):
    sandbox, session_dir = session
    shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
    command = supervised_launch_command(
        session_dir=str(session_dir),
        command=[
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1]); print('stderr', file=sys.stderr)",
            "$(literal)",
        ],
        timeout=5,
        cleanup_timeout=1,
    )
    launched = await sandbox.exec(command)
    assert launched.return_code == 0, launched.stderr
    receipt = await stop_and_confirm_cleanup(
        sandbox, session_dir=str(session_dir), workdir=str(tmp_path), timeout=2, harness="test"
    )
    assert receipt == {"cleanup_confirmed": True, "error": None, "return_code": 0, "timed_out": False}
    assert set((session_dir / OUTPUT_LOG_FILE).read_text().splitlines()) == {"$(literal)", "stderr"}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
async def test_stop_reaches_running_supervisor_through_the_stop_file(session, tmp_path):
    # The stop shell sends no signal, so the harness keeps the default TERM disposition and checkpoints.
    sandbox, session_dir = session
    shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
    ready = session_dir / "ready"
    worker = f"trap 'echo saved > checkpoint; exit 0' TERM; touch {quote(str(ready))}; while :; do sleep 0.05; done"
    command = supervised_launch_command(
        session_dir=str(session_dir), command=["bash", "-c", worker], timeout=30, cleanup_timeout=2
    )
    launch = asyncio.create_task(sandbox.exec(command, cwd=str(session_dir)))
    try:
        async with asyncio.timeout(10):
            while not ready.exists():
                await asyncio.sleep(0.01)
        receipt = await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=str(session_dir), timeout=5, harness="test"
        )
    finally:
        await asyncio.wait_for(launch, timeout=10)
    assert receipt == {"cleanup_confirmed": True, "error": None, "return_code": 0, "timed_out": False}
    assert (session_dir / "checkpoint").read_text() == "saved\n"


# Starts a non-Python tool, like an agent CLI, and waits for it.
_RUNNER = """
import pathlib, subprocess, sys
tool = subprocess.Popen(["sleep", "60"])
pathlib.Path(sys.argv[1]).write_text(str(tool.pid))
tool.wait()
"""


def _parent(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1][0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("pgrep") is None, reason="Linux /proc and pgrep")
async def test_kills_by_interpreter_name_miss_the_supervisor_and_stdlib_runner(session, tmp_path):
    """pgrep matches exactly as pkill and killall would, without signalling anything on this host."""
    sandbox, session_dir = session
    shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
    runner = session_dir / "sandbox_runner"
    runner.write_text(_RUNNER)
    tool_pid_file = session_dir / "tool.pid"
    command = supervised_launch_command(
        session_dir=str(session_dir),
        command=[sys.executable, "-I", str(runner), str(tool_pid_file)],
        timeout=60,
        cleanup_timeout=1,
        stdlib_runner=True,
    )
    launch = asyncio.create_task(sandbox.exec(command, timeout_s=90))
    try:
        async with asyncio.timeout(10):
            while not tool_pid_file.exists() or not tool_pid_file.read_text():
                await asyncio.sleep(0.01)
        supervisor_pid = int((session_dir / SUPERVISOR_PID_FILE).read_text())
        tool_pid = int(tool_pid_file.read_text())
        # The bash supervisor's polling sleeps are children too; the runner is the tool's parent.
        runner_pid = _parent(tool_pid)
        # This launch shell is no process group leader, so the recorded PID started the supervisor
        # in a new group; either way the runner's parent is the supervisor.
        harness_supervisor = _parent(runner_pid)
        assert harness_supervisor == supervisor_pid or str(harness_supervisor) in (
            Path(f"/proc/{supervisor_pid}/task/{supervisor_pid}/children").read_text().split()
        )
        # The supervisor is bash; the runner runs through a link to the base interpreter.
        for pid, name in ((supervisor_pid, "bash"), (harness_supervisor, "bash"), (runner_pid, "gym-runner")):
            assert Path(f"/proc/{pid}/comm").read_text().strip() == name
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            assert "python" not in cmdline and ".py" not in cmdline, cmdline
        assert Path(f"/proc/{runner_pid}/exe").resolve() == Path(sys.executable).resolve()
        # pkill -f python3 / python / "python3 -" / "\.py"; pkill python; killall python3 (an exact name).
        for args in (
            ["-f", "python3"],
            ["-f", "python"],
            ["-f", "python3 -"],
            ["-f", r"\.py"],
            ["python"],
            ["-x", "python3"],
        ):
            matched = {int(pid) for pid in subprocess.run(["pgrep", *args], capture_output=True).stdout.split()}
            assert not matched & {supervisor_pid, harness_supervisor, runner_pid}, args
        assert all(map(_alive, (supervisor_pid, harness_supervisor, runner_pid, tool_pid)))
    finally:
        # Gym's own stop goes through the stop file, whatever the processes are called.
        receipt = await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=str(tmp_path), timeout=5, harness="test"
        )
        await asyncio.wait_for(launch, timeout=10)
    assert receipt["cleanup_confirmed"] is True and receipt["error"] is None
    assert not any(map(_alive, (supervisor_pid, harness_supervisor, runner_pid, tool_pid)))


@pytest.mark.parametrize("confirmed", [False, "true", 1])
async def test_unconfirmed_cleanup_preserves_receipt_for_retry(session, confirmed):
    sandbox, session_dir = session
    receipt_path = session_dir / CLEANUP_RECEIPT_FILE
    receipt = {"cleanup_confirmed": confirmed, "error": "descendants remain"}
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="test harness termination was not confirmed: descendants remain"):
        await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
        )
    assert json.loads(receipt_path.read_text()) == receipt
    receipt = {"cleanup_confirmed": True, "error": None, "return_code": None, "timed_out": False}
    receipt_path.write_text(json.dumps(receipt))
    assert await stop_and_confirm_cleanup(
        sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
    ) == parse_cleanup_receipt(receipt)


async def test_terminal_negative_receipt_preserves_failure_without_signalling_a_stale_pid(session):
    sandbox, session_dir = session
    receipt = {"cleanup_confirmed": False, "return_code": None, "timed_out": False, "error": "launch failed"}
    (session_dir / CLEANUP_RECEIPT_FILE).write_text(json.dumps(receipt))
    sandbox.exec = AsyncMock(side_effect=AssertionError("must not signal a possibly reused PID"))
    with pytest.raises(HarnessTerminationError, match="launch failed") as failure:
        await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
        )
    assert failure.value.receipt == receipt
    sandbox.exec.assert_not_awaited()


async def test_stop_surfaces_terminal_negative_receipt_created_after_initial_check(session):
    sandbox, session_dir = session
    receipt_path = session_dir / CLEANUP_RECEIPT_FILE
    receipt = {"cleanup_confirmed": False, "return_code": -9, "timed_out": True, "error": "descendants remain"}
    assert not receipt_path.exists()

    async def stop(*args, **kwargs):
        # The first read sees no receipt; stopping the supervisor publishes its
        # final failure. This must take the same typed failure path as an early receipt.
        receipt_path.write_text(json.dumps(receipt))
        return SandboxExecResult("", "", 0)

    sandbox.exec = AsyncMock(side_effect=stop)
    with pytest.raises(HarnessTerminationError, match="descendants remain") as failure:
        await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
        )
    assert failure.value.receipt == receipt
    sandbox.exec.assert_awaited_once()
    assert json.loads(receipt_path.read_text()) == receipt


@pytest.mark.parametrize("timeout,close_timeout", [(1, 0.3), (2700, 30), (21600, 300)])
def test_supervision_timeouts_reserve_cleanup_and_provider_margin(timeout, close_timeout):
    cleanup_timeout, provider_timeout = supervision_timeouts(timeout=timeout, close_timeout=close_timeout)
    assert cleanup_timeout * CLEANUP_PHASE_COUNT == pytest.approx(close_timeout)
    assert provider_timeout == pytest.approx(timeout + close_timeout + 30)
    assert provider_timeout == exec_timeout(timeout=timeout, cleanup_timeout=cleanup_timeout)


@pytest.mark.parametrize(
    "invalid,normalized",
    [
        ({"return_code": "0"}, {"return_code": None}),
        ({"return_code": True}, {"return_code": None}),
        ({"timed_out": "false"}, {"timed_out": False}),
        ({"error": {"message": "failed"}}, {"error": None}),
        ({"hostname": "worker"}, {}),
    ],
)
def test_confirmed_cleanup_keeps_valid_diagnostics_and_ignores_malformed_fields(invalid, normalized):
    cleanup = {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None}
    assert parse_cleanup_receipt(cleanup | invalid) == cleanup | normalized


@pytest.mark.parametrize(
    "payload",
    [None, [], "true", {}, {"cleanup_confirmed": False}, {"cleanup_confirmed": "true"}, {"cleanup_confirmed": 1}],
)
def test_cleanup_receipt_requires_positive_boolean_evidence(payload):
    with pytest.raises(ValueError, match="termination was not confirmed"):
        parse_cleanup_receipt(payload)


def test_absent_exit_code_is_not_success(caplog):
    full = {"return_code": None, "timed_out": False, "cleanup_confirmed": True, "error": None}
    assert parse_cleanup_receipt(full) == full
    assert "diagnostic fields" not in caplog.text
    assert parse_cleanup_receipt({"cleanup_confirmed": True, "error": None}) == full
    assert "diagnostic fields" in caplog.text
    assert parse_cleanup_receipt({"cleanup_confirmed": True}) == full
    with pytest.raises(ValueError, match="termination was not confirmed"):
        parse_cleanup_receipt({"return_code": 0, "error": None})
