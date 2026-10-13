# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport contracts shared by sandboxed harness controllers."""

import asyncio
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agent_utils import process_supervisor
from nemo_gym.agent_utils.supervisor_client import (
    CLEANUP_RECEIPT_FILE,
    LAUNCH_CLAIM_FILE,
    OUTPUT_LOG_FILE,
    STOP_REQUEST_FILE,
    SUPERVISOR_FILE,
    SUPERVISOR_PID_FILE,
    parse_cleanup_receipt,
    stop_and_confirm_cleanup,
    supervised_launch_command,
    supervision_timeouts,
)
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.utils import read_text, upload_text


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
        python=sys.executable,
    )
    assert (await sandbox.exec(command)).return_code == 0
    assert not (session_dir / SUPERVISOR_PID_FILE).exists()
    assert not (session_dir / OUTPUT_LOG_FILE).exists()
    assert not (session_dir / "started").exists()
    # A fenced launch is safe to close, without claiming a successful harness exit.
    assert receipt["return_code"] is None


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
@pytest.mark.parametrize("private_python", [False, True])
async def test_launch_supervises_harness_with_selected_or_private_python(session, tmp_path, private_python):
    sandbox, session_dir = session
    runtime = tmp_path / "runtime's files"
    runtime.mkdir()
    supervisor = session_dir / SUPERVISOR_FILE
    shutil.copyfile(process_supervisor.__file__, supervisor)
    python = Path(sys.executable)
    if private_python:
        python = runtime / "python interpreter"
        python.symlink_to(sys.executable)
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
        python=str(python),
    )
    launched = await sandbox.exec(command)
    assert launched.return_code == 0, launched.stderr
    receipt = await stop_and_confirm_cleanup(
        sandbox, session_dir=str(session_dir), workdir=str(tmp_path), timeout=2, harness="test"
    )
    assert receipt == {"cleanup_confirmed": True, "error": None, "return_code": 0, "timed_out": False}
    assert set((session_dir / OUTPUT_LOG_FILE).read_text().splitlines()) == {"$(literal)", "stderr"}


@pytest.mark.parametrize("confirmed", [False, "true", 1])
async def test_unconfirmed_cleanup_preserves_receipt_for_retry(session, confirmed):
    sandbox, session_dir = session
    receipt_path = session_dir / CLEANUP_RECEIPT_FILE
    receipt = {"cleanup_confirmed": confirmed, "error": "descendants remain"}
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="test sandbox cleanup was not confirmed: descendants remain"):
        await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
        )
    assert json.loads(receipt_path.read_text()) == receipt
    receipt = {"cleanup_confirmed": True, "error": None, "return_code": None, "timed_out": False}
    receipt_path.write_text(json.dumps(receipt))
    assert await stop_and_confirm_cleanup(
        sandbox, session_dir=str(session_dir), workdir=str(session_dir.parent), timeout=1, harness="test"
    ) == parse_cleanup_receipt(receipt)


@pytest.mark.parametrize("timeout,close_timeout", [(1, 0.3), (2700, 30), (21600, 300)])
def test_supervision_timeouts_reserve_cleanup_and_provider_margin(timeout, close_timeout):
    cleanup_timeout, provider_timeout = supervision_timeouts(timeout=timeout, close_timeout=close_timeout)
    assert cleanup_timeout * process_supervisor.CLEANUP_PHASE_COUNT == pytest.approx(close_timeout)
    assert provider_timeout == pytest.approx(timeout + close_timeout + 30)
    assert provider_timeout == process_supervisor.exec_timeout(timeout=timeout, cleanup_timeout=cleanup_timeout)


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
    with pytest.raises(ValueError, match="cleanup was not confirmed"):
        parse_cleanup_receipt(payload)


def test_absent_exit_code_is_not_success(caplog):
    full = {"return_code": None, "timed_out": False, "cleanup_confirmed": True, "error": None}
    assert parse_cleanup_receipt(full) == full
    assert "diagnostic fields" not in caplog.text
    assert parse_cleanup_receipt({"cleanup_confirmed": True, "error": None}) == full
    assert "diagnostic fields" in caplog.text
    assert parse_cleanup_receipt({"cleanup_confirmed": True}) == full
    with pytest.raises(ValueError, match="cleanup was not confirmed"):
        parse_cleanup_receipt({"return_code": 0, "error": None})
