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
from pydantic import ValidationError

from nemo_gym.sandbox import process_supervisor
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.supervisor_client import (
    HarnessProcessInfo,
    parse_cleanup_receipt,
    parse_runtime_info,
    stop_and_confirm_cleanup,
    supervised_launch_command,
)
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
    directory = tmp_path / "session's files"
    directory.mkdir()
    return LocalSandbox(), directory


async def test_file_transport_keeps_contents_out_of_shell(session):
    sandbox, directory = session
    sandbox.exec = AsyncMock(side_effect=AssertionError("file transfer must not invoke a shell"))
    path = str(directory / "input.json")
    payload = '{"prompt": "$(touch unwanted); `echo surprise`"}\n'
    await upload_text(sandbox, path=path, text=payload)
    assert Path(path).read_text() == payload
    assert await read_text(sandbox, path=path) == payload
    Path(path).write_bytes(b"partial output\xff\n")
    assert await read_text(sandbox, path=path) == "partial output\ufffd\n"
    sandbox.exec.assert_not_awaited()


async def test_confirmed_receipt_does_not_signal_stored_pid(session):
    sandbox, directory = session
    receipt = {"cleanup_confirmed": True, "error": None}
    (directory / "cleanup.json").write_text(json.dumps(receipt))
    (directory / "runner.pid").write_text("12345")
    sandbox.exec = AsyncMock(side_effect=AssertionError("must not signal a possibly reused PID"))
    assert await stop_and_confirm_cleanup(
        sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
    ) == parse_cleanup_receipt(receipt)
    sandbox.exec.assert_not_awaited()


async def test_confirmed_receipt_with_malformed_diagnostics_still_allows_cleanup(session, caplog):
    sandbox, directory = session
    receipt = {"cleanup_confirmed": True, "return_code": "0", "error": {"secret": "do not log"}, "version": 2}
    (directory / "cleanup.json").write_text(json.dumps(receipt))
    sandbox.exec = AsyncMock(side_effect=AssertionError("must not signal a possibly reused PID"))
    assert await stop_and_confirm_cleanup(
        sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
    ) == {"cleanup_confirmed": True, "return_code": None, "timed_out": False, "error": None}
    assert "diagnostic fields" in caplog.text
    assert "do not log" not in caplog.text
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("receipt", [[], "stopped", None, 1])
async def test_nonobject_cleanup_receipt_is_unconfirmed_without_signalling_a_stale_pid(session, receipt):
    sandbox, directory = session
    path = directory / "cleanup.json"
    path.write_text(json.dumps(receipt))
    (directory / "runner.pid").write_text("2147483647")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cleanup receipt is not a JSON object"):
            await stop_and_confirm_cleanup(
                sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
            )
    assert json.loads(path.read_text()) == receipt
    assert not (directory / "runner.stop").exists()
    assert not (directory / "launch.claim").exists()


async def test_stop_wins_claim_and_fences_delayed_launch(session):
    sandbox, directory = session
    receipt = await stop_and_confirm_cleanup(
        sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
    )
    assert receipt == {"cleanup_confirmed": True, "error": None, "return_code": None, "timed_out": False}
    assert (directory / "launch.claim").readlink() == Path("stop")
    assert (directory / "runner.stop").exists()
    command = supervised_launch_command(
        directory=str(directory),
        command=["touch", str(directory / "started")],
        timeout=1,
        cleanup_timeout=1,
        python=sys.executable,
    )
    assert (await sandbox.exec(command)).return_code == 0
    assert not (directory / "runner.pid").exists()
    assert not (directory / "runner.log").exists()
    assert not (directory / "started").exists()
    # A fenced launch is safe to close, without pretending a worker exited successfully.
    assert receipt["return_code"] is None
    with pytest.raises(ValidationError):
        HarnessProcessInfo.model_validate(receipt)


async def test_missing_receipt_after_launch_is_not_cleanup_confirmation(session):
    sandbox, directory = session
    (directory / "launch.claim").symlink_to("launch")
    with pytest.raises(RuntimeError, match="test launch outcome is unknown"):
        await stop_and_confirm_cleanup(
            sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
        )
    assert (directory / "launch.claim").readlink() == Path("launch")
    assert not (directory / "cleanup.json").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
@pytest.mark.parametrize("private_runtime", [False, True])
async def test_launch_supervises_worker_with_selected_or_private_runtime(session, tmp_path, private_runtime):
    sandbox, directory = session
    runtime = tmp_path / "runtime's files"
    runtime.mkdir()
    supervisor = (runtime if private_runtime else directory) / "process_supervisor.py"
    shutil.copyfile(process_supervisor.__file__, supervisor)
    options = {"python": sys.executable}
    if private_runtime:
        python = runtime / "python interpreter"
        python.symlink_to(sys.executable)
        options = {"python": str(python), "supervisor_path": str(supervisor)}
    command = supervised_launch_command(
        directory=str(directory),
        command=[
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1]); print('stderr', file=sys.stderr)",
            "$(literal)",
        ],
        timeout=5,
        cleanup_timeout=1,
        **options,
    )
    launched = await sandbox.exec(command)
    assert launched.return_code == 0, launched.stderr
    receipt = await stop_and_confirm_cleanup(
        sandbox, directory=str(directory), workdir=str(tmp_path), timeout=2, harness="test"
    )
    assert receipt == {"cleanup_confirmed": True, "error": None, "return_code": 0, "timed_out": False}
    assert set((directory / "runner.log").read_text().splitlines()) == {"$(literal)", "stderr"}


@pytest.mark.parametrize("confirmed", [False, "true", 1])
async def test_unconfirmed_cleanup_preserves_receipt_for_retry(session, confirmed):
    sandbox, directory = session
    receipt_path = directory / "cleanup.json"
    receipt = {"cleanup_confirmed": confirmed, "error": "descendants remain"}
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="test sandbox cleanup was not confirmed: descendants remain"):
        await stop_and_confirm_cleanup(
            sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
        )
    assert json.loads(receipt_path.read_text()) == receipt
    receipt = {"cleanup_confirmed": True, "error": None}
    receipt_path.write_text(json.dumps(receipt))
    assert await stop_and_confirm_cleanup(
        sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
    ) == parse_cleanup_receipt(receipt)


def test_cleanup_and_runtime_are_independent():
    cleanup = {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None}
    assert parse_cleanup_receipt(cleanup) == cleanup
    runtime = HarnessProcessInfo.model_validate({"hostname": "sandbox", "pid": 123})
    assert runtime.hostname == "sandbox" and runtime.pid == 123
    assert runtime.python is None
    with pytest.raises(ValueError, match="cleanup was not confirmed"):
        parse_cleanup_receipt(runtime.model_dump())
    with pytest.raises(ValidationError):
        HarnessProcessInfo.model_validate(cleanup)


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


def test_absent_exit_code_is_not_success():
    full = {"return_code": None, "timed_out": False, "cleanup_confirmed": True, "error": None}
    assert parse_cleanup_receipt(full) == full
    assert parse_cleanup_receipt({"cleanup_confirmed": True, "error": None}) == full
    assert parse_cleanup_receipt({"cleanup_confirmed": True}) == full
    with pytest.raises(ValueError, match="cleanup was not confirmed"):
        parse_cleanup_receipt({"return_code": 0, "error": None})


@pytest.mark.parametrize("invalid", [{"pid": "123"}, {"hostname": 123}, {"python": 123}, {"return_code": 0}])
def test_runtime_info_keeps_strict_validation(invalid):
    with pytest.raises(ValidationError):
        HarnessProcessInfo.model_validate({"hostname": "sandbox", "pid": 123, **invalid})


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"hostname": "sandbox", "pid": "123"},
        {"hostname": "sandbox", "pid": 123, "python": {"secret": "do not log"}},
    ],
)
def test_malformed_runtime_info_is_optional_and_logs_without_payload(payload, caplog):
    assert parse_runtime_info(payload) is None
    assert "runtime metadata" in caplog.text
    assert "do not log" not in caplog.text


def test_hermes_runtime_uses_the_same_schema():
    payload = {"hostname": "sandbox", "pid": 123, "python": "/opt/hermes/bin/python"}
    assert HarnessProcessInfo.model_validate(payload).model_dump() == payload
    assert parse_runtime_info(payload).model_dump() == payload
    for field in ("hostname", "pid"):
        with pytest.raises(ValidationError):
            HarnessProcessInfo.model_validate({key: value for key, value in payload.items() if key != field})
