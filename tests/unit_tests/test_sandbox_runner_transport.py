# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport contracts shared by sandboxed harness controllers."""

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.runner import (
    RunnerRuntimeInfo,
    confirm_runner_cleanup,
    parse_cleanup_receipt,
    read_text,
    supervisor_command,
    upload_text,
)


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
    assert await confirm_runner_cleanup(
        sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
    ) == parse_cleanup_receipt(receipt)
    sandbox.exec.assert_not_awaited()


async def test_stop_wins_claim_and_fences_delayed_launch(session):
    sandbox, directory = session
    receipt = await confirm_runner_cleanup(
        sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
    )
    assert receipt == {"cleanup_confirmed": True, "error": None, "return_code": None, "timed_out": False}
    assert (directory / "launch.claim").readlink() == Path("stop")
    assert (directory / "runner.stop").exists()
    command = supervisor_command(
        directory=str(directory), command=["touch", str(directory / "started")], timeout=1, cleanup_timeout=1
    )
    assert (await sandbox.exec(command)).return_code == 0
    assert not (directory / "runner.pid").exists()
    assert not (directory / "runner.log").exists()
    assert not (directory / "started").exists()
    # A fenced launch is safe to close, without pretending a worker exited successfully.
    assert receipt["return_code"] is None
    with pytest.raises(ValidationError):
        RunnerRuntimeInfo.model_validate(receipt)


async def test_missing_receipt_after_launch_is_not_cleanup_confirmation(session):
    sandbox, directory = session
    (directory / "launch.claim").symlink_to("launch")
    with pytest.raises(RuntimeError, match="test launch outcome is unknown"):
        await confirm_runner_cleanup(
            sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
        )
    assert (directory / "launch.claim").readlink() == Path("launch")
    assert not (directory / "cleanup.json").exists()


@pytest.mark.parametrize("confirmed", [False, "true", 1])
async def test_unconfirmed_cleanup_preserves_receipt_for_retry(session, confirmed):
    sandbox, directory = session
    receipt_path = directory / "cleanup.json"
    receipt = {"cleanup_confirmed": confirmed, "error": "descendants remain"}
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="test sandbox cleanup was not confirmed: descendants remain"):
        await confirm_runner_cleanup(
            sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
        )
    assert json.loads(receipt_path.read_text()) == receipt
    receipt = {"cleanup_confirmed": True, "error": None}
    receipt_path.write_text(json.dumps(receipt))
    assert await confirm_runner_cleanup(
        sandbox, directory=str(directory), workdir=str(directory.parent), timeout=1, harness="test"
    ) == parse_cleanup_receipt(receipt)


def test_cleanup_and_runtime_are_independent():
    cleanup = {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None}
    assert parse_cleanup_receipt(cleanup) == cleanup
    runtime = RunnerRuntimeInfo.model_validate({"hostname": "sandbox", "pid": 123})
    assert runtime.hostname == "sandbox" and runtime.pid == 123
    assert runtime.python is None
    with pytest.raises(ValidationError):
        parse_cleanup_receipt(runtime.model_dump())
    with pytest.raises(ValidationError):
        RunnerRuntimeInfo.model_validate(cleanup)
    for field in cleanup:
        with pytest.raises(ValidationError):
            parse_cleanup_receipt({key: value for key, value in cleanup.items() if key != field})


@pytest.mark.parametrize(
    "invalid", [{"return_code": "0"}, {"timed_out": "false"}, {"cleanup_confirmed": 1}, {"hostname": "worker"}]
)
def test_cleanup_receipt_keeps_strict_validation(invalid):
    with pytest.raises(ValidationError):
        parse_cleanup_receipt(
            {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None, **invalid}
        )


def test_absent_exit_code_is_not_success():
    full = {"return_code": None, "timed_out": False, "cleanup_confirmed": True, "error": None}
    assert parse_cleanup_receipt(full) == full
    assert parse_cleanup_receipt({"cleanup_confirmed": True, "error": None}) == full
    with pytest.raises(ValidationError):
        parse_cleanup_receipt({"return_code": 0, "error": None})


@pytest.mark.parametrize("invalid", [{"pid": "123"}, {"hostname": 123}, {"python": 123}, {"return_code": 0}])
def test_runtime_info_keeps_strict_validation(invalid):
    with pytest.raises(ValidationError):
        RunnerRuntimeInfo.model_validate({"hostname": "sandbox", "pid": 123, **invalid})


def test_hermes_runtime_uses_the_same_schema():
    payload = {"hostname": "sandbox", "pid": 123, "python": "/opt/hermes/bin/python"}
    assert RunnerRuntimeInfo.model_validate(payload).model_dump() == payload
    for field in ("hostname", "pid"):
        with pytest.raises(ValidationError):
            RunnerRuntimeInfo.model_validate({key: value for key, value in payload.items() if key != field})
