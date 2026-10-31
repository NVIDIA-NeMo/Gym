# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
import shutil
import subprocess
import sys
from time import monotonic
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.agent_utils import supervisor_client
from nemo_gym.agent_utils.sandbox_session import (
    ExpiryUnsupportedError,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.agent_utils.supervisor_client import is_harness_termination_failure
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import SupportsSandboxIndependentUploads
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec
from nemo_gym.sandbox.providers.local import LocalProvider
from nemo_gym.sandbox.providers.registry import get_provider_class
from nemo_gym.typed_exceptions import NgFailureEvent


RECEIPT = {"cleanup_confirmed": True, "return_code": 0, "timed_out": False, "error": None}
OK = SandboxExecResult(stdout="", stderr="", return_code=0)
EXPIRE_MARKER = "/session/expire.request"


def _expiry_writes(session) -> int:
    return sum(1 for call in session.sandbox.upload.await_args_list if call.args[1] == EXPIRE_MARKER)


@contextlib.asynccontextmanager
async def _launched(session, launched: asyncio.Event):
    """Start an activation and wait, bounded, for its launch; cancel it on exit if it is still running."""
    activation = asyncio.create_task(execute(session))
    try:
        await asyncio.wait_for(launched.wait(), 2)
        yield activation
    finally:
        activation.cancel()
        await asyncio.gather(activation, return_exceptions=True)


@pytest.fixture
def receipt_reader(monkeypatch):
    reader = AsyncMock(return_value=json.dumps(RECEIPT))
    monkeypatch.setattr(supervisor_client, "read_text", reader)
    return reader


@pytest.fixture
def session(receipt_reader):
    sandbox = AsyncMock(spec=AsyncSandbox)
    sandbox.exec.return_value = OK
    return SandboxSession(sandbox=sandbox, session_dir="/session", workdir="/app", harness="test")


async def execute(session, *, collect=None, stage_activation=None, timeout=1):
    return await session.execute(
        stage_activation=stage_activation or AsyncMock(return_value=SandboxCommand(argv=["worker"])),
        collect=collect or AsyncMock(return_value="transcript"),
        timeout=10,
        close_timeout=timeout,
    )


@pytest.mark.parametrize("owned", [False, True])
async def test_completed_execution_captures_once_before_release(session, owned):
    session.owns_sandbox = owned
    collect = AsyncMock(return_value="transcript")
    assert await execute(session, collect=collect) == "transcript"
    assert session.cleanup == RECEIPT
    assert session.artifacts == "transcript"
    session.sandbox.stop.assert_not_awaited()
    session.sandbox.disconnect.assert_not_awaited()
    await asyncio.gather(session.close(timeout=1), session.close(timeout=1))
    await session.close(timeout=1)
    collect.assert_awaited_once()
    assert session.closed
    if owned:
        session.sandbox.stop.assert_awaited_once()
        session.sandbox.disconnect.assert_not_awaited()
        assert session.sandbox.exec.await_count == 1
    else:
        session.sandbox.stop.assert_not_awaited()
        session.sandbox.disconnect.assert_awaited_once()
        assert "mv /session /session.closed" in session.sandbox.exec.await_args.args[0]
    with pytest.raises(RuntimeError, match="closing or already activated"):
        await execute(session)


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("cancel_execution", [False, True])
async def test_interruption_captures_before_transport_cancellation_and_release(session, owned, cancel_execution):
    session.owns_sandbox = owned
    launched, collecting, captured = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []

    async def provider_exec(command, **kwargs):
        if "exec bash " in command:
            launched.set()
            try:
                await asyncio.Future()
            finally:
                assert session.cleanup == RECEIPT
                events.append("cancel_exec")
        events.append("remove_files")
        return OK

    async def collect():
        assert session.cleanup == RECEIPT
        events.append("collect")
        collecting.set()
        await captured.wait()
        return "partial transcript"

    session.sandbox.exec.side_effect = provider_exec
    session.sandbox.stop.side_effect = lambda: events.append("stop_sandbox")
    session.sandbox.disconnect.side_effect = lambda: events.append("disconnect")
    collector = AsyncMock(side_effect=collect)
    running = asyncio.create_task(execute(session, collect=collector))
    await asyncio.wait_for(launched.wait(), 2)
    if cancel_execution:
        running.cancel()
    closing = asyncio.create_task(session.close(timeout=1))
    await asyncio.wait_for(collecting.wait(), 2)
    retry = asyncio.create_task(session.close(timeout=1))
    # Losing one close waiter must not cancel collection or release early.
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert events == ["collect"]
    assert not session.closed
    captured.set()
    await asyncio.wait_for(retry, 2)
    with pytest.raises(asyncio.CancelledError):
        await running
    assert session.artifacts == "partial transcript"
    collector.assert_awaited_once()
    assert events == ["collect", "cancel_exec", *(["stop_sandbox"] if owned else ["remove_files", "disconnect"])]


async def test_close_joins_collection_started_by_normal_completion(session):
    collecting, finish = asyncio.Event(), asyncio.Event()

    async def collect():
        collecting.set()
        await finish.wait()
        return "snapshot"

    collector = AsyncMock(side_effect=collect)
    running = asyncio.create_task(execute(session, collect=collector))
    await asyncio.wait_for(collecting.wait(), 2)
    close = asyncio.create_task(session.close(timeout=1))
    await asyncio.sleep(0)
    session.sandbox.disconnect.assert_not_awaited()
    assert session.sandbox.exec.await_count == 1
    finish.set()
    assert await running == "snapshot"
    await close
    collector.assert_awaited_once()


async def test_cancellation_during_normal_finalization_waits_for_evidence(session):
    collecting, release = asyncio.Event(), asyncio.Event()

    async def collect():
        collecting.set()
        await release.wait()
        return "final snapshot"

    running = asyncio.create_task(execute(session, collect=collect))
    await collecting.wait()
    running.cancel()
    await asyncio.sleep(0)
    try:
        assert not running.done()
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await session.close(timeout=1)
    assert session.artifacts == "final snapshot"


async def test_failed_normal_finalization_waits_for_explicit_close_retry(session, monkeypatch):
    failure = RuntimeError("launch outcome is unknown")
    stop = AsyncMock(side_effect=[failure, RECEIPT])
    monkeypatch.setattr("nemo_gym.agent_utils.sandbox_session.stop_and_confirm_cleanup", stop)
    collector = AsyncMock(return_value="recovered transcript")

    with pytest.raises(RuntimeError, match="launch outcome is unknown") as error:
        await execute(session, collect=collector)
    assert error.value is failure
    stop.assert_awaited_once()
    collector.assert_not_awaited()
    assert session.cleanup is None
    session.sandbox.disconnect.assert_not_awaited()

    await session.close(timeout=1)
    assert stop.await_count == 2
    collector.assert_awaited_once()
    assert session.artifacts == "recovered transcript"
    assert session.closed


async def test_rearm_uses_fresh_launch_files_and_close_removes_the_root(session):
    await session.begin_activation(session_dir="/session/activation-1", timeout=1)
    first = AsyncMock(return_value="first")
    assert await execute(session, collect=first) == "first"
    with pytest.raises(ValueError, match="fresh supervisor directory"):
        await session.begin_activation(session_dir=session.session_dir, timeout=1)
    await session.begin_activation(session_dir="/session/activation-2", timeout=1)
    second = AsyncMock(return_value="second")
    assert await execute(session, collect=second) == "second"
    assert session.artifacts == "second"
    first.assert_awaited_once()
    second.assert_awaited_once()
    uploads = [c.args[1] for c in session.sandbox.upload.await_args_list]
    assert uploads == ["/session/activation-1/process_supervisor.sh", "/session/activation-2/process_supervisor.sh"]
    await session.close(timeout=1)
    assert "mv /session /session.closed" in session.sandbox.exec.await_args.args[0]
    with pytest.raises(RuntimeError, match="closing"):
        await session.begin_activation(session_dir="/session/activation-3", timeout=1)


async def test_rearm_rejects_unconfirmed_termination(session, receipt_reader):
    receipt_reader.return_value = json.dumps(RECEIPT | {"cleanup_confirmed": False})
    with pytest.raises(NgFailureEvent, match="termination was not confirmed") as failure:
        await execute(session)
    assert is_harness_termination_failure(failure.value)
    with pytest.raises(NgFailureEvent, match="termination was not confirmed") as rearm:
        await session.begin_activation(session_dir="/session/activation-2", timeout=1)
    assert is_harness_termination_failure(rearm.value)
    assert session.session_dir == "/session"
    await session.close(timeout=1)


async def test_deadline_during_staging_never_launches(session, receipt_reader):
    from time import monotonic

    cancelled = asyncio.Event()

    async def stage():
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    collect = AsyncMock()
    with pytest.raises(TimeoutError):
        await session.execute(
            stage_activation=stage, collect=collect, timeout=10, close_timeout=1, deadline=monotonic() + 0.01
        )
    assert cancelled.is_set() and not session.launch_started
    session.sandbox.exec.assert_not_awaited()
    receipt_reader.assert_not_awaited()
    collect.assert_not_awaited()
    await session.close(timeout=1)


async def test_deadline_shorter_than_supervisor_resolution_never_launches(session, receipt_reader):
    from time import monotonic

    collect = AsyncMock()
    with pytest.raises(TimeoutError, match="expired before launch"):
        await session.execute(
            stage_activation=AsyncMock(return_value=SandboxCommand(argv=["true"])),
            collect=collect,
            timeout=0.004,
            close_timeout=1,
            deadline=monotonic() + 10,
        )
    assert not session.launch_started
    session.sandbox.exec.assert_not_awaited()
    collect.assert_not_awaited()
    await session.close(timeout=1)


async def test_close_during_input_preparation_prevents_launch(session, receipt_reader):
    preparing = asyncio.Event()

    async def stage_activation():
        preparing.set()
        await asyncio.Future()

    collector = AsyncMock()
    running = asyncio.create_task(execute(session, stage_activation=stage_activation, collect=collector))
    await asyncio.wait_for(preparing.wait(), 2)
    await session.close(timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await running
    assert not session.launch_started
    assert session.cleanup is None
    collector.assert_not_awaited()
    receipt_reader.assert_not_awaited()
    # Only session-directory removal ran.
    assert session.sandbox.exec.await_count == 1
    assert "mv /session /session.closed" in session.sandbox.exec.await_args.args[0]


@pytest.mark.parametrize("failure", [OSError("transport failed"), "timeout", "unavailable"])
async def test_transport_failure_preserves_partial_output_and_original_error(session, failure):
    if isinstance(failure, Exception):
        session.sandbox.exec.side_effect = [failure, OK]
    else:
        session.sandbox.exec.side_effect = [SandboxExecResult("", "transport", 1, failure), OK]
    expected = OSError if isinstance(failure, Exception) else TimeoutError if failure == "timeout" else RuntimeError
    with pytest.raises(expected):
        await execute(session)
    assert session.cleanup == RECEIPT
    assert session.artifacts == "transcript"
    await session.close(timeout=1)
    assert session.closed


async def test_capture_failure_does_not_hide_execution_failure(session):
    session.sandbox.exec.side_effect = [OSError("transport failed"), OK]
    collector = AsyncMock(side_effect=ValueError("malformed snapshot"))
    with pytest.raises(OSError, match="transport failed"):
        await execute(session, collect=collector)
    assert isinstance(session.capture_error, ValueError)
    assert session.cleanup == RECEIPT
    await session.close(timeout=1)
    collector.assert_awaited_once()
    assert session.closed


async def test_capture_timeout_does_not_block_resource_release(session):
    async def collect():
        await asyncio.Future()

    with pytest.raises(TimeoutError):
        await execute(session, collect=collect, timeout=0.02)
    assert isinstance(session.capture_error, TimeoutError)
    assert session.cleanup == RECEIPT
    await session.close(timeout=1)
    assert session.closed


async def test_missing_borrowed_receipt_retains_state_and_retries_capture(session, receipt_reader):
    receipt_reader.side_effect = FileNotFoundError("no receipt yet")
    collector = AsyncMock(return_value="recovered transcript")
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await execute(session, collect=collector)
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await session.close(timeout=1)
    assert session.closing and not session.closed
    assert session.cleanup is None
    collector.assert_not_awaited()
    session.sandbox.disconnect.assert_not_awaited()
    session.sandbox.stop.assert_not_awaited()
    assert all("rm -rf" not in call.args[0] for call in session.sandbox.exec.await_args_list)
    receipt_reader.side_effect = None
    receipt_reader.return_value = json.dumps(RECEIPT)
    await session.close(timeout=1)
    collector.assert_awaited_once()
    assert session.artifacts == "recovered transcript"
    assert session.closed


async def test_transport_error_survives_failed_cleanup_until_receipt_arrives(session, receipt_reader):
    failure = OSError("transport failed")
    session.sandbox.exec.side_effect = [failure, OK, OK, OK]
    receipt_reader.side_effect = FileNotFoundError("no receipt yet")
    collector = AsyncMock(return_value="recovered transcript")

    with pytest.raises(OSError, match="transport failed") as error:
        await execute(session, collect=collector)
    assert error.value is failure
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await session.close(timeout=1)
    assert not session.closed
    assert session.cleanup is None
    collector.assert_not_awaited()
    session.sandbox.disconnect.assert_not_awaited()
    session.sandbox.stop.assert_not_awaited()
    assert all("rm -rf" not in call.args[0] for call in session.sandbox.exec.await_args_list)

    receipt_reader.side_effect = None
    await session.close(timeout=1)
    assert session.closed
    assert session.cleanup == RECEIPT
    assert session.artifacts == "recovered transcript"
    collector.assert_awaited_once()
    session.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize(
    "result", [SandboxExecResult("", "remove failed", 1), SandboxExecResult("", "timeout", 0, "timeout")]
)
async def test_directory_removal_failure_retains_borrowed_connection_for_retry(session, result):
    collector = AsyncMock(return_value="saved transcript")
    await execute(session, collect=collector)
    session.sandbox.exec.side_effect = [result, OK]

    with pytest.raises(RuntimeError, match="Could not remove test session files"):
        await session.close(timeout=1)
    assert not session.closed
    session.sandbox.disconnect.assert_not_awaited()
    session.sandbox.stop.assert_not_awaited()

    await session.close(timeout=1)
    assert session.closed
    assert session.artifacts == "saved transcript"
    collector.assert_awaited_once()
    session.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("owned", [False, True])
async def test_terminal_negative_receipt_skips_capture_but_releases_session(session, receipt_reader, owned):
    session.owns_sandbox = owned
    receipt = RECEIPT | {"cleanup_confirmed": False, "error": "Harness process group has not terminated"}
    receipt_reader.return_value = json.dumps(receipt)
    collector = AsyncMock()
    with pytest.raises(NgFailureEvent, match="harness termination was not confirmed") as failure:
        await execute(session, collect=collector)
    assert is_harness_termination_failure(failure.value)
    assert session.cleanup is None
    assert session.termination_error.receipt == receipt
    await session.close(timeout=1)
    await session.close(timeout=1)
    assert session.closed
    assert session.capture_error is session.termination_error
    collector.assert_not_awaited()
    receipt_reader.assert_awaited_once()
    if owned:
        session.sandbox.stop.assert_awaited_once()
        session.sandbox.disconnect.assert_not_awaited()
    else:
        session.sandbox.stop.assert_not_awaited()
        session.sandbox.disconnect.assert_awaited_once()
        assert "mv /session /session.closed" in session.sandbox.exec.await_args.args[0]


async def test_owned_cleanup_deadline_falls_back_to_provider_stop(session, receipt_reader):
    session.owns_sandbox = True
    launched = asyncio.Event()

    async def provider_exec(command, **kwargs):
        launched.set()
        await asyncio.Future()

    async def unavailable(*args, **kwargs):
        await asyncio.Future()

    session.sandbox.exec.side_effect = provider_exec
    receipt_reader.side_effect = unavailable
    collector = AsyncMock()
    running = asyncio.create_task(execute(session, collect=collector, timeout=0.02))
    await asyncio.wait_for(launched.wait(), 2)
    await asyncio.wait_for(session.close(timeout=0.02), 1)
    with pytest.raises(asyncio.CancelledError):
        await running
    assert session.sandbox_stopped and session.closed
    assert session.cleanup is None
    assert session.capture_error is not None
    collector.assert_not_awaited()
    session.sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("owned", [False, True])
async def test_release_failure_retries_without_recollecting(session, owned):
    session.owns_sandbox = owned
    release = session.sandbox.stop if owned else session.sandbox.disconnect
    release.side_effect = [OSError("release failed"), None]
    collector = AsyncMock(return_value="saved transcript")
    await execute(session, collect=collector)
    with pytest.raises(OSError, match="release failed"):
        await session.close(timeout=1)
    assert not session.closed
    await session.close(timeout=1)
    assert session.closed
    assert release.await_count == 2
    collector.assert_awaited_once()


@pytest.mark.parametrize("return_code", [1, None, "malformed"])
async def test_nonzero_worker_exit_still_passes_artifacts_to_adapter(session, receipt_reader, return_code):
    session.sandbox.exec.return_value = SandboxExecResult("", "worker error", 1)
    receipt_reader.return_value = json.dumps(RECEIPT | {"return_code": return_code})
    assert await execute(session) == "transcript"
    assert session.cleanup["return_code"] == (return_code if type(return_code) is int else None)


async def test_bootstrap_failure_preserves_stderr_and_still_releases(session, receipt_reader):
    receipt_reader.return_value = json.dumps(RECEIPT | {"return_code": None})
    session.sandbox.exec.return_value = SandboxExecResult("", "sh: 1: bash: not found", 127)
    collector = AsyncMock(side_effect=FileNotFoundError("no worker output"))
    with pytest.raises(RuntimeError, match="sandbox execution failed.*127.*bash: not found") as error:
        await execute(session, collect=collector)
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert session.cleanup["cleanup_confirmed"] is True
    assert session.cleanup["return_code"] is None
    collector.assert_awaited_once()
    session.sandbox.exec.return_value = OK
    await session.close(timeout=1)
    assert session.closed
    session.sandbox.disconnect.assert_awaited_once()


async def test_session_uploads_supervisor_before_launch(session):
    from pathlib import Path

    from nemo_gym.agent_utils.supervisor_client import SUPERVISOR_FILE, SUPERVISOR_SCRIPT

    events = []

    async def upload(source, destination):
        assert Path(source).read_bytes() == SUPERVISOR_SCRIPT.read_bytes()
        assert destination == f"/session/{SUPERVISOR_FILE}"
        events.append("upload")

    async def launch(*args, **kwargs):
        assert events == ["upload"]
        events.append("launch")
        return OK

    session.sandbox.upload.side_effect = upload
    session.sandbox.exec.side_effect = launch
    assert await execute(session) == "transcript"
    assert events == ["upload", "launch"]
    assert not session.closing and not session.closed
    session.sandbox.stop.assert_not_awaited()
    session.sandbox.disconnect.assert_not_awaited()


@pytest.mark.parametrize("owned", [False, True])
async def test_close_during_supervisor_upload_joins_staging_before_release(session, receipt_reader, owned):
    session.owns_sandbox = owned
    uploading = asyncio.Event()
    events = []

    async def upload(*args):
        uploading.set()
        try:
            await asyncio.Future()
        finally:
            events.append("upload_settled")

    session.sandbox.upload.side_effect = upload
    release = session.sandbox.stop if owned else session.sandbox.disconnect
    release.side_effect = lambda: events.append("release")
    collector = AsyncMock()
    running = asyncio.create_task(execute(session, collect=collector))
    await asyncio.wait_for(uploading.wait(), 2)
    await session.close(timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await running
    assert events == ["upload_settled", "release"]
    assert not session.launch_started
    assert session.closed and session.closing
    collector.assert_not_awaited()
    receipt_reader.assert_not_awaited()
    assert all("exec " not in call.args[0] for call in session.sandbox.exec.await_args_list)


async def test_supervisor_upload_failure_preserves_error_and_allows_close(session, receipt_reader):
    session.sandbox.upload.side_effect = OSError("supervisor upload failed")
    collector = AsyncMock()
    with pytest.raises(OSError, match="supervisor upload failed"):
        await execute(session, collect=collector)
    assert not session.launch_started
    assert not session.closing
    collector.assert_not_awaited()
    receipt_reader.assert_not_awaited()
    session.sandbox.exec.assert_not_awaited()
    await session.close(timeout=1)
    assert session.closed
    session.sandbox.disconnect.assert_awaited_once()


async def test_cancelled_close_stays_closing_and_can_retry(session):
    release_started = asyncio.Event()

    async def disconnect():
        release_started.set()
        await asyncio.Future()

    session.sandbox.disconnect.side_effect = disconnect
    closing = asyncio.create_task(session.close(timeout=1))
    await asyncio.wait_for(release_started.wait(), 2)
    session._close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert session.closing and not session.closed
    with pytest.raises(RuntimeError, match="closing or already activated"):
        await execute(session)
    session.sandbox.disconnect.side_effect = None
    await session.close(timeout=1)
    assert session.closed


async def test_failure_log_is_read_before_session_release(session, monkeypatch):
    from nemo_gym.agent_utils import sandbox_session

    async def read_log(sandbox, *, path):
        sandbox.disconnect.assert_not_awaited()
        assert path == "/session/output.log"
        return "diagnostic output"

    monkeypatch.setattr(sandbox_session, "read_text", read_log)

    async def collect():
        log = await session.read_output_log()
        raise FileNotFoundError(f"Missing artifacts: {log}")

    with pytest.raises(FileNotFoundError, match="Missing artifacts: diagnostic output"):
        await execute(session, collect=collect)
    await session.close(timeout=1)
    session.sandbox.disconnect.assert_awaited_once()


async def test_missing_failure_log_does_not_mask_capture_failure(session, monkeypatch):
    from nemo_gym.agent_utils import sandbox_session

    monkeypatch.setattr(sandbox_session, "read_text", AsyncMock(side_effect=OSError("download failed")))

    async def collect():
        log = await session.read_output_log()
        raise FileNotFoundError(f"Missing artifacts: {log}")

    with pytest.raises(FileNotFoundError, match="Missing artifacts"):
        await execute(session, collect=collect)
    assert session.cleanup == RECEIPT
    await session.close(timeout=1)
    assert session.closed


async def test_expire_while_running_requests_the_timeout_path_once(session, receipt_reader):
    launched = asyncio.Event()
    finish = asyncio.Event()

    async def exec_(command, **_):
        if "--expire-file /session/expire.request" in command:
            launched.set()
            await finish.wait()
        return OK

    session.sandbox.exec.side_effect = exec_
    async with _launched(session, launched) as activation:
        execs = session.sandbox.exec.await_count
        assert await session.expire(timeout=1, reason="budget") is True
        assert await session.expire(timeout=1, reason="operator") is False
        assert session.expire_reason == "budget"
        # The marker is written as a file, once, with no exec the running harness could hold up.
        assert _expiry_writes(session) == 1 and session.sandbox.exec.await_count == execs
        assert not session.expired
        session.sandbox.exec.side_effect = None
        receipt_reader.return_value = json.dumps({**RECEIPT, "return_code": -15, "timed_out": True})
        finish.set()
        assert await asyncio.wait_for(activation, 5) == "transcript"
    assert session.expired and not session.expired_before_launch
    assert await session.expire(timeout=1) is False
    await session.close(timeout=1)


async def test_expire_before_launch_marks_the_launch_expired(session, receipt_reader):
    assert await session.expire(timeout=1) is True
    session.sandbox.exec.assert_not_awaited()
    assert _expiry_writes(session) == 0
    receipt_reader.return_value = json.dumps({**RECEIPT, "return_code": None, "timed_out": True})
    assert await execute(session) == "transcript"
    assert session.expired and session.expired_before_launch
    # The marker is in place before the supervisor launches, so it never starts the harness.
    calls = [(name, args) for name, args, _ in session.sandbox.mock_calls if name in ("upload", "exec")]
    marker = next(index for index, (name, args) in enumerate(calls) if name == "upload" and args[1] == EXPIRE_MARKER)
    launch = next(index for index, (name, args) in enumerate(calls) if name == "exec" and "--expire-file" in args[0])
    assert marker < launch
    assert f"--expire-file {EXPIRE_MARKER}" in calls[launch][1][0]
    await session.close(timeout=1)


async def test_expire_refused_once_closing(session):
    await session.close(timeout=1)
    calls = session.sandbox.exec.await_count
    assert await session.expire(timeout=1) is False
    assert session.sandbox.exec.await_count == calls and _expiry_writes(session) == 0


async def test_failed_expire_request_can_be_retried(session):
    launched = asyncio.Event()
    finish = asyncio.Event()

    async def exec_(command, **_):
        if "--expire-file" in command:
            launched.set()
            await finish.wait()
        return OK

    async def upload(source, target):
        if target == EXPIRE_MARKER and upload.fail:
            upload.fail = False
            raise OSError("no dir")

    upload.fail = True
    session.sandbox.exec.side_effect = exec_
    session.sandbox.upload.side_effect = upload
    async with _launched(session, launched) as activation:
        with pytest.raises(RuntimeError, match="Could not request test expiry: no dir"):
            await session.expire(timeout=1, reason="budget")
        assert session.expire_reason is None
        assert await session.expire(timeout=1) is True
        finish.set()
        await asyncio.wait_for(activation, 5)
    await session.close(timeout=1)


async def test_expire_is_bounded_when_the_marker_cannot_be_written(session):
    launched = asyncio.Event()
    finish = asyncio.Event()

    async def exec_(command, **_):
        if "--expire-file" in command:
            launched.set()
            await finish.wait()
        return OK

    async def upload(source, target):
        if target == EXPIRE_MARKER and upload.stall:
            # A transport that never answers, or a provider queue the call cannot leave.
            await asyncio.Future()

    upload.stall = True
    session.sandbox.exec.side_effect = exec_
    session.sandbox.upload.side_effect = upload
    async with _launched(session, launched) as activation:
        started = asyncio.get_running_loop().time()
        with pytest.raises(RuntimeError, match="Could not request test expiry within 0.2 s"):
            await session.expire(timeout=0.2, reason="budget")
        assert asyncio.get_running_loop().time() - started < 1
        assert session.expire_reason is None
        upload.stall = False
        assert await session.expire(timeout=1, reason="operator") is True
        assert session.expire_reason == "operator"
        finish.set()
        await asyncio.wait_for(activation, 5)
    await session.close(timeout=1)


@pytest.mark.parametrize("marker_landed", [True, False])
async def test_failed_expire_write_keeps_the_stop_only_if_the_marker_landed(session, receipt_reader, marker_landed):
    launched = asyncio.Event()
    finish = asyncio.Event()
    written = set()

    async def exec_(command, **_):
        if "--expire-file" in command:
            launched.set()
            await finish.wait()
        if "echo present" in command:
            present = EXPIRE_MARKER in written
            return SandboxExecResult(stdout="present\n" if present else "", stderr="", return_code=0 if present else 1)
        return OK

    async def upload(source, target):
        if target == EXPIRE_MARKER:
            if marker_landed:
                written.add(target)
            raise ConnectionError("acknowledgment lost")

    session.sandbox.exec.side_effect = exec_
    session.sandbox.upload.side_effect = upload
    async with _launched(session, launched) as activation:
        with pytest.raises(RuntimeError, match="Could not request test expiry"):
            await session.expire(timeout=1, reason="budget")
        assert session.expire_reason is None
        # The harness ends at a timeout: the stop's if its marker landed, its own deadline otherwise.
        receipt_reader.return_value = json.dumps({**RECEIPT, "return_code": -15, "timed_out": True})
        finish.set()
        assert await asyncio.wait_for(activation, 5) == "transcript"
    assert session.expired is marker_landed
    assert session.expire_reason == ("budget" if marker_landed else None)
    await session.close(timeout=1)


async def test_expire_write_failing_during_finalization_still_keeps_a_landed_stop(session, receipt_reader):
    # The marker lands, the harness ends at the stop, and the write fails only once finalization has begun.
    launched, landed, fail_write = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finish = asyncio.Event()
    written = set()

    async def exec_(command, **_):
        if "--expire-file" in command:
            launched.set()
            await finish.wait()
        if "echo present" in command:
            present = EXPIRE_MARKER in written
            return SandboxExecResult(stdout="present\n" if present else "", stderr="", return_code=0 if present else 1)
        return OK

    async def upload(source, target):
        if target == EXPIRE_MARKER:
            written.add(target)
            landed.set()
            await fail_write.wait()
            raise ConnectionError("acknowledgment lost")

    session.sandbox.exec.side_effect = exec_
    session.sandbox.upload.side_effect = upload
    async with _launched(session, launched) as activation:
        stop = asyncio.create_task(session.expire(timeout=5, reason="budget"))
        await asyncio.wait_for(landed.wait(), 2)
        receipt_reader.return_value = json.dumps({**RECEIPT, "return_code": -15, "timed_out": True})
        finish.set()
        # Finalization waits for the write still in flight instead of reconciling past it.
        await asyncio.sleep(0.1)
        assert not activation.done()
        fail_write.set()
        with pytest.raises(RuntimeError, match="Could not request test expiry"):
            await asyncio.wait_for(stop, 2)
        assert await asyncio.wait_for(activation, 5) == "transcript"
    assert session.expired and session.expire_reason == "budget"
    await session.close(timeout=1)


@pytest.mark.parametrize("write_fails", [False, True])
async def test_a_stop_whose_caller_disconnects_before_its_write_starts_is_delivered_or_retryable(
    session, receipt_reader, write_fails
):
    launched = asyncio.Event()
    finish = asyncio.Event()

    async def exec_(command, **_):
        if "--expire-file" in command:
            launched.set()
            await finish.wait()
        return OK

    async def upload(source, target):
        if target == EXPIRE_MARKER and upload.fail:
            upload.fail = False
            raise OSError("no dir")

    upload.fail = write_fails
    session.sandbox.exec.side_effect = exec_
    session.sandbox.upload.side_effect = upload
    async with _launched(session, launched) as activation:
        stop = asyncio.create_task(session.expire(timeout=1, reason="budget"))
        await asyncio.sleep(0)
        # expire has created its write, which has not run yet, when the stop request disconnects.
        assert session._expire_write is not None and not session._expire_write.done()
        stop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop
        await asyncio.wait({session._expire_write}, timeout=2)
        assert _expiry_writes(session) == 1
        if write_fails:
            assert await session.expire(timeout=1, reason="operator") is True
            assert _expiry_writes(session) == 2
        else:
            assert await session.expire(timeout=1) is False
        receipt_reader.return_value = json.dumps({**RECEIPT, "return_code": -15, "timed_out": True})
        finish.set()
        assert await asyncio.wait_for(activation, 5) == "transcript"
    assert session.expired and session.expire_reason == ("operator" if write_fails else "budget")
    await session.close(timeout=1)


async def test_a_stop_write_still_pending_when_finalization_stops_waiting_stays_unattributed(session, receipt_reader):
    launched = asyncio.Event()
    finish = asyncio.Event()
    fail_write = asyncio.Event()

    async def exec_(command, **_):
        if "--expire-file" in command:
            launched.set()
            await finish.wait()
        return OK

    async def upload(source, target):
        if target == EXPIRE_MARKER:
            await fail_write.wait()
            raise ConnectionError("upload failed")

    session.sandbox.exec.side_effect = exec_
    session.sandbox.upload.side_effect = upload
    async with _launched(session, launched) as activation:
        stop = asyncio.create_task(session.expire(timeout=5, reason="budget"))
        await asyncio.sleep(0.05)
        # The harness ends at its own deadline while the write is still pending, past finalization's 1 s bound.
        receipt_reader.return_value = json.dumps({**RECEIPT, "return_code": -15, "timed_out": True})
        finish.set()
        assert await asyncio.wait_for(activation, 3) == "transcript"
        assert not session.expired and session.expire_reason is None
        fail_write.set()
        with pytest.raises(RuntimeError, match="Could not request test expiry"):
            await asyncio.wait_for(stop, 2)
    assert not session.expired
    await session.close(timeout=1)


async def test_expire_refuses_a_running_harness_whose_sandbox_uploads_can_wait_behind_it(session, receipt_reader):
    launched = asyncio.Event()
    finish = asyncio.Event()

    async def exec_(command, **_):
        if "--expire-file" in command:
            launched.set()
            await finish.wait()
        return OK

    session.sandbox.exec.side_effect = exec_
    session.sandbox.uploads_independent_of_execs = False
    async with _launched(session, launched) as activation:
        with pytest.raises(ExpiryUnsupportedError, match="cannot stop a running harness"):
            await session.expire(timeout=1, reason="budget")
        assert _expiry_writes(session) == 0 and session.expire_reason is None
        finish.set()
        assert await asyncio.wait_for(activation, 5) == "transcript"
    assert not session.expired
    await session.close(timeout=1)


async def test_expire_before_launch_works_whatever_the_sandbox_uploads(session, receipt_reader):
    session.sandbox.uploads_independent_of_execs = False
    assert await session.expire(timeout=1) is True
    receipt_reader.return_value = json.dumps({**RECEIPT, "return_code": None, "timed_out": True})
    assert await execute(session) == "transcript"
    assert session.expired_before_launch
    await session.close(timeout=1)


@pytest.mark.parametrize(
    "name,options,independent",
    [
        ("local", {}, True),
        ("opensandbox", {"operations": {"background_exec": True}}, True),
        # A running foreground command holds one of the shared HTTP pool's connections.
        ("opensandbox", {"operations": {"background_exec": False}}, False),
        # Uploads share the CLI slots or worker threads that running commands hold.
        ("docker", None, False),
        ("apptainer", None, False),
        ("enroot", None, False),
        ("openshell", None, False),
        # Not shown to be independent: their SDKs keep running commands on connection pools Gym does not bound.
        ("daytona", None, False),
        ("e2b", None, False),
        ("ecs_fargate", None, False),
    ],
)
def test_which_sandboxes_can_stop_a_running_harness(name, options, independent):
    provider_class = get_provider_class(name)
    if options is None:
        assert not issubclass(provider_class, SupportsSandboxIndependentUploads)
        provider = MagicMock(spec=provider_class)
    else:
        provider = provider_class(**options)
    assert AsyncSandbox(provider=provider, spec=SandboxSpec()).uploads_independent_of_execs is independent


@pytest.mark.skipif(sys.platform != "linux", reason="process supervision requires Linux")
async def test_stop_confirmation_is_bounded_when_every_local_exec_slot_is_held(tmp_path, monkeypatch):
    # The marker check runs during finalization; another workload holding every exec slot must not stall grading.
    monkeypatch.setattr(LocalProvider, "_semaphores", {})
    provider = LocalProvider(workspace_root=str(tmp_path), concurrency=1)
    workdir = tmp_path / "task"
    sandbox = await AsyncSandbox(provider, owns_provider=False).start(SandboxSpec(workdir=str(workdir)))
    session = SandboxSession(
        sandbox=sandbox, session_dir=str(workdir / "session"), workdir=str(workdir), harness="test", owns_sandbox=True
    )
    holder = asyncio.create_task(sandbox.exec("sleep 30", timeout_s=60))
    try:
        async with asyncio.timeout(5):
            while not provider._semaphore.locked():
                await asyncio.sleep(0.01)
        # A stop whose marker write failed, possibly after the marker landed, and a run that ended at a timeout.
        session._unconfirmed_expiry, session._unconfirmed_expire_reason = True, "budget"
        session.cleanup = {**RECEIPT, "return_code": -15, "timed_out": True}
        begun = monotonic()
        await asyncio.wait_for(session._finish(timeout=0.2), 5)
        assert monotonic() - begun < 1
        assert not session.expired and session.expire_reason is None
    finally:
        holder.cancel()
        await asyncio.gather(holder, return_exceptions=True)
        await session.close(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="process supervision requires Linux")
@pytest.mark.parametrize("sessions, concurrency", [(1, 1), (8, 8)])
async def test_expire_reaches_harnesses_that_hold_every_local_exec_slot(tmp_path, monkeypatch, sessions, concurrency):
    # Each harness's launch holds one of the provider's exec slots for as long as it runs.
    monkeypatch.setattr(LocalProvider, "_semaphores", {})
    provider = LocalProvider(workspace_root=str(tmp_path), concurrency=concurrency)
    runs = []
    for index in range(sessions):
        workdir = tmp_path / f"task-{index}"
        sandbox = await AsyncSandbox(provider, owns_provider=False).start(SandboxSpec(workdir=str(workdir)))
        session = SandboxSession(
            sandbox=sandbox,
            session_dir=str(workdir / "session"),
            workdir=str(workdir),
            harness="test",
            owns_sandbox=True,
        )
        started = workdir / "started"
        worker = f"import pathlib,time; pathlib.Path({str(started)!r}).touch(); time.sleep(60)"
        activation = asyncio.create_task(
            session.execute(
                stage_activation=AsyncMock(return_value=SandboxCommand(argv=[sys.executable, "-c", worker])),
                collect=AsyncMock(return_value="partial transcript"),
                timeout=60,
                close_timeout=5,
            )
        )
        runs.append((session, started, activation))
    try:
        async with asyncio.timeout(20):
            while not all(started.exists() for _, started, _ in runs):
                await asyncio.sleep(0.05)
        begun = monotonic()
        assert [await session.expire(timeout=2, reason="budget") for session, _, _ in runs] == [True] * sessions
        assert monotonic() - begun < 2 * sessions
        async with asyncio.timeout(20):
            results = await asyncio.gather(*(activation for _, _, activation in runs))
        assert results == ["partial transcript"] * sessions
        for session, _, _ in runs:
            assert session.cleanup["timed_out"] is True and session.cleanup["cleanup_confirmed"] is True
            assert session.expired and not session.expired_before_launch
    finally:
        for session, _, activation in runs:
            activation.cancel()
            await asyncio.gather(activation, return_exceptions=True)
            await session.close(timeout=5)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_shell_assignments_round_trip_through_bash(tmp_path):
    tricky = 'it\'s $(touch unintended) `x` "q" \\ \n new line'
    script = tmp_path / "params.sh"
    script.write_text(
        shell_assignments({"cwd": tricky, "command": ["cli", tricky, ""], "env": {"KEY": tricky, "EMPTY": ""}})
    )
    printed = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\0" "$cwd" "${#command[@]}" "${command[@]}" "${env[@]}"', "_", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    ).stdout.split("\0")[:-1]
    assert printed == [tricky, "3", "cli", tricky, "", f"KEY={tricky}", "EMPTY="]
    assert not (tmp_path / "unintended").exists()


@pytest.mark.parametrize(
    "values",
    [{"bad-name": "x"}, {"env": {"BAD KEY": "x"}}, {"cwd": "a\0b"}, {"command": ["cli", 1]}, {"env": {"K": None}}],
)
def test_shell_assignments_reject_names_bash_cannot_assign(values):
    with pytest.raises(ValueError, match="Invalid"):
        shell_assignments(values)
