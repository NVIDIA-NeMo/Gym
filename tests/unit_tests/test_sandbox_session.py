# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agent_utils import supervisor_client
from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult


RECEIPT = {"cleanup_confirmed": True, "return_code": 0, "timed_out": False, "error": None}
OK = SandboxExecResult(stdout="", stderr="", return_code=0)


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
        stage_activation=stage_activation
        or AsyncMock(return_value=SandboxCommand(argv=["worker"], python="/runtime/python")),
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
        if "trap '' TERM" in command:
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


async def test_borrowed_cleanup_failure_retains_state_and_retries_capture(session, receipt_reader):
    receipt_reader.return_value = json.dumps(RECEIPT | {"cleanup_confirmed": False})
    collector = AsyncMock(return_value="recovered transcript")
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await execute(session, collect=collector)
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await session.close(timeout=1)
    assert session.closing and not session.closed
    assert session.cleanup is None
    collector.assert_not_awaited()
    session.sandbox.disconnect.assert_not_awaited()
    session.sandbox.stop.assert_not_awaited()
    assert all("rm -rf" not in call.args[0] for call in session.sandbox.exec.await_args_list)
    receipt_reader.return_value = json.dumps(RECEIPT)
    await session.close(timeout=1)
    collector.assert_awaited_once()
    assert session.artifacts == "recovered transcript"
    assert session.closed


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
    session.sandbox.exec.return_value = SandboxExecResult("", "/runtime/python: not found", 127)
    collector = AsyncMock(side_effect=FileNotFoundError("no worker output"))
    with pytest.raises(RuntimeError, match="sandbox execution failed.*127.*python: not found") as error:
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

    from nemo_gym.agent_utils import process_supervisor
    from nemo_gym.agent_utils.supervisor_client import SUPERVISOR_FILE

    events = []

    async def upload(source, destination):
        assert Path(source).read_bytes() == Path(process_supervisor.__file__).read_bytes()
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
