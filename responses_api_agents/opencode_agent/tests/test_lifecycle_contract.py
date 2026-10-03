# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cleanup evidence must not depend on worker metadata or parseable harness output."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import Request
from pydantic import ValidationError

from responses_api_agents.opencode_agent.tests.test_native_sessions import seed, setup  # noqa: F401


async def session(setup):
    agent, sandbox = setup
    body = seed()
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    state = agent._native_sessions[body.agent_session_id]
    payload = {"directory": state.directory, "cwd": state.workdir, "prompt": "task", "command": [], "env": {}}
    return state, sandbox, payload


@pytest.mark.parametrize(
    "runtime",
    [
        {},
        {"hostname": "worker", "pid": "123"},
        {"hostname": 1, "pid": 123},
        {"hostname": "worker", "pid": 123, "return_code": 0},
    ],
)
async def test_invalid_runtime_fails_execution_but_not_confirmed_close(setup, runtime):
    state, sandbox, payload = await session(setup)
    sandbox.runtime_info = runtime
    with pytest.raises(ValidationError):
        await state.execute(payload, timeout=5, close_timeout=1)
    assert state.cleanup == {"cleanup_confirmed": True, "return_code": 0, "error": None, "timed_out": False}
    await state.close(1)
    assert state.closed
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("invalid", [{"return_code": "0"}, {"cleanup_confirmed": 1}, {"hostname": "mixed"}])
async def test_bad_cleanup_keeps_state_retryable_without_reading_runtime(setup, invalid):
    state, sandbox, payload = await session(setup)
    sandbox.result.update(invalid)
    with pytest.raises((RuntimeError, ValidationError)):
        await state.execute(payload, timeout=5, close_timeout=1)
    assert state.cleanup is None
    assert state.runtime_info is None
    with pytest.raises((RuntimeError, ValidationError)):
        await state.close(1)
    assert not state.closed
    sandbox.disconnect.assert_not_awaited()
    sandbox.files[f"{state.directory}/cleanup.json"] = json.dumps({"cleanup_confirmed": True, "error": None})
    # A stop-before-launch receipt suffices for cleanup, not execution success.
    await state.close(1)
    assert state.closed
    assert state.runtime_info is None


async def test_output_failure_does_not_erase_cleanup_confirmation(setup):
    state, sandbox, payload = await session(setup)
    execute = sandbox.exec.side_effect

    async def fail_output(command, **kwargs):
        if "--snapshot" in command:
            assert state.cleanup["cleanup_confirmed"] is True
            return SimpleNamespace(return_code=1, error_type=None, stderr="database unreadable")
        return await execute(command, **kwargs)

    sandbox.exec.side_effect = fail_output
    with pytest.raises(RuntimeError, match="transcript capture failed"):
        await state.execute(payload, timeout=5, close_timeout=1)
    assert state.cleanup["cleanup_confirmed"] is True
    assert state.cleanup["error"] is None
    await state.close(1)
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("capture_fails", [False, True])
async def test_cancel_captures_partial_output_only_after_cleanup_and_keeps_original_error(setup, capture_fails):
    state, sandbox, payload = await session(setup)
    sandbox.blocked = True
    execute = sandbox.exec.side_effect
    snapshots = []

    async def capture(command, **kwargs):
        if "--snapshot" in command:
            assert state.cleanup["cleanup_confirmed"] is True
            snapshots.append(command)
            if capture_fails:
                raise RuntimeError("snapshot unavailable")
        return await execute(command, **kwargs)

    sandbox.exec.side_effect = capture
    state.task = asyncio.create_task(state.execute(payload, timeout=5, close_timeout=1))
    await asyncio.wait_for(sandbox.started.wait(), 2)
    state.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await state.task
    assert len(snapshots) == 1
    assert state.cleanup["error"] is None
    await state.close(1)
    assert state.closed
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize(
    "receipt", [None, {"cleanup_confirmed": False, "return_code": 0, "error": None, "timed_out": False}]
)
async def test_snapshot_requires_positive_cleanup_before_any_io(setup, receipt):
    state, sandbox, _ = await session(setup)
    state.cleanup = receipt
    sandbox.exec.reset_mock()
    with pytest.raises(RuntimeError, match="requires confirmed cleanup"):
        await state.snapshot(1)
    sandbox.exec.assert_not_awaited()
    await state.close(1)
