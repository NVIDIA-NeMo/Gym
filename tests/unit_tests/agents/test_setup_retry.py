# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents.utils.setup_retry import exec_idempotent
from nemo_gym.sandbox.providers.base import SandboxExecResult


OK = SandboxExecResult(stdout="ok", stderr="", return_code=0)


async def test_a_raised_exec_is_retried_and_its_result_returned(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [ConnectionError("502: The sandbox could not be reached"), OK]
    assert await exec_idempotent(sandbox, "git rev-parse HEAD", what="Reading HEAD", delay_s=0, cwd="/w") is OK
    assert sandbox.exec.await_count == 2 and sandbox.exec.await_args.kwargs == {"cwd": "/w"}
    assert "Reading HEAD failed (attempt 1/3)" in caplog.text


async def test_a_non_zero_exit_is_returned_without_a_retry() -> None:
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="not a git repository", return_code=128)
    result = await exec_idempotent(sandbox, "git rev-parse HEAD", what="Reading HEAD", delay_s=0)
    assert result.return_code == 128 and sandbox.exec.await_count == 1


async def test_the_last_error_is_raised_when_attempts_run_out() -> None:
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [ConnectionError("a"), ConnectionError("b"), ConnectionError("c")]
    with pytest.raises(ConnectionError, match="c"):
        await exec_idempotent(sandbox, "true", what="Probe", delay_s=0)
    assert sandbox.exec.await_count == 3


async def test_cancellation_is_never_retried() -> None:
    sandbox = AsyncMock()
    sandbox.exec.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await exec_idempotent(sandbox, "true", what="Probe", delay_s=0)
    assert sandbox.exec.await_count == 1
