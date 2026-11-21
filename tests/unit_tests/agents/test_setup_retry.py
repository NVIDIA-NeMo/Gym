# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents.utils import setup_retry
from nemo_gym.agents.utils.setup_retry import exec_idempotent
from nemo_gym.sandbox.providers.base import SandboxExecResult


OK = SandboxExecResult(stdout="ok", stderr="", return_code=0)


async def test_a_raised_exec_is_retried_and_its_result_returned(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [ConnectionError("502: The sandbox could not be reached"), OK]
    assert await exec_idempotent(sandbox, "git rev-parse HEAD", what="Reading HEAD", delay_s=0, cwd="/w") is OK
    assert sandbox.exec.await_count == 2
    assert sandbox.exec.await_args.kwargs == {"cwd": "/w", "env": None, "timeout_s": None}
    assert "Reading HEAD failed (attempt 1/3)" in caplog.text


async def test_the_working_directory_environment_and_timeout_reach_every_attempt() -> None:
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [ConnectionError("502: The sandbox could not be reached"), OK]
    result = await exec_idempotent(
        sandbox, "git status", what="Reading the status", delay_s=0, cwd="/w", env={"GIT_PAGER": "cat"}, timeout_s=7
    )
    assert result is OK
    calls = sandbox.exec.await_args_list
    assert [call.args for call in calls] == [("git status",)] * 2
    assert [(call.kwargs["cwd"], call.kwargs["env"]) for call in calls] == [("/w", {"GIT_PAGER": "cat"})] * 2
    # Each attempt gets what is left of the one budget.
    assert 0 < calls[1].kwargs["timeout_s"] <= calls[0].kwargs["timeout_s"] <= 7


async def test_attempts_and_the_waits_between_them_share_one_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 0.0

    async def exec_that_times_out(command: str, **kwargs: object) -> SandboxExecResult:
        nonlocal now
        now += 4
        raise TimeoutError("the provider gave up")

    async def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(setup_retry, "monotonic", lambda: now)
    monkeypatch.setattr(setup_retry, "asyncio", SimpleNamespace(sleep=sleep))
    sandbox = AsyncMock()
    sandbox.exec.side_effect = exec_that_times_out
    with pytest.raises(TimeoutError, match="the provider gave up"):
        await exec_idempotent(sandbox, "apt-get install -y bash", what="Install", delay_s=1, timeout_s=10)
    # 4 s, a 1 s wait, 4 s more: the 2 s wait before a third attempt would pass the 10 s budget.
    assert [call.kwargs["timeout_s"] for call in sandbox.exec.await_args_list] == [10, 5]


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
