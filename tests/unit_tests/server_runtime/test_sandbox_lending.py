# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LentSandboxes on the local sandbox provider, in a temporary working directory."""

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import pytest

from nemo_gym.core.episode_types import TaskId
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.server_runtime import sandbox_lending
from nemo_gym.server_runtime.sandbox_lending import LentSandboxes


TASK = TaskId(taskset="lending:test", task_id="t" * 80)


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(sandbox_lending, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    # The local provider cannot be reattached from another process, so it has no descriptor of its own.
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    return tmp_path / "workspace"


@pytest.fixture
def started(monkeypatch) -> list:
    specs = []
    real_start = AsyncSandbox.start

    async def start(self: AsyncSandbox, spec) -> None:
        specs.append(spec)
        await real_start(self, spec)

    monkeypatch.setattr(AsyncSandbox, "start", start)
    return specs


@pytest.fixture
def stopped(monkeypatch) -> list[AsyncSandbox]:
    stopped: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stopped.append(self)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return stopped


def _lender(workdir: Path) -> LentSandboxes:
    config = SandboxConfig(ttl_s=60, metadata={"team": "gym"})
    return LentSandboxes("sandbox", config, image="unused-by-the-local-provider", workdir=str(workdir))


async def test_lend_starts_one_sandbox_per_session_and_release_stops_it(workdir, started, stopped) -> None:
    lender = _lender(workdir)

    access = await lender.lend("session", TASK, files={str(workdir / "images" / "0.png"): b"png"})
    again = await lender.lend("session", TASK, files={str(workdir / "other.png"): b"ignored"})

    assert access == again
    assert access.workdir == str(workdir)
    assert access.connection.provider_config_ref == "sandbox"
    assert access.connection.descriptor == {"sandbox_id": "local"}
    # The harness gets an operate descriptor: it may use the sandbox, but only this server stops it.
    AsyncSandbox.serialize.assert_awaited_with(scope="operate")
    assert (workdir / "images" / "0.png").read_bytes() == b"png"
    assert not (workdir / "other.png").exists(), "a repeated lend must not start a second sandbox"
    assert len(started) == 1
    spec = started[0]
    assert spec.ttl_s == 60 and spec.workdir == str(workdir)
    assert spec.metadata == {"team": "gym", "task_id": "t" * 63}

    sandbox = lender.get("session")
    await lender.release("session")
    await lender.release("session")
    assert stopped == [sandbox]
    assert lender.get("session") is None


async def test_lend_can_start_an_empty_sandbox(workdir, started, stopped) -> None:
    lender = _lender(workdir)

    await lender.lend("session", TASK)

    assert started[0].files == {}
    await lender.release_all()
    assert len(stopped) == 1


async def test_a_failed_stop_keeps_the_sandbox_so_a_close_can_retry(workdir, started, monkeypatch) -> None:
    lender = _lender(workdir)
    await lender.lend("session", TASK)
    real_stop = AsyncSandbox.stop
    monkeypatch.setattr(AsyncSandbox, "stop", AsyncMock(side_effect=RuntimeError("provider down")))

    with pytest.raises(RuntimeError, match="provider down"):
        await lender.release("session")
    assert lender.get("session") is not None

    monkeypatch.setattr(AsyncSandbox, "stop", real_stop)
    await lender.release("session")
    assert lender.get("session") is None


async def test_release_all_stops_every_sandbox_and_logs_a_failed_stop(workdir, started, monkeypatch, caplog) -> None:
    lender = _lender(workdir)
    await lender.lend("ok", TASK)
    await lender.lend("stuck", TASK)
    stuck = lender.get("stuck")
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        if self is stuck:
            raise RuntimeError("provider down")
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    with caplog.at_level(logging.ERROR, logger=sandbox_lending.__name__):
        await lender.release_all()

    assert lender.get("ok") is None
    assert lender.get("stuck") is stuck
    assert "resources session stuck" in caplog.text


async def test_a_sandbox_that_cannot_be_described_is_stopped_and_not_kept(
    workdir, started, stopped, monkeypatch
) -> None:
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(side_effect=RuntimeError("no descriptor")))
    lender = _lender(workdir)

    with pytest.raises(RuntimeError, match="no descriptor"):
        await lender.lend("session", TASK)

    assert len(stopped) == 1
    assert lender.get("session") is None


async def test_release_finishes_the_stop_when_the_close_is_cancelled(workdir, started, monkeypatch) -> None:
    lender = _lender(workdir)
    await lender.lend("session", TASK)
    sandbox = lender.get("session")
    finished: list[AsyncSandbox] = []

    async def stop(self: AsyncSandbox) -> None:
        # A checkpoint where an unshielded stop would be cancelled with its request.
        await anyio.sleep(0)
        finished.append(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    with anyio.CancelScope() as scope:
        scope.cancel()
        await lender.release("session")

    assert finished == [sandbox]
    assert lender.get("session") is None


async def test_a_stop_that_hangs_times_out_and_keeps_the_sandbox(workdir, started, monkeypatch) -> None:
    lender = _lender(workdir)
    await lender.lend("session", TASK)
    monkeypatch.setattr(sandbox_lending, "STOP_TIMEOUT_S", 0.05)

    async def stop(self: AsyncSandbox) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    release = asyncio.create_task(lender.release("session"))
    done, _ = await asyncio.wait({release}, timeout=5)

    assert done == {release}, "release() must give up on a stop that hangs"
    with pytest.raises(TimeoutError):
        release.result()
    assert lender.get("session") is not None


async def test_release_all_stops_the_sandboxes_concurrently(workdir, started, monkeypatch) -> None:
    lender = _lender(workdir)
    for session in ("a", "b", "c"):
        await lender.lend(session, TASK)
    entered: list[AsyncSandbox] = []
    all_entered = asyncio.Event()

    async def stop(self: AsyncSandbox) -> None:
        # Each stop waits until all three have started, so stops run one at a time would never finish.
        entered.append(self)
        if len(entered) == 3:
            all_entered.set()
        await all_entered.wait()

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    await asyncio.wait_for(lender.release_all(), timeout=5)

    assert len(entered) == 3
    assert all(lender.get(session) is None for session in ("a", "b", "c"))
