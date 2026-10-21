# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint polling wakes existing commands and stays local to the stopping sandbox."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


pytestmark = pytest.mark.sandbox


class Commands:
    def __init__(self, *, polls_to_finish: int | None = None) -> None:
        self.finished = False
        self.submissions = 0
        self.status_calls = 0
        self.polls_to_finish = polls_to_finish

    async def run(self, command: str, *, opts: Any) -> Any:
        self.submissions += 1
        return SimpleNamespace(id=f"exec-{self.submissions}")

    async def get_command_status(self, execution_id: str) -> Any:
        self.status_calls += 1
        done = self.finished or (self.polls_to_finish is not None and self.status_calls >= self.polls_to_finish)
        return SimpleNamespace(running=not done, exit_code=0 if done else None, error=None)

    async def get_background_command_logs(self, execution_id: str) -> Any:
        return SimpleNamespace(content="finished")


def make_provider(monkeypatch: pytest.MonkeyPatch) -> opensandbox_provider.OpenSandboxProvider:
    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, SimpleNamespace, object, object)
    )
    return opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={
            "background_exec": True,
            "background_poll_initial_s": 30,
            "background_poll_interval_s": 30,
            "checkpoint_background_poll_interval_s": 1,
        },
    )


def handle(name: str, commands: Commands) -> SandboxHandle:
    return SandboxHandle(sandbox_id=name, provider_name="opensandbox", raw=SimpleNamespace(commands=commands))


@pytest.mark.asyncio
async def test_checkpoint_wakes_long_sleep_without_affecting_other_sandboxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: asyncio.Queue[tuple[float, asyncio.Event]] = asyncio.Queue()

    async def sleep(delay: float) -> None:
        release = asyncio.Event()
        sleeps.put_nowait((delay, release))
        await release.wait()

    monkeypatch.setattr(asyncio, "sleep", sleep)
    provider = make_provider(monkeypatch)
    first, second = Commands(), Commands()
    first_handle, second_handle = handle("first", first), handle("second", second)
    task1 = asyncio.create_task(provider.exec(first_handle, "work", timeout_s=30))
    delay1, _ = await asyncio.wait_for(sleeps.get(), 1)
    task2 = asyncio.create_task(provider.exec(second_handle, "work", timeout_s=30))
    delay2, release2 = await asyncio.wait_for(sleeps.get(), 1)
    assert (delay1, delay2) == (30, 30)
    try:
        # Never release the original 30-second sleep: entering the scope must wake it itself.
        with provider.checkpoint_exec_polling(first_handle):
            first.finished = True
            result = await asyncio.wait_for(task1, 1)
            assert result.return_code == 0
            assert not task2.done()
            assert not release2.is_set()
        second.finished = True
        release2.set()
        await asyncio.wait_for(task2, 1)
        assert first.submissions == second.submissions == 1
        assert first.status_calls == second.status_calls == 2
        assert provider._command_polls == {}
        assert provider._checkpoint_poll_scopes == {}
    finally:
        task1.cancel()
        task2.cancel()
        await asyncio.gather(task1, task2, return_exceptions=True)


@pytest.mark.asyncio
async def test_checkpoint_wakes_every_command_in_the_same_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: asyncio.Queue[float] = asyncio.Queue()

    async def sleep(delay: float) -> None:
        sleeps.put_nowait(delay)
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "sleep", sleep)
    provider = make_provider(monkeypatch)
    commands = Commands()
    sandbox_handle = handle("shared", commands)
    tasks = [asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=30)) for _ in range(2)]
    try:
        assert await asyncio.wait_for(sleeps.get(), 1) == 30
        assert await asyncio.wait_for(sleeps.get(), 1) == 30
        with provider.checkpoint_exec_polling(sandbox_handle):
            commands.finished = True
            results = await asyncio.wait_for(asyncio.gather(*tasks), 1)
        assert [result.return_code for result in results] == [0, 0]
        assert commands.submissions == 2
        assert commands.status_calls == 4
        assert provider._command_polls == {}
        assert provider._checkpoint_poll_scopes == {}
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_nested_checkpoint_scope_caps_new_command_polls_and_restores_normal_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    provider = make_provider(monkeypatch)
    commands = Commands(polls_to_finish=4)
    sandbox_handle = handle("nested", commands)
    sandbox = AsyncSandbox(provider=provider)
    sandbox._handle = sandbox_handle
    sandbox._stopped = False
    with sandbox.checkpoint_exec_polling():
        with sandbox.checkpoint_exec_polling():
            pass
        await provider.exec(sandbox_handle, "work", timeout_s=30)
    assert delays == [1, 1, 1]
    commands.status_calls = 0
    await provider.exec(sandbox_handle, "work", timeout_s=30)
    assert delays == [1, 1, 1, 30, 30, 30]
    assert provider._command_polls == {}
    assert provider._checkpoint_poll_scopes == {}


@pytest.mark.asyncio
async def test_cancelled_checkpoint_command_cleans_poll_waiters_and_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered_sleep = asyncio.Event()
    asleep = 0

    async def sleep(delay: float) -> None:
        nonlocal asleep
        asleep += 1
        entered_sleep.set()
        try:
            await asyncio.Event().wait()
        finally:
            asleep -= 1

    monkeypatch.setattr(asyncio, "sleep", sleep)
    provider = make_provider(monkeypatch)
    sandbox_handle = handle("cancelled", Commands())
    with provider.checkpoint_exec_polling(sandbox_handle):
        task = asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=30))
        await asyncio.wait_for(entered_sleep.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert asleep == 0
    assert provider._command_polls == {}
    assert provider._checkpoint_poll_scopes == {}


def test_provider_without_checkpoint_polling_keeps_existing_behavior() -> None:
    sandbox = AsyncSandbox(provider=SimpleNamespace())
    sandbox._handle = handle("plain", Commands())
    sandbox._stopped = False
    assert isinstance(sandbox.checkpoint_exec_polling(), nullcontext)
    with sandbox.checkpoint_exec_polling():
        pass


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_checkpoint_poll_interval_must_be_positive(interval: float) -> None:
    with pytest.raises(ValueError, match="checkpoint_background_poll_interval_s"):
        opensandbox_provider.OpenSandboxProvider(operations={"checkpoint_background_poll_interval_s": interval})
