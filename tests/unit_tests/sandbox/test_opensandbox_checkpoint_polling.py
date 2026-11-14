# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint polling wakes existing commands and stays local to the stopping sandbox."""

import asyncio
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.core.failure_kinds import (
    PROVIDER_COMMAND_OUTCOME_UNKNOWN,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_STATUS_UNAVAILABLE,
)
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


pytestmark = pytest.mark.sandbox

# A guard against a hang, not a bound on the provider: the first typed failure imports the SDK on the event loop.
HANG_GUARD_S = 30


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
    task1 = asyncio.create_task(provider.exec(first_handle, "work", timeout_s=300))
    delay1, _ = await asyncio.wait_for(sleeps.get(), 1)
    task2 = asyncio.create_task(provider.exec(second_handle, "work", timeout_s=300))
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
    tasks = [asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=300)) for _ in range(2)]
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
        await provider.exec(sandbox_handle, "work", timeout_s=300)
    assert delays == [1, 1, 1]
    commands.status_calls = 0
    await provider.exec(sandbox_handle, "work", timeout_s=300)
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


class ScriptedStatus(Commands):
    """Each status request runs the next scripted step; ``asked`` is set once a request has started."""

    def __init__(self, *steps: Any) -> None:
        super().__init__()
        self.steps = list(steps)
        self.asked = asyncio.Event()

    async def get_command_status(self, execution_id: str) -> Any:
        self.status_calls += 1
        self.asked.set()
        return await self.steps.pop(0)()


async def running() -> Any:
    return SimpleNamespace(running=True, exit_code=None, error=None)


async def finished() -> Any:
    return SimpleNamespace(running=False, exit_code=0, error=None)


async def unanswered() -> Any:
    """A sandbox that does not answer, as one being snapshotted: the provider's status timeout ends the request."""
    await asyncio.Event().wait()


def make_parked_provider(monkeypatch: pytest.MonkeyPatch) -> opensandbox_provider.OpenSandboxProvider:
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
            "status_poll_timeout_s": 0.05,
            "status_recovery_timeout_s": 0.3,
            "retries": 0,
        },
    )


@pytest.mark.asyncio
async def test_a_parked_sandbox_is_not_asked_until_the_park_ends_and_then_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: asyncio.Queue[asyncio.Event] = asyncio.Queue()

    async def sleep(delay: float) -> None:
        release = asyncio.Event()
        sleeps.put_nowait(release)
        await release.wait()

    monkeypatch.setattr(asyncio, "sleep", sleep)
    provider = make_parked_provider(monkeypatch)
    commands = ScriptedStatus(running, finished)
    sandbox_handle = handle("parked", commands)
    task = asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=300))
    try:
        backoff = await asyncio.wait_for(sleeps.get(), 1)
        with provider.parked_exec_polling(sandbox_handle):
            # Every poll interval that passes while parked holds the poll again instead of asking.
            for _ in range(3):
                backoff.set()
                backoff = await asyncio.wait_for(sleeps.get(), 1)
            assert commands.status_calls == 1
        result = await asyncio.wait_for(task, 1)
        assert result.return_code == 0
        assert commands.status_calls == 2, "asked once, right after the park, without waiting out the interval"
        assert not backoff.is_set()
        assert provider._command_polls == {} and provider._parked_poll_scopes == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_status_timeout_during_a_park_does_not_fail_the_command(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider = make_parked_provider(monkeypatch)
    commands = ScriptedStatus(unanswered, finished)
    sandbox_handle = handle("snapshotted", commands)
    with caplog.at_level(logging.INFO, logger=opensandbox_provider.__name__):
        task = asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=300))
        try:
            await asyncio.wait_for(commands.asked.wait(), 1)
            # The park begins while the request is unanswered, as when a snapshot freezes the sandbox mid-poll.
            with provider.parked_exec_polling(sandbox_handle):
                await asyncio.sleep(0.3)  # well past the 0.05 s status timeout
                assert not task.done(), "the timed-out request was forgiven and the poll is held"
                assert commands.status_calls == 1
            result = await asyncio.wait_for(task, 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert result.return_code == 0
    assert commands.status_calls == 2
    [forgiven] = [record for record in caplog.records if "Forgave" in record.getMessage()]
    assert "'snapshotted'" in forgiven.getMessage()
    assert "Timed out during OpenSandbox command status" in forgiven.getMessage()


@pytest.mark.asyncio
async def test_a_status_request_that_outlasts_the_park_is_asked_again_with_a_fresh_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = make_parked_provider(monkeypatch)
    commands = ScriptedStatus(unanswered, finished)
    sandbox_handle = handle("slow-to-recover", commands)
    task = asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=300))
    try:
        await asyncio.wait_for(commands.asked.wait(), 1)
        # The park opens and closes while the request is still unanswered; it then times out after the park.
        with provider.parked_exec_polling(sandbox_handle):
            pass
        result = await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert result.return_code == 0
    assert commands.status_calls == 2


@pytest.mark.asyncio
async def test_a_status_timeout_without_a_park_still_fails_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_parked_provider(monkeypatch)
    commands = ScriptedStatus(unanswered)
    # Without a park the status-recovery window runs out, as for any sandbox that stops answering.
    with pytest.raises(NgFailureEvent) as failed:
        await asyncio.wait_for(provider.exec(handle("lost", commands), "work", timeout_s=300), HANG_GUARD_S)
    assert failed.value.failure_kind == PROVIDER_STATUS_UNAVAILABLE
    assert commands.status_calls == 1


@pytest.mark.asyncio
async def test_a_park_forgives_one_failed_request_and_a_sandbox_still_lost_afterwards_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = make_parked_provider(monkeypatch)
    commands = ScriptedStatus(unanswered, unanswered)
    sandbox_handle = handle("lost-after-park", commands)
    task = asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=300))
    try:
        await asyncio.wait_for(commands.asked.wait(), 1)
        with provider.parked_exec_polling(sandbox_handle):
            # Longer than the status-recovery window: the park, not the sandbox, kept the request unanswered.
            await asyncio.sleep(0.5)
        with pytest.raises(NgFailureEvent) as failed:
            await asyncio.wait_for(task, HANG_GUARD_S)
        assert failed.value.failure_kind == PROVIDER_STATUS_UNAVAILABLE
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert commands.status_calls == 2
    assert provider._command_polls == {} and provider._parked_poll_scopes == {}


@pytest.mark.asyncio
async def test_a_park_that_outlasts_the_commands_client_wait_fails_the_command_while_still_parked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A park holds the polls, not the command's time limit: the wait ends at its deadline, not when the park does."""
    provider = make_parked_provider(monkeypatch)
    object.__setattr__(provider._operations, "command_timeout_headroom_s", 0.1)
    commands = ScriptedStatus(running)
    sandbox_handle = handle("parked-too-long", commands)
    task = asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=0.2))
    try:
        await asyncio.wait_for(commands.asked.wait(), 1)
        with provider.parked_exec_polling(sandbox_handle):
            # The client wait is the 0.2 s timeout plus the 0.1 s headroom; the exec's hard cap is 30 s later.
            with pytest.raises(NgFailureEvent) as failed:
                await asyncio.wait_for(task, 5)
            assert failed.value.failure_kind == PROVIDER_COMMAND_OUTCOME_UNKNOWN
            assert "client wait expired" in str(failed.value)
            assert commands.status_calls == 1, "the parked sandbox was not asked again"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert provider._command_polls == {} and provider._parked_poll_scopes == {}


@pytest.mark.asyncio
async def test_a_command_without_a_timeout_waits_for_the_park_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_parked_provider(monkeypatch)
    commands = ScriptedStatus(running, finished)
    sandbox_handle = handle("parked-without-a-limit", commands)
    task = asyncio.create_task(provider.exec(sandbox_handle, "work"))
    try:
        await asyncio.wait_for(commands.asked.wait(), 1)
        with provider.parked_exec_polling(sandbox_handle):
            await asyncio.sleep(0.4)
            assert not task.done() and commands.status_calls == 1
        result = await asyncio.wait_for(task, 5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert result.return_code == 0 and commands.status_calls == 2


class ScriptedLogs(ScriptedStatus):
    """Status steps as ``ScriptedStatus``; each log request runs the next log step.

    ``log_asked`` is set once a log request has started, and a request that starts while the sandbox is parked is
    counted and never answered, as by a sandbox being snapshotted.
    """

    def __init__(self, provider: Any, sandbox_id: str, *, status: list[Any], logs: list[Any]) -> None:
        super().__init__(*status)
        self.provider = provider
        self.sandbox_id = sandbox_id
        self.log_steps = list(logs)
        self.log_calls = 0
        self.log_calls_while_parked = 0
        self.log_asked = asyncio.Event()

    async def get_background_command_logs(self, execution_id: str) -> Any:
        self.log_calls += 1
        self.log_asked.set()
        if self.sandbox_id in self.provider._parked_poll_scopes:
            self.log_calls_while_parked += 1
            await asyncio.Event().wait()
        return await self.log_steps.pop(0)()


async def output() -> Any:
    return SimpleNamespace(content="the command's output")


@pytest.mark.asyncio
async def test_the_logs_of_a_command_seen_finished_during_a_park_are_read_once_the_park_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = make_parked_provider(monkeypatch)
    # The status request below stays in flight across the start of the park, so it must not time out first.
    object.__setattr__(provider._operations, "status_poll_timeout_s", HANG_GUARD_S)
    answer = asyncio.Event()

    async def finished_with_exit_code_3() -> Any:
        await answer.wait()
        return SimpleNamespace(running=False, exit_code=3, error=None)

    commands = ScriptedLogs(provider, "parked", status=[finished_with_exit_code_3], logs=[output])
    sandbox_handle = handle("parked", commands)
    task = asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=300))
    try:
        await asyncio.wait_for(commands.asked.wait(), 1)
        with provider.parked_exec_polling(sandbox_handle):
            answer.set()
            # Long enough for the status answer to be handled and a log request, if any, to start.
            for _ in range(20):
                await asyncio.sleep(0)
            assert commands.log_calls == 0, "a parked sandbox is not asked for the logs"
            assert not task.done()
        result = await asyncio.wait_for(task, HANG_GUARD_S)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert (result.return_code, result.stdout) == (3, "the command's output")
    assert (commands.log_calls, commands.log_calls_while_parked) == (1, 0)
    assert provider._command_polls == {} and provider._parked_poll_scopes == {}


@pytest.mark.asyncio
async def test_the_logs_are_read_at_the_commands_deadline_when_the_park_outlasts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The log read waits for the park only as long as the command's client wait: then it is tried anyway."""
    provider = make_parked_provider(monkeypatch)
    object.__setattr__(provider._operations, "command_timeout_headroom_s", 0.1)
    object.__setattr__(provider._connection, "request_timeout_s", 0.05)
    # The status request below stays in flight across the start of the park, so it must not time out first.
    object.__setattr__(provider._operations, "status_poll_timeout_s", HANG_GUARD_S)
    answer = asyncio.Event()

    async def finished_once_parked() -> Any:
        await answer.wait()
        return SimpleNamespace(running=False, exit_code=0, error=None)

    commands = ScriptedLogs(provider, "parked-too-long", status=[finished_once_parked], logs=[output])
    sandbox_handle = handle("parked-too-long", commands)
    task = asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=0.2))
    try:
        await asyncio.wait_for(commands.asked.wait(), 1)
        with provider.parked_exec_polling(sandbox_handle):
            answer.set()
            with pytest.raises(NgFailureEvent) as failed:
                await asyncio.wait_for(task, 5)
            # The parked sandbox did not answer the read, which fails the command as an unanswered read does.
            assert failed.value.failure_kind == PROVIDER_IO_TIMEOUT
            assert (commands.log_calls, commands.log_calls_while_parked) == (1, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_log_timeout_without_a_park_still_fails_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_parked_provider(monkeypatch)
    # The log read uses the connection's request timeout; the fixture's 5 s would only slow the test.
    object.__setattr__(provider._connection, "request_timeout_s", 0.05)
    commands = ScriptedLogs(provider, "lost", status=[finished], logs=[unanswered])
    with pytest.raises(NgFailureEvent) as failed:
        await asyncio.wait_for(provider.exec(handle("lost", commands), "work", timeout_s=300), HANG_GUARD_S)
    assert failed.value.failure_kind == PROVIDER_IO_TIMEOUT
    assert commands.log_calls == 1, "retries is 0 and no park forgives the read"


@pytest.mark.asyncio
async def test_a_checkpoint_scope_opened_during_a_status_request_is_not_slept_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "sleep", sleep)
    provider = make_provider(monkeypatch)
    answer = asyncio.Event()

    async def answered_late() -> Any:
        await answer.wait()
        return SimpleNamespace(running=True, exit_code=None, error=None)

    commands = ScriptedStatus(answered_late, finished)
    sandbox_handle = handle("mid-request", commands)
    task = asyncio.create_task(provider.exec(sandbox_handle, "work", timeout_s=300))
    try:
        await asyncio.wait_for(commands.asked.wait(), 1)
        # The scope opens while the first status request is in flight: no poll is waiting to be woken yet.
        with provider.checkpoint_exec_polling(sandbox_handle):
            answer.set()
            result = await asyncio.wait_for(task, 1)
        assert result.return_code == 0
        assert commands.status_calls == 2
        assert sleeps == [], "the wakeup delivered before the wait replaced the 30-second backoff"
        assert provider._command_polls == {} and provider._checkpoint_poll_scopes == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_a_sandbox_delegates_its_parked_scope_to_a_provider_that_has_one(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_parked_provider(monkeypatch)
    sandbox = AsyncSandbox(provider=provider)
    sandbox._handle = handle("parked", Commands())
    sandbox._stopped = False

    with sandbox.parked_exec_polling():
        with sandbox.parked_exec_polling():
            assert provider._parked_poll_scopes == {"parked": 2}
        assert provider._parked_poll_scopes == {"parked": 1}
    assert provider._parked_poll_scopes == {}


def test_provider_without_parked_polling_gets_a_no_op_scope() -> None:
    sandbox = AsyncSandbox(provider=SimpleNamespace())
    sandbox._handle = handle("plain", Commands())
    sandbox._stopped = False
    assert isinstance(sandbox.parked_exec_polling(), nullcontext)
