# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A sandbox's releases (stop, disconnect, a one-shot terminal session) finish under the request's cancellation and
are bounded in time.

A client that disconnects cancels its request's anyio cancel scope, and anyio then cancels every later await in it, so
an unshielded release in the request's cleanup would raise before the provider let go of the sandbox.
"""

import asyncio
import gc
from pathlib import Path
from typing import Any

import anyio
import pytest

from nemo_gym.sandbox import api as sandbox_api
from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus


class _SlowProvider:
    """A provider whose close and client release have to await, as real HTTP providers do."""

    name = "slow"

    def __init__(
        self, *, close_takes_s: float = 0.01, close_error: Exception | None = None, aclose_takes_s: float = 0.01
    ) -> None:
        self.close_takes_s = close_takes_s
        self.aclose_takes_s = aclose_takes_s
        self.close_error = close_error
        self.closed: list[str] = []
        self.client_closed = False
        self.sessions: list["_Session"] = []
        self.session_close_takes_s = 0.01
        self.uploading = asyncio.Event()

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return SandboxHandle(sandbox_id="sb-1", provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self.uploading.set()
        await asyncio.Event().wait()

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        pass

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        await anyio.sleep(self.close_takes_s)
        if self.close_error is not None:
            raise self.close_error
        self.closed.append(handle.sandbox_id)

    async def aclose(self) -> None:
        await anyio.sleep(self.aclose_takes_s)
        self.client_closed = True

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: dict[str, Any]) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw=None)

    async def create_pty(self, handle: SandboxHandle, spec: Any) -> "_Session":
        session = _Session(close_takes_s=self.session_close_takes_s)
        self.sessions.append(session)
        return session


class _Session:
    """A terminal session whose command never ends and whose close has to await."""

    def __init__(self, *, close_takes_s: float) -> None:
        self.close_takes_s = close_takes_s
        self.closed = False
        self.reading = asyncio.Event()

    async def read(self) -> bytes:
        self.reading.set()
        await asyncio.Event().wait()
        return b""

    async def read_stderr(self) -> bytes:
        await asyncio.Event().wait()
        return b""

    async def wait_exit(self) -> int:
        await asyncio.Event().wait()
        return 0

    async def close(self) -> None:
        await anyio.sleep(self.close_takes_s)
        self.closed = True


async def _started(provider: _SlowProvider) -> AsyncSandbox:
    sandbox = AsyncSandbox(provider, owns_provider=True)
    await sandbox.start(SandboxSpec(image="image"))
    return sandbox


async def test_a_stop_in_a_cancelled_request_still_stops_the_sandbox() -> None:
    provider = _SlowProvider()
    sandbox = await _started(provider)

    with anyio.fail_after(5):
        with anyio.CancelScope() as request_scope:
            request_scope.cancel()
            await sandbox.stop()
            await anyio.sleep(0)
            pytest.fail("the request's cancellation did not reach the first await after the stop")

    assert provider.closed == ["sb-1"]
    assert provider.client_closed
    assert request_scope.cancelled_caught


async def test_a_stop_that_runs_past_its_time_limit_raises_and_can_be_retried() -> None:
    # Finite and short, so a stop that is not bounded finishes late and fails the test instead of hanging it.
    provider = _SlowProvider(close_takes_s=2)
    sandbox = await _started(provider)

    with pytest.raises(TimeoutError):
        await sandbox.stop(timeout_s=0.05)

    assert provider.closed == []
    provider.close_takes_s = 0.01
    await sandbox.stop()
    assert provider.closed == ["sb-1"]


async def test_no_time_limit_lets_the_stop_finish() -> None:
    provider = _SlowProvider(close_takes_s=0.05)
    sandbox = await _started(provider)

    await sandbox.stop(timeout_s=None)

    assert provider.closed == ["sb-1"]


async def test_a_stop_that_fails_raises_its_error() -> None:
    sandbox = await _started(_SlowProvider(close_error=RuntimeError("provider refused the stop")))

    with pytest.raises(RuntimeError, match="provider refused the stop"):
        await sandbox.stop()


async def test_a_start_cancelled_during_its_upload_stops_the_sandbox() -> None:
    provider = _SlowProvider()
    sandbox = AsyncSandbox(provider, owns_provider=False)

    async with anyio.create_task_group() as request:
        request.start_soon(sandbox.start, SandboxSpec(image="image", files={"/task": "x"}))
        await asyncio.wait_for(provider.uploading.wait(), timeout=5)
        request.cancel_scope.cancel()

    assert provider.closed == ["sb-1"]


async def test_a_disconnect_in_a_cancelled_request_still_releases_the_client() -> None:
    provider = _SlowProvider()
    borrowed = await AsyncSandbox.connect({"sandbox_id": "sb-1"}, provider=provider, owns_provider=True)

    with anyio.fail_after(5):
        with anyio.CancelScope() as request_scope:
            request_scope.cancel()
            await borrowed.disconnect()

    assert provider.client_closed
    assert provider.closed == [], "a disconnect never stops the sandbox"


async def test_a_disconnect_that_runs_past_its_time_limit_raises() -> None:
    # Finite and short, so a disconnect that is not bounded finishes late and fails the test instead of hanging it.
    provider = _SlowProvider(aclose_takes_s=2)
    borrowed = await AsyncSandbox.connect({"sandbox_id": "sb-1"}, provider=provider, owns_provider=True)

    with pytest.raises(TimeoutError):
        await borrowed.disconnect(timeout_s=0.05)

    assert not provider.client_closed


async def test_a_one_shot_terminal_session_close_that_hangs_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_api, "STOP_TIMEOUT_S", 0.05)
    provider = _SlowProvider()
    provider.session_close_takes_s = 2
    sandbox = await _started(provider)
    started = asyncio.get_running_loop().time()

    with anyio.CancelScope() as request_scope:

        async def cancel_once_running() -> None:
            await _until(lambda: provider.sessions and provider.sessions[0].reading.is_set())
            request_scope.cancel()

        canceller = asyncio.create_task(cancel_once_running())
        with pytest.raises(TimeoutError):
            await sandbox.pty.exec("sleep infinity", timeout_s=60)
    await canceller

    assert asyncio.get_running_loop().time() - started < 1
    (session,) = provider.sessions
    assert not session.closed


async def test_a_one_shot_terminal_session_is_closed_when_its_request_is_cancelled() -> None:
    provider = _SlowProvider()
    sandbox = await _started(provider)

    async with anyio.create_task_group() as request:
        request.start_soon(lambda: sandbox.pty.exec("sleep infinity", env={"A": "1"}, timeout_s=60))
        await asyncio.wait_for(_until(lambda: provider.sessions and provider.sessions[0].reading.is_set()), timeout=5)
        request.cancel_scope.cancel()

    (session,) = provider.sessions
    assert session.closed


async def test_a_compose_group_stop_is_bounded_and_a_retry_joins_it() -> None:
    group = AsyncSandboxCompose(object(), None)
    finish = asyncio.Event()
    stops = 0

    async def slow_stop() -> None:
        nonlocal stops
        stops += 1
        await finish.wait()

    group._stop = slow_stop  # type: ignore[method-assign]
    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        # The outer wait turns a stop that ignores its limit into a failure here instead of a hung test.
        await asyncio.wait_for(group.stop(timeout_s=0.05), timeout=2)
    assert asyncio.get_running_loop().time() - started < 1, "the group stop gave up at its own limit"

    finish.set()
    await group.stop()
    assert stops == 1, "the retry waits for the stop already running instead of starting another"


async def test_a_compose_group_stop_in_a_cancelled_request_waits_for_the_group_to_stop() -> None:
    group = AsyncSandboxCompose(object(), None)
    stopped = asyncio.Event()

    async def slow_stop() -> None:
        await asyncio.sleep(0.05)
        stopped.set()

    group._stop = slow_stop  # type: ignore[method-assign]
    with anyio.fail_after(5):
        with anyio.CancelScope() as request_scope:
            request_scope.cancel()
            await group.stop()

    assert stopped.is_set(), "the cancelled request did not return before the group stopped"


async def test_a_compose_group_stop_the_caller_gave_up_on_fails_quietly_and_a_retry_runs_again() -> None:
    group = AsyncSandboxCompose(object(), None)
    finish = asyncio.Event()
    stops = 0
    unretrieved: list[dict[str, Any]] = []
    asyncio.get_running_loop().set_exception_handler(lambda _, context: unretrieved.append(context))

    async def failing_stop() -> None:
        nonlocal stops
        stops += 1
        if stops == 1:
            await finish.wait()
            raise RuntimeError("provider refused")

    group._stop = failing_stop  # type: ignore[method-assign]
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(group.stop(timeout_s=0.05), timeout=5)
    finish.set()
    await _until(lambda: group._stop_task.done())
    await group.stop()
    gc.collect()

    assert stops == 2, "a stop that failed in the background is started again"
    assert unretrieved == [], "the background failure is retrieved, not reported as never retrieved"


async def _until(condition: Any) -> None:
    while not condition():
        await asyncio.sleep(0.001)
