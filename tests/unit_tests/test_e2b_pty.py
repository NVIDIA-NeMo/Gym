# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""Unit tests for the e2b PTY session (SDK faked; no network)."""

import asyncio
from dataclasses import dataclass

import pytest

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxPtyError, SandboxPtySpec, SupportsSandboxPty
from nemo_gym.sandbox.providers.e2b import provider as e2b_provider
from nemo_gym.sandbox.providers.e2b.provider import E2BProvider


@dataclass
class FakePtySize:
    rows: int
    cols: int


class FakeExit(Exception):
    def __init__(self, exit_code: int) -> None:
        super().__init__(f"exit {exit_code}")
        self.exit_code = exit_code


class FakeResult:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code


class FakePtyHandle:
    pid = 4242

    def __init__(self) -> None:
        self.exited = asyncio.Event()
        self.exit_code = 0

    async def wait(self) -> FakeResult:
        await self.exited.wait()
        if self.exit_code:
            raise FakeExit(self.exit_code)
        return FakeResult(0)


class FakePty:
    def __init__(self) -> None:
        self.handle = FakePtyHandle()
        self.on_data = None
        self.create_kwargs: dict = {}
        self.stdin: list[bytes] = []
        self.sizes: list[FakePtySize] = []
        self.killed = 0

    async def create(self, size, on_data, **kwargs):
        self.create_kwargs = {"size": size, **kwargs}
        self.on_data = on_data
        return self.handle

    async def send_stdin(self, pid: int, data: bytes) -> None:
        assert pid == self.handle.pid
        self.stdin.append(data)

    async def resize(self, pid: int, size) -> None:
        self.sizes.append(size)

    async def kill(self, pid: int) -> bool:
        self.killed += 1
        self.handle.exited.set()
        return True


class FakeSandbox:
    sandbox_id = "sbx-1"

    def __init__(self) -> None:
        self.pty = FakePty()


@pytest.fixture
def provider_and_handle(monkeypatch: pytest.MonkeyPatch):
    sdk = type("SDK", (), {"PtySize": FakePtySize})
    monkeypatch.setattr(e2b_provider, "_require_e2b_sdk", lambda: sdk)
    sandbox = FakeSandbox()
    return E2BProvider(), SandboxHandle(sandbox_id="sbx-1", provider_name="e2b", raw=sandbox), sandbox


def test_provider_advertises_pty_support() -> None:
    assert isinstance(E2BProvider(), SupportsSandboxPty)


async def test_session_streams_output_and_input(provider_and_handle) -> None:
    provider, handle, sandbox = provider_and_handle
    session = await provider.create_pty(handle, SandboxPtySpec(rows=30, cols=100, cwd="/work", env={"A": "1"}))

    assert session.mode == "pty"
    kwargs = sandbox.pty.create_kwargs
    assert (kwargs["size"].rows, kwargs["size"].cols) == (30, 100)
    assert kwargs["cwd"] == "/work"
    assert kwargs["envs"] == {"A": "1"}
    # The SDK's default 60 s bound on the running stream would kill long sessions.
    assert kwargs["timeout"] == 0

    sandbox.pty.on_data(b"hello\r\n")
    assert await session.read(timeout_s=1) == b"hello\r\n"

    await session.write(b"ls\n")
    assert sandbox.pty.stdin == [b"ls\n"]

    await session.resize(50, 132)
    await session.resize(50, 132)  # unchanged size is not re-sent
    assert [(s.rows, s.cols) for s in sandbox.pty.sizes] == [(50, 132)]

    sandbox.pty.handle.exit_code = 7
    sandbox.pty.handle.exited.set()
    assert await session.wait_exit(timeout_s=1) == 7
    assert await session.read(timeout_s=1) == b""
    assert [chunk async for chunk in session] == []
    await session.close()


async def test_clean_exit_reports_zero(provider_and_handle) -> None:
    provider, handle, sandbox = provider_and_handle
    session = await provider.create_pty(handle, SandboxPtySpec())
    sandbox.pty.handle.exited.set()
    assert await session.wait_exit(timeout_s=1) == 0
    await session.close()


async def test_command_runs_as_one_quoted_unit(provider_and_handle) -> None:
    provider, handle, sandbox = provider_and_handle
    session = await provider.create_pty(handle, SandboxPtySpec(command="echo it's; exit 3"))
    # `exec a; b` alone would replace the shell with `a` and never run `b`.
    assert sandbox.pty.stdin == [b"exec bash -c 'echo it'\"'\"'s; exit 3'\n"]
    await session.close()


async def test_signals(provider_and_handle) -> None:
    provider, handle, sandbox = provider_and_handle
    session = await provider.create_pty(handle, SandboxPtySpec())

    await session.send_signal("SIGINT")
    assert sandbox.pty.stdin == [b"\x03"]
    with pytest.raises(NotImplementedError):
        await session.send_signal("SIGHUP")
    await session.send_signal("SIGKILL")
    assert sandbox.pty.killed == 1
    await session.close()


async def test_close_kills_a_live_session_and_blocks_further_use(provider_and_handle) -> None:
    provider, handle, sandbox = provider_and_handle
    session = await provider.create_pty(handle, SandboxPtySpec())
    await session.close()
    await session.close()  # idempotent

    assert session.closed
    assert sandbox.pty.killed == 1
    with pytest.raises(SandboxPtyError):
        await session.write(b"x")
    with pytest.raises(SandboxPtyError):
        await session.resize(1, 1)


async def test_pipe_mode_is_rejected(provider_and_handle) -> None:
    provider, handle, _ = provider_and_handle
    with pytest.raises(ValueError, match="pipe mode"):
        await provider.create_pty(handle, SandboxPtySpec(pty=False))


async def test_detached_runs_are_unsupported(provider_and_handle) -> None:
    provider, handle, _ = provider_and_handle
    session = await provider.create_pty(handle, SandboxPtySpec())
    with pytest.raises(NotImplementedError):
        await session.run_detached("true")
    await session.close()
