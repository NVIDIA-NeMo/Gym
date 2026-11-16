# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import contextlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox import shell


pytestmark = pytest.mark.sandbox


class FakeSession:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.resizes: list[tuple[int, int]] = []
        self.closed = False
        self._eof = asyncio.Event()

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.closed = True

    async def write(self, data: bytes) -> None:
        self.writes.append(data)
        if data == b"\x04":
            self._eof.set()

    async def resize(self, rows: int, cols: int) -> None:
        self.resizes.append((rows, cols))

    async def wait_exit(self) -> int:
        await self._eof.wait()
        return 7

    async def _output(self):
        yield b"ready\r\n"
        await self._eof.wait()

    def __aiter__(self):
        return self._output()


def test_parse_and_load_provider_config(tmp_path: Path) -> None:
    config = tmp_path / "sandbox.yaml"
    config.write_text("sandbox:\n  local:\n    workspace_root: /tmp/workspace\n  default_metadata:\n    owner: test\n")

    args = shell._parse_args(["--config", str(config)])
    provider, metadata = shell._load_provider_config(args.config)

    assert args.image == shell.DEFAULT_IMAGE
    assert provider == {"local": {"workspace_root": "/tmp/workspace"}}
    assert metadata == {"owner": "test"}

    with pytest.raises(SystemExit):
        shell._parse_args(["--config", str(config), "--image", " "])

    config.write_text("- local\n")
    with pytest.raises(ValueError, match="must contain a YAML object"):
        shell._load_provider_config(config)


def test_terminal_io_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    attributes = ["original"]
    monkeypatch.setattr(shell.termios, "tcgetattr", lambda fd: attributes)
    monkeypatch.setattr(shell.tty, "setraw", lambda fd: calls.append(("raw", fd)))
    monkeypatch.setattr(shell.termios, "tcsetattr", lambda *args: calls.append(args))
    monkeypatch.setattr(shell.os, "get_terminal_size", lambda fd: os.terminal_size((120, 40)))

    with shell._raw_terminal(4):
        assert shell._terminal_size(4) == (40, 120)

    assert calls == [("raw", 4), (4, shell.termios.TCSADRAIN, attributes)]

    writes = iter([2, 3])
    chunks: list[bytes] = []

    def write(_fd: int, data: memoryview) -> int:
        chunks.append(bytes(data))
        return next(writes)

    monkeypatch.setattr(shell.os, "write", write)
    shell._write_all(5, b"hello")
    assert chunks == [b"hello", b"llo"]


async def test_read_and_bridge_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"input")
        assert await shell._read_fd(read_fd, 5) == b"input"
    finally:
        os.close(read_fd)
        os.close(write_fd)

    reads = iter([b"typed", b""])
    output: list[tuple[int, bytes]] = []
    session = FakeSession()
    monkeypatch.setattr(shell, "_read_fd", lambda _fd: anext(_async_iter(reads)))
    monkeypatch.setattr(shell, "_write_all", lambda fd, data: output.append((fd, data)))

    assert await shell._bridge_terminal(session, 0, 1) == 7
    assert session.writes == [b"typed", b"\x04"]
    assert output == [(1, b"ready\r\n")]


async def test_bridge_handles_remote_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def blocked_read(_fd: int) -> bytes:
        await asyncio.sleep(3600)
        return b""

    class ExitFirstSession(FakeSession):
        async def wait_exit(self) -> int:
            return 4

        async def _output(self):
            await asyncio.sleep(0)
            yield b"done"

    class OutputFirstSession(FakeSession):
        async def wait_exit(self) -> int:
            await asyncio.sleep(0.01)
            return 3

        async def _output(self):
            if False:
                yield b""

    monkeypatch.setattr(shell, "_read_fd", blocked_read)
    monkeypatch.setattr(shell, "_write_all", lambda _fd, _data: None)

    assert await shell._bridge_terminal(ExitFirstSession(), 0, 1) == 4
    assert await shell._bridge_terminal(OutputFirstSession(), 0, 1) == 3


async def _async_iter(values: Any):
    for value in values:
        yield value


def test_run_shell_orchestrates_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession()
    captured: dict[str, Any] = {}

    class FakePty:
        async def create(self, *, rows: int, cols: int) -> FakeSession:
            captured["size"] = (rows, cols)
            return session

    class FakeSandbox:
        def __init__(self, provider: dict[str, Any], spec: object) -> None:
            captured["provider"] = provider
            captured["spec"] = spec
            self.pty = FakePty()

        async def __aenter__(self) -> "FakeSandbox":
            return self

        async def __aexit__(self, *_args: object) -> None:
            captured["sandbox_closed"] = True

        async def start(self) -> None:
            captured["started"] = True

    class FakeLoop:
        def add_signal_handler(self, _signal: int, callback: Any) -> None:
            callback()

        def remove_signal_handler(self, removed_signal: int) -> None:
            captured["removed_signal"] = removed_signal

    sizes = iter([(24, 80), (30, 100)])

    async def bridge(received: FakeSession, stdin_fd: int, stdout_fd: int) -> int:
        captured["bridge"] = (received, stdin_fd, stdout_fd)
        await asyncio.sleep(0)
        return 9

    monkeypatch.setattr(shell, "_load_provider_config", lambda _path: ({"local": {}}, {"owner": "test"}))
    monkeypatch.setattr(shell, "AsyncSandbox", FakeSandbox)
    monkeypatch.setattr(shell, "_terminal_size", lambda _fd: next(sizes))
    monkeypatch.setattr(shell, "_raw_terminal", lambda _fd: contextlib.nullcontext())
    monkeypatch.setattr(shell, "_bridge_terminal", bridge)
    monkeypatch.setattr(shell.asyncio, "get_running_loop", lambda: FakeLoop())
    stdout = shell.sys.stdout
    monkeypatch.setattr(shell.sys, "stdin", SimpleNamespace(fileno=lambda: 10))
    monkeypatch.setattr(
        shell.sys,
        "stdout",
        SimpleNamespace(
            fileno=lambda: 11,
            write=stdout.write,
            flush=stdout.flush,
        ),
    )

    result = asyncio.run(shell._run_shell(argparse.Namespace(config=Path("config.yaml"), image="image:test")))

    assert result == 9
    assert captured["provider"] == {"local": {}}
    assert captured["size"] == (24, 80)
    assert captured["bridge"] == (session, 10, 11)
    assert captured["started"] and captured["sandbox_closed"]
    assert captured["spec"].metadata == {"owner": "test", "purpose": "interactive-shell"}
    assert session.resizes == [(30, 100)]
    assert session.closed


@pytest.mark.parametrize(
    ("is_tty", "outcome", "expected"),
    [(False, 0, 2), (True, 5, 5), (True, KeyboardInterrupt(), 130), (True, RuntimeError("failed"), 1)],
)
def test_main_exit_codes(
    monkeypatch: pytest.MonkeyPatch, is_tty: bool, outcome: int | BaseException, expected: int
) -> None:
    monkeypatch.setattr(shell.sys.stdin, "isatty", lambda: is_tty)
    monkeypatch.setattr(shell.sys.stdout, "isatty", lambda: is_tty)

    async def run(_args: argparse.Namespace) -> int:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(shell, "_run_shell", run)
    assert shell.main(["--config", "config.yaml"]) == expected
