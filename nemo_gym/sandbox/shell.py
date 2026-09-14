# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# limitations under the License.

"""Open an interactive shell in an ephemeral NeMo Gym sandbox."""

import argparse
import asyncio
import contextlib
import os
import signal
import sys
import termios
import tty
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxPtySession,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)


DEFAULT_IMAGE = "docker.io/library/python:3.12-slim"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        metavar="PATH",
        help="YAML file containing a named sandbox provider block.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Sandbox image (default: {DEFAULT_IMAGE}). Image syntax is provider-specific.",
    )
    args = parser.parse_args(argv)
    if not args.image.strip():
        parser.error("--image must not be empty")
    return args


def _load_provider_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = OmegaConf.load(config_path)
    resolved = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(resolved, Mapping):
        raise ValueError(f"Sandbox config {config_path} must contain a YAML object")
    return (
        resolve_provider_config("sandbox", resolved),
        resolve_provider_metadata("sandbox", resolved),
    )


def _terminal_size(fd: int) -> tuple[int, int]:
    size = os.get_terminal_size(fd)
    return size.lines, size.columns


@contextlib.contextmanager
def _raw_terminal(fd: int) -> Iterator[None]:
    original = termios.tcgetattr(fd)
    tty.setraw(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


async def _read_fd(fd: int, size: int = 65536) -> bytes:
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[bytes] = loop.create_future()

    def on_ready() -> None:
        loop.remove_reader(fd)
        try:
            ready.set_result(os.read(fd, size))
        except BaseException as error:
            ready.set_exception(error)

    loop.add_reader(fd, on_ready)
    try:
        return await ready
    finally:
        loop.remove_reader(fd)


def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        remaining = remaining[os.write(fd, remaining) :]


async def _pump_stdin(session: SandboxPtySession, fd: int) -> None:
    while data := await _read_fd(fd):
        await session.write(data)


async def _pump_stdout(session: SandboxPtySession, fd: int) -> None:
    async for chunk in session:
        _write_all(fd, chunk)


async def _bridge_terminal(session: SandboxPtySession, stdin_fd: int, stdout_fd: int) -> int:
    stdin_task = asyncio.create_task(_pump_stdin(session, stdin_fd))
    stdout_task = asyncio.create_task(_pump_stdout(session, stdout_fd))
    exit_task = asyncio.create_task(session.wait_exit())
    tasks = {stdin_task, stdout_task, exit_task}
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if exit_task in done:
            return_code = exit_task.result()
            await stdout_task
            return return_code
        if stdout_task in done:
            stdout_task.result()
            return await exit_task

        # A closed stdin should act like terminal EOF so the remote shell can
        # exit normally instead of leaving the sandbox alive until its TTL.
        stdin_task.result()
        await session.write(b"\x04")
        return_code = await exit_task
        await stdout_task
        return return_code
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_shell(args: argparse.Namespace) -> int:
    provider_config, provider_metadata = _load_provider_config(args.config)
    spec = SandboxSpec(
        image=args.image,
        ttl_s=1800,
        ready_timeout_s=1200,
        workdir="/tmp",
        metadata={**provider_metadata, "purpose": "interactive-shell"},
    )

    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()
    rows, cols = _terminal_size(stdin_fd)

    print(f"Creating ephemeral sandbox from {args.image!r}...", flush=True)
    async with AsyncSandbox(provider_config, spec) as sandbox:
        await sandbox.start()
        print("Sandbox ready. Exit the remote shell to destroy it.", flush=True)
        try:
            session = await sandbox.pty.create(rows=rows, cols=cols)
        except NotImplementedError as error:
            raise RuntimeError(
                f"Configured sandbox provider does not support interactive PTY sessions: {error}"
            ) from error

        loop = asyncio.get_running_loop()
        resize_tasks: set[asyncio.Task[None]] = set()

        def resize() -> None:
            current_rows, current_cols = _terminal_size(stdin_fd)
            task = asyncio.create_task(session.resize(current_rows, current_cols))
            resize_tasks.add(task)
            task.add_done_callback(resize_tasks.discard)

        loop.add_signal_handler(signal.SIGWINCH, resize)
        try:
            async with session:
                with _raw_terminal(stdin_fd):
                    return await _bridge_terminal(session, stdin_fd, stdout_fd)
        finally:
            loop.remove_signal_handler(signal.SIGWINCH)
            if resize_tasks:
                await asyncio.gather(*resize_tasks, return_exceptions=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("sandbox shell requires an interactive stdin and stdout", file=sys.stderr)
        return 2
    try:
        return asyncio.run(_run_shell(args))
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"sandbox shell failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
