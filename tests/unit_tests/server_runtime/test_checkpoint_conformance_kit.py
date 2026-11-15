# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The local provider and helpers of the harness checkpoint conformance kit."""

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.sandbox.providers.base import SandboxSpec, SandboxStatus
from nemo_gym.server_runtime.testing.checkpoint_conformance import LocalExecProvider, _kill, _process_state, _until


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None, reason="Linux /proc and bash are required"
)


def _gone(pid: int) -> bool:
    return _process_state(pid) in (None, "Z", "X")


async def _wait_gone(pid: int) -> None:
    async with asyncio.timeout(10):
        while not _gone(pid):
            await asyncio.sleep(0.02)


async def test_the_local_provider_runs_commands_and_copies_files(tmp_path: Path) -> None:
    provider = LocalExecProvider()
    handle = await provider.create(MagicMock(spec=SandboxSpec))
    result = await provider.exec(
        handle, 'printf %s "$GREETING"; echo problem >&2; exit 3', cwd=str(tmp_path), env={"GREETING": "hi"}
    )
    # Not equality on stderr: some images print shell start-up noise there.
    assert (result.stdout, result.return_code, result.error_type) == ("hi", 3, None) and "problem" in result.stderr
    assert provider.commands == ['printf %s "$GREETING"; echo problem >&2; exit 3']

    source = tmp_path / "source.txt"
    source.write_text("payload")
    await provider.upload_file(handle, source, str(tmp_path / "new" / "nested" / "target.txt"))
    await provider.download_file(handle, str(tmp_path / "new" / "nested" / "target.txt"), tmp_path / "back.txt")
    assert (tmp_path / "back.txt").read_text() == "payload"

    assert await provider.status(handle) is SandboxStatus.RUNNING
    connected = await provider.connect(await provider.serialize_handle(handle))
    assert (connected.sandbox_id, connected.provider_name) == (handle.sandbox_id, "local-exec")
    await provider.close(handle)
    await provider.aclose()


async def test_a_command_past_its_timeout_is_killed_with_its_children(tmp_path: Path) -> None:
    provider = LocalExecProvider()
    handle = await provider.create(MagicMock(spec=SandboxSpec))
    result = await provider.exec(handle, "sleep 60 & echo $! > child.pid; wait", cwd=str(tmp_path), timeout_s=1)
    assert (result.return_code, result.error_type, result.stdout) == (-1, "timeout", None)
    await _wait_gone(int((tmp_path / "child.pid").read_text()))


async def test_a_cancelled_command_leaves_nothing_running(tmp_path: Path) -> None:
    provider = LocalExecProvider()
    handle = await provider.create(MagicMock(spec=SandboxSpec))
    running = asyncio.create_task(
        provider.exec(handle, "sleep 60 & echo $! > child.pid; wait", cwd=str(tmp_path), timeout_s=60)
    )
    async with asyncio.timeout(10):
        while not (tmp_path / "child.pid").exists() or not (tmp_path / "child.pid").read_text().strip():
            await asyncio.sleep(0.02)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 10)
    await _wait_gone(int((tmp_path / "child.pid").read_text()))


async def test_until_polls_plain_and_awaitable_conditions_and_fails_at_its_deadline() -> None:
    answers = iter([False, True])
    await _until(lambda: next(answers))

    async def ready() -> bool:
        return True

    await _until(ready)
    started = time.monotonic()
    with pytest.raises(AssertionError, match="not reached in time"):
        await _until(lambda: False, timeout=0.1)
    assert time.monotonic() - started < 5


def test_process_state_and_kill() -> None:
    process = subprocess.Popen(["sleep", "60"])
    try:
        assert _process_state(process.pid) in ("S", "R")
        _kill(process.pid)
        process.wait(timeout=10)
        assert _process_state(process.pid) is None
        _kill(process.pid)  # already gone: no error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
