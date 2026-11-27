# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The local sandbox provider and the process helper of the harness checkpoint conformance kit."""

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.agents.testing.checkpoint_conformance import HarnessFixture, process_state
from nemo_gym.sandbox.providers.base import SandboxSpec, SandboxStatus


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None, reason="Linux /proc and bash are required"
)


def _gone(pid: int) -> bool:
    return process_state(pid) in (None, "Z", "X")


async def _wait_gone(pid: int) -> None:
    async with asyncio.timeout(10):
        while not _gone(pid):
            await asyncio.sleep(0.02)


async def _wait_for_pid(path: Path) -> int:
    async with asyncio.timeout(10):
        while not path.exists() or not path.read_text().strip():
            await asyncio.sleep(0.02)
    return int(path.read_text())


async def test_the_local_provider_runs_commands_and_copies_files(tmp_path: Path) -> None:
    provider = HarnessFixture().provider
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


def test_a_fixture_keeps_one_provider_and_checks_no_conversation_by_default() -> None:
    fixture = HarnessFixture()
    assert fixture.provider is fixture.provider
    assert HarnessFixture().provider is not fixture.provider
    # An adapter that cannot tell whether recovery continued its conversation is checked for the relaunch only.
    assert fixture.crash_recovery_assertion() is None


async def test_a_command_past_its_timeout_is_killed_with_its_children(tmp_path: Path) -> None:
    provider = HarnessFixture().provider
    handle = await provider.create(MagicMock(spec=SandboxSpec))
    result = await provider.exec(handle, "sleep 60 & echo $! > child.pid; wait", cwd=str(tmp_path), timeout_s=1)
    assert (result.return_code, result.error_type, result.stdout) == (-1, "timeout", None)
    await _wait_gone(int((tmp_path / "child.pid").read_text()))


async def test_a_cancelled_command_leaves_nothing_running(tmp_path: Path) -> None:
    provider = HarnessFixture().provider
    handle = await provider.create(MagicMock(spec=SandboxSpec))
    running = asyncio.create_task(
        provider.exec(handle, "sleep 60 & echo $! > child.pid; wait", cwd=str(tmp_path), timeout_s=60)
    )
    child = await _wait_for_pid(tmp_path / "child.pid")
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 10)
    await _wait_gone(child)


async def test_the_provider_finds_the_processes_its_commands_left_and_no_others(tmp_path: Path) -> None:
    provider = HarnessFixture().provider
    other = HarnessFixture().provider
    handle = await provider.create(MagicMock(spec=SandboxSpec))
    # The child outlives the command that started it, as a harness outlives the command that asked for its launch.
    result = await provider.exec(
        handle, "sleep 60 >/dev/null 2>&1 & echo $! > child.pid", cwd=str(tmp_path), timeout_s=30
    )
    assert result.return_code == 0, result.stderr
    child = await _wait_for_pid(tmp_path / "child.pid")
    try:
        assert provider.processes() == [child]
        assert other.processes() == []
        # A stopped process still counts, and a kill ends it.
        os.kill(child, signal.SIGSTOP)
        async with asyncio.timeout(10):
            while process_state(child) != "T":
                await asyncio.sleep(0.02)
        assert provider.processes() == [child]
        os.kill(child, signal.SIGKILL)
        await _wait_gone(child)
        assert provider.processes() == []
    finally:
        if not _gone(child):
            os.kill(child, signal.SIGKILL)


async def test_handles_made_before_a_disconnect_refuse_and_later_ones_work(tmp_path: Path) -> None:
    provider = HarnessFixture().provider
    created = await provider.create(MagicMock(spec=SandboxSpec))
    connected = await provider.connect({"sandbox_id": "local"})
    source = tmp_path / "source.txt"
    source.write_text("payload")

    provider.disconnect_handles()

    for handle in (created, connected):
        with pytest.raises(ConnectionError, match="crashed"):
            await provider.exec(handle, "touch ran", cwd=str(tmp_path))
        with pytest.raises(ConnectionError, match="crashed"):
            await provider.upload_file(handle, source, str(tmp_path / "uploaded.txt"))
        with pytest.raises(ConnectionError, match="crashed"):
            await provider.download_file(handle, str(source), tmp_path / "downloaded.txt")
    assert not (tmp_path / "ran").exists() and not (tmp_path / "uploaded.txt").exists()
    assert not (tmp_path / "downloaded.txt").exists() and provider.commands == []

    again = await provider.connect({"sandbox_id": "local"})
    result = await provider.exec(again, "touch ran", cwd=str(tmp_path))
    assert result.return_code == 0 and (tmp_path / "ran").exists()


def test_process_state_reads_a_running_a_stopped_and_a_missing_process() -> None:
    process = subprocess.Popen(["sleep", "60"])
    try:
        assert process_state(process.pid) in ("S", "R")
        process.send_signal(signal.SIGSTOP)
        deadline = time.monotonic() + 10
        while process_state(process.pid) != "T":
            assert time.monotonic() < deadline, "the process did not stop"
            time.sleep(0.02)
        process.kill()
        process.wait(timeout=10)
        assert process_state(process.pid) is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
