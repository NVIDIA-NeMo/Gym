# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from nemo_gym.utils.install_lock import (
    adopt_recorded_install,
    async_install_lock,
    install_lock,
    record_install,
)


def _hold_lock(lock_path: Path, log_path: Path) -> None:
    with install_lock(lock_path):
        with log_path.open("a") as log:
            log.write(f"enter {os.getpid()}\n")
        time.sleep(0.3)
        with log_path.open("a") as log:
            log.write(f"exit {os.getpid()}\n")


def test_processes_holding_the_lock_never_overlap(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_hold_lock, args=(tmp_path / "x.lock", tmp_path / "log")) for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)

    assert [process.exitcode for process in processes] == [0, 0, 0]
    events = [line.split() for line in (tmp_path / "log").read_text().splitlines()]
    # Each process's exit comes right after its enter, so no two critical sections interleave.
    assert [kind for kind, _ in events] == ["enter", "exit"] * 3
    assert all(events[index][1] == events[index + 1][1] for index in range(0, 6, 2))


async def test_async_waiter_waits_without_blocking_and_can_be_cancelled(tmp_path: Path) -> None:
    lock_path = tmp_path / "x.lock"
    with install_lock(lock_path):

        async def wait_for_lock() -> None:
            async with async_install_lock(lock_path, poll_seconds=0.01):
                pass

        waiter = asyncio.create_task(wait_for_lock())
        # The event loop keeps running while the waiter polls.
        await asyncio.sleep(0.05)
        assert not waiter.done()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    # The cancelled waiter left nothing holding the lock.
    async with async_install_lock(lock_path, poll_seconds=0.01):
        pass


def test_a_recorded_install_is_adopted_only_while_it_holds_the_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    marker = tmp_path / "tool.installed"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    assert not adopt_recorded_install(marker, "tool", version="1.0.0")

    record_install(marker, bin_dir, version="1.0.0")
    assert not adopt_recorded_install(marker, "tool", version="1.0.0")
    assert os.environ["PATH"] == "/usr/bin"

    (bin_dir / "tool").write_text("#!/bin/sh\n")
    assert adopt_recorded_install(marker, "tool", version="1.0.0")
    assert os.environ["PATH"] == f"{bin_dir}{os.pathsep}/usr/bin"


def test_a_recorded_install_is_adopted_only_for_the_version_it_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    marker = tmp_path / "tool.installed"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "tool").write_text("#!/bin/sh\n")

    # A pin bump must reinstall rather than run the tool another version left behind.
    record_install(marker, bin_dir, version="1.0.0")
    assert not adopt_recorded_install(marker, "tool", version="2.0.0")
    assert not adopt_recorded_install(marker, "tool", version=None)

    # An unpinned install is reused only by an unpinned request.
    record_install(marker, bin_dir, version=None)
    assert not adopt_recorded_install(marker, "tool", version="1.0.0")
    assert adopt_recorded_install(marker, "tool", version=None)

    # A record that names only the directory, as records did before they named a version, is not adopted.
    marker.write_text(str(bin_dir))
    assert not adopt_recorded_install(marker, "tool", version=None)
    assert os.environ["PATH"] == f"{bin_dir}{os.pathsep}/usr/bin"
