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
"""Tests for the AppWorld worker-process pool.

These spawn real subprocesses, but of ``fake_appworld_server.py`` rather than
AppWorld — it speaks the same two things the pool needs (a readiness probe on
``GET /`` and ``{"output": ...}`` on POST), so spawn, readiness, leasing,
respawn-on-death and shutdown are all covered without a 193 MB download.
"""

import os
import stat
import sys
from pathlib import Path

import pytest

from resources_servers.appworld.worker_pool import (
    AppWorldWorker,
    AppWorldWorkerError,
    AppWorldWorkerPool,
)


FAKE_SERVER = Path(__file__).resolve().parent / "fake_appworld_server.py"
STARTUP_TIMEOUT_SECS = 60.0


def make_executable(tmp_path: Path, name: str, body: str) -> str:
    """Write an executable shell wrapper standing in for the ``appworld`` CLI."""
    script = tmp_path / name
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


@pytest.fixture
def fake_appworld(tmp_path: Path) -> str:
    return make_executable(tmp_path, "fake-appworld", f'exec "{sys.executable}" "{FAKE_SERVER}" "$@"')


def make_pool(tmp_path: Path, executable: str, num_workers: int = 1, **overrides) -> AppWorldWorkerPool:
    return AppWorldWorkerPool(
        num_workers=num_workers,
        port_start=0,  # always ask the OS for a free port
        root=str(tmp_path),
        executable=executable,
        startup_timeout_secs=overrides.pop("startup_timeout_secs", STARTUP_TIMEOUT_SECS),
        request_timeout_secs=overrides.pop("request_timeout_secs", 30.0),
        **overrides,
    )


@pytest.mark.asyncio
async def test_pool_starts_workers_and_leases_them_exclusively(tmp_path, fake_appworld):
    pool = make_pool(tmp_path, fake_appworld, num_workers=2)
    try:
        await pool.start()
        first = await pool.acquire()
        second = await pool.acquire()

        assert first is not second
        assert all(worker.alive for worker in pool.workers)
        # Both leased out: nothing left in the free queue.
        assert pool._free.qsize() == 0

        await pool.release(first)
        assert (await pool.acquire()) is first
    finally:
        await pool.stop()

    assert pool.workers == []


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path, fake_appworld):
    pool = make_pool(tmp_path, fake_appworld)
    try:
        await pool.start()
        processes = [worker.process for worker in pool.workers]
        await pool.start()

        assert [worker.process for worker in pool.workers] == processes
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_acquire_starts_the_pool_on_first_use(tmp_path, fake_appworld):
    pool = make_pool(tmp_path, fake_appworld)
    try:
        worker = await pool.acquire()

        assert worker.alive
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_call_unwraps_the_output_envelope(tmp_path, fake_appworld):
    pool = make_pool(tmp_path, fake_appworld)
    try:
        worker = await pool.acquire()

        assert await worker.call("/execute", {"code": "print(1)"}) == {"code": "print(1)"}
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_call_returns_the_body_when_there_is_no_output_key(tmp_path, fake_appworld):
    pool = make_pool(tmp_path, fake_appworld)
    try:
        worker = await pool.acquire()

        assert await worker.call("/bare", {"a": 1}) == {"echo": {"a": 1}}
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_call_raises_on_an_http_error(tmp_path, fake_appworld):
    import aiohttp

    pool = make_pool(tmp_path, fake_appworld)
    try:
        worker = await pool.acquire()

        with pytest.raises(aiohttp.ClientResponseError):
            await worker.call("/boom", {})
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_a_hanging_worker_fails_the_call_instead_of_the_rollout(tmp_path, fake_appworld):
    """`request()` retries connection errors forever, so calls must be bounded."""
    pool = make_pool(tmp_path, fake_appworld, request_timeout_secs=0.5)
    try:
        worker = await pool.acquire()

        with pytest.raises(AppWorldWorkerError, match="did not answer /slow"):
            await worker.call("/slow", {})
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_a_dead_worker_is_respawned_on_the_next_lease(tmp_path, fake_appworld):
    pool = make_pool(tmp_path, fake_appworld)
    try:
        worker = await pool.acquire()
        worker.process.kill()
        worker.process.wait()
        assert not worker.alive
        await pool.release(worker)

        respawned = await pool.acquire()

        assert respawned is worker
        assert worker.alive
        assert await worker.call("/execute", {"code": "1"}) == {"code": "1"}
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_startup_failure_is_reported_and_leaves_no_workers(tmp_path):
    exits_immediately = make_executable(tmp_path, "exits", "exit 3")
    pool = make_pool(tmp_path, exits_immediately)

    with pytest.raises(AppWorldWorkerError, match="exited during startup"):
        await pool.start()

    assert pool._free is None


@pytest.mark.asyncio
async def test_a_worker_that_never_serves_times_out(tmp_path):
    never_serves = make_executable(tmp_path, "hangs", "sleep 30")
    pool = make_pool(tmp_path, never_serves, startup_timeout_secs=1.0)

    with pytest.raises(AppWorldWorkerError, match="did not become ready"):
        await pool.start()


@pytest.mark.asyncio
async def test_terminate_kills_a_worker_that_ignores_sigterm(tmp_path, monkeypatch):
    import resources_servers.appworld.worker_pool as worker_pool_module

    # `trap "" TERM` in the parent shell is inherited by the exec'd python, so
    # SIGTERM is ignored and the SIGKILL fallback has to fire.
    ignores_sigterm = make_executable(
        tmp_path,
        "stubborn",
        f'trap "" TERM\nexec "{sys.executable}" "{FAKE_SERVER}" "$@"',
    )
    monkeypatch.setattr(worker_pool_module, "TERMINATE_TIMEOUT_SECS", 0.5)
    pool = make_pool(tmp_path, ignores_sigterm)
    await pool.start()
    process = pool.workers[0].process

    await pool.stop()

    assert process.poll() is not None


@pytest.mark.asyncio
async def test_worker_logs_land_in_the_root(tmp_path, fake_appworld):
    pool = make_pool(tmp_path, fake_appworld)
    try:
        await pool.start()
    finally:
        await pool.stop()

    assert (tmp_path / ".nemo_gym_worker_logs" / "worker_0.log").is_file()


@pytest.mark.asyncio
async def test_workers_inherit_the_appworld_root(tmp_path, fake_appworld, monkeypatch):
    monkeypatch.setenv("APPWORLD_ROOT", "/somewhere/else")
    pool = make_pool(tmp_path, fake_appworld)
    try:
        await pool.start()
    finally:
        await pool.stop()

    # The pool's root wins over whatever is in the ambient environment.
    assert os.environ["APPWORLD_ROOT"] == "/somewhere/else"
    assert pool.root == str(tmp_path)


def test_worker_url_is_loopback_only(tmp_path):
    worker = AppWorldWorker(
        index=0,
        port=21000,
        root=str(tmp_path),
        executable="appworld",
        log_fpath=tmp_path / "w.log",
        request_timeout_secs=1.0,
    )

    assert worker.url == "http://127.0.0.1:21000"
