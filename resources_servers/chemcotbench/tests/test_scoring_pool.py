# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys

import pytest

from resources_servers.chemcotbench.scoring_pool import ScoringPool, ScoringWorkerError


@pytest.fixture
async def pool():
    instance = ScoringPool(2)
    yield instance
    await instance.close()


@pytest.fixture
def command(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("""import json, os, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if request.get("crash"):
        sys.stderr.buffer.write(b"scorer failed\\xff")
        sys.exit(1)
    if request.get("bad_json"):
        print("bad JSON", flush=True)
        continue
    if request.get("stderr"):
        sys.stderr.write("x" * 300000)
        sys.stderr.flush()
    time.sleep(request.get("sleep", 0))
    print(json.dumps({"pid": os.getpid(), "value": request.get("value"), "cwd": os.getcwd()}), flush=True)
""")
    return [sys.executable, str(worker)]


async def test_reuses_worker_and_drains_stderr(pool, command):
    first = await pool.score(command, {"value": "first", "stderr": True}, cwd=None, timeout=5)
    second = await pool.score(command, {"value": "second"}, cwd=None, timeout=5)
    assert first["pid"] == second["pid"]
    assert (first["value"], second["value"]) == ("first", "second")
    assert max(len(slot.stderr_tail) for slot in pool._slots) <= 2000


async def test_concurrency_limit_and_response_isolation(pool, command):
    results = await asyncio.gather(
        *[pool.score(command, {"value": i, "sleep": 0.02}, cwd=None, timeout=5) for i in range(8)]
    )
    assert len({result["pid"] for result in results}) == 2
    assert [result["value"] for result in results] == list(range(8))


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"crash": True}, "upstream_error"),
        ({"bad_json": True}, "invalid_result"),
    ],
)
async def test_replaces_failed_worker(pool, command, payload, error):
    before = await pool.score(command, {}, cwd=None, timeout=5)
    with pytest.raises(ScoringWorkerError) as caught:
        await pool.score(command, payload, cwd=None, timeout=5)
    assert caught.value.code == error
    if error == "upstream_error":
        assert "scorer failed" in str(caught.value)
    after = await pool.score(command, {"value": "recovered"}, cwd=None, timeout=5)
    assert after["pid"] != before["pid"] and after["value"] == "recovered"


async def test_timeout_reaps_worker_without_stale_response(pool, command):
    before = await pool.score(command, {}, cwd=None, timeout=5)
    process = next(slot.process for slot in pool._slots if slot.process)
    with pytest.raises(TimeoutError):
        await pool.score(command, {"sleep": 10, "value": "late"}, cwd=None, timeout=0.05)
    assert process.returncode is not None
    after = await pool.score(command, {"value": "next"}, cwd=None, timeout=5)
    assert after["pid"] != before["pid"] and after["value"] == "next"


async def test_cancellation_reaps_worker(pool, command):
    await pool.score(command, {}, cwd=None, timeout=5)
    process = next(slot.process for slot in pool._slots if slot.process)
    pending = asyncio.create_task(pool.score(command, {"sleep": 10}, cwd=None, timeout=30))
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert process.returncode is not None
    result = await pool.score(command, {"value": "next"}, cwd=None, timeout=5)
    assert result["value"] == "next"


async def test_runtime_switch_stays_bounded_and_shutdown_reaps(pool, command, tmp_path):
    paths = [tmp_path / str(i) for i in range(3)]
    for path in paths:
        path.mkdir()
    results = [await pool.score(command, {}, cwd=path, timeout=5) for path in paths]
    assert [result["cwd"] for result in results] == [str(path) for path in paths]
    processes = [slot.process for slot in pool._slots if slot.process]
    assert len(processes) == 2
    await pool.close()
    assert all(process.returncode is not None for process in processes)
    with pytest.raises(ScoringWorkerError, match="closed"):
        await pool.score(command, {}, cwd=None, timeout=5)


async def test_shutdown_cancels_active_and_rejects_queued_requests(pool, command):
    pending = [asyncio.create_task(pool.score(command, {"sleep": 10}, cwd=None, timeout=30)) for _ in range(3)]
    async with asyncio.timeout(5):
        while sum(slot.process is not None for slot in pool._slots) < 2:
            await asyncio.sleep(0.01)
        processes = [slot.process for slot in pool._slots]
        await pool.close()
        results = await asyncio.gather(*pending, return_exceptions=True)
    assert sum(isinstance(result, asyncio.CancelledError) for result in results) == 2
    assert sum(isinstance(result, ScoringWorkerError) for result in results) == 1
    assert all(process.returncode is not None for process in processes)
