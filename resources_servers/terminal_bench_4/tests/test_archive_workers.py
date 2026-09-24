# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import threading
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_4 import transfers
from resources_servers.terminal_bench_4.app import TerminalBench4Config, TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.archive_workers import ArchiveWorkers
from resources_servers.terminal_bench_4.models import TerminalBench4RunRequest
from resources_servers.terminal_bench_4.tests.test_environment import environment_config
from resources_servers.terminal_bench_4.tests.test_transfer_metadata import synthetic_archive


async def test_pool_bounds_running_work_without_blocking_other_coroutines():
    loop = asyncio.get_running_loop()
    main_thread = threading.get_ident()
    release = threading.Event()
    saturated = asyncio.Event()
    lock = threading.Lock()
    running = maximum = 0

    def operation():
        nonlocal running, maximum
        assert threading.get_ident() != main_thread
        with lock:
            running += 1
            maximum = max(maximum, running)
            if running == 2:
                loop.call_soon_threadsafe(saturated.set)
        try:
            assert release.wait(10), "Test failed to release archive workers"
            return 42
        finally:
            with lock:
                running -= 1

    async with ArchiveWorkers(2) as workers:
        tasks = [asyncio.create_task(workers.run(operation)) for _ in range(6)]
        try:
            await asyncio.wait_for(saturated.wait(), 5)
            # Both workers are still blocked, but unrelated async work completes.
            assert await asyncio.wait_for(asyncio.sleep(0, result="responsive"), 1) == "responsive"
            assert maximum == 2 and all(not task.done() for task in tasks)
        finally:
            release.set()
            results = await asyncio.gather(*tasks)
    assert results == [42] * 6 and maximum == 2


async def test_cancelled_queued_work_never_starts():
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    queued_operation = MagicMock()

    def blocked():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(10)

    async with ArchiveWorkers(1) as workers:
        active = asyncio.create_task(workers.run(blocked))
        queued = None
        try:
            await asyncio.wait_for(started.wait(), 5)
            queued = asyncio.create_task(workers.run(queued_operation))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
        finally:
            release.set()
            await active
        assert await workers.run(lambda: "slot reusable") == "slot reusable"
    queued_operation.assert_not_called()


async def test_shutdown_drains_threads_and_rejects_new_or_queued_work():
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    threads = []

    def blocked():
        threads.append(threading.current_thread())
        loop.call_soon_threadsafe(started.set)
        assert release.wait(10)

    workers = ArchiveWorkers(1)
    active = asyncio.create_task(workers.run(blocked))
    queued = closing = None
    try:
        await asyncio.wait_for(started.wait(), 5)
        queued = asyncio.create_task(workers.run(lambda: pytest.fail("Queued work ran during shutdown")))
        await asyncio.sleep(0)
        closing = asyncio.create_task(workers.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        with pytest.raises(RuntimeError, match="shutting down"):
            await workers.run(lambda: None)
    finally:
        release.set()
        await active
        if queued:
            with pytest.raises(RuntimeError, match="shutting down"):
                await queued
        if closing:
            await closing
        await workers.aclose()  # Idempotent, including after the first join.
    assert threads and all(not thread.is_alive() for thread in threads)


async def test_worker_errors_propagate_and_context_is_preserved():
    context = ContextVar("archive_test_rollout", default="missing")
    token = context.set("rollout-one")

    def fail():
        raise OSError("synthetic disk error")

    try:
        async with ArchiveWorkers(1) as workers:
            with pytest.raises(OSError, match="synthetic disk error"):
                await workers.run(fail)
            assert await workers.run(context.get) == "rollout-one"
    finally:
        context.reset(token)


@pytest.mark.parametrize("operation", ["trusted", "upload", "download"])
async def test_transfer_archive_work_uses_shared_pool_and_keeps_loop_responsive(tmp_path, monkeypatch, operation):
    source = tmp_path / "source"
    source.mkdir()
    (source / "solve.sh").write_text("exit 0\n")
    archive = tmp_path / "original.tar.gz"
    synthetic_archive(archive)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    main_thread = threading.get_ident()
    helper_name = {
        "trusted": "_pack_trusted_directory",
        "upload": "_pack_directory",
        "download": "_extract_directory",
    }[operation]
    original = getattr(transfers, helper_name)

    def blocked(*args, **kwargs):
        assert threading.get_ident() != main_thread
        loop.call_soon_threadsafe(started.set)
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(transfers, helper_name, blocked)

    async def download(_remote, destination):
        destination.write_bytes(archive.read_bytes())

    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0)), upload=AsyncMock(), download=download
    )
    metadata = tmp_path / "metadata.json"
    async with ArchiveWorkers(1) as workers:
        if operation == "download":
            coroutine = transfers.download_dir(
                sandbox,
                "/app",
                tmp_path / "view",
                metadata_path=metadata,
                shared_archive="/logs/snapshot.tar.gz",
                archive_workers=workers,
            )
        elif operation == "trusted":
            coroutine = transfers.stage_trusted_directory(sandbox, source, "/solution", archive_workers=workers)
        else:
            coroutine = transfers.upload_dir(sandbox, source, "/app", archive_workers=workers)
        task = asyncio.create_task(coroutine)
        try:
            await asyncio.wait_for(started.wait(), 5)
            assert await asyncio.wait_for(asyncio.sleep(0, result=True), 1)
            assert not task.done()
        finally:
            release.set()
            result = await task
    if operation == "download":
        assert result == hashlib.sha256(archive.read_bytes()).hexdigest()
        assert json.loads(metadata.read_text())["nested/tool"] == [1001, 2002, 0o775]


@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancelled_extraction_keeps_archive_and_slot_until_worker_finishes(tmp_path, monkeypatch, worker_fails):
    archive = tmp_path / "original.tar.gz"
    synthetic_archive(archive)
    original = transfers._extract_directory
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    paths = []
    next_job = MagicMock(return_value="next")

    def blocked(local_archive, *args, **kwargs):
        paths.append(local_archive)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(10)
        assert local_archive.exists(), "Caller deleted an archive still in use"
        if worker_fails:
            raise OSError("synthetic extraction failure after cancellation")
        return original(local_archive, *args, **kwargs)

    monkeypatch.setattr(transfers, "_extract_directory", blocked)

    async def download(_source, destination):
        destination.write_bytes(archive.read_bytes())

    sandbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0)), download=download)
    async with ArchiveWorkers(1) as workers:
        task = asyncio.create_task(transfers.download_dir(sandbox, "/app", tmp_path / "view", archive_workers=workers))
        following = None
        try:
            await asyncio.wait_for(started.wait(), 5)
            following = asyncio.create_task(workers.run(next_job))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()  # Repeated shutdown cancellation must also be deferred.
            await asyncio.sleep(0)
            assert not task.done() and paths[0].exists()
            next_job.assert_not_called()
            assert not any(call.args[0].startswith("rm -f") for call in sandbox.exec.await_args_list)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            if following:
                assert await following == "next"
    assert not paths[0].exists()
    assert sandbox.exec.await_args.args[0].startswith("rm -f")


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_archive_concurrency_is_rejected(limit):
    with pytest.raises(ValueError, match="positive"):
        ArchiveWorkers(limit)
    with pytest.raises(ValidationError, match="max_concurrent_archive_operations"):
        TerminalBench4Config(
            host="localhost",
            port=1,
            name="tb4",
            entrypoint="app.py",
            environment=environment_config(),
            max_concurrent_archive_operations=limit,
        )


def test_benchmark_archive_override_is_independent_of_task_concurrency():
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(root / "benchmarks/terminal_bench_4/resources.yaml")
    resource = config.terminal_bench_4.resources_servers.terminal_bench_4
    assert resource.max_concurrent_archive_operations == 2
    config.tb4_concurrency = 50
    config.tb4_archive_concurrency = 3
    assert resource.max_concurrent_sessions == 50
    assert resource.max_concurrent_archive_operations == 3


async def test_sessions_share_one_configured_pool_and_lifespan_closes_it(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"ref": "sha256:" + "a" * 64, "tasks": []}))
    server = TerminalBench4ResourcesServer(
        config=TerminalBench4Config(
            host="localhost",
            port=1,
            name="tb4",
            entrypoint="app.py",
            manifest_path=manifest,
            artifacts_dir=tmp_path / "results",
            environment=environment_config(),
            max_concurrent_archive_operations=3,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    body = TerminalBench4RunRequest(
        task_name="synthetic",
        task_ref="sha256:" + "b" * 64,
        dataset_ref="sha256:" + "a" * 64,
        rollout_id="one",
        responses_create_params={"input": []},
    )
    one = server._new_session("one", "owner", body, "tb4-one")
    two = server._new_session("two", "owner", body, "tb4-two")
    assert one.archive_workers is two.archive_workers is server._archive_workers
    app = server.setup_webserver()
    async with app.router.lifespan_context(app):
        assert await one.archive_workers.run(lambda: 42) == 42
    with pytest.raises(RuntimeError, match="shutting down"):
        await server._archive_workers.run(lambda: None)
