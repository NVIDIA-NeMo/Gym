# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading

import pytest

from nemo_gym.web.operation_runner import DirectWebOperationRunner, ThreadAffineWebOperationRunner


@pytest.mark.asyncio
async def test_direct_runner_executes_inline_and_closes_idempotently():
    runner = DirectWebOperationRunner()
    event_loop_thread = threading.get_ident()

    assert await runner.run(threading.get_ident) == event_loop_thread
    await runner.close()
    await runner.close()


@pytest.mark.asyncio
async def test_thread_affine_runner_serializes_calls_on_one_worker():
    runner = ThreadAffineWebOperationRunner(thread_name_prefix="test-web-runtime")
    event_loop_thread = threading.get_ident()

    worker_threads = await asyncio.gather(*(runner.run(threading.get_ident) for _ in range(8)))

    assert len(set(worker_threads)) == 1
    assert worker_threads[0] != event_loop_thread
    await runner.close()


@pytest.mark.asyncio
async def test_thread_affine_runner_runs_finalizer_on_its_worker():
    calls: list[tuple[str, int]] = []
    runner = ThreadAffineWebOperationRunner(
        finalizer=lambda: calls.append(("finalize", threading.get_ident())),
    )

    worker_thread = await runner.run(threading.get_ident)
    await runner.close()

    assert calls == [("finalize", worker_thread)]
    await runner.close()

    with pytest.raises(RuntimeError, match="already stopped"):
        await runner.run(threading.get_ident)
