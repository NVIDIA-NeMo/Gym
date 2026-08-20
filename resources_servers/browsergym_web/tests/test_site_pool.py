# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from nemo_gym.web.models import WebBenchmark, WebTask
from resources_servers.browsergym_web.site_pool import LocalSiteLockPool


def _task(task_id: str, sites: list[str], mutation_class: str) -> WebTask:
    return WebTask(
        benchmark=WebBenchmark.VISUALWEBARENA,
        task_id=task_id,
        sites=sites,
        site_locks=sites,
        mutation_class=mutation_class,
    )


@pytest.mark.asyncio
async def test_local_site_pool_shares_readers_and_serializes_same_site_writer():
    pool = LocalSiteLockPool()
    first = await pool.acquire("reader-a", _task("a", ["reddit"], "read_only"))
    second = await pool.acquire("reader-b", _task("b", ["reddit"], "read_only"))

    writer_future = asyncio.create_task(pool.acquire("writer", _task("c", ["reddit"], "state_changing")))
    await asyncio.sleep(0)
    assert not writer_future.done()

    other_site = await pool.acquire(
        "shopping-writer",
        _task("d", ["shopping"], "state_changing"),
    )
    assert other_site.metadata["access"] == "writer"

    await pool.release(first, healthy=True)
    await asyncio.sleep(0)
    assert not writer_future.done()
    await pool.release(second, healthy=True)
    writer = await asyncio.wait_for(writer_future, timeout=1)

    assert writer.metadata["sites"] == ["reddit"]
    await pool.release(writer, healthy=True)
    await pool.release(other_site, healthy=True)
    assert (await pool.health())["active_leases"] == 0


@pytest.mark.asyncio
async def test_cross_site_writer_acquires_all_sites_atomically():
    pool = LocalSiteLockPool()
    shopping_reader = await pool.acquire(
        "shopping-reader",
        _task("a", ["shopping"], "read_only"),
    )
    cross_future = asyncio.create_task(
        pool.acquire(
            "cross-writer",
            _task("b", ["shopping", "reddit"], "state_changing"),
        )
    )
    await asyncio.sleep(0)
    assert not cross_future.done()

    reddit_reader = await pool.acquire("reddit-reader", _task("c", ["reddit"], "read_only"))
    await pool.release(shopping_reader, healthy=True)
    await asyncio.sleep(0)
    assert not cross_future.done()
    await pool.release(reddit_reader, healthy=True)

    cross_writer = await asyncio.wait_for(cross_future, timeout=1)
    health = await pool.health()
    assert health["writer_sites"] == ["reddit", "shopping"]
    await pool.release(cross_writer, healthy=True)
