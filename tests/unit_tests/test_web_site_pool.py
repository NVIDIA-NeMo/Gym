# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from nemo_gym.web.models import WebBenchmark, WebTask
from nemo_gym.web.site_pool import LocalSiteLockPool, SiteLease, UnmanagedSitePool


def _task(task_id: str, sites: list[str], mutation_class: str = "state_changing") -> WebTask:
    return WebTask(
        benchmark=WebBenchmark.WEBVOYAGER,
        task_id=task_id,
        sites=sites,
        site_locks=sites,
        mutation_class=mutation_class,
    )


@pytest.mark.asyncio
async def test_unmanaged_site_pool_tracks_active_sessions() -> None:
    pool = UnmanagedSitePool()
    lease = await pool.acquire("session-a", _task("a", ["reddit"]))

    assert lease == SiteLease(
        lease_id="unmanaged:session-a",
        isolated=False,
        metadata={"benchmark": "webvoyager", "sites": ["reddit"]},
    )
    assert (await pool.health())["active_leases"] == 1
    await pool.release(lease, healthy=False)
    assert (await pool.health())["active_leases"] == 0


@pytest.mark.asyncio
async def test_local_site_pool_shares_readers_and_serializes_same_site_writer() -> None:
    pool = LocalSiteLockPool()
    first = await pool.acquire("reader-a", _task("a", ["Reddit"], "read_only"))
    second = await pool.acquire("reader-b", _task("b", ["reddit"], "session_only"))

    with pytest.raises(RuntimeError, match="duplicate site lease"):
        await pool.acquire("reader-a", _task("duplicate", ["reddit"], "read_only"))

    writer_future = asyncio.create_task(pool.acquire("writer", _task("c", ["reddit"])))
    await asyncio.sleep(0)
    assert not writer_future.done()

    other_site = await pool.acquire("shopping-writer", _task("d", ["shopping"]))
    assert other_site.metadata["access"] == "writer"

    await pool.release(first, healthy=True)
    await asyncio.sleep(0)
    assert not writer_future.done()
    await pool.release(second, healthy=True)
    writer = await asyncio.wait_for(writer_future, timeout=1)

    assert writer.metadata["sites"] == ["reddit"]
    assert (await pool.health())["writer_sites"] == ["reddit", "shopping"]
    await pool.release(writer, healthy=True)
    await pool.release(other_site, healthy=True)
    await pool.release(other_site, healthy=False)
    assert (await pool.health())["active_leases"] == 0


@pytest.mark.asyncio
async def test_cross_site_writer_acquires_all_sites_atomically() -> None:
    pool = LocalSiteLockPool()
    shopping_reader = await pool.acquire("shopping-reader", _task("a", ["shopping"], "read_only"))
    cross_future = asyncio.create_task(pool.acquire("cross-writer", _task("b", ["shopping", "reddit"])))
    await asyncio.sleep(0)
    assert not cross_future.done()

    reddit_reader = await pool.acquire("reddit-reader", _task("c", ["reddit"], "read_only"))
    await pool.release(shopping_reader, healthy=True)
    await asyncio.sleep(0)
    assert not cross_future.done()
    await pool.release(reddit_reader, healthy=True)

    cross_writer = await asyncio.wait_for(cross_future, timeout=1)
    assert (await pool.health())["writer_sites"] == ["reddit", "shopping"]
    await pool.release(cross_writer, healthy=True)


@pytest.mark.asyncio
async def test_empty_site_list_uses_benchmark_lock() -> None:
    pool = LocalSiteLockPool()
    lease = await pool.acquire("fallback", _task("empty", []))

    assert lease.metadata["sites"] == ["benchmark:webvoyager"]
    await pool.release(lease, healthy=True)
