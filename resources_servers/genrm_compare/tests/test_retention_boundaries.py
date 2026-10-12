# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retries across independent result and attempt-record retention boundaries."""

import asyncio
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import resources_servers.genrm_compare.app as genrm
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import member


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


async def complete(server, group="group", attempt=0):
    return await asyncio.gather(*(server.verify(member(i, group=group, attempt=attempt)) for i in range(2)))


async def test_slow_judge_result_keeps_newest_attempt_fenced(server, clock):
    server.config.cohort_result_ttl_s = 10
    started, release = asyncio.Event(), asyncio.Event()

    async def compare(**kwargs):
        started.set()
        await release.wait()
        return [3.0, 3.0], {}, [], []

    server._run_compare = compare
    latest = asyncio.create_task(complete(server, attempt=1))
    await started.wait()
    clock[0] = 108
    release.set()
    await latest
    clock[0] = 111
    with pytest.raises(HTTPException) as error:
        await asyncio.wait_for(server.verify(member(0, attempt=0)), 0.1)
    assert error.value.status_code == 409
    assert (await server.verify(member(0, attempt=1))).reward == 3


@pytest.mark.parametrize("eviction", ["ttl", "count"])
@pytest.mark.parametrize("duplicates", [1, 2])
async def test_finished_attempt_with_missing_result_is_rejected_without_rejudging(server, clock, eviction, duplicates):
    judge = AsyncMock(return_value=(3.0, 3.0, 3.5))
    server._run_single_comparison = judge
    await complete(server, "a")
    if eviction == "ttl":
        server.config.cohort_result_ttl_s = 10
        clock[0] = 108
        assert (await server.verify(member(0, group="a"))).reward == 3
        clock[0] = 111
    else:
        server.config.max_terminal_cohorts = 2
        # Failed legacy groups occupy result slots without explicit attempt records.
        server.config.cohort_collection_timeout_s = 0.01
        for legacy_attempt in range(2):
            with pytest.raises(HTTPException):
                await server.verify(member(0, group=None, attempt=legacy_attempt))
    calls = judge.await_count
    for _ in range(2):
        outcomes = await asyncio.wait_for(
            asyncio.gather(*(server.verify(member(i, group="a")) for i in range(duplicates)), return_exceptions=True),
            0.1,
        )
        assert all(isinstance(r, HTTPException) and r.status_code == 409 for r in outcomes)
        assert judge.await_count == calls
    assert [r.reward for r in await complete(server, "a", attempt=1)] == [3, 3]


@pytest.mark.parametrize("conflicting_field", [None, "prompt", "principle"])
async def test_count_eviction_keeps_retained_result_replayable(server, clock, conflicting_field):
    server.config.max_terminal_cohorts = 2
    server._run_single_comparison = AsyncMock(return_value=(3.0, 3.0, 3.5))
    await complete(server, "a")
    clock[0] = 101
    await complete(server, "b", attempt=1)
    clock[0] = 102
    await server.verify(member(0, group="a"))
    clock[0] = 103
    await complete(server, "c")
    server._prune_terminal_cohorts()
    # Replaying a refreshed its attempt record only, putting b first in that eviction order.
    assert "b" not in server._latest_group_attempts
    calls = server._run_single_comparison.await_count
    if conflicting_field is not None:
        invalid = member(0, group="b", attempt=1)
        if conflicting_field == "prompt":
            invalid.responses_create_params.input[0].content = "Different question"
        else:
            invalid.principle = "Different judging instructions"
        with pytest.raises(HTTPException) as error:
            await server.verify(invalid)
        assert error.value.status_code == 409
        assert "inconsistent prompt or principle" in error.value.detail
    replay = await asyncio.wait_for(complete(server, "b", attempt=1), 0.1)
    assert [r.reward for r in replay] == [3, 3]
    assert server._run_single_comparison.await_count == calls


async def test_rejected_expired_duplicate_does_not_extend_attempt_retention(server, clock):
    server.config.cohort_result_ttl_s = 10
    server._run_single_comparison = AsyncMock(return_value=(3.0, 3.0, 3.5))
    await complete(server)
    clock[0] = 108
    await server.verify(member(0))
    clock[0] = 111
    with pytest.raises(HTTPException) as error:
        await asyncio.wait_for(server.verify(member(0)), 0.1)
    assert error.value.status_code == 409
    clock[0] = 119
    server._prune_terminal_cohorts()
    assert not server._latest_group_attempts


async def test_cleanup_does_not_scan_active_prefix_with_finished_group(server, clock):
    class NoScan(dict):
        def items(self):
            raise AssertionError("scanned active groups")

        def values(self):
            raise AssertionError("scanned active groups")

        def __iter__(self):
            raise AssertionError("scanned active groups")

    class CountedOrder(OrderedDict):
        visits = 0

        def items(self):
            for item in super().items():
                self.visits += 1
                yield item

    server.config.cohort_result_ttl_s = 10
    # Resolve normal group identities without starting one thousand timers.
    for i in range(1000):
        body = member(0, group=f"active-{i}")
        await server._resolve_verify_cohort(
            body=body, prompt_key=server._group_cohort_key(body.group_id, 0), prompt_digest="p"
        )
    clock[0] = 111
    server._run_single_comparison = AsyncMock(return_value=(3.0, 3.0, 3.5))
    await complete(server, "finished")
    original = server._latest_group_attempts
    server._latest_group_attempts = NoScan(original)
    idle = CountedOrder(server._idle_groups)
    server._idle_groups = idle
    try:
        for _ in range(3):
            server._prune_terminal_cohorts()
        assert idle.visits == 3
        assert len(server._latest_group_attempts) == 1001
        clock[0] = 122
        server._prune_terminal_cohorts()
        assert "finished" not in server._latest_group_attempts
        assert len(server._latest_group_attempts) == 1000
    finally:
        server._latest_group_attempts = dict(server._latest_group_attempts)


@pytest.mark.parametrize("whole_group", [False, True])
@pytest.mark.parametrize("phase", ["collecting", "evaluating"])
async def test_active_groups_do_not_evict_a_finished_result(server, whole_group, phase):
    server.config.max_terminal_cohorts = 2
    judge_started = 0
    judging = asyncio.Event()
    release = asyncio.Event()

    async def compare(*, conversation_history, **kwargs):
        nonlocal judge_started
        if conversation_history[0]["content"].startswith("active"):
            judge_started += 1
            if judge_started == 3:
                judging.set()
            await release.wait()
        return [3.0, 3.0], {}, [], []

    server._run_compare = AsyncMock(side_effect=compare)
    active = []
    for group in range(3):
        for index in range(2 if phase == "evaluating" else 1):
            request = member(index, group=f"active-{group}")
            request.responses_create_params.input[0].content = f"active {group}"
            active.append(asyncio.create_task(server.verify(request)))
    try:
        if phase == "evaluating":
            await asyncio.wait_for(judging.wait(), 1)
        else:
            await asyncio.sleep(0)
        assert [r.reward for r in await complete(server, "done")] == [3, 3]
        calls = server._run_compare.await_count
        indices = range(2) if whole_group else [0]
        retry = await asyncio.wait_for(asyncio.gather(*(server.verify(member(i, group="done")) for i in indices)), 0.1)
        assert [r.reward for r in retry] == [3] * len(indices)
        assert server._run_compare.await_count == calls
    finally:
        release.set()
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


async def test_superseded_result_neither_uses_a_slot_nor_replays_after_eviction(server, clock):
    server.config.max_terminal_cohorts = 3
    server.config.cohort_collection_timeout_s = 0.02
    judge = AsyncMock(return_value=(3.0, 3.0, 3.5))
    server._run_single_comparison = judge
    await complete(server, "x1")
    await complete(server, "x2")
    clock[0] = 101
    await complete(server, "a")
    await complete(server, "a", attempt=1)
    assert (await asyncio.wait_for(server.verify(member(0, group="x1")), 0.1)).reward == 3
    clock[0] = 102
    await server.verify(member(0, group="x2"))
    clock[0] = 103
    await complete(server, "y")
    server._prune_terminal_cohorts()
    assert "a" not in server._latest_group_attempts
    assert server._group_cohort_key("a", 0) not in server._verify_cohorts
    calls = judge.await_count
    # Once the attempt record is gone, the caller owns stale-attempt detection.
    # This result was removed while its attempt was still tracked and must not replay.
    with pytest.raises(HTTPException) as error:
        await server.verify(member(0, group="a"))
    assert error.value.status_code == 503
    assert judge.await_count == calls
