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

"""Regression coverage for cohort deadlines, transport retries and ownership."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import resources_servers.genrm_compare.app as genrm
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


def member(index, *, group="group", attempt=0, response_id=None):
    return genrm.GenRMCompareVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "2+2?"}]),
        response=NeMoGymResponse(
            id=response_id or f"answer-{index}",
            created_at=0.0,
            model="test",
            tools=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            output=[],
            object="response",
        ),
        group_id=group,
        group_attempt=attempt,
        rollout_index=index,
    )


@pytest.mark.parametrize(
    "field", ["cohort_collection_timeout_s", "cohort_evaluation_timeout_s", "judge_request_timeout_s"]
)
@pytest.mark.parametrize("value", [None, 0, -1, float("inf"), float("nan")])
def test_deadline_must_be_finite_positive(config, field, value):
    with pytest.raises(ValidationError):
        genrm.GenRMCompareConfig.model_validate(config.model_dump() | {field: value})


async def test_judging_waits_for_full_group_and_exact_retry_replays(server):
    judge = AsyncMock(return_value=(4.0, 2.0, 1.0))
    server._run_single_comparison = judge
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    assert judge.await_count == 0
    result = await server.verify(member(1))
    assert result.reward == (await first).reward == 3.0
    assert (await server.verify(member(0))).reward == 3.0
    assert judge.await_count == 2


@pytest.mark.parametrize("during_judging", [False, True])
async def test_disconnect_reattaches_without_replacing_answer(server, during_judging):
    started, release = asyncio.Event(), asyncio.Event()

    async def compare(*args, **kwargs):
        started.set()
        await release.wait()
        return [1.0, 2.0], {}, [], []

    server._run_compare = AsyncMock(side_effect=compare)
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    second = asyncio.create_task(server.verify(member(1))) if during_judging else None
    if second:
        await started.wait()
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    cohort = next(iter(server._verify_cohorts.values()))
    assert cohort.members[0].body is not None and not cohort.members[0].waiters
    with pytest.raises(HTTPException) as error:
        await server.verify(member(0, response_id="different-answer"))
    assert error.value.status_code == 409
    retry = asyncio.create_task(server.verify(member(0)))
    if second is None:
        second = asyncio.create_task(server.verify(member(1)))
    release.set()
    assert [r.reward for r in await asyncio.gather(retry, second)] == [1.0, 2.0]
    server._run_compare.assert_awaited_once()


async def test_judge_has_separate_deadline_and_drains_comparisons(server):
    server.config.cohort_collection_timeout_s = 0.05
    server.config.cohort_evaluation_timeout_s = 0.2
    started, cancelled = set(), set()

    async def judge(*args, pair_idx, **kwargs):
        started.add(pair_idx)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(pair_idx)

    server._run_single_comparison = judge
    tasks = [asyncio.create_task(server.verify(member(i))) for i in range(2)]
    await asyncio.sleep(0.08)
    assert all(not task.done() for task in tasks)
    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
    assert all(isinstance(r, genrm.JudgeError) and "evaluation deadline" in str(r) for r in results)
    assert started == cancelled == {(0, 1), (1, 0)}
    assert all(m.body is None and not m.waiters for c in server._verify_cohorts.values() for m in c.members.values())


async def test_simultaneous_groups_do_not_mix(server):
    server._run_single_comparison = AsyncMock(return_value=(4.0, 2.0, 1.0))
    a = asyncio.create_task(server.verify(member(0, group="run-a")))
    b = asyncio.create_task(server.verify(member(1, group="run-b")))
    await asyncio.sleep(0)
    assert not a.done() and not b.done()
    await server.verify(member(1, group="run-a"))
    assert (await a).reward == 3.0 and not b.done()
    await server.verify(member(0, group="run-b"))
    assert (await b).reward == 3.0


@pytest.mark.parametrize("delay", [0, 0.008, 0.012])
async def test_final_arrival_deadline_race_never_publishes_partial_reward(server, delay):
    server.config.cohort_collection_timeout_s = 0.01
    server._run_single_comparison = AsyncMock(return_value=(4.0, 2.0, 1.0))
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(delay)
    results = await asyncio.wait_for(asyncio.gather(first, server.verify(member(1)), return_exceptions=True), 1)
    cohort = next(iter(server._verify_cohorts.values()))
    if cohort.phase == "completed":
        assert [r.reward for r in results] == [3.0, 3.0]
    else:
        assert all(isinstance(r, HTTPException) for r in results)
        assert not cohort.rewards
    assert all(m.body is None and not m.waiters for m in cohort.members.values())


async def test_shutdown_fails_waiters_and_prevents_new_work(server):
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    await server.aclose()
    with pytest.raises(HTTPException, match="shutting down"):
        await first
    with pytest.raises(HTTPException, match="shutting down"):
        await server.verify(member(1))
    assert not server._cohort_tasks and not server._verify_cohorts


async def test_abandoned_group_expires_without_another_request(server):
    server.config.cohort_collection_timeout_s = 0.02
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    cohort = next(iter(server._verify_cohorts.values()))
    timer = cohort.collection_timeout_task
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    assert cohort.members[0].body is not None and not cohort.members[0].waiters
    await asyncio.wait_for(timer, 1)
    assert cohort.phase == "failed" and not cohort.rewards
    assert all(m.body is None and not m.waiters for m in cohort.members.values())


async def test_late_judge_result_cannot_publish_after_supersession(server):
    started = asyncio.Event()
    calls = 0

    async def compare(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # A backend may already have produced a reply when cancellation arrives.
                return [99.0, 99.0], {}, [], []
        return [1.0, 2.0], {}, [], []

    server._run_compare = compare
    old = [asyncio.create_task(server.verify(member(i))) for i in range(2)]
    await asyncio.wait_for(started.wait(), 1)
    new = await asyncio.gather(*(server.verify(member(i, attempt=1)) for i in range(2)))
    old_results = await asyncio.gather(*old, return_exceptions=True)
    assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in old_results)
    assert [r.reward for r in new] == [1.0, 2.0]
    retired = next(c for c in server._verify_cohorts.values() if c.group_attempt == 0)
    assert retired.phase == "failed" and not retired.rewards


async def test_failed_legacy_group_restarts_and_ignores_late_completion(server):
    server.config.cohort_collection_timeout_s = 0.02
    first = asyncio.create_task(server.verify(member(0, group=None)))
    await asyncio.sleep(0)
    old = next(iter(server._verify_cohorts.values()))
    with pytest.raises(HTTPException, match="did not collect"):
        await first
    assert not server._verify_cohorts
    assert all(m.body is None and not m.waiters for m in old.members.values())

    server._run_single_comparison = AsyncMock(return_value=(4.0, 2.0, 1.0))
    retry = asyncio.create_task(server.verify(member(0, group=None, response_id="new-answer")))
    await asyncio.sleep(0)
    replacement = next(iter(server._verify_cohorts.values()))
    await server._publish_verify_cohort(old.key, old, {0: 99.0, 1: 99.0})
    assert replacement.phase == "collecting" and not replacement.rewards
    assert next(iter(server._verify_cohorts.values())) is replacement
    second = await server.verify(member(1, group=None))
    assert second.reward == (await retry).reward == 3.0
    assert not server._verify_cohorts
