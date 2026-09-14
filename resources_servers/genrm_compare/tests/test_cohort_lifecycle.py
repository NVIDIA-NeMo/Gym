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
import logging
from types import SimpleNamespace
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


@pytest.fixture
def judging(server, monkeypatch):
    judge = AsyncMock(return_value=(4.0, 2.0, 1.0))
    monkeypatch.setattr(server, "_run_single_comparison", judge)
    return judge


@pytest.mark.parametrize("value", [None, 0, -1, float("inf"), float("nan")])
def test_deadline_must_be_finite_positive(config, value):
    with pytest.raises(ValidationError):
        genrm.GenRMCompareConfig.model_validate(config.model_dump() | {"cohort_timeout_s": value})


@pytest.mark.parametrize("field", ["group_attempt", "rollout_index", "group_member_index"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_identity_counters_are_strict_integers(field, value):
    with pytest.raises(ValidationError):
        genrm.GenRMCompareVerifyRequest.model_validate(member(0).model_dump() | {field: value})


async def test_pairwise_judging_overlaps_generation_and_waits_for_all(server, judging):
    server.config.num_rollouts_per_prompt = 4
    server.config.comparison_strategy = "all_pairs"
    release, started = asyncio.Event(), asyncio.Event()

    async def compare(*args, **kwargs):
        started.set()
        await release.wait()
        return (4.0, 2.0, 1.0)

    judging.side_effect = compare
    waiters = [asyncio.create_task(server.verify(member(i))) for i in (1, 0)]
    await started.wait()
    assert judging.await_count == 1
    assert all(not waiter.done() for waiter in waiters)
    waiters += [asyncio.create_task(server.verify(member(i))) for i in (3, 2)]
    await asyncio.sleep(0)
    assert all(not waiter.done() for waiter in waiters)
    release.set()
    results = await asyncio.gather(*waiters)
    assert [r.reward for r in results] == [10 / 3, 4.0, 2.0, 8 / 3]
    assert judging.await_count == 6
    assert len({call.kwargs["pair_idx"] for call in judging.await_args_list}) == 6
    cohort = next(iter(server._verify_cohorts.values()))
    assert cohort.phase == "completed"
    assert not cohort.members and not cohort.rewards and not cohort.comparisons
    with pytest.raises(HTTPException, match="complete") as error:
        await server.verify(member(0))
    assert error.value.status_code == 409


async def test_duplicate_disconnect_keeps_attached_original(server, judging):
    started, release = asyncio.Event(), asyncio.Event()

    async def judge(*args, **kwargs):
        started.set()
        await release.wait()
        return (4.0, 2.0, 1.0)

    judging.side_effect = judge
    original = [asyncio.create_task(server.verify(member(i))) for i in range(2)]
    await started.wait()
    duplicate = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    with pytest.raises(HTTPException) as error:
        await server.verify(member(0, response_id="different"))
    assert error.value.status_code == 409
    duplicate.cancel()
    with pytest.raises(asyncio.CancelledError):
        await duplicate
    release.set()
    assert [r.reward for r in await asyncio.gather(*original)] == [3.0, 3.0]
    assert judging.await_count == 2


@pytest.mark.parametrize("during_judging", [False, True])
async def test_last_waiter_disconnect_fails_group_and_allows_new_attempt(server, judging, during_judging):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def judge(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    judging.side_effect = judge
    server.config.num_rollouts_per_prompt = 3
    waiters = [asyncio.create_task(server.verify(member(i))) for i in range(2 if during_judging else 1)]
    if during_judging:
        await started.wait()
    else:
        await asyncio.sleep(0)
    waiters[0].cancel()
    results = await asyncio.gather(*waiters, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in results[1:])
    if during_judging:
        await cancelled.wait()
    with pytest.raises(HTTPException, match="last waiter"):
        await server.verify(member(0, response_id="regenerated"))
    judging.side_effect = None
    current = await asyncio.gather(*(server.verify(member(i, attempt=1)) for i in range(3)))
    assert len(current) == 3 and all(r.reward == 3.0 for r in current)


async def test_missing_member_fails_all_duplicates_with_one_bounded_event(server, judging, caplog):
    server.config.cohort_timeout_s = 0.01
    with caplog.at_level(logging.WARNING):
        results = await asyncio.gather(server.verify(member(0)), server.verify(member(0)), return_exceptions=True)
    assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in results)
    judging.assert_not_awaited()
    cohort = next(iter(server._verify_cohorts.values()))
    assert cohort.phase == "failed" and not cohort.members and cohort.timeout_handle is None
    events = [r.message for r in caplog.records if "GenRM cohort disposition=" in r.message]
    assert events == ["GenRM cohort disposition=failed reason=cohort_incomplete arrived=1 expected=2"]


@pytest.mark.parametrize("expired", [False, True])
async def test_final_arrival_deadline_race(server, judging, monkeypatch, expired):
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    cohort = next(iter(server._verify_cohorts.values()))
    if expired:
        monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: cohort.deadline))
    results = await asyncio.gather(first, server.verify(member(1)), return_exceptions=True)
    if expired:
        assert all(isinstance(r, HTTPException) for r in results)
        judging.assert_not_awaited()
    else:
        assert [r.reward for r in results] == [3.0, 3.0]
        monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: cohort.deadline))
        assert not server._expire_verify_cohort(cohort)
    assert not cohort.members and cohort.timeout_handle is None


async def test_publication_after_deadline_fails_even_before_timer(server, judging, monkeypatch):
    async def judge(*args, **kwargs):
        cohort = next(iter(server._verify_cohorts.values()))
        monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: cohort.deadline))
        return (4.0, 2.0, 1.0)

    judging.side_effect = judge
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(r, HTTPException) for r in results)
    assert not next(iter(server._verify_cohorts.values())).rewards


async def test_late_old_judge_cannot_publish_after_replacement_and_eviction(server, judging):
    started, finished = asyncio.Event(), asyncio.Event()
    server.config.max_terminal_cohorts = 1

    async def judge(history, first, second, **kwargs):
        if first["id"] == "old":
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                finished.set()
                return (99.0, 99.0, 1.0)
        return (4.0, 2.0, 1.0)

    judging.side_effect = judge
    old = [asyncio.create_task(server.verify(member(i, response_id="old"))) for i in range(2)]
    await started.wait()
    previous = next(iter(server._verify_cohorts.values()))
    new = await asyncio.gather(*(server.verify(member(i, attempt=1)) for i in range(2)))
    await finished.wait()
    assert [r.reward for r in new] == [3.0, 3.0]
    assert all(isinstance(r, HTTPException) for r in await asyncio.gather(*old, return_exceptions=True))
    assert previous.phase == "failed" and not previous.rewards
    assert len(server._verify_cohorts) == 1
    with pytest.raises(HTTPException, match="superseded"):
        await server.verify(member(0))


async def test_concurrent_groups_and_prompt_mismatch(server, judging):
    waiting = asyncio.create_task(server.verify(member(0, group="other")))
    await asyncio.sleep(0)
    changed = member(1, group="other", attempt=1)
    changed.responses_create_params.input = "different prompt"
    with pytest.raises(HTTPException, match="inconsistent"):
        await server.verify(changed)
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)))
    assert [r.reward for r in results] == [3.0, 3.0]
    assert not waiting.done()
    await server.aclose()
    with pytest.raises(HTTPException, match="shutting down"):
        await waiting


@pytest.mark.parametrize("rewards", [[1.0], [1.0, float("nan")], [float("inf"), 2.0]])
async def test_invalid_aggregation_never_publishes_rewards(server, judging, monkeypatch, rewards):
    monkeypatch.setattr(genrm, "aggregate_scores", lambda **kwargs: (rewards, {}, [], []))
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in results)


async def test_early_judge_failure_releases_incomplete_group(server, judging):
    server.config.num_rollouts_per_prompt = 4
    judging.side_effect = ValueError("judge failed")
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(r, HTTPException) and "judge failed" in r.detail for r in results)
    assert not server._cohort_tasks


async def test_judge_timeout_cancels_pairs_and_releases_group(server, judging):
    server.config.cohort_timeout_s = 0.02
    cancelled = asyncio.Event()

    async def judge(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    judging.side_effect = judge
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    await cancelled.wait()
    assert all(isinstance(r, HTTPException) and "evaluating" in r.detail for r in results)


async def test_shutdown_drains_judging_and_settles_waiters(server, judging):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def judge(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    judging.side_effect = judge
    waiters = [asyncio.create_task(server.verify(member(i))) for i in range(2)]
    await started.wait()
    await server.aclose()
    assert cancelled.is_set()
    assert all(isinstance(r, HTTPException) for r in await asyncio.gather(*waiters, return_exceptions=True))
    assert not server._cohort_tasks and not server._verify_cohorts
    with pytest.raises(HTTPException, match="shutting down"):
        await server.verify(member(0))


async def test_cancel_before_evaluation_first_step_releases_waiters(server, judging):
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    second = asyncio.create_task(server.verify(member(1)))
    await asyncio.sleep(0)
    cohort = next(iter(server._verify_cohorts.values()))
    cohort.evaluation_task.cancel()
    results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 1)
    assert all(isinstance(r, HTTPException) for r in results)
    assert cohort.timeout_handle is None


async def test_terminal_retention_is_bounded_by_count_and_ttl(server, judging, monkeypatch):
    server.config.max_terminal_cohorts = 2
    for group in ("a", "b", "c"):
        await asyncio.gather(*(server.verify(member(i, group=group)) for i in range(2)))
    assert len(server._verify_cohorts) == len(server._latest_group_attempts) == 2
    assert all(not c.members and not c.rewards for c in server._verify_cohorts.values())
    now = genrm.time.monotonic() + server.config.cohort_result_ttl_s
    monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: now))
    server._prune_terminal_cohorts()
    assert not server._verify_cohorts and not server._latest_group_attempts and not server._group_cohort_counts


async def test_single_member_preserves_default_score(server, judging):
    server.config.num_rollouts_per_prompt = 1
    assert (await server.verify(member(0))).reward == server.config.default_score
    judging.assert_not_awaited()


def test_group_verification_rejects_multiple_workers(config):
    with pytest.raises(ValidationError, match="one HTTP worker"):
        genrm.GenRMCompareConfig.model_validate(config.model_dump() | {"num_workers": 2})
