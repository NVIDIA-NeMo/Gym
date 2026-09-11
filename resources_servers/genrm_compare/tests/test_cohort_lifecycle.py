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
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import resources_servers.genrm_compare.app as genrm
from nemo_gym.config_types import ModelServerRef
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
def config():
    return genrm.GenRMCompareConfig(
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="rlhf",
        name="genrm_compare",
        genrm_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        num_rollouts_per_prompt=2,
        cohort_timeout_s=1.0,
    )


@pytest.fixture
async def server(config):
    server = genrm.GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
    yield server
    await server.aclose()


@pytest.mark.parametrize("value", [None, 0, -1, float("inf"), float("nan")])
def test_deadline_must_be_finite_positive(config, value):
    with pytest.raises(ValidationError):
        genrm.GenRMCompareConfig.model_validate(config.model_dump() | {"cohort_timeout_s": value})


@pytest.mark.parametrize("field", ["group_attempt", "rollout_index"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_identity_counters_are_strict_integers(field, value):
    with pytest.raises(ValidationError):
        genrm.GenRMCompareVerifyRequest.model_validate(member(0).model_dump() | {field: value})


async def test_duplicate_disconnect_does_not_cancel_scoring(server, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    async def judge(**kwargs):
        started.set()
        await release.wait()
        return [0.0, 2.0], None, None, None

    judge_mock = AsyncMock(side_effect=judge)
    monkeypatch.setattr(server, "_run_compare", judge_mock)
    original = asyncio.create_task(server.verify(member(0)))
    sibling = asyncio.create_task(server.verify(member(1)))
    await started.wait()
    duplicate = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    duplicate.cancel()
    with pytest.raises(asyncio.CancelledError):
        await duplicate
    release.set()
    results = await asyncio.gather(original, sibling)
    assert [result.reward for result in results] == [0.0, 2.0]
    assert (await server.verify(member(0))).reward == 0.0
    judge_mock.assert_awaited_once()


async def test_missing_member_fails_every_waiter_without_reward(server, monkeypatch, caplog):
    server.config.cohort_timeout_s = 0.01
    judge = AsyncMock()
    monkeypatch.setattr(server, "_run_compare", judge)
    with caplog.at_level(logging.WARNING):
        results = await asyncio.wait_for(
            asyncio.gather(
                server.verify(member(0)),
                server.verify(member(0)),
                return_exceptions=True,
            ),
            timeout=1,
        )
    assert all(isinstance(result, HTTPException) and result.status_code == 503 for result in results)
    assert all("1/2 members" in result.detail for result in results)
    judge.assert_not_awaited()
    cohort = next(iter(server._verify_cohorts.values()))
    assert not cohort.rewards
    assert cohort.timeout_handle is None
    assert all(item.body is None and not item.waiters for item in cohort.members.values())
    assert sum("reason=cohort_incomplete" in record.message for record in caplog.records) == 1
    with pytest.raises(HTTPException):
        await server.verify(member(1))
    judge.assert_not_awaited()


async def test_orphaned_collecting_group_expires(server):
    server.config.cohort_timeout_s = 0.01
    request = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    cohort = next(iter(server._verify_cohorts.values()))
    assert not cohort.members[0].waiters
    await asyncio.sleep(0.03)
    assert cohort.phase == "failed"
    assert cohort.members[0].body is None
    assert cohort.timeout_handle is None


async def test_deadline_remains_active_during_judging(server, monkeypatch):
    server.config.cohort_timeout_s = 0.02
    cancelled = asyncio.Event()

    async def judge(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(server, "_run_compare", judge)
    results = await asyncio.wait_for(
        asyncio.gather(
            server.verify(member(0)),
            server.verify(member(1)),
            return_exceptions=True,
        ),
        timeout=1,
    )
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    assert all(isinstance(result, HTTPException) and "during evaluating" in result.detail for result in results)
    assert not next(iter(server._verify_cohorts.values())).rewards


@pytest.mark.parametrize("expired", [False, True])
async def test_final_arrival_and_deadline_have_one_disposition(server, monkeypatch, expired):
    judge = AsyncMock(return_value=([1.0, 2.0], None, None, None))
    monkeypatch.setattr(server, "_run_compare", judge)
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    cohort = next(iter(server._verify_cohorts.values()))
    if expired:
        monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: cohort.deadline))
    results = await asyncio.gather(first, server.verify(member(1)), return_exceptions=True)
    if expired:
        assert all(isinstance(result, HTTPException) for result in results)
        judge.assert_not_awaited()
        assert cohort.phase == "failed"
    else:
        assert [result.reward for result in results] == [1.0, 2.0]
        # A previously scheduled timeout callback cannot change a completed result.
        monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: cohort.deadline))
        assert not server._expire_verify_cohort(cohort)
        assert (await server.verify(member(0))).reward == 1.0
    assert cohort.timeout_handle is None
    assert all(not item.waiters and item.body is None for item in cohort.members.values())


async def test_late_judge_publication_checks_deadline_even_before_timer_runs(server, monkeypatch):
    async def judge(**kwargs):
        cohort = next(iter(server._verify_cohorts.values()))
        monkeypatch.setattr(genrm, "time", SimpleNamespace(monotonic=lambda: cohort.deadline))
        return [1.0, 2.0], None, None, None

    monkeypatch.setattr(server, "_run_compare", judge)
    results = await asyncio.gather(server.verify(member(0)), server.verify(member(1)), return_exceptions=True)
    assert all(isinstance(result, HTTPException) for result in results)
    assert not next(iter(server._verify_cohorts.values())).rewards


async def test_old_judge_cannot_publish_after_replacement(server, monkeypatch):
    started, old_finished = asyncio.Event(), asyncio.Event()

    async def judge(**kwargs):
        if kwargs["response_objs"][0]["id"] == "old":
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                old_finished.set()
                return [99.0, 99.0], None, None, None
        return [1.0, 2.0], None, None, None

    monkeypatch.setattr(server, "_run_compare", judge)
    old = [
        asyncio.create_task(server.verify(member(0, response_id="old"))),
        asyncio.create_task(server.verify(member(1))),
    ]
    await started.wait()
    new = await asyncio.gather(server.verify(member(0, attempt=1)), server.verify(member(1, attempt=1)))
    await old_finished.wait()
    assert [result.reward for result in new] == [1.0, 2.0]
    assert all(isinstance(result, HTTPException) for result in await asyncio.gather(*old, return_exceptions=True))
    assert (await server.verify(member(0, attempt=1))).reward == 1.0
    assert all(not cohort.rewards for cohort in server._verify_cohorts.values() if cohort.group_attempt == 0)


async def test_concurrent_groups_do_not_wake_each_other(server, monkeypatch):
    judge = AsyncMock(return_value=([1.0, 2.0], None, None, None))
    monkeypatch.setattr(server, "_run_compare", judge)
    waiting = asyncio.create_task(server.verify(member(0, group="incomplete")))
    await asyncio.sleep(0)
    complete = await asyncio.gather(
        server.verify(member(0, group="complete")), server.verify(member(1, group="complete"))
    )
    assert [result.reward for result in complete] == [1.0, 2.0]
    assert not waiting.done()
    await server.aclose()
    with pytest.raises(HTTPException):
        await waiting


async def test_judge_failure_and_replacement_group(server, monkeypatch):
    judge = AsyncMock(side_effect=[ValueError("aggregation failed"), ([1.0, 2.0], None, None, None)])
    monkeypatch.setattr(server, "_run_compare", judge)
    failed = await asyncio.gather(server.verify(member(0)), server.verify(member(1)), return_exceptions=True)
    assert all(isinstance(result, HTTPException) and "aggregation failed" in result.detail for result in failed)
    with pytest.raises(HTTPException):
        await server.verify(member(0))
    new = await asyncio.gather(server.verify(member(0, attempt=1)), server.verify(member(1, attempt=1)))
    assert [result.reward for result in new] == [1.0, 2.0]
    assert judge.await_count == 2


@pytest.mark.parametrize("rewards", [[1.0], [1.0, float("nan")], [float("inf"), 2.0]])
async def test_invalid_rewards_fail_entire_group(server, monkeypatch, rewards):
    monkeypatch.setattr(server, "_run_compare", AsyncMock(return_value=(rewards, None, None, None)))
    results = await asyncio.gather(server.verify(member(0)), server.verify(member(1)), return_exceptions=True)
    assert all(isinstance(result, HTTPException) for result in results)
    assert not next(iter(server._verify_cohorts.values())).rewards


async def test_comparison_failure_cancels_other_comparisons(server, monkeypatch):
    other_started, other_cancelled = asyncio.Event(), asyncio.Event()

    async def compare(*args, pair_idx, **kwargs):
        if pair_idx == (0, 1):
            await other_started.wait()
            raise ValueError("comparison failed")
        other_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            other_cancelled.set()

    monkeypatch.setattr(server, "_run_single_comparison", compare)
    # Three circular pairs ensure a sibling call is already in progress.
    with pytest.raises(ValueError, match="comparison failed"):
        await server._run_compare([], [{}, {}, {}])
    assert other_cancelled.is_set()


async def test_shutdown_settles_waiters_and_drains_judge_tasks(server, monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def judge(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(server, "_run_compare", judge)
    waiters = [asyncio.create_task(server.verify(member(i))) for i in range(2)]
    await started.wait()
    await server.aclose()
    assert cancelled.is_set()
    assert all(isinstance(result, HTTPException) for result in await asyncio.gather(*waiters, return_exceptions=True))
    assert not server._verify_cohorts and not server._latest_group_attempts and not server._cohort_tasks
    with pytest.raises(HTTPException, match="shutting down"):
        await server.verify(member(0, group="later"))


async def test_eviction_of_old_attempt_preserves_new_attempt_fence(server, monkeypatch):
    server.config.max_terminal_cohorts = 1
    monkeypatch.setattr(server, "_run_compare", AsyncMock(return_value=([1.0, 2.0], None, None, None)))
    for attempt in range(2):
        await asyncio.gather(server.verify(member(0, attempt=attempt)), server.verify(member(1, attempt=attempt)))
    assert len(server._verify_cohorts) == 1
    with pytest.raises(HTTPException, match="superseded"):
        await server.verify(member(0, attempt=0))


def test_http_timeout_returns_failure_without_reward(config):
    from fastapi.testclient import TestClient

    config.cohort_timeout_s = 0.01
    server = genrm.GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/verify", json=member(0).model_dump(mode="json", by_alias=True))
        assert response.status_code == 503
        assert "deadline exceeded" in response.json()["detail"]
        assert "reward" not in response.json()
        assert "response" not in response.json()
    assert not server._verify_cohorts and not server._cohort_tasks


def test_http_complete_group_echoes_coordinates_and_caches_results(config, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    server = genrm.GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
    judge = AsyncMock(return_value=([1.0, 2.0], None, None, None))
    monkeypatch.setattr(server, "_run_compare", judge)
    with TestClient(server.setup_webserver()) as client, ThreadPoolExecutor(max_workers=2) as pool:

        def send(index):
            return client.post("/verify", json=member(index, attempt=3).model_dump(mode="json", by_alias=True))

        responses = list(pool.map(send, (1, 0)))
        assert [response.status_code for response in responses] == [200, 200]
        for response, index in zip(responses, (1, 0)):
            result = response.json()
            assert result["_ng_group_id"] == "group"
            assert result["_ng_group_attempt"] == 3
            assert result["_ng_rollout_index"] == index
            assert result["reward"] == [1.0, 2.0][index]
        assert send(0).json()["reward"] == 1.0
    judge.assert_awaited_once()
    assert not server._verify_cohorts


async def test_evaluation_cancelled_before_first_step_releases_waiters(server, monkeypatch):
    judge = AsyncMock()
    monkeypatch.setattr(server, "_run_compare", judge)
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    second = asyncio.create_task(server.verify(member(1)))
    await asyncio.sleep(0)
    cohort = next(iter(server._verify_cohorts.values()))
    cohort.evaluation_task.cancel()
    results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), timeout=1)
    assert all(isinstance(result, HTTPException) and "cancelled" in result.detail for result in results)
    judge.assert_not_awaited()
    assert cohort.timeout_handle is None


async def test_legacy_task_attempts_are_fenced(server, monkeypatch):
    monkeypatch.setattr(server, "_run_compare", AsyncMock(return_value=([1.0, 2.0], None, None, None)))

    def legacy(index, attempt):
        return member(index, attempt=attempt).model_copy(update={"group_id": None, "task_index": 42})

    previous = asyncio.create_task(server.verify(legacy(0, 0)))
    await asyncio.sleep(0)
    current = await asyncio.gather(server.verify(legacy(0, 1)), server.verify(legacy(1, 1)))
    assert [result.reward for result in current] == [1.0, 2.0]
    with pytest.raises(HTTPException, match="superseded"):
        await previous
    with pytest.raises(HTTPException, match="superseded"):
        await server.verify(legacy(1, 0))


async def test_single_rollout_verification_preserves_default_score(server):
    server.config.num_rollouts_per_prompt = 1
    result = await server.verify(member(0))
    assert result.reward == server.config.default_score
    assert not server._verify_cohorts


async def test_batch_compare_preserves_pair_metadata_and_rewards(server, monkeypatch):
    server.config.debug_logging = True
    compare = AsyncMock(return_value=(4.0, 2.0, 1.0))
    monkeypatch.setattr(server, "_run_single_comparison", compare)
    result = await server.compare(
        genrm.GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "question"}],
            response_objs=[member(0).response.model_dump(), member(1).response.model_dump()],
        )
    )
    assert result.rewards == [3.0, 3.0]
    assert [(item["response_i"], item["response_j"]) for item in result.comparison_results] == [(0, 1), (1, 0)]
    assert all(item["score_1"] == 4.0 and item["score_2"] == 2.0 for item in result.comparison_results)
    assert compare.await_count == 2


@pytest.mark.parametrize("recovers", [False, True])
async def test_judge_parse_retries_preserve_existing_scoring_policy(server, recovers):
    server.config.genrm_parse_retries = 1
    server.config.genrm_parse_retry_sleep_s = 0

    def output(text):
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    response = AsyncMock()
    response.json.side_effect = [
        output("invalid"),
        output('{"score_1":4,"score_2":2,"ranking":1}' if recovers else "invalid"),
    ]
    server.server_client.post = AsyncMock(return_value=response)
    result = await server._run_single_comparison([], member(0).response.model_dump(), member(1).response.model_dump())
    assert result == (
        (4.0, 2.0, 1.0)
        if recovers
        else (server.config.default_score, server.config.default_score, server.config.default_ranking)
    )
    assert server.server_client.post.await_count == 2


def test_group_verification_rejects_multiple_http_workers(config):
    with pytest.raises(ValidationError, match="one HTTP worker"):
        genrm.GenRMCompareConfig.model_validate(config.model_dump() | {"num_workers": 2})
