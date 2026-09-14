# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi import HTTPException
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

import resources_servers.genrm_compare.app as genrm
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import RewardProfiler
from nemo_gym.rollout_correlation import current_rollout_id, rollout_context
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import member


def failing_judge(status):
    response = MagicMock(ok=False)
    response.content.read = AsyncMock(return_value=b'{"error":"judge offline"}')
    url = URL("http://judge/v1/responses")
    response.request_info = RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url)
    response.raise_for_status.side_effect = ClientResponseError(response.request_info, (), status=status)
    response.json = AsyncMock()
    return response


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_judge_http_failure_never_completes_cohort(server, status):
    response = failing_judge(status)
    server.server_client.post = AsyncMock(return_value=response)
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(result, HTTPException) and result.status_code == 503 for result in results)
    cohort = next(iter(server._verify_cohorts.values()))
    assert cohort.phase == "failed" and not cohort.rewards
    response.json.assert_not_awaited()
    with pytest.raises(HTTPException):
        await server.verify(member(0))
    assert server.server_client.post.await_count <= 2  # no parse retries on failed HTTP


@pytest.mark.parametrize("error", [ConnectionError("offline"), TimeoutError("judge timeout")])
async def test_judge_transport_failure_never_defaults(server, error):
    server.server_client.post = AsyncMock(side_effect=error)
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(result, HTTPException) for result in results)
    assert not next(iter(server._verify_cohorts.values())).rewards


@pytest.mark.parametrize(
    "payload", [None, {}, {"status": "incomplete"}, {"status": "failed"}, {"status": "cancelled"}]
)
async def test_unsuccessful_http_200_judge_is_failure(server, payload):
    response = MagicMock(ok=True)
    response.json = AsyncMock(return_value=payload)
    server.server_client.post = AsyncMock(return_value=response)
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in results)
    with pytest.raises(HTTPException) as error:
        await server.compare(genrm.GenRMCompareRequest(conversation_history=[], response_objs=[{}, {}]))
    assert error.value.status_code == 503


@pytest.mark.parametrize("recovers", [False, True])
async def test_nonempty_parse_retries_preserve_existing_fallback(server, recovers):
    server.config.genrm_parse_retries = 1
    server.config.genrm_parse_retry_sleep_s = 0

    def output(text):
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    response = MagicMock(ok=True)
    response.json = AsyncMock(
        side_effect=[output("invalid"), output('{"score_1":4,"score_2":2,"ranking":1}' if recovers else "invalid")]
    )
    server.server_client.post = AsyncMock(return_value=response)
    result = await server._run_single_comparison([], {}, {})
    assert result == ((4.0, 2.0, 1.0) if recovers else (3.0, 3.0, 3.5))
    assert server.server_client.post.await_count == 2


async def test_global_rollout_indices_and_local_slots_are_separate(server):
    server._run_single_comparison = AsyncMock(return_value=(4.0, 2.0, 1.0))
    rows = [member(16 + i).model_copy(update={"group_member_index": i}) for i in range(2)]
    results = await asyncio.gather(*(server.verify(row) for row in reversed(rows)))
    assert [r.rollout_index for r in results] == [17, 16]
    assert [r.reward for r in results] == [3.0, 3.0]
    for result in results:
        data = result.model_dump(by_alias=True) | {"_ng_task_index": 0}
        metrics = RewardProfiler().rollout_info_from_result(data)
        assert "_ng_group_member_index" not in metrics and "_ng_group_attempt" not in metrics


async def test_prompt_id_with_member_slots_remains_supported(server):
    server._run_single_comparison = AsyncMock(return_value=(4.0, 2.0, 1.0))
    rows = [member(i).model_copy(update={"group_id": None, "prompt_id": "legacy-prompt"}) for i in range(2)]
    results = await asyncio.gather(*(server.verify(row) for row in rows))
    assert [result.reward for result in results] == [3.0, 3.0]


@pytest.mark.parametrize(
    "input_value", ["Hello", [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]]
)
async def test_actual_judge_receives_normalized_prompt(server, input_value):
    server._run_single_comparison = AsyncMock(return_value=(4.0, 2.0, 1.0))
    params = NeMoGymResponseCreateParamsNonStreaming(input=input_value)
    await asyncio.gather(
        *(server.verify(member(i).model_copy(update={"responses_create_params": params})) for i in range(2))
    )
    assert server._run_single_comparison.await_args.args[0] == [{"role": "user", "content": "Hello"}]


async def test_shared_judging_has_no_member_capture_context(server):
    seen = []

    async def judge(*args, **kwargs):
        seen.append(current_rollout_id())
        return (4.0, 2.0, 1.0)

    server._run_single_comparison = AsyncMock(side_effect=judge)
    with rollout_context("member-a"):
        first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    with rollout_context("member-b"):
        second = asyncio.create_task(server.verify(member(1)))
    await asyncio.gather(first, second)
    assert seen == [None, None]


def test_migration_logging_is_bounded_and_not_a_python_warning(caplog):
    genrm._warn_legacy_attempt.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i in range(20):
            body = member(0, group=f"group-{i}").model_dump(by_alias=True)
            del body["_ng_group_attempt"]
            genrm.GenRMCompareVerifyRequest.model_validate(body)
    assert sum("GenRM group attempt omitted" in r.message for r in caplog.records) == 1


async def test_batch_compare_returns_pair_metadata_and_cancels_failed_siblings(server):
    server._run_single_comparison = AsyncMock(return_value=(4.0, 2.0, 1.0))
    server.config.debug_logging = True
    body = genrm.GenRMCompareRequest(conversation_history=[], response_objs=[{}, {}])
    result = await server.compare(body)
    assert result.rewards == [3.0, 3.0]
    assert [(p["response_i"], p["response_j"]) for p in result.comparison_results] == [(0, 1), (1, 0)]
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def comparison(*args, pair_idx, **kwargs):
        if pair_idx == (0, 1):
            await started.wait()
            raise ValueError("pair failed")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    server._run_single_comparison = comparison
    with pytest.raises(ValueError, match="pair failed"):
        await server._run_compare([], [{}, {}])
    assert cancelled.is_set()
