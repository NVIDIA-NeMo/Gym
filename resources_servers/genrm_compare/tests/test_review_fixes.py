# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi import HTTPException
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

import resources_servers.genrm_compare.app as genrm
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import RewardProfiler
from nemo_gym.rollout_correlation import current_rollout_id, rollout_context
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import config, member, server  # noqa: F401


def failing_judge(status):
    response = MagicMock(ok=False)
    response.content.read = AsyncMock(return_value=b'{"error":"judge offline"}')
    url = URL("http://judge/v1/responses")
    response.request_info = RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url)
    response.raise_for_status.side_effect = ClientResponseError(response.request_info, (), status=status)
    response.json = AsyncMock()
    return response


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_judge_http_failure_never_completes_cohort(server, status):  # noqa: F811
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
async def test_judge_transport_failure_never_defaults(server, error):  # noqa: F811
    server.server_client.post = AsyncMock(side_effect=error)
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(result, HTTPException) for result in results)
    assert not next(iter(server._verify_cohorts.values())).rewards


def test_collector_opt_in_returns_tagged_saved_answers_on_judge_failure(config):  # noqa: F811
    instance = genrm.GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
    instance.server_client.post = AsyncMock(return_value=failing_judge(500))
    with TestClient(instance.setup_webserver()) as client, ThreadPoolExecutor(max_workers=2) as pool:

        def send(i):
            body = member(i).model_dump(mode="json", by_alias=True)
            body["_ng_cohort_failure_mode"] = "row"
            return client.post("/verify", json=body)

        results = list(pool.map(send, range(2)))
        assert all(result.status_code == 200 for result in results)
        for i, result in enumerate(results):
            data = result.json()
            assert data["_ng_failure_class"] == "judge_failed"
            assert data["_ng_failure_kind"] == "judge_failed"
            assert data["response"]["id"] == member(i).response.id
            assert data["_ng_group_id"] == "group" and data["_ng_group_attempt"] == 0
        assert not next(iter(instance._verify_cohorts.values())).rewards


def test_collector_timeout_reports_cohort_incomplete(config):  # noqa: F811
    config.cohort_timeout_s = 0.01
    instance = genrm.GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
    with TestClient(instance.setup_webserver()) as client:
        body = member(0).model_dump(mode="json", by_alias=True) | {"_ng_cohort_failure_mode": "row"}
        result = client.post("/verify", json=body)
        assert result.status_code == 200
        assert result.json()["_ng_failure_kind"] == "cohort_incomplete"
        assert result.json()["_ng_failure_class"] == "judge_failed"
        assert result.json()["response"]["id"] == member(0).response.id


async def test_global_rollout_indices_and_local_slots_are_separate(server):  # noqa: F811
    server._run_compare = AsyncMock(return_value=([1.0, 2.0], None, None, None))
    rows = [member(16 + i).model_copy(update={"group_member_index": i}) for i in range(2)]
    results = await asyncio.gather(*(server.verify(row) for row in reversed(rows)))
    assert [r.rollout_index for r in results] == [17, 16]
    assert [r.reward for r in results] == [2.0, 1.0]
    for result in results:
        data = result.model_dump(by_alias=True) | {"_ng_task_index": 0}
        metrics = RewardProfiler().rollout_info_from_result(data)
        assert "_ng_group_member_index" not in metrics and "_ng_group_attempt" not in metrics


async def test_prompt_id_with_member_slots_remains_supported(server):  # noqa: F811
    server._run_compare = AsyncMock(return_value=([1.0, 2.0], None, None, None))
    rows = [member(i).model_copy(update={"group_id": None, "prompt_id": "legacy-prompt"}) for i in range(2)]
    results = await asyncio.gather(*(server.verify(row) for row in rows))
    assert [result.reward for result in results] == [1.0, 2.0]


@pytest.mark.parametrize(
    "input_value", ["Hello", [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]]
)
async def test_actual_judge_receives_normalized_prompt(server, input_value):  # noqa: F811
    server._run_compare = AsyncMock(return_value=([1.0, 2.0], None, None, None))
    params = NeMoGymResponseCreateParamsNonStreaming(input=input_value)
    await asyncio.gather(
        *(server.verify(member(i).model_copy(update={"responses_create_params": params})) for i in range(2))
    )
    assert server._run_compare.await_args.kwargs["conversation_history"] == [{"role": "user", "content": "Hello"}]


async def test_shared_judging_has_no_member_capture_context(server):  # noqa: F811
    seen = []

    async def judge(**kwargs):
        seen.append(current_rollout_id())
        return [1.0, 2.0], None, None, None

    server._run_compare = AsyncMock(side_effect=judge)
    with rollout_context("member-a"):
        first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    with rollout_context("member-b"):
        second = asyncio.create_task(server.verify(member(1)))
    await asyncio.gather(first, second)
    assert seen == [None]


def test_migration_logging_is_bounded_and_not_a_python_warning(caplog):
    genrm._warn_legacy_attempt.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i in range(20):
            body = member(0, group=f"group-{i}").model_dump(by_alias=True)
            del body["_ng_group_attempt"]
            genrm.GenRMCompareVerifyRequest.model_validate(body)
    assert sum("GenRM group attempt omitted" in r.message for r in caplog.records) == 1
