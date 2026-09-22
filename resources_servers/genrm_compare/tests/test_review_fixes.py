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

import asyncio
import json
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi import HTTPException
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

import resources_servers.genrm_compare.app as genrm
from nemo_gym.judge import judge_failsafe
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
    response.read = AsyncMock()
    return response


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_judge_http_failure_never_completes_cohort(server, status):
    response = failing_judge(status)
    server.server_client.post = AsyncMock(return_value=response)
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(result, genrm.JudgeError) and "judge offline" in str(result) for result in results)
    cohort = next(iter(server._verify_cohorts.values()))
    assert cohort.phase == "failed" and not cohort.rewards
    response.read.assert_not_awaited()
    with pytest.raises(genrm.JudgeError, match="judge offline"):
        await server.verify(member(0, response_id="regenerated-answer"))
    calls_after_failure = server.server_client.post.await_count
    max_calls = 2 * (1 if status == 401 else server.config.genrm_parse_retries + 1)
    assert 1 <= calls_after_failure <= max_calls
    server._run_compare = AsyncMock(return_value=([1.0, 2.0], {}, [], []))
    recovered = await asyncio.gather(*(server.verify(member(i, attempt=1)) for i in range(2)))
    assert [result.reward for result in recovered] == [1.0, 2.0]
    server._run_compare.assert_awaited_once()


@pytest.mark.parametrize("error", [ConnectionError("offline"), TimeoutError("judge timeout")])
async def test_judge_transport_failure_never_defaults(server, error):
    server.server_client.post = AsyncMock(side_effect=error)
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(result, genrm.JudgeError) for result in results)
    assert not next(iter(server._verify_cohorts.values())).rewards


@pytest.mark.parametrize(
    "payload", [None, {}, {"status": "incomplete"}, {"status": "failed"}, {"status": "cancelled"}]
)
async def test_unsuccessful_http_200_judge_is_failure(server, payload):
    response = MagicMock(ok=True)
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    server.server_client.post = AsyncMock(return_value=response)
    results = await asyncio.gather(*(server.verify(member(i)) for i in range(2)), return_exceptions=True)
    assert all(isinstance(r, genrm.JudgeError) for r in results)
    with pytest.raises(HTTPException, match="503"):
        await server.compare(genrm.GenRMCompareRequest(conversation_history=[], response_objs=[{}, {}]))


@pytest.mark.parametrize("recovers", [False, True])
async def test_nonempty_parse_retries_preserve_existing_fallback(server, recovers):
    server.config.genrm_parse_retries = 1
    server.config.genrm_parse_retry_sleep_s = 0

    def output(text):
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    response = MagicMock(ok=True)
    response.read = AsyncMock(
        side_effect=[
            json.dumps(payload).encode()
            for payload in (
                [output("invalid"), output('{"score_1":4,"score_2":2,"ranking":1}' if recovers else "invalid")]
            )
        ]
    )
    server.server_client.post = AsyncMock(return_value=response)
    result = await server._run_single_comparison([], {}, {})
    assert result == ((4.0, 2.0, 1.0) if recovers else (3.0, 3.0, 3.5))
    assert server.server_client.post.await_count == 2


async def test_group_attempt_metadata_is_not_a_reward_metric(server):
    server._run_single_comparison = AsyncMock(return_value=(4.0, 2.0, 1.0))
    rows = [member(i, attempt=2) for i in range(2)]
    results = await asyncio.gather(*(server.verify(row) for row in reversed(rows)))
    assert [r.rollout_index for r in results] == [1, 0]
    assert [r.reward for r in results] == [3.0, 3.0]
    for result in results:
        data = result.model_dump(by_alias=True) | {"_ng_task_index": 0}
        metrics = RewardProfiler().rollout_info_from_result(data)
        assert "_ng_group_attempt" not in metrics
        assert metrics["_ng_rollout_index"] == result.rollout_index


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


@pytest.mark.parametrize("first", ["empty", "incomplete"])
@pytest.mark.parametrize("recovers", [False, True])
async def test_empty_or_incomplete_judge_uses_configured_retries(server, first, recovers):
    server.config.genrm_parse_retries = 1
    valid = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": '{"score_1":4,"score_2":2,"ranking":1}'}]}
        ]
    }
    failed = {"output": []} if first == "empty" else valid | {"status": "incomplete"}
    response = MagicMock(ok=True)
    response.read = AsyncMock(
        side_effect=[json.dumps(payload).encode() for payload in ([failed, valid if recovers else failed])]
    )
    server.server_client.post = AsyncMock(return_value=response)
    if recovers:
        assert await server._run_single_comparison([], {}, {}) == (4, 2, 1)
    else:
        with pytest.raises(genrm.JudgeError, match="after 2 attempts"):
            await server._run_single_comparison([], {}, {})
    assert server.server_client.post.await_count == 2


def test_reasoning_is_not_a_fake_user_message():
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"type": "reasoning", "id": "r", "summary": []},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Earlier answer", "annotations": []}],
                "type": "message",
                "id": "a",
                "status": "completed",
            },
            {"role": "user", "content": [{"type": "input_text", "text": "Follow up"}]},
        ]
    )
    assert genrm._input_to_conversation_history(params.input) == [
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Follow up"},
    ]


def test_response_digest_accepts_surrogates():
    assert genrm.GenRMCompareResourcesServer._response_digest({"text": "\ud800"})


async def test_null_judge_status_accepts_parseable_answer(server):
    response = MagicMock(ok=True)
    response.read = AsyncMock(
        return_value=json.dumps(
            {
                "status": None,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"score_1":4,"score_2":2,"ranking":1}'}],
                    }
                ],
            }
        ).encode()
    )
    server.server_client.post = AsyncMock(return_value=response)
    assert await server._run_single_comparison([], {}, {}) == (4, 2, 1)
    server.server_client.post.assert_awaited_once()


@pytest.mark.parametrize("malformed_first", [False, True])
async def test_parse_fallback_does_not_depend_on_retry_order(server, malformed_first):
    server.config.genrm_parse_retries = 1
    malformed = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "malformed"}]}]}
    empty = {"output": []}
    response = MagicMock(ok=True)
    response.read = AsyncMock(
        side_effect=[
            json.dumps(payload).encode() for payload in ([malformed, empty] if malformed_first else [empty, malformed])
        ]
    )
    server.server_client.post = AsyncMock(return_value=response)
    assert await server._run_single_comparison([], {}, {}) == (3, 3, 3.5)
    assert server.server_client.post.await_count == 2


@pytest.mark.parametrize("index", [0, 1])
async def test_late_member_receives_original_timeout_failure(server, index):
    server.config.cohort_collection_timeout_s = 0.01
    server._run_single_comparison = AsyncMock()
    with pytest.raises(HTTPException) as first:
        await server.verify(member(0))
    with pytest.raises(HTTPException) as late:
        await server.verify(member(index, response_id="new-answer"))
    assert first.value.status_code == late.value.status_code == 503
    assert late.value.detail == first.value.detail
    assert "did not collect" in late.value.detail
    server._run_single_comparison.assert_not_awaited()


async def test_judge_failure_preserves_existing_instance_config_and_masks_all_members(server):
    server._run_compare = AsyncMock(side_effect=genrm.JudgeError("judge offline"))
    bodies = [member(i) for i in range(2)]
    bodies[0].instance_config = {"task": "keep", "mask_sample": False}
    results = await asyncio.gather(*(judge_failsafe(server.verify)(body) for body in bodies))
    rows = [json.loads(result.body) for result in results]
    assert rows[0]["instance_config"] == {"task": "keep", "mask_sample": True}
    assert rows[1]["instance_config"] == {"mask_sample": True}
    assert all(row["mask_sample"] is True and row["failure_kind"] == "judge_failed" for row in rows)
    assert bodies[0].instance_config == {"task": "keep", "mask_sample": False}


async def test_empty_batch_has_no_rewards(server):
    result = await server.compare(genrm.GenRMCompareRequest(conversation_history=[], response_objs=[]))
    assert result.rewards == []


async def test_null_text_is_retried_as_unusable_judge_output(server):
    server.config.genrm_parse_retries = 1
    response = MagicMock(ok=True)
    response.read = AsyncMock(
        return_value=json.dumps(
            {
                "output": [
                    {"type": "reasoning", "summary": [{"text": None}]},
                    {"type": "message", "content": [{"type": "output_text", "text": None}]},
                ]
            }
        ).encode()
    )
    server.server_client.post = AsyncMock(return_value=response)
    with pytest.raises(genrm.JudgeError, match="no completed answer after 2 attempts"):
        await server._run_single_comparison([], {}, {})
    assert server.server_client.post.await_count == 2


@pytest.mark.parametrize("status", [408, 429, 500, 503, 599])
async def test_transient_judge_http_error_retries_within_existing_budget(server, status):
    valid = MagicMock(ok=True)
    valid.read = AsyncMock(
        return_value=json.dumps(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"score_1":4,"score_2":2,"ranking":1}'}],
                    }
                ]
            }
        ).encode()
    )
    server.server_client.post = AsyncMock(side_effect=[failing_judge(status), valid])
    assert await server._run_single_comparison([], {}, {}) == (4, 2, 1)
    assert server.server_client.post.await_count == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 429, 500, 503])
async def test_http_retry_budget_exhaustion_never_becomes_default_score(server, status):
    server.config.genrm_parse_retries = 2
    server.server_client.post = AsyncMock(return_value=failing_judge(status))
    with pytest.raises(genrm.JudgeError, match="judge offline"):
        await server._run_single_comparison([], {}, {})
    assert server.server_client.post.await_count == (3 if status in (408, 429, 500, 503) else 1)


async def test_parse_and_http_failures_share_one_budget(server):
    server.config.genrm_parse_retries = 1
    malformed = MagicMock(ok=True)
    malformed.read = AsyncMock(
        return_value=json.dumps(
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "invalid"}]}]}
        ).encode()
    )
    server.server_client.post = AsyncMock(side_effect=[malformed, failing_judge(503)])
    with pytest.raises(genrm.JudgeError, match="judge offline"):
        await server._run_single_comparison([], {}, {})
    assert server.server_client.post.await_count == 2
