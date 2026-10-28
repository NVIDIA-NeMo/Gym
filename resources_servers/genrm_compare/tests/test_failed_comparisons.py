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

"""A failed judge comparison is dropped; the group fails only when a rollout has no comparison left."""

import asyncio

import pytest
from fastapi import HTTPException
from pytest import approx

import resources_servers.genrm_compare.app as genrm
from resources_servers.genrm_compare.tests.support import request_with_session
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import comparison_result, member


# One distinct score per rollout; ranking 3.5 with unequal scores skips the tiebreaker, so each
# rollout's reward is its score whatever subset of its comparisons survives.
SCORES = [1.0 + 0.25 * index for index in range(16)]
NO_VERDICT = "Judge returned no completed answer after 2 attempts"


def judge_failing(failing_pairs, *, error=None, blocked=None, cancelled=None):
    """Judge stub: listed pairs fail; ``blocked`` pairs wait until cancelled; the rest score with SCORES."""

    async def judge(_conversation, _response_1, _response_2, *, pair_idx, **_kwargs):
        if pair_idx in failing_pairs:
            raise error or genrm.JudgeError(NO_VERDICT)
        if blocked is not None and pair_idx in blocked:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(pair_idx)
        i, j = pair_idx
        return comparison_result(SCORES[i], SCORES[j], 3.5)

    return judge


def response_obj(text):
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


async def verify_cohort(server, size):
    server.config.num_rollouts_per_prompt = size
    return await asyncio.gather(
        *(server.verify(request_with_session(), member(i)) for i in range(size)), return_exceptions=True
    )


async def test_one_failed_comparison_scores_every_rollout(server, caplog):
    server._run_single_comparison = judge_failing({(5, 6)})
    with caplog.at_level("WARNING"):
        results = await verify_cohort(server, 16)
    # Rollouts 5 and 6 lost (5, 6) and keep one comparison each; nobody is masked.
    assert [r.reward for r in results] == approx(SCORES)
    assert not any(r.mask_sample for r in results)
    assert all(r.genrm_failed_comparison_rate_per_group == 1 / 16 for r in results)
    assert all(r.genrm_parse_failure_rate_per_group == 0.0 for r in results)
    assert "dropped comparison (5, 6)" in caplog.text and NO_VERDICT in caplog.text


async def test_non_adjacent_failures_score_every_rollout(server):
    server._run_single_comparison = judge_failing({(1, 2), (6, 7)})
    results = await verify_cohort(server, 16)
    assert [r.reward for r in results] == approx(SCORES)
    assert not any(r.mask_sample for r in results)
    assert all(r.genrm_failed_comparison_rate_per_group == 2 / 16 for r in results)


async def test_adjacent_failures_fail_the_group_without_waiting(server):
    cancelled = set()
    # Both of rollout 1's comparisons fail; (2, 3) and (3, 0) would never finish on their own.
    server._run_single_comparison = judge_failing({(0, 1), (1, 2)}, blocked={(2, 3), (3, 0)}, cancelled=cancelled)
    results = await asyncio.wait_for(verify_cohort(server, 4), 1)
    assert all(isinstance(r, genrm.JudgeError) for r in results)
    assert all("rollout_index=1 has no successful comparison" in str(r) and NO_VERDICT in str(r) for r in results)
    assert cancelled == {(2, 3), (3, 0)}
    (cohort,) = server._verify_cohorts.values()
    assert cohort.phase == "failed" and cohort.failure_kind == "judge"


async def test_every_failed_comparison_fails_the_group(server):
    server._run_single_comparison = judge_failing({(0, 1), (1, 2), (2, 3), (3, 0)})
    results = await verify_cohort(server, 4)
    assert all(isinstance(r, genrm.JudgeError) and NO_VERDICT in str(r) for r in results)


async def test_non_judge_errors_fail_the_whole_group(server):
    server._run_single_comparison = judge_failing({(1, 2)}, error=ValueError("bad comparison"))
    results = await verify_cohort(server, 4)
    assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in results)
    assert all("ValueError: bad comparison" in r.detail for r in results)


async def test_compare_endpoint_keeps_results_aligned_with_pairs(server):
    server._run_single_comparison = judge_failing({(1, 2)})
    request = genrm.GenRMCompareRequest(
        conversation_history=[], response_objs=[response_obj(str(i)) for i in range(4)]
    )
    response = await server.compare(request)
    assert response.rewards == approx(SCORES[:4])
    assert [(r["response_i"], r["response_j"]) for r in response.comparison_results] == [(0, 1), (2, 3), (3, 0)]
    expected_scores = [(SCORES[0], SCORES[1]), (SCORES[2], SCORES[3]), (SCORES[3], SCORES[0])]
    assert [(r["score_1"], r["score_2"]) for r in response.comparison_results] == expected_scores
    assert response.metrics["genrm_failed_comparison_rate_per_group"] == 0.25


async def test_compare_endpoint_fails_when_a_response_keeps_no_comparison(server):
    server._run_single_comparison = judge_failing({(0, 1), (1, 2)})
    request = genrm.GenRMCompareRequest(
        conversation_history=[], response_objs=[response_obj(str(i)) for i in range(4)]
    )
    with pytest.raises(HTTPException) as raised:
        await server.compare(request)
    assert raised.value.status_code == 503 and "rollout_index=1 has no successful comparison" in raised.value.detail


@pytest.mark.parametrize("failing_passes", [{0}, {0, 1}])
async def test_fixed_baseline_drops_a_failed_judge_pass(server, failing_passes):
    server.config.comparison_mode = "fixed_baseline"
    server.config.num_judges_per_comparison = 2
    passes = {}

    async def judge(_conversation, _response_1, _response_2, *, pair_idx, **_kwargs):
        rollout = pair_idx[0]
        attempt = passes[rollout] = passes.get(rollout, -1) + 1
        if rollout == 0 and attempt in failing_passes:
            raise genrm.JudgeError(NO_VERDICT)
        return comparison_result(5.0, 5.0, 3.5)

    server._run_single_comparison = judge
    response_objs = [response_obj("first"), response_obj("second")]
    if failing_passes == {0, 1}:
        with pytest.raises(genrm.JudgeError, match="rollout_index=0 has no successful comparison"):
            await server._run_fixed_baseline_compare([], response_objs, "baseline", None, "prompt")
        return
    rewards, *_, metrics, _, _, _, _ = await server._run_fixed_baseline_compare(
        [], response_objs, "baseline", None, "prompt"
    )
    assert rewards == approx([5.0, 5.0])
    assert metrics["genrm_failed_comparison_rate_per_group"] == 0.25
