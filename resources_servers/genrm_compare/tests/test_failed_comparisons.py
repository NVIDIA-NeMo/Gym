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

"""A failed judge comparison masks its two rollouts; every other rollout is scored from all its comparisons."""

import asyncio

import pytest
from fastapi import HTTPException
from pytest import approx

import resources_servers.genrm_compare.app as genrm
from resources_servers.genrm_compare.tests.support import request_with_session
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import comparison_result, member


# One distinct score per rollout; ranking 3.5 with unequal scores skips the tiebreaker, so each
# scored rollout's reward is its score.
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


def masked_and_scored(results):
    masked = {index for index, r in enumerate(results) if isinstance(r, genrm.JudgeError)}
    scored = {index: r for index, r in enumerate(results) if index not in masked}
    return masked, scored


async def test_one_failed_comparison_masks_its_two_rollouts(server, caplog):
    server._run_single_comparison = judge_failing({(5, 6)})
    with caplog.at_level("WARNING"):
        results = await verify_cohort(server, 16)
    masked, scored = masked_and_scored(results)
    # Rollouts 5 and 6 lost (5, 6); every other rollout keeps both of its comparisons.
    assert masked == {5, 6}
    assert all(
        "lost a comparison to a judge failure" in str(results[i]) and NO_VERDICT in str(results[i]) for i in masked
    )
    assert [r.reward for r in scored.values()] == approx([SCORES[i] for i in scored])
    assert not any(r.mask_sample for r in scored.values())
    assert all(r.genrm_failed_comparison_rate_per_group == 1 / 16 for r in scored.values())
    assert all(r.genrm_masked_rollout_rate_per_group == 2 / 16 for r in scored.values())
    assert all(r.genrm_parse_failure_rate_per_group == 0.0 for r in scored.values())
    assert "dropped comparison (5, 6)" in caplog.text and "masked one rollout" in caplog.text


async def test_neighbors_of_a_masked_rollout_keep_both_comparisons(server):
    """Rollout 4 is still scored from (3, 4) and (4, 5), although rollout 5 is masked by (5, 6)."""
    seen = []
    judge = judge_failing({(5, 6)})

    async def recording_judge(*args, pair_idx, **kwargs):
        seen.append(pair_idx)
        return await judge(*args, pair_idx=pair_idx, **kwargs)

    server._run_single_comparison = recording_judge
    results = await verify_cohort(server, 16)
    assert (4, 5) in seen and (6, 7) in seen
    assert results[4].reward == approx(SCORES[4]) and results[7].reward == approx(SCORES[7])


async def test_non_adjacent_failures_mask_their_rollouts(server):
    server._run_single_comparison = judge_failing({(1, 2), (6, 7)})
    results = await verify_cohort(server, 16)
    masked, scored = masked_and_scored(results)
    assert masked == {1, 2, 6, 7}
    assert [r.reward for r in scored.values()] == approx([SCORES[i] for i in scored])
    assert all(r.genrm_failed_comparison_rate_per_group == 2 / 16 for r in scored.values())
    assert all(r.genrm_masked_rollout_rate_per_group == 4 / 16 for r in scored.values())


async def test_masked_members_replay_without_judging_again(server):
    server._run_single_comparison = judge_failing({(0, 1), (1, 2)})
    results = await verify_cohort(server, 4)
    masked, scored = masked_and_scored(results)
    assert masked == {0, 1, 2} and scored[3].reward == approx(SCORES[3])
    assert "rollout_index=1 lost a comparison to a judge failure (2 of 4 comparisons failed)" in str(results[1])

    (cohort,) = server._verify_cohorts.values()
    assert cohort.phase == "completed" and set(cohort.member_failures) == {0, 1, 2} and set(cohort.results) == {3}
    # An exact retry replays the same outcome without judging again.
    server._run_single_comparison = judge_failing(set(), error=AssertionError("judged again"))
    with pytest.raises(genrm.JudgeError, match="rollout_index=1"):
        await server.verify(request_with_session(), member(1))
    assert (await server.verify(request_with_session(), member(3))).reward == approx(SCORES[3])


async def test_a_non_judge_error_cancels_the_comparisons_still_running(server):
    cancelled = set()
    # (2, 3) and (3, 0) would never finish on their own.
    server._run_single_comparison = judge_failing(
        {(1, 2)}, error=ValueError("bad comparison"), blocked={(2, 3), (3, 0)}, cancelled=cancelled
    )
    results = await asyncio.wait_for(verify_cohort(server, 4), 1)
    assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in results)
    assert cancelled == {(2, 3), (3, 0)}


async def test_masked_rollouts_do_not_shift_group_adjustments(server):
    """The masked rollouts never enter the length penalty that the scored rollouts share."""
    server.config.group_answer_length_penalty_coeff = 0.5
    texts = ["x" * 5000, "y" * 3000, "medium answer", "a"]
    metadata = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)]
    raw_results = [genrm.JudgeError(NO_VERDICT)] + [comparison_result(3.0, 3.0, 3.5)] * 3
    rewards, *_, masked = server._aggregate_results([response_obj(t) for t in texts], raw_results, metadata)

    alone, *_ = server._score_results(
        [response_obj(t) for t in texts[2:]], [comparison_result(3.0, 3.0, 3.5)], [(0, 1, 0)]
    )
    assert set(masked) == {0, 1}
    assert rewards[:2] == [None, None]
    assert rewards[2:] == approx(alone)
    assert rewards[2] != approx(rewards[3])


@pytest.mark.parametrize("failing_pairs", [{(0, 1)}, {(0, 1), (2, 3)}], ids=["one_failure", "every_rollout_masked"])
async def test_all_pairs_masks_the_rollouts_of_a_failed_pair(server, failing_pairs):
    # Under all_pairs each rollout has three comparisons in a group of 4, not two as under circular.
    server.config.comparison_strategy = "all_pairs"
    server._run_single_comparison = judge_failing(failing_pairs)
    results = await verify_cohort(server, 4)
    if len(failing_pairs) == 2:
        assert all(isinstance(r, genrm.JudgeError) for r in results)
        return
    masked, scored = masked_and_scored(results)
    assert masked == {0, 1}
    # Rollouts 2 and 3 keep all three comparisons, including the ones against 0 and 1.
    assert [r.reward for r in scored.values()] == approx(SCORES[2:4])
    assert all(r.genrm_failed_comparison_rate_per_group == approx(1 / 6) for r in scored.values())


async def test_every_failed_comparison_fails_the_group(server):
    server._run_single_comparison = judge_failing({(0, 1), (1, 2), (2, 3), (3, 0)})
    results = await verify_cohort(server, 4)
    assert all(isinstance(r, genrm.JudgeError) and NO_VERDICT in str(r) for r in results)
    (cohort,) = server._verify_cohorts.values()
    assert cohort.phase == "failed" and cohort.failure_kind == "judge"


async def test_non_judge_errors_fail_the_whole_group(server):
    server._run_single_comparison = judge_failing({(1, 2)}, error=ValueError("bad comparison"))
    results = await verify_cohort(server, 4)
    assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in results)
    assert all("ValueError: bad comparison" in r.detail for r in results)


async def test_compare_endpoint_fails_on_a_failed_comparison(server):
    # /compare has no per-response mask.
    server._run_single_comparison = judge_failing({(1, 2)})
    request = genrm.GenRMCompareRequest(
        conversation_history=[], response_objs=[response_obj(str(i)) for i in range(4)]
    )
    with pytest.raises(HTTPException) as raised:
        await server.compare(request)
    assert raised.value.status_code == 503 and "rollout_index=1 lost a comparison" in raised.value.detail


@pytest.mark.parametrize("failing_passes", [{0}, {0, 1}])
async def test_fixed_baseline_masks_a_rollout_with_a_failed_pass(server, failing_passes):
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
    rewards, *_, metrics, _, _, _, _, masked = await server._run_fixed_baseline_compare(
        [], [response_obj("first"), response_obj("second")], "baseline", None, "prompt"
    )
    assert rewards[0] is None and rewards[1] == approx(5.0)
    assert set(masked) == {0}
    assert metrics["genrm_failed_comparison_rate_per_group"] == len(failing_passes) / 4
    assert metrics["genrm_masked_rollout_rate_per_group"] == 0.5


async def test_fixed_baseline_cohort_masks_the_member_through_verify(server):
    server.config.comparison_mode = "fixed_baseline"

    async def judge(_conversation, _response_1, _response_2, *, pair_idx, **_kwargs):
        if pair_idx[0] == 0:
            raise genrm.JudgeError(NO_VERDICT)
        return comparison_result(5.0, 5.0, 3.5)

    server._run_single_comparison = judge

    def baseline_member(index):
        body = member(index)
        body.responses_create_params.metadata = {"baseline_response": "baseline"}
        return body

    server.config.num_rollouts_per_prompt = 2
    first, second = await asyncio.gather(
        *(server.verify(request_with_session(), baseline_member(index)) for index in range(2)), return_exceptions=True
    )
    assert isinstance(first, genrm.JudgeError) and "rollout_index=0" in str(first)
    assert second.reward == approx(5.0) and not second.mask_sample
