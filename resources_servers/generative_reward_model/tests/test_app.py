# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from unittest.mock import MagicMock

from pytest import approx, fixture

from nemo_gym.server_utils import ServerClient
from resources_servers.generative_reward_model.app import (
    GenerativeRewardModelResourcesServer,
    GenerativeRewardModelResourcesServerConfig,
    GenerativeRewardModelVerifyRequest,
)


def _rubric(rubric_id, s1, s2, ranking):
    return {
        "rubric_id": rubric_id,
        "response_1_analysis": f"R1 analysis for rubric {rubric_id}",
        "response_2_analysis": f"R2 analysis for rubric {rubric_id}",
        "score_1": s1,
        "score_2": s2,
        "ranking": ranking,
    }


def _overall(s1, s2, ranking):
    return {
        "response_1_analysis": "Overall R1",
        "response_2_analysis": "Overall R2",
        "score_1": s1,
        "score_2": s2,
        "ranking": ranking,
    }


def _make_output(rubric_evals, overall):
    return json.dumps({"rubric_evaluations": rubric_evals, "overall": overall})


class TestGenerativeRewardModelApp:
    @fixture
    def config(self) -> GenerativeRewardModelResourcesServerConfig:
        return GenerativeRewardModelResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            name="generative_reward_model",
            score_weight=1.0,
            ranking_weight=2.0,
            rubric_weight=0.5,
            parse_failure_penalty=-100.0,
        )

    def _make_verify_request(
        self,
        response_text: str,
        gt_overall=None,
        gt_rubric_scores=None,
    ) -> GenerativeRewardModelVerifyRequest:
        data = {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Evaluate these responses..."}],
            },
            "response": {
                "id": "resp_123",
                "created_at": 0.0,
                "model": "test-model",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": response_text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
            "id": "test-id-1",
        }
        if gt_overall is not None:
            data["ground_truth_overall"] = gt_overall
        if gt_rubric_scores is not None:
            data["ground_truth_rubric_scores"] = gt_rubric_scores
        return GenerativeRewardModelVerifyRequest.model_validate(data)

    # --- Single rubric (per-principle) tests ---

    async def test_single_rubric_perfect(self, config) -> None:
        """Perfect prediction on single rubric → reward = 0."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        output = _make_output([_rubric(1, 5, 1, 1)], _overall(5, 1, 1))
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
            gt_rubric_scores=[{"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0}],
        )
        res = await rs.verify(body)

        assert res.reward == approx(0.0)
        assert res.format_correct == approx(1.0)
        assert res.overall_score_1_l1 == approx(0.0)
        assert res.rubric_score_l1 == approx(0.0)
        assert res.rubric_ranking_l1 == approx(0.0)

    async def test_single_rubric_wrong_scores(self, config) -> None:
        """Wrong scores on single rubric."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        # Predict (3, 3, 3) vs GT (5, 1, 1)
        output = _make_output([_rubric(1, 3, 3, 3)], _overall(3, 3, 3))
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
            gt_rubric_scores=[{"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0}],
        )
        res = await rs.verify(body)

        # overall: 1*2 + 1*2 + 2*2 = 8
        # rubric:  0.5 * (1*2 + 1*2 + 2*2) = 0.5 * 8 = 4
        # total: -(8 + 4) = -12
        assert res.reward == approx(-12.0)
        assert res.format_correct == approx(1.0)

    # --- Multi rubric (combined) tests ---

    async def test_multi_rubric_perfect(self, config) -> None:
        """Perfect prediction on all rubrics → reward = 0."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        output = _make_output(
            [_rubric(1, 5, 1, 1), _rubric(2, 3, 4, 5)],
            _overall(4, 2, 2),
        )
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 4.0, "score_2": 2.0, "ranking": 2.0},
            gt_rubric_scores=[
                {"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
                {"rubric_id": 2, "score_1": 3.0, "score_2": 4.0, "ranking": 5.0},
            ],
        )
        res = await rs.verify(body)

        assert res.reward == approx(0.0)
        assert res.format_correct == approx(1.0)

    async def test_multi_rubric_perfect_overall_bad_rubrics(self, config) -> None:
        """Perfect overall but wrong rubric scores → only rubric penalty."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        # GT rubrics: (5,1,1) and (3,4,5). Predict (3,3,3) for both.
        output = _make_output(
            [_rubric(1, 3, 3, 3), _rubric(2, 3, 3, 3)],
            _overall(4, 2, 2),  # perfect overall
        )
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 4.0, "score_2": 2.0, "ranking": 2.0},
            gt_rubric_scores=[
                {"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
                {"rubric_id": 2, "score_1": 3.0, "score_2": 4.0, "ranking": 5.0},
            ],
        )
        res = await rs.verify(body)

        # overall: 0
        # rubric 1: 1*2 + 1*2 + 2*2 = 8
        # rubric 2: 1*0 + 1*1 + 2*2 = 5
        # avg rubric penalty: (8 + 5) / 2 = 6.5
        # rubric contribution: 0.5 * 6.5 = 3.25
        assert res.reward == approx(-3.25)
        assert res.overall_score_1_l1 == approx(0.0)

    async def test_multi_rubric_bad_overall_perfect_rubrics(self, config) -> None:
        """Bad overall but perfect rubric scores → only overall penalty."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        output = _make_output(
            [_rubric(1, 5, 1, 1), _rubric(2, 3, 4, 5)],  # perfect rubrics
            _overall(2, 4, 4),  # bad overall (GT is 4, 2, 2)
        )
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 4.0, "score_2": 2.0, "ranking": 2.0},
            gt_rubric_scores=[
                {"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
                {"rubric_id": 2, "score_1": 3.0, "score_2": 4.0, "ranking": 5.0},
            ],
        )
        res = await rs.verify(body)

        # overall: 1*2 + 1*2 + 2*2 = 8
        # rubric: 0
        assert res.reward == approx(-8.0)
        assert res.rubric_score_l1 == approx(0.0)
        assert res.rubric_ranking_l1 == approx(0.0)

    # --- Parse failure tests ---

    async def test_parse_failure_garbage(self, config) -> None:
        """Unparseable output → parse failure penalty."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        body = self._make_verify_request(
            response_text="I cannot evaluate these responses.",
            gt_overall={"score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
            gt_rubric_scores=[{"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0}],
        )
        res = await rs.verify(body)

        assert res.reward == approx(-100.0)
        assert res.format_correct == approx(0.0)
        assert res.predicted_score_1 is None

    async def test_parse_failure_missing_rubrics(self, config) -> None:
        """Output with overall but no rubric_evaluations → parse failure."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        output = json.dumps({"overall": _overall(5, 1, 1)})
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
            gt_rubric_scores=[{"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0}],
        )
        res = await rs.verify(body)

        assert res.reward == approx(-100.0)
        assert res.format_correct == approx(0.0)

    async def test_parse_failure_old_flat_format(self, config) -> None:
        """Old flat format → parse failure."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        output = json.dumps(
            {
                "response_1_analysis": "A",
                "response_2_analysis": "B",
                "score_1": 5,
                "score_2": 1,
                "ranking": 1,
            }
        )
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
            gt_rubric_scores=[{"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0}],
        )
        res = await rs.verify(body)

        assert res.reward == approx(-100.0)
        assert res.format_correct == approx(0.0)

    # --- Wrong rubric ID tests ---

    async def test_wrong_rubric_ids_is_parse_failure(self, config) -> None:
        """Predicted rubric IDs don't match GT → parse failure."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        # GT expects rubric_id 1 and 2, model outputs 1 and 99
        output = _make_output(
            [_rubric(1, 5, 1, 1), _rubric(99, 3, 4, 5)],
            _overall(4, 2, 2),
        )
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 4.0, "score_2": 2.0, "ranking": 2.0},
            gt_rubric_scores=[
                {"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
                {"rubric_id": 2, "score_1": 3.0, "score_2": 4.0, "ranking": 5.0},
            ],
        )
        res = await rs.verify(body)

        assert res.reward == approx(-100.0)
        assert res.format_correct == approx(0.0)

    async def test_extra_rubric_is_parse_failure(self, config) -> None:
        """Model outputs more rubrics than GT → parse failure."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        output = _make_output(
            [_rubric(1, 5, 1, 1), _rubric(2, 3, 4, 5), _rubric(3, 4, 4, 3)],
            _overall(4, 2, 2),
        )
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 4.0, "score_2": 2.0, "ranking": 2.0},
            gt_rubric_scores=[
                {"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
                {"rubric_id": 2, "score_1": 3.0, "score_2": 4.0, "ranking": 5.0},
            ],
        )
        res = await rs.verify(body)

        assert res.reward == approx(-100.0)
        assert res.format_correct == approx(0.0)

    async def test_fewer_rubrics_is_parse_failure(self, config) -> None:
        """Model outputs fewer rubrics than GT → parse failure."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        output = _make_output(
            [_rubric(1, 5, 1, 1)],  # only 1, GT has 2
            _overall(4, 2, 2),
        )
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 4.0, "score_2": 2.0, "ranking": 2.0},
            gt_rubric_scores=[
                {"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
                {"rubric_id": 2, "score_1": 3.0, "score_2": 4.0, "ranking": 5.0},
            ],
        )
        res = await rs.verify(body)

        assert res.reward == approx(-100.0)
        assert res.format_correct == approx(0.0)

    # --- Config tests ---

    async def test_custom_rubric_weight(self, config) -> None:
        """rubric_weight scales rubric penalty."""
        config.rubric_weight = 1.0  # equal to overall
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        # Predict (3, 3, 3) for rubric, perfect overall
        output = _make_output([_rubric(1, 3, 3, 3)], _overall(5, 1, 1))
        body = self._make_verify_request(
            response_text=output,
            gt_overall={"score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
            gt_rubric_scores=[{"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0}],
        )
        res = await rs.verify(body)

        # overall: 0
        # rubric: 1.0 * (1*2 + 1*2 + 2*2) = 8
        assert res.reward == approx(-8.0)

    async def test_custom_parse_failure_penalty(self, config) -> None:
        config.parse_failure_penalty = -50.0
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        body = self._make_verify_request(
            response_text="garbage",
            gt_overall={"score_1": 5.0, "score_2": 1.0, "ranking": 1.0},
            gt_rubric_scores=[{"rubric_id": 1, "score_1": 5.0, "score_2": 1.0, "ranking": 1.0}],
        )
        res = await rs.verify(body)

        assert res.reward == approx(-50.0)
        assert res.format_correct == approx(0.0)

    # --- Regression: perfectly scored rubrics must still count toward the mean ---

    async def test_perfect_rubrics_count_toward_the_mean(self, config) -> None:
        """A rubric predicted perfectly contributes 0.0 to the mean, it is not dropped from it.

        Dropping zero-penalty rubrics shrinks the denominator as the policy improves, which made
        the rubric term flat: one wrong rubric of five scored the same as five wrong of five.
        """
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        gt_rubrics = [{"rubric_id": i, "ranking": 1.0} for i in range(1, 6)]
        # Four rubrics exactly right, one off by 4 on ranking.
        rubric_evals = [_rubric(i, 3, 3, 5.0 if i == 1 else 1.0) for i in range(1, 6)]
        body = self._make_verify_request(
            response_text=_make_output(rubric_evals, _overall(3, 3, 3)),
            gt_overall={},
            gt_rubric_scores=gt_rubrics,
        )
        res = await rs.verify(body)

        # rubric penalties = [2.0*4, 0, 0, 0, 0]; mean over ALL five = 8/5 = 1.6
        # reward = -(rubric_weight * 1.6) = -(0.5 * 1.6) = -0.8
        assert res.reward == approx(-0.8)

    async def test_rubric_penalty_scales_with_how_many_are_wrong(self, config) -> None:
        """More wrong rubrics must cost strictly more; the old code was flat across all counts."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        gt_rubrics = [{"rubric_id": i, "ranking": 1.0} for i in range(1, 6)]
        rewards = []
        for num_wrong in range(0, 6):
            rubric_evals = [_rubric(i, 3, 3, 5.0 if i <= num_wrong else 1.0) for i in range(1, 6)]
            body = self._make_verify_request(
                response_text=_make_output(rubric_evals, _overall(3, 3, 3)),
                gt_overall={},
                gt_rubric_scores=gt_rubrics,
            )
            rewards.append((await rs.verify(body)).reward)

        assert rewards[0] == approx(0.0)
        assert rewards[5] == approx(-4.0)
        # Strictly decreasing: fixing any rubric must improve the reward.
        assert all(a > b for a, b in zip(rewards, rewards[1:])), rewards

    async def test_improving_one_rubric_never_lowers_the_reward(self, config) -> None:
        """Monotonicity: the old mean-over-imperfect-only made fixing a rubric score worse."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        gt_rubrics = [{"rubric_id": 1, "ranking": 1.0}, {"rubric_id": 2, "ranking": 1.0}]

        async def reward_for(ranking_2: float) -> float:
            rubric_evals = [
                _rubric(1, 3, 3, 5.0),
                _rubric(2, 3, 3, ranking_2),
            ]
            body = self._make_verify_request(
                response_text=_make_output(rubric_evals, _overall(3, 3, 3)),
                gt_overall={},
                gt_rubric_scores=gt_rubrics,
            )
            return (await rs.verify(body)).reward

        slightly_off = await reward_for(2.0)  # rubric 2 off by 1
        exactly_right = await reward_for(1.0)  # rubric 2 correct
        assert exactly_right > slightly_off

    # --- Regression: duplicate rubric IDs must not slip past the match check ---

    async def test_duplicate_rubric_ids_are_a_parse_failure(self, config) -> None:
        """Repeating a rubric_id collapses in the lookup dict, so count the raw predictions."""
        server_mock = MagicMock(spec=ServerClient)
        rs = GenerativeRewardModelResourcesServer(config=config, server_client=server_mock)

        # Ground truth has a single rubric; the verdict answers it twice, wrong then right.
        rubric_evals = [
            _rubric(1, 3, 3, 5.0),
            _rubric(1, 3, 3, 1.0),
        ]
        body = self._make_verify_request(
            response_text=_make_output(rubric_evals, _overall(3, 3, 3)),
            gt_overall={},
            gt_rubric_scores=[{"rubric_id": 1, "ranking": 1.0}],
        )
        res = await rs.verify(body)

        assert res.reward == approx(config.parse_failure_penalty)
        assert res.format_correct == approx(0.0)
