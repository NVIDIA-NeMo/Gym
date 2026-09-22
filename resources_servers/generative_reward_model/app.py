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
from typing import Dict, List, Optional, Union

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.generative_reward_model.score_parser import extract_scores


class GenerativeRewardModelResourcesServerConfig(BaseResourcesServerConfig):
    score_weight: float = 1.0  # weight for individual score L1
    ranking_weight: float = 2.0  # weight for ranking L1
    rubric_weight: float = 0.5  # multiplier on averaged rubric penalty (2x less than overall)
    parse_failure_penalty: float = -100.0


class GenerativeRewardModelRunRequest(BaseRunRequest):
    id: Union[int, str]
    ground_truth_overall: Optional[Dict] = None  # {"score_1", "score_2", "ranking"}
    ground_truth_rubric_scores: Optional[List[Dict]] = None  # [{"rubric_id", "score_1", "score_2", "ranking"}, ...]


class GenerativeRewardModelVerifyRequest(GenerativeRewardModelRunRequest, BaseVerifyRequest):
    pass


class GenerativeRewardModelVerifyResponse(BaseVerifyResponse):
    predicted_score_1: Optional[float]
    predicted_score_2: Optional[float]
    predicted_ranking: Optional[float]
    ground_truth_overall: Optional[Dict]
    ground_truth_rubric_scores: Optional[List[Dict]]
    # Numeric metrics (auto-averaged by rollout collection for wandb)
    format_correct: float  # 1.0 or 0.0
    overall_score_1_l1: Optional[float]
    overall_score_2_l1: Optional[float]
    overall_ranking_l1: Optional[float]
    rubric_score_l1: Optional[float]  # mean L1 across rubrics for scores
    rubric_ranking_l1: Optional[float]  # mean L1 across rubrics for rankings
    overall_ranking_binary_acc: Optional[float]  # 1.0 if predicted ranking direction matches GT


class GenerativeRewardModelResourcesServer(SimpleResourcesServer):
    config: GenerativeRewardModelResourcesServerConfig

    def _fail(self, body) -> GenerativeRewardModelVerifyResponse:
        """Return a parse-failure response."""
        return GenerativeRewardModelVerifyResponse(
            **body.model_dump(),
            reward=self.config.parse_failure_penalty,
            predicted_score_1=None,
            predicted_score_2=None,
            predicted_ranking=None,
            format_correct=0.0,
            overall_score_1_l1=None,
            overall_score_2_l1=None,
            overall_ranking_l1=None,
            rubric_score_l1=None,
            rubric_ranking_l1=None,
            overall_ranking_binary_acc=None,
        )

    async def verify(self, body: GenerativeRewardModelVerifyRequest) -> GenerativeRewardModelVerifyResponse:
        # Extract text from message output items, skipping reasoning items.
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                assistant_responses.append(content_item.text)

        scores = extract_scores("".join(assistant_responses))
        cfg = self.config

        if scores is None:
            return self._fail(body)

        reward = 0.0
        overall_score_1_l1 = None
        overall_score_2_l1 = None
        overall_ranking_l1 = None
        rubric_score_l1 = None
        rubric_ranking_l1 = None
        overall_ranking_binary_acc = None

        # --- Overall reward (only penalize fields present in GT) ---
        gt_overall = body.ground_truth_overall
        if gt_overall is not None:
            if gt_overall.get("score_1") is not None:
                overall_score_1_l1 = abs(gt_overall["score_1"] - scores.score_1)
                reward -= cfg.score_weight * overall_score_1_l1
            if gt_overall.get("score_2") is not None:
                overall_score_2_l1 = abs(gt_overall["score_2"] - scores.score_2)
                reward -= cfg.score_weight * overall_score_2_l1
            if gt_overall.get("ranking") is not None:
                overall_ranking_l1 = abs(gt_overall["ranking"] - scores.ranking)
                reward -= cfg.ranking_weight * overall_ranking_l1
                # Binary accuracy: did the model get the direction right?
                # Rankings 1-3 = one direction, 4-6 = the other; midpoint is 3.5
                gt_dir = gt_overall["ranking"] < 3.5
                pred_dir = scores.ranking < 3.5
                overall_ranking_binary_acc = 1.0 if gt_dir == pred_dir else 0.0

        # --- Per-rubric reward ---
        gt_rubrics = body.ground_truth_rubric_scores
        if gt_rubrics:
            gt_by_id = {r["rubric_id"]: r for r in gt_rubrics}
            pred_by_id = {r.rubric_id: r for r in scores.rubric_scores}

            # Predicted rubrics must match GT exactly: same count, same IDs
            if len(pred_by_id) != len(gt_by_id) or set(pred_by_id.keys()) != set(gt_by_id.keys()):
                return self._fail(body)

            score_errors = []
            ranking_errors = []
            rubric_penalties = []
            for rid, gt in gt_by_id.items():
                pred = pred_by_id[rid]
                penalty = 0.0
                if gt.get("score_1") is not None:
                    s1_err = abs(gt["score_1"] - pred.score_1)
                    score_errors.append(s1_err)
                    penalty += cfg.score_weight * s1_err
                if gt.get("score_2") is not None:
                    s2_err = abs(gt["score_2"] - pred.score_2)
                    score_errors.append(s2_err)
                    penalty += cfg.score_weight * s2_err
                if gt.get("ranking") is not None:
                    r_err = abs(gt["ranking"] - pred.ranking)
                    ranking_errors.append(r_err)
                    penalty += cfg.ranking_weight * r_err
                if penalty > 0:
                    rubric_penalties.append(penalty)

            if rubric_penalties:
                reward -= cfg.rubric_weight * sum(rubric_penalties) / len(rubric_penalties)
            if score_errors:
                rubric_score_l1 = sum(score_errors) / len(score_errors)
            if ranking_errors:
                rubric_ranking_l1 = sum(ranking_errors) / len(ranking_errors)

        return GenerativeRewardModelVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            predicted_score_1=scores.score_1,
            predicted_score_2=scores.score_2,
            predicted_ranking=scores.ranking,
            format_correct=1.0,
            overall_score_1_l1=overall_score_1_l1,
            overall_score_2_l1=overall_score_2_l1,
            overall_ranking_l1=overall_ranking_l1,
            rubric_score_l1=rubric_score_l1,
            rubric_ranking_l1=rubric_ranking_l1,
            overall_ranking_binary_acc=overall_ranking_binary_acc,
        )


if __name__ == "__main__":
    GenerativeRewardModelResourcesServer.run_webserver()
