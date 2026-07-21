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

# Resources server for short-answer question answering benchmarks like
# HotpotQA (closed-book and open-book). Verification is fully deterministic:
#   1. Extract a JSON-formatted answer from the model output (last valid JSON
#      object containing an "answer" key wins, matching the prompt).
#   2. Compute SQuAD-style answer EM and token-overlap F1 against the
#      ground-truth answer.
#   3. Compute alternative-aware substring match (lenient + strict) over
#      surface-form variants of the ground truth.
#   4. Mark unreliable ground truths (long answers / multi-word names) so
#      downstream aggregation can report both "all" and "filtered" metrics.
#
# All three scoring routines are faithful ports of Skills'
# `nemo_skills/evaluation/metrics/hotpotqa_metrics.py` and
# `nemo_skills/evaluation/metrics/hotpotqa_filtering.py`.

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from resources_servers.hotpotqa_qa.scoring import (
    answer_exact_match,
    answer_f1_score,
    is_correct,
    is_correct_strict,
    normalize_gt,
    parse_generation,
)


class HotpotQAQAResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "hotpotqa_qa"
    # When True, /aggregate_metrics also reports "filtered_*" channels that
    # exclude tasks whose ground-truth answers are unreliable for substring
    # evaluation (long strings or multi-word proper names). Mirrors Skills'
    # HotpotQAMetrics.get_metrics behavior.
    report_filtered_metrics: bool = True


class HotpotQAQARunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    expected_answer: str


class HotpotQAQAVerifyRequest(HotpotQAQARunRequest, BaseVerifyRequest):
    pass


class HotpotQAQAVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    expected_answer: str
    # Parsed answer extracted from the model's JSON output.
    extracted_answer: Optional[str]
    # SQuAD-normalized exact match (1.0 / 0.0).
    answer_em: float
    # SQuAD token-overlap F1 (continuous in [0, 1]).
    answer_f1: float
    # Alternative-aware substring match (lenient).
    is_correct: float
    # Alternative-aware substring match (strict — word-boundary + position guards).
    is_correct_strict: float
    # True if the GT is flagged as unreliable for substring evaluation.
    gt_should_remove: bool
    # Reason the GT was flagged ('' if not removed). One of:
    # 'gt_too_long', 'multi_word_name', or '' (kept).
    gt_remove_reason: str


class HotpotQAQAResourcesServer(SimpleResourcesServer):
    """Deterministic verification for short-answer QA (HotpotQA-style).

    The reward emitted on `/verify` is `is_correct_strict` — the strictest
    of the four scores above — which is what the closed-book benchmark
    primarily ranks on. The full set of scores is returned in the response
    so downstream metric aggregation can compute pass@k for every channel.
    """

    config: HotpotQAQAResourcesServerConfig

    async def verify(self, body: HotpotQAQAVerifyRequest) -> HotpotQAQAVerifyResponse:
        # Concatenate all output_text content blocks from the assistant turn.
        parts: List[str] = []
        for output in body.response.output:
            if output.type != "message":
                continue
            for item in output.content:
                if item.type == "output_text":
                    parts.append(item.text)
        generation = "".join(parts)

        pred_answer, _ = parse_generation(generation)
        expected_answer = body.expected_answer

        ans_em = answer_exact_match(pred_answer, expected_answer)
        ans_f1, _, _ = answer_f1_score(pred_answer, expected_answer)

        gt_info = normalize_gt(expected_answer)
        alternatives = gt_info["alternatives"]
        alt_correct = float(is_correct(alternatives, pred_answer))
        alt_correct_strict = float(is_correct_strict(alternatives, pred_answer))

        return HotpotQAQAVerifyResponse(
            **body.model_dump(),
            reward=alt_correct_strict,
            extracted_answer=pred_answer if pred_answer else None,
            answer_em=ans_em,
            answer_f1=ans_f1,
            is_correct=alt_correct,
            is_correct_strict=alt_correct_strict,
            gt_should_remove=gt_info["should_remove"],
            gt_remove_reason=gt_info["remove_reason"],
        )

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _hotpotqa_score_fn(r: Dict[str, Any]) -> Dict[str, float]:
        """Pull HotpotQA's four scoring channels out of a verify response."""
        scores: Dict[str, float] = {}
        for key in ("answer_em", "answer_f1", "is_correct", "is_correct_strict"):
            if key in r:
                scores[key] = float(r[key])
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute HotpotQA pass@k / majority@k for both unfiltered and filtered tasks.

        The ``filtered_*`` channels exclude tasks whose ground-truth answer is
        flagged as unreliable for substring evaluation. This matches Skills'
        HotpotQAMetrics, which reports both modes side by side.
        """
        metrics: Dict[str, Any] = {}

        unfiltered_metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._hotpotqa_score_fn,
            answer_key="extracted_answer",
        )
        metrics.update(unfiltered_metrics)

        if self.config.report_filtered_metrics:
            filtered_tasks = [rollouts for rollouts in tasks if rollouts and not _task_should_remove(rollouts)]
            if filtered_tasks:
                filtered_metrics, _, _, _ = compute_pass_majority_metrics(
                    filtered_tasks,
                    score_fn=self._hotpotqa_score_fn,
                    answer_key="extracted_answer",
                )
                # Drop the inner per_sample_aggregate to keep the dict shallow
                # (it is preserved on the unfiltered side, which is the canonical
                # variance source for pass@1 across runs).
                filtered_metrics.pop("per_sample_aggregate", None)
                for k, v in filtered_metrics.items():
                    metrics[f"filtered_{k}"] = v
                metrics["filtered_num_tasks"] = len(filtered_tasks)
            else:
                metrics["filtered_num_tasks"] = 0
            metrics["unfiltered_num_tasks"] = len(tasks)

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for HotpotQA-style benchmarks."""
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        # Pull out the highest-k pass@k / pass@1[avg-of-k] / majority@k for the
        # canonical HotpotQA score channels (both unfiltered and filtered).
        score_names = [
            "answer_em",
            "answer_f1",
            "is_correct",
            "is_correct_strict",
        ]
        for prefix in ("", "filtered_"):
            key.update(
                highest_k_metrics(
                    agent_metrics,
                    f"{prefix}pass@1[avg-of-{{k}}]",
                    score_names=score_names,
                )
            )
            key.update(
                highest_k_metrics(
                    agent_metrics,
                    f"{prefix}pass@{{k}}",
                    score_names=score_names,
                )
            )
            key.update(
                highest_k_metrics(
                    agent_metrics,
                    f"{prefix}majority@{{k}}",
                    score_names=score_names,
                )
            )

        return key


def _task_should_remove(rollouts: List[Dict[str, Any]]) -> bool:
    """A task is filtered out iff its GT was flagged as unreliable.

    Reads the flag off the first rollout's verify response; all rollouts for
    a task share the same expected_answer so the flag is identical across
    them.
    """
    return bool(rollouts[0].get("gt_should_remove", False))


if __name__ == "__main__":
    HotpotQAQAResourcesServer.run_webserver()
