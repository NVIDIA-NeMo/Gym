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
"""PolyMath resources server.

Subclasses ``math_with_judge`` to add the two pieces of metric machinery
that NeMo Skills' ``WeightedMathMetrics`` provides for the PolyMath
benchmark and that stock ``math_with_judge`` does not:

  1. Difficulty-weighted aggregation. PolyMath rows carry a ``weight``
     field (low=1, medium=2, high=4, top=8). Pass@k, pass@1[avg-of-k],
     and majority@k are reported both unweighted (default Gym
     behaviour) and weighted (Skills' headline numbers) so reviewers
     can cross-check either side.

  2. Per-language stratification. Skills' pipeline-level
     ``subset_for_metrics`` mechanism reports one set of metrics per
     language; we replicate that with ``compute_subset_metrics`` keyed
     on the JSONL ``language`` field.

Verification itself (math-verify symbolic check, optional LLM judge
fallback) is unchanged from ``math_with_judge``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    LibraryJudgeMathVerifyRequest,
    LibraryJudgeMathVerifyResponse,
)


class PolyMathResourcesServerConfig(LibraryJudgeMathResourcesServerConfig):
    pass


class PolyMathVerifyRequest(LibraryJudgeMathVerifyRequest):
    # Per-row metadata that PolyMath's metrics consume. Both fields are
    # optional so the server can still verify rows without them (e.g.
    # the math_with_judge example data, which has neither).
    weight: Optional[float] = None
    language: Optional[str] = None


class PolyMathVerifyResponse(LibraryJudgeMathVerifyResponse):
    weight: Optional[float] = None
    language: Optional[str] = None


class PolyMathResourcesServer(LibraryJudgeMathResourcesServer):
    """math_with_judge + difficulty-weighted aggregation + per-language metrics."""

    config: PolyMathResourcesServerConfig

    async def verify(self, body: PolyMathVerifyRequest) -> PolyMathVerifyResponse:  # type: ignore[override]
        base = await super().verify(
            LibraryJudgeMathVerifyRequest(
                responses_create_params=body.responses_create_params,
                response=body.response,
                question=body.question,
                expected_answer=body.expected_answer,
            )
        )
        return PolyMathVerifyResponse(
            **base.model_dump(),
            weight=body.weight,
            language=body.language,
        )

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics
    # ──────────────────────────────────────────────────────────

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute pass@k metrics with PolyMath's weighted + per-language extensions.

        Three metric families are emitted:

          * Tier 1 (unweighted, all languages pooled): the same keys
            ``math_with_judge`` produces — ``pass@k/symbolic_accuracy``,
            ``pass@1[avg-of-k]/symbolic_accuracy``, ``majority@k/symbolic_accuracy``,
            etc.
          * Tier 2 (per-language): same keys, prefixed by language —
            ``en/pass@k/symbolic_accuracy``, ``zh/pass@k/symbolic_accuracy``, …
          * Tier 3 (weighted by ``weight``): keys suffixed with
            ``_weighted`` (e.g. ``pass@1[avg-of-4]/symbolic_accuracy_weighted``).
            Mirrors Skills' ``weighted_<score_method>`` headline metrics.
        """
        if not tasks:
            return {}

        metrics = compute_pass_majority_metrics(
            tasks,
            score_fn=self._math_score_fn,
            answer_key="extracted_answer",
        )[0]
        metrics.update(
            compute_subset_metrics(
                tasks,
                subset_key="language",
                score_fn=self._math_score_fn,
                answer_key="extracted_answer",
            )
        )
        metrics.update(
            self._compute_weighted_pass_majority_metrics(
                tasks,
                score_fn=self._math_score_fn,
                answer_key="extracted_answer",
            )
        )
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Pick headline metrics: tokens + highest-k pass@1[avg]/pass@k/majority@k.

        Both unweighted and ``_weighted`` variants flow through; the
        weighted ones are PolyMath's published headline.
        """
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))

        return key

    # ──────────────────────────────────────────────────────────
    # Weighted aggregation (port of WeightedMathMetrics)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_weighted_pass_majority_metrics(
        tasks: List[List[Dict[str, Any]]],
        score_fn,
        answer_key: str,
    ) -> Dict[str, float]:
        """Difficulty-weighted analogue of ``compute_pass_majority_metrics``.

        Mirrors NeMo Skills' ``WeightedMathMetrics`` in
        ``nemo_skills/evaluation/metrics/weighted_math_metrics.py``.

        Each task contributes its rollouts weighted by ``weight`` from
        the first rollout (the value is per-question, so all rollouts of
        a task share the same weight). Tasks missing a weight default
        to 1.0 — the same fallback Skills uses.

        Returns a flat dict with keys
        ``pass@k/{score}_weighted``, ``pass@1[avg-of-k]/{score}_weighted``,
        and ``majority@k/{score}_weighted`` (all percentages 0-100).
        """
        if not tasks:
            return {}

        per_task = []
        for rollouts in tasks:
            if not rollouts:
                continue
            weight = float(rollouts[0].get("weight") or 1.0)
            score_dicts = [score_fn(r) for r in rollouts]
            answers = [r.get(answer_key) for r in rollouts]
            per_task.append((weight, score_dicts, answers))

        if not per_task:
            return {}

        max_k = max(len(sds) for _, sds, _ in per_task)
        score_names = sorted({n for _, sds, _ in per_task for sd in sds for n in sd})

        out: Dict[str, float] = {}
        for k in range(1, max_k + 1):
            for name in score_names:
                pass_k_num = 0.0
                pass1_avg_num = 0.0
                majority_num = 0.0
                contributing_weight = 0.0

                for weight, score_dicts, answers in per_task:
                    vals = [bool(sd.get(name, False)) for sd in score_dicts if name in sd]
                    if not vals or k > len(vals):
                        continue

                    n_incorrect = sum(1 for v in vals if not v)
                    if n_incorrect < k:
                        pass_k_num += weight
                    else:
                        pass_k_num += weight * (1.0 - math.comb(n_incorrect, k) / math.comb(len(vals), k))

                    pass1_avg_num += weight * sum(1 if v else 0 for v in vals[:k]) / k

                    if k >= 2:
                        valid = [(a, v) for a, v in zip(answers[:k], vals[:k]) if a is not None]
                        if valid:
                            counter = Counter(valid)
                            top = counter.most_common(1)[0][1]
                            tied = [v for (_, v), c in counter.items() if c == top]
                            majority_num += weight * sum(1 if v else 0 for v in tied) / len(tied)

                    contributing_weight += weight

                if contributing_weight > 0:
                    out[f"pass@{k}/{name}_weighted"] = 100.0 * pass_k_num / contributing_weight
                    out[f"pass@1[avg-of-{k}]/{name}_weighted"] = 100.0 * pass1_avg_num / contributing_weight
                    if k >= 2:
                        out[f"majority@{k}/{name}_weighted"] = 100.0 * majority_num / contributing_weight

        return out


if __name__ == "__main__":
    PolyMathResourcesServer.run_webserver()
