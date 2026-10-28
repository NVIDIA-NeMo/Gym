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

"""IMO-GradingBench resource server.

The policy model acts as a JUDGE of math proofs across **four** ordinal
grade buckets: ``correct`` (7) > ``almost`` (6) > ``partial`` (1) >
``incorrect`` (0). Verification is fully deterministic — no external
LLM judge — so this server regex-extracts the last word of the model's
response and compares it to the gold grade passed via ``expected_answer``.

Ported from NeMo Skills' ``GradingBenchMetrics``
(``nemo_skills/evaluation/metrics/gradingbench_metrics.py``):

* ``extract_grade()`` mirrors ``_extract_grade`` — strip markdown /
  punctuation from the last whitespace-delimited token, lowercase, and
  match against ``{correct, almost, partial, incorrect}``.
* ``verify()`` returns ``reward = 1.0`` iff the parsed grade equals the
  expected grade, plus ``binary_match`` (high vs low bucket agreement)
  and ``score_diff`` (absolute difference between the GRADE_TO_SCORE
  values, used by the MAE aggregator).
* ``compute_metrics()`` produces ``pass@k`` / ``pass@1[avg-of-k]`` /
  ``majority@k`` for both ``exact_accuracy`` and ``binarized_accuracy``
  (Tier 1+2 via ``compute_pass_majority_metrics``), then adds the
  Tier-3 ``mae`` metric whose denominator is the count of valid
  (pred, gold) pairs across all rollouts (Skills-side ``mae_count``).

CoT handling: CoT stripping happens at the vLLM server layer via
``--reasoning-parser`` (e.g. ``deepseek_r1`` for ``<think>…</think>``
models). With the parser active, ``/v1/responses`` returns reasoning
in a separate ``type="reasoning"`` output item; this server only reads
``type="message"`` items, so the last-word grade extractor never sees
CoT. The server README documents the required vLLM invocation.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


# ---------------------------------------------------------------------------
# Grade extraction (mirrors GradingBenchMetrics._extract_grade)
# ---------------------------------------------------------------------------

# Skills strips this exact set of markdown / punctuation chars from the last
# token before lowercasing. Keep the character class character-for-character
# identical so the two pipelines parse the same edge cases the same way.
_PUNCT_RE = re.compile(r"[*_`.,;:!?()[\]{}]+")

GRADE_TO_SCORE = {
    "correct": 7,
    "almost": 6,
    "partial": 1,
    "incorrect": 0,
}
GRADE_TO_BINARY = {
    "correct": "high",
    "almost": "high",
    "partial": "low",
    "incorrect": "low",
}
VALID_GRADES = frozenset(GRADE_TO_SCORE)


def extract_grade(text: Optional[str]) -> Optional[str]:
    """Extract the last-word grade from a free-form judge response.

    Mirrors NeMo Skills' ``GradingBenchMetrics._extract_grade``:

    1. ``text.strip().split()`` — whitespace-delimited tokens.
    2. Take the last token, strip ``[*_`.,;:!?()[\\]{}]+`` characters,
       and lowercase.
    3. If the result is in ``{correct, almost, partial, incorrect}``,
       return it; otherwise return ``None``.

    Returns ``None`` on empty / non-string input.
    """
    if not text or not isinstance(text, str):
        return None

    words = text.strip().split()
    if not words:
        return None

    last_word = _PUNCT_RE.sub("", words[-1]).lower()
    if last_word not in VALID_GRADES:
        return None

    return last_word


def normalize_expected_grade(text: Optional[str]) -> Optional[str]:
    """Normalize the gold ``expected_answer`` to a valid grade word.

    Skills lowercases + strips and rejects values outside the four-grade
    set (logging a warning). We mirror the same: a malformed gold value
    yields ``None`` and the row contributes ``reward=0`` with no MAE
    contribution.
    """
    if not text or not isinstance(text, str):
        return None
    candidate = text.strip().lower()
    if candidate not in VALID_GRADES:
        return None
    return candidate


def _extract_assistant_text(response: NeMoGymResponse) -> str:
    """Concatenate the assistant message text from a Responses-API payload.

    Reasoning content is expected to be filtered out by vLLM's
    ``--reasoning-parser`` at the server layer, which routes
    ``<think>…</think>`` tokens to a separate ``ResponseReasoningItem``
    (type ``"reasoning"``). We only read ``type == "message"`` items
    here, so the last-word grade extractor never sees CoT.
    """
    if response is None or not getattr(response, "output", None):
        return ""
    texts: List[str] = []
    for out in response.output:
        if getattr(out, "type", None) != "message":
            continue
        if getattr(out, "role", None) != "assistant":
            continue
        content = getattr(out, "content", None) or []
        if isinstance(content, str):
            texts.append(content)
            continue
        for c in content:
            t = getattr(c, "text", None)
            if isinstance(t, str):
                texts.append(t)
    return "\n".join(texts).strip()


# ---------------------------------------------------------------------------
# Config, request/response models
# ---------------------------------------------------------------------------


class ImoGradingBenchConfig(BaseResourcesServerConfig):
    """Config for the imo_gradingbench server.

    No tunable fields. The 4-class grade extractor is hardcoded; for
    reasoning models, enable vLLM's ``--reasoning-parser`` so
    ``<think>…</think>`` tokens are routed to a separate reasoning
    output item and never reach the grade regex.
    """


class ImoGradingBenchRunRequest(BaseRunRequest):
    """Run-time fields carried through the verify response for metrics."""

    model_config = ConfigDict(extra="allow")

    expected_answer: Optional[str] = None
    grading_id: Optional[str] = None
    problem_id: Optional[str] = None


class ImoGradingBenchVerifyRequest(ImoGradingBenchRunRequest, BaseVerifyRequest):
    pass


class ImoGradingBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    expected_answer: Optional[str] = None
    extracted_grade: Optional[str] = None
    # Bucket-level (high/low) agreement — feeds binarized_accuracy.
    binary_match: bool = False
    # |GRADE_TO_SCORE[pred] - GRADE_TO_SCORE[gold]| when both are valid,
    # else None. compute_metrics() averages this for the MAE metric.
    score_diff: Optional[float] = None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class ImoGradingBenchResourcesServer(SimpleResourcesServer):
    config: ImoGradingBenchConfig

    # --- verify ------------------------------------------------------------

    async def verify(self, body: ImoGradingBenchVerifyRequest) -> ImoGradingBenchVerifyResponse:
        text = _extract_assistant_text(body.response)
        pred_grade = extract_grade(text)
        expected_grade = normalize_expected_grade(body.expected_answer)

        exact_match = pred_grade is not None and pred_grade == expected_grade

        pred_binary = GRADE_TO_BINARY.get(pred_grade)
        expected_binary = GRADE_TO_BINARY.get(expected_grade)
        binary_match = pred_binary is not None and expected_binary is not None and pred_binary == expected_binary

        score_diff: Optional[float]
        if pred_grade is not None and expected_grade is not None:
            score_diff = float(abs(GRADE_TO_SCORE[pred_grade] - GRADE_TO_SCORE[expected_grade]))
        else:
            score_diff = None

        return ImoGradingBenchVerifyResponse(
            **body.model_dump(exclude={"reward"}),
            reward=1.0 if exact_match else 0.0,
            extracted_grade=pred_grade,
            binary_match=binary_match,
            score_diff=score_diff,
        )

    # --- metrics -----------------------------------------------------------

    @staticmethod
    def _score_fn(result: dict) -> dict:
        """Per-rollout scores fed into ``compute_pass_majority_metrics``.

        ``exact_accuracy`` (= reward) is the primary metric. Skills also
        reports ``binarized_accuracy`` — high (correct/almost) vs low
        (partial/incorrect) bucket agreement — which Gym computes for
        free as a second named score.
        """
        return {
            "exact_accuracy": float(result.get("reward", 0.0)),
            "binarized_accuracy": float(bool(result.get("binary_match", False))),
        }

    def compute_metrics(self, tasks: List[List[dict]]) -> dict:
        metrics, _, _, max_k = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_grade",
        )
        metrics.update(self._compute_mae(tasks, max_k))
        return metrics

    @staticmethod
    def _compute_mae(tasks: List[List[dict]], max_k: int) -> dict:
        """MAE / mae_count over the ``GRADE_TO_SCORE`` ordinal mapping.

        Skills' MAE counts only rollouts with a valid (pred, gold) pair,
        then averages ``|pred_score - gold_score|``. Three views:

        * ``pass@1[avg-of-k]/mae`` — pool all valid pairs across the
          first k rollouts of every task and average.
        * ``pass@k/mae`` — pick each task's closest prediction within
          the first k rollouts and average those task-level minima.
        * Bare ``mae`` / ``mae_count`` — Skills-parity all-rollouts
          pooled view (== ``pass@1[avg-of-{max_k}]/mae``). Kept for
          consumers that read the un-suffixed key.
        """
        if max_k == 0 or not tasks:
            return {}

        out: dict = {}
        for k in range(1, max_k + 1):
            pooled: List[float] = []
            best_per_task: List[float] = []
            for rollouts in tasks:
                task_errors = [float(r["score_diff"]) for r in rollouts[:k] if r.get("score_diff") is not None]
                pooled.extend(task_errors)
                if task_errors:
                    best_per_task.append(min(task_errors))
            if pooled:
                out[f"pass@1[avg-of-{k}]/mae"] = sum(pooled) / len(pooled)
                out[f"pass@1[avg-of-{k}]/mae_count"] = float(len(pooled))
            if best_per_task:
                out[f"pass@{k}/mae"] = sum(best_per_task) / len(best_per_task)
                out[f"pass@{k}/mae_count"] = float(len(best_per_task))

        if f"pass@1[avg-of-{max_k}]/mae" in out:
            out["mae"] = out[f"pass@1[avg-of-{max_k}]/mae"]
            out["mae_count"] = out[f"pass@1[avg-of-{max_k}]/mae_count"]

        return out

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        """Surface highest-k pass@1 / pass@k / majority@k entries plus MAE."""
        key = {}
        if "mean/reward" in agent_metrics:
            key["mean/reward"] = agent_metrics["mean/reward"]

        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@1[avg-of-{k}]",
                score_names=["exact_accuracy", "binarized_accuracy", "mae", "no_answer"],
            )
        )
        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@{k}",
                score_names=["exact_accuracy", "binarized_accuracy", "mae"],
            )
        )
        key.update(
            highest_k_metrics(
                agent_metrics,
                "majority@{k}",
                score_names=["exact_accuracy", "binarized_accuracy"],
            )
        )
        if "mae" in agent_metrics:
            key["mae"] = agent_metrics["mae"]
            key["mae_count"] = agent_metrics.get("mae_count", 0.0)
        return key


if __name__ == "__main__":
    ImoGradingBenchResourcesServer.run_webserver()
