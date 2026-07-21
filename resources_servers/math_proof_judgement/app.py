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

"""Math Proof Judgement resource server.

The policy model acts as a JUDGE of math proofs: given a problem + candidate
proof, it must output "Judgement: Yes" (correct) or "Judgement: No"
(incorrect). Verification is deterministic — no external LLM judge — so this
server just regex-parses the generated response and compares to the gold
label passed via ``expected_judgement``.

Ported from NeMo Skills' ``AnswerJudgementMetrics`` + ``is_correct_judgement``
(format 1 only; the boxed and <points> variants live in sibling judge benchmarks
and are not needed here). The aggregate metrics are binary-classification
style — accuracy, false_positives/false_negatives, precision/recall/F1 — and
are averaged across the K rollouts the same way Skills does.

CoT handling: CoT stripping happens at the vLLM server layer via
``--reasoning-parser`` (e.g. ``deepseek_r1`` for ``<think>…</think>`` models).
With the parser active, ``/v1/responses`` returns reasoning in a separate
``type="reasoning"`` output item; this server only reads ``type="message"``
items, so the Judgement regex never sees CoT text. The server README
documents the required vLLM invocation.
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
# Judgement parsing (mirrors nemo_skills.evaluation.metrics.utils.is_correct_judgement,
# format 1: "Judgement: Yes/No", with optional ** markdown bold).
# ---------------------------------------------------------------------------

_JUDGEMENT_PREFIX_RE = re.compile(r"\*{0,2}Judgement\*{0,2}\s*:", re.IGNORECASE)


def parse_judgement(text: Optional[str]) -> Optional[bool]:
    """Parse the first ``Judgement: Yes/No`` out of a text blob.

    Returns ``True`` for Yes, ``False`` for No, or ``None`` if no valid judgement
    is found. Matches the Skills ``is_correct_judgement`` format-1 regex exactly
    so the two pipelines agree on what counts as an invalid judgement.
    """
    if not text:
        return None
    match = _JUDGEMENT_PREFIX_RE.search(text)
    if not match:
        return None
    verdict = text[match.end() :].strip().lstrip("*").strip()
    if verdict.lower().startswith("yes"):
        return True
    if verdict.lower().startswith("no"):
        return False
    return None


def _extract_assistant_text(response: NeMoGymResponse) -> str:
    """Concatenate the assistant message text from a Responses-API payload.

    Reasoning content is expected to be filtered out by vLLM's
    ``--reasoning-parser`` at the server layer, which routes ``<think>…</think>``
    tokens to a separate ``ResponseReasoningItem`` (type ``"reasoning"``).
    We only read ``type == "message"`` items here, so the Judgement regex never
    sees CoT. See the benchmark README for the required vLLM invocation.
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


class MathProofJudgementConfig(BaseResourcesServerConfig):
    """Configuration for the math_proof_judgement server.

    No tunable fields — the Judgement: regex runs over the assistant message
    content returned by the upstream model server. For reasoning models,
    enable vLLM's ``--reasoning-parser`` (e.g. ``deepseek_r1``) so
    ``<think>…</think>`` tokens are routed to a separate reasoning output
    item and never reach the regex.
    """


class MathProofJudgementRunRequest(BaseRunRequest):
    """Run-time fields carried through the verify response for metrics.

    Besides the gold ``expected_judgement`` (always required), optional fields
    let downstream analysis re-identify rows without re-parsing the prompt.
    """

    model_config = ConfigDict(extra="allow")

    expected_judgement: Optional[str] = None
    problem_id: Optional[str] = None
    problem: Optional[str] = None
    proof: Optional[str] = None


class MathProofJudgementVerifyRequest(MathProofJudgementRunRequest, BaseVerifyRequest):
    pass


class MathProofJudgementVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    expected_judgement: Optional[str] = None
    # Normalised Yes/No/None verdict string — used as answer_key for majority@k.
    extracted_judgement: Optional[str] = None
    # One-hot classification counters (floats so avg-of-k is meaningful).
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0


# ---------------------------------------------------------------------------
# Classification-metric helpers (match Skills' averaging across K samples).
# ---------------------------------------------------------------------------


def _verdict_label(v: Optional[bool]) -> Optional[str]:
    """Stable answer_key value: "Yes" / "No" / None for unparseable."""
    if v is True:
        return "Yes"
    if v is False:
        return "No"
    return None


def _confusion(pred: Optional[bool], gt: Optional[bool]) -> tuple[float, float, float, float]:
    """Return (tp, fp, fn, tn) one-hot. None pred or gt → all zeros."""
    if pred is None or gt is None:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(pred is True and gt is True),
        float(pred is True and gt is False),
        float(pred is False and gt is True),
        float(pred is False and gt is False),
    )


def _per_sample_precision_recall_f1(
    per_sample: List[tuple[float, float, float]],
    total_positives: int,
) -> tuple[float, float, float]:
    """Average precision / recall / F1 across K samples.

    Mirrors Skills' ``_compute_precision_recall_f1``: compute precision/recall/F1
    once per sample-index k from aggregated TP/FP/FN across all datapoints, then
    average the K resulting values ("unbiased" — i.e. macro across K).

    When a denominator is 0 the convention matches Skills:
    - precision defaults to 1.0 when there are no positive predictions
    - recall defaults to 1.0 when there are no positive golds
    - F1 defaults to 0.0 when precision+recall == 0
    """
    if not per_sample:
        return 0.0, 0.0, 0.0
    precisions, recalls, f1s = [], [], []
    for tp, fp, fn in per_sample:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / total_positives if total_positives > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    n = len(per_sample)
    return (
        100.0 * sum(precisions) / n,
        100.0 * sum(recalls) / n,
        100.0 * sum(f1s) / n,
    )


def _select_pass_verdict(preds: List[Optional[bool]], gt: Optional[bool]) -> Optional[bool]:
    """Match Skills ``_update_score_metrics_for_pass``: prefer gt if present in preds,
    else first non-None pred, else None."""
    if gt is not None and gt in preds:
        return gt
    non_none = [p for p in preds if p is not None]
    return non_none[0] if non_none else None


def _select_majority_verdict(preds: List[Optional[bool]]) -> Optional[bool]:
    """Majority vote with tie-break = first vote (mirrors Skills' majority handling).

    Skills' ``_update_score_metrics_for_majority`` uses the majority answer chosen
    by the ``_compute_majority_at_k`` helper, which picks the most common valid
    answer; on ties it picks the first-encountered tied answer. None votes don't
    participate in the majority.
    """
    yes = sum(1 for p in preds if p is True)
    no = sum(1 for p in preds if p is False)
    if yes == 0 and no == 0:
        return None
    if yes > no:
        return True
    if no > yes:
        return False
    # Tie: first non-None prediction wins.
    for p in preds:
        if p is not None:
            return p
    return None


def _label_to_bool(label: Optional[str]) -> Optional[bool]:
    """Inverse of ``_verdict_label`` — used when post-processing rollouts."""
    if label == "Yes":
        return True
    if label == "No":
        return False
    return None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class MathProofJudgementResourcesServer(SimpleResourcesServer):
    config: MathProofJudgementConfig

    # --- verify ------------------------------------------------------------

    async def verify(self, body: MathProofJudgementVerifyRequest) -> MathProofJudgementVerifyResponse:
        text = _extract_assistant_text(body.response)
        pred = parse_judgement(text)
        gt = parse_judgement(body.expected_judgement)

        reward = 1.0 if (pred is not None and gt is not None and pred == gt) else 0.0
        tp, fp, fn, tn = _confusion(pred, gt)

        return MathProofJudgementVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_judgement=_verdict_label(pred),
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
        )

    # --- metrics -----------------------------------------------------------

    @staticmethod
    def _score_fn(result: dict) -> dict:
        """Per-rollout scores fed into ``compute_pass_majority_metrics``.

        Returns accuracy (=reward) plus the four confusion-matrix one-hots.
        Gym automatically derives pass@k / pass@1[avg-of-k] / majority@k for each.
        """
        return {
            "accuracy": float(result.get("reward", 0.0)),
            "true_positive": float(result.get("tp", 0.0)),
            "false_positive": float(result.get("fp", 0.0)),
            "false_negative": float(result.get("fn", 0.0)),
            "true_negative": float(result.get("tn", 0.0)),
        }

    def compute_metrics(self, tasks: List[List[dict]]) -> dict:
        metrics = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_judgement",
        )[0]
        # Tier-3 Skills parity: precision/recall/F1 averaged across K samples.
        metrics.update(self._compute_classification_metrics(tasks))
        return metrics

    def _compute_classification_metrics(self, tasks: List[List[dict]]) -> dict:
        """Compute precision/recall/F1 for pass@1[avg-of-k], pass@k, majority@k.

        Matches Skills' ``_compute_precision_recall_f1``: one TP/FP/FN measurement
        per (datapoint, sample_k), aggregated across datapoints then macro-averaged
        across K.
        """
        if not tasks:
            return {}
        max_k = max(len(t) for t in tasks)
        if max_k == 0:
            return {}

        # Gold judgement per task. All rollouts for task i carry the same
        # expected_judgement, so read it once.
        gold_per_task: List[Optional[bool]] = []
        for rollouts in tasks:
            if not rollouts:
                gold_per_task.append(None)
                continue
            gold_per_task.append(parse_judgement(rollouts[0].get("expected_judgement")))
        total_positives = sum(1 for g in gold_per_task if g is True)

        out: dict[str, float] = {}

        # --- pass@1[avg-of-k] ---
        # K per-sample TP/FP/FN aggregations, then macro-average across K.
        per_sample: List[tuple[float, float, float]] = []
        for sample_idx in range(max_k):
            tp = fp = fn = 0.0
            for rollouts, gt in zip(tasks, gold_per_task):
                if sample_idx >= len(rollouts):
                    continue
                r = rollouts[sample_idx]
                pred = _label_to_bool(r.get("extracted_judgement"))
                a, b, c, _ = _confusion(pred, gt)
                tp += a
                fp += b
                fn += c
            per_sample.append((tp, fp, fn))

        for k in range(1, max_k + 1):
            prec, rec, f1 = _per_sample_precision_recall_f1(per_sample[:k], total_positives)
            out[f"pass@1[avg-of-{k}]/precision"] = prec
            out[f"pass@1[avg-of-{k}]/recall"] = rec
            out[f"pass@1[avg-of-{k}]/f1"] = f1

        # --- pass@k (best-of-K): one per-datapoint verdict per k ---
        for k in range(1, max_k + 1):
            tp = fp = fn = 0.0
            for rollouts, gt in zip(tasks, gold_per_task):
                preds = [_label_to_bool(r.get("extracted_judgement")) for r in rollouts[:k]]
                pred = _select_pass_verdict(preds, gt)
                a, b, c, _ = _confusion(pred, gt)
                tp += a
                fp += b
                fn += c
            prec, rec, f1 = _per_sample_precision_recall_f1([(tp, fp, fn)], total_positives)
            out[f"pass@{k}/precision"] = prec
            out[f"pass@{k}/recall"] = rec
            out[f"pass@{k}/f1"] = f1

        # --- majority@k ---
        for k in range(1, max_k + 1):
            tp = fp = fn = 0.0
            for rollouts, gt in zip(tasks, gold_per_task):
                preds = [_label_to_bool(r.get("extracted_judgement")) for r in rollouts[:k]]
                pred = _select_majority_verdict(preds)
                a, b, c, _ = _confusion(pred, gt)
                tp += a
                fp += b
                fn += c
            prec, rec, f1 = _per_sample_precision_recall_f1([(tp, fp, fn)], total_positives)
            out[f"majority@{k}/precision"] = prec
            out[f"majority@{k}/recall"] = rec
            out[f"majority@{k}/f1"] = f1

        # Skills also reports total_positives — keep as an integer count.
        out["total_positives"] = float(total_positives)
        return out

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        """Expose the highest-k pass@1 / pass@k / majority@k plus F1 and accuracy."""
        key = {}
        if "mean/reward" in agent_metrics:
            key["mean/reward"] = agent_metrics["mean/reward"]
        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@1[avg-of-{k}]",
                score_names=["accuracy", "f1", "precision", "recall", "no_answer"],
            )
        )
        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@{k}",
                score_names=["accuracy", "f1", "no_answer"],
            )
        )
        key.update(
            highest_k_metrics(
                agent_metrics,
                "majority@{k}",
                score_names=["accuracy", "f1"],
            )
        )
        return key


if __name__ == "__main__":
    MathProofJudgementResourcesServer.run_webserver()
