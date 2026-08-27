# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-row binary scoring and run-level PrimeVul paired metrics."""

from __future__ import annotations

from typing import Any, Optional

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME


# PrimeVul's four pair-wise outcomes (paper §IV-B2), mapped to the metric keys we report. They
# partition every complete pair, so the four rates sum to 1:
#
#   P-C  correct     both members classified correctly — the reported benchmark number
#   P-V  vulnerable  both members called vulnerable — the degenerate "always say vulnerable" policy
#   P-B  benign      both members called benign — the mirror-image degenerate policy
#   P-R  reversed    the labels are inverted: the fix is flagged and the vulnerability is missed
PAIRWISE_OUTCOME_KEYS = {
    "correct": "mean/paired_accuracy",
    "vulnerable": "mean/pairwise_vulnerable_rate",
    "benign": "mean/pairwise_benign_rate",
    "reversed": "mean/pairwise_reversed_rate",
}


def score_verdict(verdict: dict, verifier_metadata: dict) -> dict:
    """Score the parsed YES/NO verdict against one required gold label."""
    gold_is_vulnerable = verifier_metadata["gold_is_vulnerable"]
    parse_error = bool(verdict.get("parse_error"))
    pred_is_vulnerable = None if parse_error else bool(verdict.get("is_vulnerable"))
    correct = pred_is_vulnerable is not None and pred_is_vulnerable == gold_is_vulnerable
    return {
        "reward": float(correct),
        "correct": correct,
        "parse_error": parse_error,
        "pair_id": verifier_metadata["pair_id"],
        "pred_is_vulnerable": pred_is_vulnerable,
        "gold_is_vulnerable": gold_is_vulnerable,
    }


def aggregate_paired(tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Run-level metrics from verify responses grouped by task.

    `tasks[i]` is the list of rollouts for task *i* (length = `--num-repeats`). Paired
    accuracy is computed **per repeat and then averaged**, not over all rollouts pooled.

    Pooling would put all `2 * k` rollouts of a pair into one group and require every one of
    them to be correct — silently turning "both members classified correctly" into "both members
    classified correctly *k* times running". That is a different, much stricter metric, and it
    decays as `--num-repeats` rises, so the same policy would score lower purely for having
    been measured more carefully and runs at different repeat counts would not be comparable.
    """
    flat = [row for rollouts in tasks for row in rollouts]
    if not flat:
        return {}

    metrics: dict[str, Any] = {
        "mean/parse_error_rate": sum(bool(row.get("parse_error")) for row in flat) / len(flat),
        "n_rollouts": len(flat),
    }
    metrics.update(_binary_metrics(flat))

    indexed_tasks = [
        {row.get(ROLLOUT_INDEX_KEY_NAME, position): row for position, row in enumerate(rollouts)} for rollouts in tasks
    ]
    repeats = sorted({repeat for rollouts in indexed_tasks for repeat in rollouts})
    per_repeat = []
    for index in repeats:
        rows = [rollouts[index] for rollouts in indexed_tasks if index in rollouts]
        outcomes = _pairwise_outcomes(rows)
        if outcomes is not None:
            per_repeat.append(outcomes)
    if per_repeat:
        for key in PAIRWISE_OUTCOME_KEYS.values():
            metrics[key] = sum(outcomes[key] for outcomes in per_repeat) / len(per_repeat)
        metrics["n_pairs"] = _complete_pair_count([rollouts[0] for rollouts in tasks if rollouts])
    return metrics


def _predicted_label(row: dict) -> bool:
    """The predicted binary label, coercing an unparseable verdict to the wrong label.

    A row the model never answered must never count as correct, so it is scored as whichever
    label the gold label is not.
    """
    gold = bool(row.get("gold_is_vulnerable"))
    pred = row.get("pred_is_vulnerable")
    return (not gold) if pred is None else bool(pred)


def _binary_metrics(rows: list[dict]) -> dict[str, Any]:
    """Vulnerable-vs-benign accuracy, precision, recall, F1 and the confusion counts.

    The confusion counts are reported because they are what exposes the degenerate
    "always say vulnerable" policy at a glance.
    """
    tp = fp = fn = tn = 0
    for row in rows:
        gold = bool(row.get("gold_is_vulnerable"))
        pred = _predicted_label(row)
        if gold and pred:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "mean/binary_accuracy": (tp + tn) / len(rows),
        "mean/binary_precision": precision,
        "mean/binary_recall": recall,
        "mean/binary_f1": f1,
        "n_true_positives": tp,
        "n_false_positives": fp,
        "n_false_negatives": fn,
        "n_true_negatives": tn,
    }


def _group_by_pair(rows: list[dict]) -> dict[Any, list[dict]]:
    grouped: dict[Any, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("pair_id"), []).append(row)
    return grouped


def _complete_pair_count(rows: list[dict]) -> int:
    return sum(1 for members in _group_by_pair(rows).values() if len(members) >= 2)


def _pairwise_outcomes(rows: list[dict]) -> Optional[dict[str, float]]:
    """The rates of PrimeVul's four pair-wise outcomes, or None without a complete pair.

    The four outcomes partition every complete pair, so the rates sum to 1. `correct` is the
    reported metric; the other three are what distinguish a model that understands the code from
    one that has collapsed onto a constant answer, which a single accuracy number cannot show.

    Incomplete pairs are skipped rather than counted as failures: a pair split across an eval
    subset (`--limit`, a filtered split) is a measurement artifact, not a model error.
    """
    complete = [members for members in _group_by_pair(rows).values() if len(members) >= 2]
    if not complete:
        return None

    counts = dict.fromkeys(PAIRWISE_OUTCOME_KEYS, 0)
    for members in complete:
        counts[_pair_outcome(members)] += 1
    return {PAIRWISE_OUTCOME_KEYS[outcome]: count / len(complete) for outcome, count in counts.items()}


def _pair_outcome(members: list[dict]) -> str:
    """Classify one complete pair into exactly one of PrimeVul's four pair-wise outcomes."""
    if all(member.get("correct") for member in members):
        return "correct"
    predictions = [_predicted_label(member) for member in members]
    if all(predictions):
        return "vulnerable"
    if not any(predictions):
        return "benign"
    # Neither constant nor correct: each member got the other's label.
    return "reversed"
