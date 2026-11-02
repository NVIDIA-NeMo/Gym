# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score JUDGEBENCH response-pair preferences."""

import logging
from typing import Any


logger = logging.getLogger(__name__)


def base_pair_id(pair_id: str) -> str:
    return pair_id.split("-flipped", 1)[0].split("-flip", 1)[0]


def category_for_source(source: str) -> str:
    if source.startswith("mmlu-pro"):
        return "knowledge"
    if source.startswith("livebench-reasoning"):
        return "reasoning"
    if source.startswith("livebench-math"):
        return "math"
    if source.startswith("livecodebench"):
        return "code"
    return "unknown"


def judgebench_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault((base_pair_id(result["pair_id"]), result.get("_ng_rollout_index", 0)), []).append(result)

    category_scores: dict[str, list[int]] = {}
    no_answer = 0
    incomplete_pairs = 0
    for pair_id, samples in grouped.items():
        if len(samples) > 2 or len({sample["pair_id"] for sample in samples}) != len(samples):
            raise ValueError(f"Duplicate JudgeBench orientations for {pair_id}")
        if len(samples) == 2 and {sample.get("label") for sample in samples} != {"A>B", "B>A"}:
            raise ValueError(f"JudgeBench pair {pair_id} must have opposite labels")
        balance = 0
        if len(samples) != 2:
            logger.warning("Expected 2 samples for %s, got %d", pair_id, len(samples))
            incomplete_pairs += 1
        for sample in samples:
            ranking = sample.get("predicted_ranking")
            if ranking is None:
                no_answer += 1
                continue
            label = sample.get("label")
            correct = (label == "A>B" and ranking <= 3.5) or (label == "B>A" and ranking > 3.5)
            # Upstream: correct +1, incorrect -1, unparseable 0.
            balance += 1 if correct else -1
        category = category_for_source(samples[0].get("source", "unknown"))
        category_scores.setdefault(category, []).append(int(len(samples) == 2 and balance > 0))

    metrics: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    all_scores: list[int] = []
    for category, scores in sorted(category_scores.items()):
        metrics[category] = sum(scores) / len(scores) if scores else 0.0
        counts[category] = {"total": len(scores), "correct": sum(scores)}
        all_scores.extend(scores)
    metrics["overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    counts["overall"] = {"total": len(all_scores), "correct": sum(all_scores)}
    return {"metrics": metrics, "counts": counts, "no_answer": no_answer, "incomplete_pairs": incomplete_pairs}


def score(labels: dict[str, Any], ranking: float | None) -> float:
    if ranking is None:
        return 0.0
    return float((labels.get("label") == "A>B" and ranking <= 3.5) or (labels.get("label") == "B>A" and ranking > 3.5))


def compute_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    results = [
        row["verifier_metadata"]
        | {
            "predicted_ranking": row.get("predicted_ranking"),
            "request_error": row.get("request_error", "predicted_ranking" not in row),
            "_ng_rollout_index": row.get("_ng_rollout_index", index),
        }
        for task in tasks
        for index, row in enumerate(task)
    ]
    return judgebench_metrics(results)
