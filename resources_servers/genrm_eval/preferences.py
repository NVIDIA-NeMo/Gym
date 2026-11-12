# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score response-pair preference direction."""

from typing import Any


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Require an expressed preference on the 1–6 scale."""
    label = metadata.get("ground_truth_label")
    if isinstance(label, bool) or not isinstance(label, (int, float)) or not 1 <= label <= 6 or label == 3.5:
        raise ValueError("Expected a preference from 1 to 6, excluding ties")
    if not isinstance(metadata.get("data_type", "unknown"), str):
        raise ValueError("Expected data_type to be a string")


def preference_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "overall": {"total": 0, "correct": 0, "accuracy": 0.0, "no_answer": 0},
        "by_data_type": {},
    }
    for result in results:
        ground_truth = result.get("ground_truth_label")
        if ground_truth is None:
            continue
        data_type = result.get("data_type", "unknown")
        if not isinstance(data_type, str):
            data_type = "unknown"
        by_type = metrics["by_data_type"].setdefault(
            data_type, {"total": 0, "correct": 0, "accuracy": 0.0, "no_answer": 0}
        )
        ranking = result.get("predicted_ranking")
        metrics["overall"]["total"] += 1
        by_type["total"] += 1
        if ranking is None:
            metrics["overall"]["no_answer"] += 1
            by_type["no_answer"] += 1
            continue
        if (ranking < 3.5) == (float(ground_truth) < 3.5):
            metrics["overall"]["correct"] += 1
            by_type["correct"] += 1
    if metrics["overall"]["total"]:
        metrics["overall"]["accuracy"] = metrics["overall"]["correct"] / metrics["overall"]["total"]
    for by_type in metrics["by_data_type"].values():
        if by_type["total"]:
            by_type["accuracy"] = by_type["correct"] / by_type["total"]
    return metrics


def score(labels: dict[str, Any], ranking: float | None) -> float:
    if ranking is None:
        return 0.0
    return float(
        labels.get("ground_truth_label") is not None and (ranking < 3.5) == (labels.get("ground_truth_label") < 3.5)
    )


def compute_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    results = [
        row["verifier_metadata"]
        | {
            "predicted_ranking": row.get("predicted_ranking"),
        }
        for task in tasks
        for row in task
    ]
    return preference_metrics(results)
