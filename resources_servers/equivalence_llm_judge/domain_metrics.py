# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HLE domain and modality summaries using Gym's shared aggregation."""

import math
from typing import Any
from urllib.parse import quote

from nemo_gym.reward_profile import compute_aggregate_metrics


def _modality(row: dict[str, Any]) -> str | None:
    value = row.get("modality")
    if value in ("text", "vision"):
        return value
    has_image = row.get("has_image")
    return ("vision" if has_image else "text") if type(has_image) is bool else None


def repeat_coverage_reason(questions: list[list[dict[str, Any]]]) -> str | None:
    """Require the same labeled questions in every repeat before publishing CIs."""
    coverage = []
    for task in questions:
        repeats = [row.get("_ng_rollout_index") for row in task]
        if any(type(repeat) is not int or repeat < 0 for repeat in repeats):
            return "Missing or invalid repeat identity"
        if len(set(repeats)) != len(repeats):
            return "Duplicate question within a repeat"
        coverage.append(set(repeats))
    if any(repeats != coverage[0] for repeats in coverage[1:]):
        return "Repeat question coverage differs"
    if not coverage or len(coverage[0]) < 2:
        return "At least two evaluation repeats are required"
    return None


def compute_domain_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate category, raw subject and text/vision slices of scored responses.

    The parent aggregator has already excluded masked outcomes. Each slice calls
    the same public aggregator without this custom hook, so its means and CIs use
    Gym's normal repeat machinery, without recursive slicing or separate CI math.
    """
    metrics = _compute_slices(tasks, ("category", "raw_subject", "modality"))
    for modality in ("text", "vision"):
        questions = [task for task in tasks if task and _modality(task[0]) == modality]
        scoped = _compute_slices(questions, ("category", "raw_subject"))
        metrics.update({key.replace("domains/", f"domains/{modality}/", 1): value for key, value in scoped.items()})
    metrics["domains/ci_method"] = "Gym Student t interval across repeat-level estimates"
    return metrics


def _compute_slices(tasks: list[list[dict[str, Any]]], fields: tuple[str, ...]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for field in fields:
        namespace = field
        groups: dict[str, list[list[dict[str, Any]]]] = {}
        missing = 0
        for task in tasks:
            labels = [_modality(row) if field == "modality" else row.get(field) for row in task]
            if not labels:
                continue
            if any(label != labels[0] or type(label) is not type(labels[0]) for label in labels):
                raise ValueError(f"Inconsistent {field} labels across repeats of a question")
            label = labels[0]
            if not isinstance(label, str) or not label.strip():
                missing += len(task)
                continue
            groups.setdefault(label, []).append(task)
        metrics[f"domains/{namespace}/missing_label_rollouts"] = missing
        for label, questions in sorted(groups.items()):
            prefix = f"domains/{namespace}/{quote(label, safe='')}/"
            rows = [row for task in questions for row in task]
            metrics[prefix + "question_count"] = len(questions)
            metrics[prefix + "rollout_count"] = len(rows)
            coverage_reason = repeat_coverage_reason(questions)
            aggregate = compute_aggregate_metrics(rows).agent_metrics
            for metric in ("reward", "output_tokens"):
                values = [
                    row.get("reward")
                    if metric == "reward"
                    else ((row.get("response") or {}).get("usage") or {}).get("output_tokens")
                    for row in rows
                ]
                missing_values = sum(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    or (metric == "reward" and value > 1)
                    for value in values
                )
                metrics[prefix + f"missing/{metric}"] = missing_values
                metrics[prefix + f"ci_available/{metric}"] = False
                if missing_values:
                    metrics[prefix + f"ci_unavailable_reason/{metric}"] = "Missing or invalid observations"
                    continue
                metrics[prefix + f"mean/{metric}"] = aggregate[f"mean/{metric}"]
                low = aggregate.get(f"ci_low_95_across_repeats/mean/{metric}")
                high = aggregate.get(f"ci_high_95_across_repeats/mean/{metric}")
                if coverage_reason or low is None or high is None:
                    metrics[prefix + f"ci_unavailable_reason/{metric}"] = (
                        coverage_reason or "Gym did not emit a repeat interval"
                    )
                    continue
                metrics[prefix + f"ci_available/{metric}"] = True
                metrics[prefix + f"ci_low_95_across_repeats/{metric}"] = low
                metrics[prefix + f"ci_high_95_across_repeats/{metric}"] = high
                metrics[prefix + f"repeat_count/{metric}"] = aggregate["num_repeats"]
    return metrics
