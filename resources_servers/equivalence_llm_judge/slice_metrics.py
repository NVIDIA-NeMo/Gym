# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configured label and cross-label summaries using Gym's shared aggregation."""

import math
from statistics import fmean
from typing import Any
from urllib.parse import quote

from nemo_gym.metrics import compute_aggregate_metrics


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


def compute_slice_metrics(
    tasks: list[list[dict[str, Any]]],
    fields: list[str],
    cross_fields: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Summarize configured labels using the public aggregator's repeat statistics.

    Masked rows are excluded by the parent aggregator. Each question contributes
    its matching rows to a slice; labels may differ across repeats, but a CI is
    only available when the slice covers the same questions in each repeat.
    """
    metrics: dict[str, Any] = {}
    dimensions = [(field,) for field in fields] + [tuple(pair) for pair in cross_fields]
    for columns in dict.fromkeys(dimensions):
        namespace = "/".join(quote(field, safe="") for field in columns)
        groups: dict[tuple[str, ...], list[list[dict[str, Any]]]] = {}
        missing = 0
        for task in tasks:
            by_label: dict[tuple[str, ...], list[dict[str, Any]]] = {}
            for row in task:
                values = [row.get(field) for field in columns]
                if any(value is None or (isinstance(value, str) and not value.strip()) for value in values):
                    missing += 1
                    continue
                labels = tuple(str(value) for value in values)
                by_label.setdefault(labels, []).append(row)
            for labels, rows in by_label.items():
                groups.setdefault(labels, []).append(rows)
        metrics[f"slices/{namespace}/missing_label_rollouts"] = missing
        for labels, questions in sorted(groups.items()):
            label_path = "/".join(quote(label, safe="") for label in labels)
            prefix = f"slices/{namespace}/{label_path}/"
            rows = [row for task in questions for row in task]
            metrics[prefix + "question_count"] = len(questions)
            metrics[prefix + "rollout_count"] = len(rows)
            coverage_reason = repeat_coverage_reason(questions)
            # Invalid repeat identities cannot be passed to Gym's repeat grouper.
            # Row means remain meaningful, but no interval is available.
            aggregate = compute_aggregate_metrics(rows).agent_metrics if coverage_reason is None else {}
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
                    or (metric == "output_tokens" and value < 0)
                    for value in values
                )
                metrics[prefix + f"missing/{metric}"] = missing_values
                metrics[prefix + f"ci_available/{metric}"] = False
                if missing_values:
                    metrics[prefix + f"ci_unavailable_reason/{metric}"] = "Missing or invalid observations"
                    continue
                metrics[prefix + f"mean/{metric}"] = fmean(values)
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
