# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Question-clustered HLE domain summaries from scored verifier responses."""

import math
import random
from typing import Any
from urllib.parse import quote


def question_bootstrap_interval(clusters: list[tuple[float, int]]) -> dict | None:
    """95% percentile interval for a rollout mean, resampling whole questions.

    Each pair is a question's sum and rollout count. Repeated generations stay
    together. This estimates question-population uncertainty, not repeat noise.
    """
    if len(clusters) < 2:
        return None
    rng = random.Random(0)
    estimates = []
    for _ in range(2000):
        sampled = rng.choices(clusters, k=len(clusters))
        estimates.append(sum(value for value, _ in sampled) / sum(count for _, count in sampled))
    estimates.sort()

    def percentile(q: float) -> float:
        index = q * (len(estimates) - 1)
        low = math.floor(index)
        return estimates[low] + (estimates[math.ceil(index)] - estimates[low]) * (index - low)

    return {
        "level": 0.95,
        "lower": percentile(0.025),
        "upper": percentile(0.975),
        "method": "percentile bootstrap over question clusters",
        "sampling_unit": "question (all repeated generations kept together)",
        "resamples": 2000,
        "seed": 0,
        "question_count": len(clusters),
    }


def compute_domain_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Emit flat, exporter-friendly category and raw-subject accuracy/token metrics.

    Called after masked rollouts have been excluded by compute_aggregate_metrics.
    A domain is never guessed; missing labels and token usage have explicit counts.
    """
    metrics: dict[str, Any] = {}
    for field in ("category", "raw_subject"):
        groups: dict[str, list[list[dict[str, Any]]]] = {}
        missing = 0
        for task in tasks:
            labels = {row.get(field) for row in task}
            if len(labels) != 1:
                raise ValueError(f"Inconsistent {field} labels across repeats of a question")
            label = next(iter(labels), None)
            if not isinstance(label, str) or not label.strip():
                missing += len(task)
                continue
            groups.setdefault(label, []).append(task)
        metrics[f"domains/{field}/missing_label_rollouts"] = missing
        for label, questions in sorted(groups.items()):
            prefix = f"domains/{field}/{quote(label, safe='')}/"
            metrics[prefix + "question_count"] = len(questions)
            metrics[prefix + "rollout_count"] = sum(map(len, questions))
            for metric in ("reward", "output_tokens"):
                clusters = []
                missing_values = 0
                for task in questions:
                    values = []
                    for row in task:
                        value = (
                            row.get("reward")
                            if metric == "reward"
                            else ((row.get("response") or {}).get("usage") or {}).get("output_tokens")
                        )
                        valid = (
                            not isinstance(value, bool)
                            and isinstance(value, (int, float))
                            and math.isfinite(value)
                            and value >= 0
                            and (metric != "reward" or value <= 1)
                        )
                        if valid:
                            values.append(value)
                        else:
                            missing_values += 1
                    clusters.append((sum(values), len(values)))
                metrics[prefix + f"missing/{metric}"] = missing_values
                if missing_values:
                    continue
                metrics[prefix + f"mean/{metric}"] = sum(v for v, _ in clusters) / sum(n for _, n in clusters)
                interval = question_bootstrap_interval(clusters)
                metrics[prefix + f"ci_available/{metric}"] = interval is not None
                if interval:
                    metrics[prefix + f"ci_low_95_question_bootstrap/{metric}"] = interval["lower"]
                    metrics[prefix + f"ci_high_95_question_bootstrap/{metric}"] = interval["upper"]
    metrics["domains/ci_method"] = "percentile bootstrap over question clusters; 2000 resamples; seed 0"
    return metrics
