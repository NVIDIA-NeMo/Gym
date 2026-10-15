# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Question-clustered HLE domain summaries from scored verifier responses."""

import math
import random
from typing import Any
from urllib.parse import quote

from scipy.stats import t


def repeat_confidence_interval(observations: list[tuple]) -> tuple[dict | None, str | None]:
    """Student-t interval across repeats with identical question coverage."""
    repeats = {}
    for question, repeat, value in observations:
        if type(repeat) is not int or repeat < 0:
            return None, "Missing or invalid repeat identity"
        samples = repeats.setdefault(repeat, {})
        if question in samples:
            return None, "Duplicate question within a repeat"
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None, "Missing or invalid observations"
        samples[question] = value
    if len(repeats) < 2:
        return None, "At least two evaluation repeats are required"
    coverage = [set(samples) for samples in repeats.values()]
    if any(keys != coverage[0] for keys in coverage[1:]):
        return None, "Repeat question coverage differs"
    means = [sum(samples.values()) / len(samples) for _, samples in sorted(repeats.items())]
    mean = sum(means) / len(means)
    sem = math.sqrt(sum((value - mean) ** 2 for value in means) / (len(means) - 1) / len(means))
    half_width = float(t.ppf(0.975, len(means) - 1)) * sem if sem else 0.0
    return {
        "level": 0.95,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "half_width": half_width,
        "method": "Student t interval across repeat-level estimates",
        "kind": "across_repeats",
        "sampling_unit": "evaluation repeat on the fixed benchmark",
        "repeat_count": len(means),
    }, None


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
                observations = []
                missing_values = 0
                for question_index, task in enumerate(questions):
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
                            observations.append((question_index, row.get("_ng_rollout_index"), value))
                        else:
                            missing_values += 1
                    clusters.append((sum(values), len(values)))
                metrics[prefix + f"missing/{metric}"] = missing_values
                if missing_values:
                    continue
                metrics[prefix + f"mean/{metric}"] = sum(v for v, _ in clusters) / sum(n for _, n in clusters)
                repeat_interval, reason = repeat_confidence_interval(observations)
                metrics[prefix + f"ci_available/{metric}"] = repeat_interval is not None
                if repeat_interval:
                    metrics[prefix + f"ci_low_95_across_repeats/{metric}"] = repeat_interval["lower"]
                    metrics[prefix + f"ci_high_95_across_repeats/{metric}"] = repeat_interval["upper"]
                    metrics[prefix + f"repeat_count/{metric}"] = repeat_interval["repeat_count"]
                else:
                    metrics[prefix + f"ci_unavailable_reason/{metric}"] = reason
                interval = question_bootstrap_interval(clusters)
                metrics[prefix + f"question_ci_available/{metric}"] = interval is not None
                if interval:
                    metrics[prefix + f"ci_low_95_question_bootstrap/{metric}"] = interval["lower"]
                    metrics[prefix + f"ci_high_95_question_bootstrap/{metric}"] = interval["upper"]
    metrics["domains/ci_method"] = "Student t interval across repeat-level estimates (default)"
    metrics["domains/question_ci_method"] = "percentile bootstrap over question clusters; 2000 resamples; seed 0"
    return metrics
