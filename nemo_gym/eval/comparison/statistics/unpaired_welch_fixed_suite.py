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
"""Normalize repeat scores and estimate fixed-suite unpaired Welch intervals.

Baseline and candidate repeats are independent samples. Benchmarks are a fixed
suite, not a sample of possible benchmarks. Multiple metrics within a benchmark
are averaged *within each repeat* before estimating sample variance, preserving
their covariance. The weighted aggregate uses independent variance contributions
and Welch–Satterthwaite effective degrees of freedom.

This module contains numerical inputs and outputs only; it neither loads Gym
artifacts nor decides which benchmarks are eligible for comparison.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite, sqrt
from statistics import fmean, variance

from scipy.stats import t


@dataclass(frozen=True, slots=True)
class MetricScale:
    """Theoretical metric bounds and direction for conversion to a 0–100 score."""

    low: float
    high: float
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.low) or not isfinite(self.high) or self.high <= self.low:
            raise ValueError("Metric bounds must be finite and high must be greater than low.")
        if not isfinite(self.high - self.low):
            raise ValueError("Metric bounds must have a finite span (high minus low).")
        if not isinstance(self.higher_is_better, bool):
            raise ValueError("Metric higher_is_better must be a boolean.")


@dataclass(frozen=True, slots=True)
class SampleSummary:
    """Mean and unbiased sample variance of independent repeat scores."""

    count: int
    mean: float
    sample_variance: float

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 2:
            raise ValueError("A sample summary requires at least two repeat scores.")
        if not isfinite(self.mean) or not isfinite(self.sample_variance) or self.sample_variance < 0:
            raise ValueError("A sample summary requires a finite mean and finite nonnegative sample variance.")


@dataclass(frozen=True, slots=True)
class BenchmarkSamples:
    """Independent baseline and candidate repeat summaries for one benchmark cell."""

    baseline: SampleSummary
    candidate: SampleSummary


@dataclass(frozen=True, slots=True)
class VarianceContribution:
    """One cell's weighted mean-variance contributions, each equal to w²s²/n."""

    weight: float
    baseline: float
    candidate: float


@dataclass(frozen=True, slots=True)
class WelchResult:
    """Weighted scores, candidate-minus-baseline estimate, and unpaired Welch interval."""

    baseline_mean: float
    candidate_mean: float
    delta: float
    variance: float
    standard_error: float
    degrees_of_freedom: float | None
    ci_low: float | None
    ci_high: float | None
    confidence_level: float
    contributions: tuple[VarianceContribution, ...]


def normalize_score(value: float, scale: MetricScale) -> float:
    """Scale a finite metric to percentage points, reversing lower-is-better metrics.

    Bounds describe the metric, not observed extrema. Scores outside these
    theoretical bounds are rejected; confidence intervals remain unclipped.
    """
    if not isfinite(value):
        raise ValueError("Metric scores must be finite.")
    if not scale.low <= value <= scale.high:
        raise ValueError(f"Metric score {value} is outside theoretical bounds [{scale.low}, {scale.high}].")
    normalized = 100.0 * ((value - scale.low) / (scale.high - scale.low))
    normalized = normalized if scale.higher_is_better else 100.0 - normalized
    if not isfinite(normalized):
        raise ValueError("Metric score and bounds produced a nonfinite normalized score.")
    return normalized


def normalize_and_average_metrics_per_repeat(
    repeat_values: Sequence[Sequence[float]], scales: Sequence[MetricScale]
) -> tuple[float, ...]:
    """Average equally weighted normalized metrics within each repeat.

    Rows represent repeats; columns correspond to ``scales``. Averaging before
    summarizing keeps covariance between metrics from the same repeat. A cell's
    benchmark weight is independent of its number of selected metrics.
    """
    if not scales:
        raise ValueError("At least one metric scale is required.")
    if not repeat_values:
        raise ValueError("At least one repeat row is required.")
    scores = []
    for values in repeat_values:
        if len(values) != len(scales):
            raise ValueError("Every repeat must contain exactly one value per metric scale.")
        scores.append(fmean(normalize_score(value, scale) for value, scale in zip(values, scales, strict=True)))
    return tuple(scores)


def summarize_sample(values: Sequence[float]) -> SampleSummary:
    """Summarize finite repeat scores using sample variance with divisor n − 1."""
    if len(values) < 2:
        raise ValueError("At least two repeat scores are required to estimate sample variance.")
    if not all(isfinite(value) for value in values):
        raise ValueError("Repeat scores must be finite.")
    return SampleSummary(count=len(values), mean=fmean(values), sample_variance=variance(values))


def aggregate_welch(
    comparisons: Sequence[BenchmarkSamples],
    *,
    weights: Sequence[float] | None = None,
    confidence_level: float = 0.90,
) -> WelchResult:
    """Estimate a weighted candidate-minus-baseline delta with Welch–Satterthwaite uncertainty.

    For each cell, ``a = w²s_C²/n_C`` and ``b = w²s_B²/n_B``. Total
    variance is ``V = sum(a + b)``, and effective degrees of freedom are
    ``V² / sum(a²/(n_C−1) + b²/(n_B−1))``. The two-sided interval uses
    Student-t quantile ``(1 + confidence_level) / 2``.

    Weights default to equal values and must sum to one when provided. Constant
    cells retain their weights. If all contributions are zero, the estimate and
    zero standard error are returned with no interval or degrees of freedom.
    """
    if isinstance(confidence_level, bool) or not isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be finite and strictly between zero and one.")
    if not comparisons:
        raise ValueError("At least one benchmark comparison is required.")
    resolved_weights = tuple(weights) if weights is not None else (1.0 / len(comparisons),) * len(comparisons)
    if len(resolved_weights) != len(comparisons):
        raise ValueError("There must be exactly one weight per benchmark comparison.")
    if any(not isfinite(weight) or weight <= 0 for weight in resolved_weights):
        raise ValueError("Benchmark weights must be finite and positive.")
    if not isclose(fsum(resolved_weights), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Benchmark weights must sum to one.")

    contributions = tuple(
        VarianceContribution(
            weight=weight,
            baseline=weight**2 * cell.baseline.sample_variance / cell.baseline.count,
            candidate=weight**2 * cell.candidate.sample_variance / cell.candidate.count,
        )
        for cell, weight in zip(comparisons, resolved_weights, strict=True)
    )
    baseline_mean = fsum(
        weight * cell.baseline.mean for cell, weight in zip(comparisons, resolved_weights, strict=True)
    )
    candidate_mean = fsum(
        weight * cell.candidate.mean for cell, weight in zip(comparisons, resolved_weights, strict=True)
    )
    delta = fsum(
        weight * (cell.candidate.mean - cell.baseline.mean)
        for cell, weight in zip(comparisons, resolved_weights, strict=True)
    )
    total_variance = fsum(part.baseline + part.candidate for part in contributions)
    standard_error = sqrt(total_variance)
    degrees_of_freedom = ci_low = ci_high = None
    if total_variance > 0:
        # Divide contributions by V before squaring: the equivalent dimensionless
        # expression avoids underflow in V² and the squared variance contributions.
        denominator = fsum(
            (part.baseline / total_variance) ** 2 / (cell.baseline.count - 1)
            + (part.candidate / total_variance) ** 2 / (cell.candidate.count - 1)
            for cell, part in zip(comparisons, contributions, strict=True)
        )
        degrees_of_freedom = 1 / denominator
        margin = float(t.isf((1 - confidence_level) / 2, degrees_of_freedom)) * standard_error
        ci_low, ci_high = delta - margin, delta + margin
    return WelchResult(
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=delta,
        variance=total_variance,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=confidence_level,
        contributions=contributions,
    )
