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

import math

import pytest
from scipy import stats

from nemo_gym.comparison.statistics.unpaired_welch_fixed_suite import (
    BenchmarkSamples,
    MetricScale,
    SampleSummary,
    aggregate_welch,
    normalize_and_average_metrics_per_repeat,
    normalize_score,
    summarize_sample,
)


def _cell(baseline: list[float], candidate: list[float]) -> BenchmarkSamples:
    return BenchmarkSamples(summarize_sample(baseline), summarize_sample(candidate))


@pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.80])
def test_single_benchmark_matches_scipy_with_unequal_counts_and_variances(confidence_level: float) -> None:
    baseline = [10.0, 10.0, 20.0]
    candidate = [10.0, 30.0, 50.0, 70.0, 90.0]
    reference = stats.ttest_ind(candidate, baseline, equal_var=False)
    interval = reference.confidence_interval(confidence_level=confidence_level)

    result = aggregate_welch([_cell(baseline, candidate)], confidence_level=confidence_level)

    assert result.baseline_mean == pytest.approx(40 / 3)
    assert result.candidate_mean == 50
    assert result.delta == pytest.approx(50 - 40 / 3)
    assert result.degrees_of_freedom == pytest.approx(reference.df)
    assert (result.ci_low, result.ci_high) == pytest.approx((interval.low, interval.high))
    assert result.confidence_level == confidence_level


def test_two_benchmark_fixture_matches_hand_calculated_welch_contributions() -> None:
    cells = [_cell([0, 20, 40], [20, 40, 60, 80]), _cell([20, 40], [40, 60, 80])]

    result = aggregate_welch(cells)

    assert result.baseline_mean == 25
    assert result.candidate_mean == 55
    assert result.delta == 30
    assert [part.weight for part in result.contributions] == [0.5, 0.5]
    assert [(part.baseline, part.candidate) for part in result.contributions] == [
        pytest.approx((100 / 3, 125 / 3)),
        pytest.approx((25, 100 / 3)),
    ]
    assert result.variance == pytest.approx(400 / 3)
    assert result.standard_error == pytest.approx(math.sqrt(400 / 3))
    assert result.degrees_of_freedom == pytest.approx(7.68)
    assert (result.ci_low, result.ci_high) == pytest.approx((8.411227252222858, 51.58877274777714))


def test_custom_weights_affect_means_linearly_and_variance_quadratically() -> None:
    cell_a = _cell([0, 20, 40], [20, 40, 60])
    cell_b = _cell([20, 40, 60], [60, 80, 100])

    result = aggregate_welch([cell_a, cell_b], weights=[0.25, 0.75])

    assert result.baseline_mean == 35
    assert result.candidate_mean == 70
    assert result.delta == 35
    assert result.contributions[0].baseline == pytest.approx(25 / 3)
    assert result.contributions[1].baseline == 75
    assert result.variance == pytest.approx(500 / 3)


def test_reordering_cells_preserves_result_and_swapping_sides_reverses_interval() -> None:
    cells = [_cell([0, 20, 40], [20, 40, 60, 80]), _cell([20, 40], [40, 60, 80])]
    original = aggregate_welch(cells)
    reordered = aggregate_welch(list(reversed(cells)))
    reversed_sides = aggregate_welch([BenchmarkSamples(cell.candidate, cell.baseline) for cell in cells])

    assert reordered.delta == original.delta
    assert reordered.variance == original.variance
    assert reordered.degrees_of_freedom == original.degrees_of_freedom
    assert (reordered.ci_low, reordered.ci_high) == (original.ci_low, original.ci_high)
    assert reversed_sides.delta == -original.delta
    assert (reversed_sides.ci_low, reversed_sides.ci_high) == pytest.approx((-original.ci_high, -original.ci_low))


def test_normalization_uses_theoretical_bounds_and_reverses_lower_is_better() -> None:
    assert normalize_score(0, MetricScale(-20, 80)) == 20
    assert normalize_score(0, MetricScale(-20, 80, higher_is_better=False)) == 80
    assert normalize_score(-20, MetricScale(-20, 80)) == 0
    assert normalize_score(80, MetricScale(-20, 80)) == 100
    assert normalize_and_average_metrics_per_repeat(
        [[0, 100], [50, 50]], [MetricScale(0, 100), MetricScale(0, 100)]
    ) == (50, 50)


def test_identical_metrics_preserve_repeat_variance_instead_of_halving_it() -> None:
    single = normalize_and_average_metrics_per_repeat([[0], [0.5], [1]], [MetricScale(0, 1)])
    duplicated = normalize_and_average_metrics_per_repeat([[0, 0], [0.5, 0.5], [1, 1]], [MetricScale(0, 1)] * 2)

    assert duplicated == single == (0, 50, 100)
    assert summarize_sample(duplicated).sample_variance == 2500
    assert summarize_sample(duplicated) == summarize_sample(single)


def test_anticorrelated_metrics_cancel_variance_within_repeats() -> None:
    composites = normalize_and_average_metrics_per_repeat([[0, 1], [0.5, 0.5], [1, 0]], [MetricScale(0, 1)] * 2)

    assert composites == (50, 50, 50)
    assert summarize_sample(composites) == SampleSummary(count=3, mean=50, sample_variance=0)


def test_benchmarks_keep_equal_weight_when_metric_counts_differ() -> None:
    baseline_a = normalize_and_average_metrics_per_repeat([[0], [0.2], [0.4]], [MetricScale(0, 1)])
    candidate_a = normalize_and_average_metrics_per_repeat([[0.2], [0.4], [0.6]], [MetricScale(0, 1)])
    baseline_b = normalize_and_average_metrics_per_repeat(
        [[0, 0, 0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]], [MetricScale(0, 1)] * 3
    )
    candidate_b = normalize_and_average_metrics_per_repeat(
        [[0.4, 0.4, 0.4], [0.6, 0.6, 0.6], [0.8, 0.8, 0.8]], [MetricScale(0, 1)] * 3
    )

    result = aggregate_welch(
        [
            BenchmarkSamples(summarize_sample(baseline_a), summarize_sample(candidate_a)),
            BenchmarkSamples(summarize_sample(baseline_b), summarize_sample(candidate_b)),
        ]
    )

    assert result.delta == 30
    assert result.baseline_mean == 20
    assert result.candidate_mean == 50
    assert [part.weight for part in result.contributions] == [0.5, 0.5]


def test_sample_summary_uses_unbiased_variance() -> None:
    assert summarize_sample([10, 20, 30]) == SampleSummary(count=3, mean=20, sample_variance=100)


def test_zero_variance_keeps_estimate_and_reports_no_interval() -> None:
    result = aggregate_welch([_cell([10, 10], [30, 30, 30])])

    assert result.delta == 20
    assert result.variance == result.standard_error == 0
    assert result.degrees_of_freedom is result.ci_low is result.ci_high is None


def test_constant_benchmark_keeps_its_weight_and_variable_benchmark_determines_uncertainty() -> None:
    result = aggregate_welch([_cell([0, 0], [100, 100]), _cell([20, 40, 60], [20, 40, 60])])

    assert result.delta == 50
    assert result.contributions[0].weight == 0.5
    assert result.contributions[0].baseline == result.contributions[0].candidate == 0
    assert result.standard_error == pytest.approx(math.sqrt(200 / 3))
    assert result.degrees_of_freedom == pytest.approx(4)
    assert result.ci_low < 50 < result.ci_high


def test_raising_confidence_level_widens_interval_without_changing_estimate() -> None:
    cell = _cell([0, 20, 40], [20, 40, 60])
    default = aggregate_welch([cell])
    wider = aggregate_welch([cell], confidence_level=0.95)

    assert default.confidence_level == 0.90
    assert default.delta == wider.delta
    assert default.standard_error == wider.standard_error
    assert wider.ci_low < default.ci_low < default.ci_high < wider.ci_high


def test_tiny_nonzero_variance_preserves_degrees_of_freedom_and_interval() -> None:
    ordinary = aggregate_welch([_cell([0, 1, 2], [0, 3, 6])])
    tiny = aggregate_welch([_cell([0, 1e-100, 2e-100], [0, 3e-100, 6e-100])])

    assert tiny.variance > 0
    assert tiny.degrees_of_freedom == pytest.approx(100 / 41)
    assert tiny.standard_error == pytest.approx(math.sqrt(10 / 3) * 1e-100, abs=0)
    assert tiny.delta == pytest.approx(2e-100, abs=0)
    assert (tiny.ci_low / 1e-100, tiny.ci_high / 1e-100) == pytest.approx((ordinary.ci_low, ordinary.ci_high))


def test_confidence_level_just_below_one_produces_finite_interval() -> None:
    confidence = math.nextafter(1.0, 0.0)
    result = aggregate_welch([_cell([0, 1], [0, 1])], confidence_level=confidence)
    tail = (1 - confidence) / 2
    # Analytic inverse survival function for Student-t with two degrees of freedom.
    expected_quantile = (1 - 2 * tail) / math.sqrt(2 * tail * (1 - tail))
    expected_margin = expected_quantile / math.sqrt(2)

    assert result.degrees_of_freedom == 2
    assert math.isfinite(result.ci_low) and math.isfinite(result.ci_high)
    assert (result.ci_low, result.ci_high) == pytest.approx((-expected_margin, expected_margin))


def test_finite_bounds_with_overflowing_span_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite span"):
        MetricScale(-1e308, 1e308)


@pytest.mark.parametrize("low,high", [(0, 0), (1, 0), (math.nan, 1), (0, math.inf), (-math.inf, 1)])
def test_invalid_theoretical_bounds_are_rejected(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="bounds"):
        MetricScale(low, high)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_scores_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_score(value, MetricScale(0, 1))
    with pytest.raises(ValueError, match="finite"):
        summarize_sample([0, value])


@pytest.mark.parametrize("value", [-21, 81])
def test_scores_outside_theoretical_bounds_are_rejected_instead_of_clipped(value: float) -> None:
    with pytest.raises(ValueError, match="outside theoretical bounds"):
        normalize_score(value, MetricScale(-20, 80))


@pytest.mark.parametrize("values", [[], [1]])
def test_insufficient_repeats_are_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError, match="At least two"):
        summarize_sample(values)


@pytest.mark.parametrize("values,scales", [([], [MetricScale(0, 1)]), ([[1]], []), ([[1, 2]], [MetricScale(0, 1)])])
def test_composites_require_nonempty_aligned_repeat_rows(values: list[list[float]], scales: list[MetricScale]) -> None:
    with pytest.raises(ValueError):
        normalize_and_average_metrics_per_repeat(values, scales)


@pytest.mark.parametrize("confidence_level", [0, 1, -0.1, 1.1, math.nan, math.inf, True])
def test_invalid_confidence_levels_are_rejected(confidence_level: float) -> None:
    with pytest.raises(ValueError, match="Confidence level"):
        aggregate_welch([_cell([0, 1], [0, 1])], confidence_level=confidence_level)


@pytest.mark.parametrize("weights", [[0], [-1], [math.nan], [math.inf], [0.5], [0.5, 0.5]])
def test_invalid_weights_are_rejected(weights: list[float]) -> None:
    with pytest.raises(ValueError, match="weight"):
        aggregate_welch([_cell([0, 1], [0, 1])], weights=weights)


def test_empty_suite_is_rejected() -> None:
    with pytest.raises(ValueError, match="At least one benchmark"):
        aggregate_welch([])


@pytest.mark.parametrize("count,mean,sample_variance", [(1, 0, 0), (2, math.inf, 0), (2, 0, -1), (2, 0, math.nan)])
def test_summary_inputs_cannot_bypass_inference_validation(count: int, mean: float, sample_variance: float) -> None:
    with pytest.raises(ValueError):
        SampleSummary(count, mean, sample_variance)
