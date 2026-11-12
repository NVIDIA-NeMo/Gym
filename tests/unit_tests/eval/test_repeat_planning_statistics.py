# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks of prospective power, integer allocation and actual decisions."""

from dataclasses import replace
from math import sqrt

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2, nct, norm, t

from nemo_gym.eval.comparison.statistics.repeat_planning import (
    PlanningCell,
    PlanningParameters,
    plan_repeats,
    power_for_counts,
    simulate_plan,
)
from nemo_gym.eval.comparison.statistics.unpaired_welch_fixed_suite import (
    BenchmarkSamples,
    SampleSummary,
    aggregate_welch,
)


def test_prospective_welch_uses_same_uncertainty_as_gym_and_hand_arithmetic() -> None:
    cells = [PlanningCell("model", "one", 9.0, 4.5), PlanningCell("model", "two", 18.0, 27.0)]
    parameters = PlanningParameters(margin_points=0.5, expected_delta_points=-0.25)
    result = power_for_counts(cells, [3, 6], [3, 6], parameters=parameters, weights=[1 / 3, 2 / 3])
    gym = aggregate_welch(
        [
            BenchmarkSamples(SampleSummary(3, 0.0, 9.0), SampleSummary(3, 0.0, 4.5)),
            BenchmarkSamples(SampleSummary(6, 0.0, 18.0), SampleSummary(6, 0.0, 27.0)),
        ],
        weights=[1 / 3, 2 / 3],
        confidence_level=0.90,
    )
    assert result.variance == pytest.approx(23 / 6)
    assert result.degrees_of_freedom == pytest.approx(5290 / 441)
    assert result.standard_error == gym.standard_error
    assert result.half_width == gym.ci_high
    assert result.critical_value == pytest.approx(t.isf(0.05, 5290 / 441))
    assert result.power == pytest.approx(nct.sf(result.critical_value, 5290 / 441, 0.25 / sqrt(23 / 6)))


def test_nct_power_matches_independent_normal_chi_square_integral() -> None:
    # Four scores per arm, variance four: mean-difference variance two and df six.
    result = power_for_counts([PlanningCell("m", "b", 4.0, 4.0)], [4], [4], parameters=PlanningParameters())
    critical = t.isf(0.05, 6)
    expected = quad(
        lambda value: norm.sf(critical * sqrt(value / 6) - 1 / sqrt(2)) * chi2.pdf(value, 6),
        0,
        np.inf,
    )[0]
    assert result.power == pytest.approx(expected, abs=1e-9)


def test_constant_cells_keep_their_weights_but_all_zero_variance_is_unavailable() -> None:
    cells = [PlanningCell("m", "constant", 0.0, 0.0), PlanningCell("m", "variable", 4.0, 4.0)]
    result = power_for_counts(cells, [4, 4], [4, 4], parameters=PlanningParameters())
    assert result.variance == 0.5
    assert result.degrees_of_freedom == 6
    with pytest.raises(ValueError, match="all one-pass variances are zero"):
        plan_repeats(cells[:1], parameters=PlanningParameters())


@pytest.mark.parametrize("experiments,step,variance", [(1, 1, 1.0), (7, 2, 0.7), (48, 1, 1.0)])
def test_joint_allocation_matches_independent_exhaustive_cost_oracle(
    experiments: int, step: int, variance: float
) -> None:
    parameters = PlanningParameters(candidate_experiments=experiments, max_repeats=64)
    cell = PlanningCell("m", "b", variance, variance * 1.5, repeat_step=step, repeat_cost=2.5)
    result = plan_repeats([cell], parameters=parameters)
    # This independent SciPy oracle enumerates the whole bounded rectangle,
    # whereas production prunes using a feasible incumbent and batches by arm.
    legal = np.arange(max(step, int(np.ceil(2 / step)) * step), 65, step)
    baseline, candidate = np.meshgrid(legal, legal, indexing="ij")
    a, b = variance / baseline, variance * 1.5 / candidate
    degrees = (a + b) ** 2 / (a**2 / (baseline - 1) + b**2 / (candidate - 1))
    power = nct.sf(t.isf(0.05, degrees), degrees, 1 / np.sqrt(a + b))
    optimum = np.min((baseline + experiments * candidate)[power >= 0.8])
    assert result.total_additional_passes == optimum
    assert result.total_cost == 2.5 * optimum
    assert result.statistics.power >= 0.8
    assert result.baseline_counts[0] % step == result.candidate_counts[0] % step == 0
    assert "exhaustive minimum" in result.method


def test_one_candidate_does_not_force_equal_integer_arm_counts() -> None:
    cell = PlanningCell("m", "b", 1.0, 1.0)
    parameters = PlanningParameters(max_repeats=40)
    result = plan_repeats([cell], parameters=parameters)
    assert result.total_additional_passes == 27
    assert result.statistics.power >= 0.8
    assert power_for_counts([cell], [13], [13], parameters=parameters).power < 0.8
    assert power_for_counts([cell], [14], [14], parameters=parameters).power > 0.8


def test_joint_search_checks_unbalanced_designs_when_no_balanced_seed_is_feasible() -> None:
    cell = PlanningCell("m", "b", 1.0, 0.00001)
    parameters = PlanningParameters(margin_points=1.2589254117941675, target_power=0.426738, max_repeats=3)
    assert power_for_counts([cell], [3], [3], parameters=parameters).power < parameters.target_power
    result = plan_repeats([cell], parameters=parameters)
    assert result.baseline_counts == (3,)
    assert result.candidate_counts == (2,)
    assert result.statistics.power >= parameters.target_power


@pytest.mark.parametrize("mode", ["optimize", "existing"])
def test_infeasible_scalar_bound_avoids_exhaustive_noncentral_t_work(monkeypatch, mode: str) -> None:
    cell = PlanningCell("m", "b", 25.0, 25.0, existing_baseline_repeats=20_000 if mode == "existing" else None)
    parameters = PlanningParameters(baseline_mode=mode, margin_points=0.1)
    real_survival = nct.sf
    evaluations = 0

    def counted_survival(*args, **kwargs):
        nonlocal evaluations
        evaluations += np.broadcast(*args[:3]).size
        # Fail promptly if pruning regresses, instead of running the 400-million-pair search.
        assert evaluations < 1_000
        return real_survival(*args, **kwargs)

    monkeypatch.setattr(nct, "sf", counted_survival)
    with pytest.raises(ValueError, match="max_repeats=20000.*known-variance power upper bound"):
        plan_repeats([cell], parameters=parameters)
    assert evaluations == 0


def test_scalar_bound_preserves_exhaustive_checks_near_roundoff_boundary(monkeypatch) -> None:
    target = 0.8
    distance = float(norm.isf(0.05) + norm.ppf(target) - 2e-12)
    cell = PlanningCell("m", "b", 1.0, 1.0)
    parameters = PlanningParameters(margin_points=distance, max_repeats=2)
    real_survival = nct.sf
    evaluations = 0

    def counted_survival(*args, **kwargs):
        nonlocal evaluations
        evaluations += np.broadcast(*args[:3]).size
        return real_survival(*args, **kwargs)

    monkeypatch.setattr(nct, "sf", counted_survival)
    with pytest.raises(ValueError, match=r"No feasible allocation found within max_repeats=2\.$"):
        plan_repeats([cell], parameters=parameters)
    # The normal bound is just below target; the buffer must leave the real NCT test in charge.
    assert evaluations > 0


def test_existing_baseline_filter_preserves_the_first_feasible_candidate_count(monkeypatch) -> None:
    cell = PlanningCell("m", "b", 1.0, 1.5, existing_baseline_repeats=50)
    parameters = PlanningParameters(baseline_mode="existing", max_repeats=64)
    candidates = np.arange(2, 65)
    baseline_variance, candidate_variance = 1 / 50, 1.5 / candidates
    total_variance = baseline_variance + candidate_variance
    degrees = total_variance**2 / (baseline_variance**2 / 49 + candidate_variance**2 / (candidates - 1))
    independent_power = nct.sf(t.isf(0.05, degrees), degrees, 1 / np.sqrt(total_variance))
    expected = candidates[np.flatnonzero(independent_power >= parameters.target_power)[0]]
    normal_upper = norm.sf(norm.isf(0.05) - 1 / np.sqrt(total_variance))
    assert normal_upper[0] < parameters.target_power - 1e-12

    real_survival = nct.sf
    evaluations = 0

    def counted_survival(*args, **kwargs):
        nonlocal evaluations
        evaluations += np.broadcast(*args[:3]).size
        return real_survival(*args, **kwargs)

    monkeypatch.setattr(nct, "sf", counted_survival)
    result = plan_repeats([cell], parameters=parameters)
    assert result.baseline_counts == (50,)
    assert result.candidate_counts == (expected,)
    assert result.statistics.power >= parameters.target_power
    # Initial rows really were filtered; all retained probabilities use the real SciPy calculation.
    assert 0 < evaluations < len(candidates)


def test_existing_baseline_search_handles_nonmonotone_power() -> None:
    cell = PlanningCell("m", "b", 0.01, 0.01, existing_baseline_repeats=2)
    parameters = PlanningParameters(baseline_mode="existing", margin_points=0.5, target_power=0.95, max_repeats=100)
    result = plan_repeats([cell], parameters=parameters)
    assert result.baseline_counts == (2,)
    assert result.candidate_counts == (3,)
    assert result.statistics.power == pytest.approx(0.9690008427917992)
    assert power_for_counts([cell], [2], [2], parameters=parameters).power < 0.95
    assert power_for_counts([cell], [2], [100], parameters=parameters).power < 0.95
    assert result.additional_baseline_counts == (0,)
    assert result.total_cost == result.total_additional_passes == 3


def test_existing_baseline_need_not_follow_new_repeat_step_or_count_cap() -> None:
    cell = PlanningCell("m", "b", 0.01, 0.01, repeat_step=4, repeat_cost=2.0, existing_baseline_repeats=31)
    result = plan_repeats(
        [cell], parameters=PlanningParameters(baseline_mode="existing", candidate_experiments=3, max_repeats=8)
    )
    assert result.baseline_counts == (31,)
    assert result.candidate_counts == (4,)
    assert result.total_additional_passes == 12
    assert result.total_cost == 24


def test_count_limit_and_infeasible_search_fail_with_actionable_context() -> None:
    with pytest.raises(ValueError, match="max_repeats=2"):
        plan_repeats([PlanningCell("m", "b", 100.0, 100.0)], parameters=PlanningParameters(max_repeats=2))
    with pytest.raises(ValueError, match="No feasible candidate count.*max_repeats=20"):
        plan_repeats(
            [PlanningCell("m", "b", 10.0, 10.0, existing_baseline_repeats=2)],
            parameters=PlanningParameters(baseline_mode="existing", max_repeats=20),
        )
    with pytest.raises(ValueError, match="No legal new count"):
        plan_repeats([PlanningCell("m", "b", 1.0, 1.0, repeat_step=8)], parameters=PlanningParameters(max_repeats=4))


def test_suite_buys_more_cheap_passes_and_explicitly_reports_local_solution() -> None:
    cells = [
        PlanningCell("m", "cheap", 9.0, 9.0, repeat_step=2, repeat_cost=1.0),
        PlanningCell("m", "expensive", 9.0, 9.0, repeat_cost=20.0),
    ]
    parameters = PlanningParameters(max_repeats=200)
    result = plan_repeats(cells, parameters=parameters)
    assert result.statistics.power >= 0.8
    assert (
        result.baseline_counts[0] + result.candidate_counts[0] > result.baseline_counts[1] + result.candidate_counts[1]
    )
    assert result.baseline_counts[0] % 2 == result.candidate_counts[0] % 2 == 0
    assert result.total_cost == sum(result.baseline_counts[:1] + result.candidate_counts[:1]) + 20 * sum(
        result.baseline_counts[1:] + result.candidate_counts[1:]
    )
    assert "global minimum not established" in result.method
    assert result == plan_repeats(cells, parameters=parameters)


def test_suite_existing_baseline_is_fixed_and_sunk_in_cost() -> None:
    cells = [
        PlanningCell("m", "one", 4.0, 4.0, repeat_step=2, repeat_cost=1.0, existing_baseline_repeats=20),
        PlanningCell("m", "two", 9.0, 9.0, repeat_step=4, repeat_cost=3.0, existing_baseline_repeats=30),
    ]
    parameters = PlanningParameters(baseline_mode="existing", candidate_experiments=3, max_repeats=200)
    result = plan_repeats(cells, parameters=parameters)
    assert result.baseline_counts == (20, 30)
    assert result.additional_baseline_counts == (0, 0)
    assert result.statistics.power >= 0.8
    assert result.total_additional_passes == 3 * sum(result.candidate_counts)
    assert result.total_cost == 3 * (result.candidate_counts[0] + 3 * result.candidate_counts[1])


def test_existing_suite_cannot_increase_power_with_deterministic_candidates() -> None:
    cells = [
        PlanningCell("m", "one", 10.0, 0.0, existing_baseline_repeats=2),
        PlanningCell("m", "two", 10.0, 0.0, existing_baseline_repeats=2),
    ]
    with pytest.raises(ValueError, match="new repeats cannot reduce.*baseline uncertainty"):
        plan_repeats(cells, parameters=PlanningParameters(baseline_mode="existing"))


def test_suite_keeps_zero_variance_coordinates_at_their_legal_floors() -> None:
    cells = [PlanningCell("m", "one", 1.0, 0.0), PlanningCell("m", "two", 0.0, 1.0)]
    result = plan_repeats(cells, parameters=PlanningParameters(max_repeats=80))
    assert result.candidate_counts[0] == result.baseline_counts[1] == 2
    assert result.statistics.power >= 0.8


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_experiments", True),
        ("candidate_experiments", 1.5),
        ("max_repeats", 1),
        ("alpha", 0.5),
        ("alpha", np.nan),
        ("target_power", 1.0),
        ("target_power", True),
        ("margin_points", 0),
        ("expected_delta_points", -1.0),
        ("baseline_mode", "fixed"),
    ],
)
def test_invalid_parameters_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(PlanningParameters(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [("baseline_variance", -1), ("candidate_variance", np.inf), ("repeat_cost", 0), ("repeat_step", True)],
)
def test_invalid_cell_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(PlanningCell("m", "b", 1.0, 1.0), **{field: value})


@pytest.mark.parametrize("baseline,candidate", [([True], [4]), ([2], [1]), ([2.5], [4]), ([4], [21]), ([], [4])])
def test_invalid_counts_fail_before_calculation(baseline: list[object], candidate: list[object]) -> None:
    with pytest.raises(ValueError):
        power_for_counts(
            [PlanningCell("m", "b", 1.0, 1.0)], baseline, candidate, parameters=PlanningParameters(max_repeats=20)
        )


def test_identity_weights_and_missing_existing_counts_are_validated() -> None:
    cell = PlanningCell("m", "b", 1.0, 1.0)
    with pytest.raises(ValueError, match="unique"):
        plan_repeats([cell, cell], parameters=PlanningParameters())
    with pytest.raises(ValueError, match="sum to one"):
        plan_repeats([cell], parameters=PlanningParameters(), weights=[0.5])
    with pytest.raises(ValueError, match="existing_baseline_repeats"):
        plan_repeats([cell], parameters=PlanningParameters(baseline_mode="existing"))


def test_simulation_checks_actual_welch_power_separately_from_nct_approximation() -> None:
    cell = PlanningCell("glm", "mmlongbench", 0.21983263058853658, 0.21983263058853658)
    parameters = PlanningParameters(margin_points=2.0, max_repeats=8)
    allocation = plan_repeats([cell], parameters=parameters)
    assert allocation.baseline_counts == allocation.candidate_counts == (2,)
    assert allocation.statistics.power == pytest.approx(0.8402184443678733)
    result = simulate_plan([cell], allocation, parameters=parameters, trials=4_000, seed=20261109)
    # Independent quadrature over the random Welch variance ratio gives this
    # actual-normal power, rather than the fixed-df NCT approximation above.
    assert result.power.estimate == pytest.approx(0.6137932911861711, abs=0.025)
    assert 0.015 < result.false_positive.estimate < 0.045
    assert result.coverage.estimate == 1 - result.false_positive.estimate
    assert result.coverage.successes + result.false_positive.successes == result.trials == 4_000
    assert result.power.ci_low < result.power.estimate < result.power.ci_high
    assert result.seed == 20261109


@pytest.mark.parametrize("field,value", [("trials", 0), ("trials", True), ("seed", -1), ("seed", 1.5)])
def test_simulation_rejects_invalid_trial_count_or_seed(field: str, value: object) -> None:
    cell = PlanningCell("m", "b", 1.0, 1.0)
    parameters = PlanningParameters(max_repeats=40)
    allocation = plan_repeats([cell], parameters=parameters)
    with pytest.raises(ValueError):
        simulate_plan([cell], allocation, parameters=parameters, **{field: value})
