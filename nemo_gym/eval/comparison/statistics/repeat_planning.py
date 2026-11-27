# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan complete benchmark passes with prospective unpaired Welch power.

Inputs are one-pass variances in normalized score points squared. The
noncentral-t power approximation holds those variances fixed. Simulations
separately evaluate the actual Welch decision with newly estimated variances;
they never change an allocation. This module performs no artifact I/O.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite, sqrt
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import nct, norm, t

from nemo_gym.eval.comparison.statistics.unpaired_welch_fixed_suite import (
    BenchmarkSamples,
    SampleSummary,
    aggregate_welch,
)


def _real(value: object, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")


def _integer(value: object, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")


@dataclass(frozen=True, slots=True)
class PlanningCell:
    """One model/benchmark's pass variances, legal increment and per-pass cost.

    Baseline and candidate costs share the same arbitrary positive cost unit.
    Existing baseline passes are already available and are a sunk cost.
    """

    model: str
    benchmark: str
    baseline_variance: float
    candidate_variance: float
    repeat_step: int = 1
    repeat_cost: float = 1.0
    existing_baseline_repeats: int | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.model, self.benchmark)):
            raise ValueError("Model and benchmark must be nonempty names.")
        for name in ("baseline_variance", "candidate_variance"):
            value = getattr(self, name)
            _real(value, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        _integer(self.repeat_step, "repeat_step")
        _real(self.repeat_cost, "repeat_cost", positive=True)
        if self.existing_baseline_repeats is not None:
            _integer(self.existing_baseline_repeats, "existing_baseline_repeats", minimum=2)


@dataclass(frozen=True, slots=True)
class PlanningParameters:
    """Fixed, pre-run non-inferiority design settings.

    ``expected_delta_points`` is candidate minus baseline. Success means a
    one-sided lower bound exceeds ``-margin_points``. ``alpha=.05`` uses the
    lower endpoint of a two-sided 90% interval. Candidate experiments share
    one baseline and each receive the returned candidate counts; power remains
    marginal for one comparison. ``max_repeats`` bounds newly planned counts.
    """

    baseline_mode: Literal["optimize", "existing"] = "optimize"
    candidate_experiments: int = 1
    margin_points: float = 1.0
    target_power: float = 0.8
    alpha: float = 0.05
    expected_delta_points: float = 0.0
    max_repeats: int = 20_000

    def __post_init__(self) -> None:
        if self.baseline_mode not in ("optimize", "existing"):
            raise ValueError("baseline_mode must be optimize or existing.")
        _integer(self.candidate_experiments, "candidate_experiments")
        _integer(self.max_repeats, "max_repeats", minimum=2)
        _real(self.margin_points, "margin_points", positive=True)
        _real(self.expected_delta_points, "expected_delta_points")
        _real(self.target_power, "target_power")
        _real(self.alpha, "alpha")
        if not 0 < self.target_power < 1 or not 0 < self.alpha < 0.5:
            raise ValueError("target_power must be between zero and one; alpha must be between zero and 0.5.")
        distance = self.margin_points + self.expected_delta_points
        if not isfinite(distance) or distance <= 0:
            raise ValueError("Expected delta must be strictly above the negative non-inferiority margin.")


@dataclass(frozen=True, slots=True)
class PlanningStatistics:
    """Prospective Welch uncertainty and approximate noncentral-t power."""

    variance: float
    standard_error: float
    degrees_of_freedom: float
    critical_value: float
    half_width: float
    power: float


@dataclass(frozen=True, slots=True)
class RepeatAllocation:
    """Counts in input-cell order, with baseline paid once and candidates K times."""

    baseline_counts: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    statistics: PlanningStatistics
    additional_baseline_counts: tuple[int, ...]
    total_additional_passes: int
    total_cost: float
    method: str


@dataclass(frozen=True, slots=True)
class SimulationRate:
    """A simulated event frequency and its pointwise 95% Wilson interval."""

    successes: int
    trials: int
    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Actual-Welch diagnostics at fixed supplied population variances."""

    trials: int
    seed: int
    power: SimulationRate
    false_positive: SimulationRate
    coverage: SimulationRate


def _inputs(cells: Sequence[PlanningCell], weights: Sequence[float] | None) -> tuple[float, ...]:
    if not cells or any(not isinstance(cell, PlanningCell) for cell in cells):
        raise ValueError("At least one PlanningCell is required.")
    identities = {(cell.model, cell.benchmark) for cell in cells}
    if len(identities) != len(cells):
        raise ValueError("Model/benchmark cells must be unique.")
    cell_weights = (1.0 / len(cells),) * len(cells) if weights is None else tuple(weights)
    if len(cell_weights) != len(cells):
        raise ValueError("Exactly one weight is required per cell.")
    for weight in cell_weights:
        _real(weight, "weight", positive=True)
    try:
        total_weight = fsum(cell_weights)
    except OverflowError as error:
        raise ValueError("Cell weights overflowed; they must sum to one.") from error
    if not isclose(total_weight, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Cell weights must sum to one.")
    if not any(cell.baseline_variance > 0 or cell.candidate_variance > 0 for cell in cells):
        raise ValueError("Power is unavailable when all one-pass variances are zero.")
    return cell_weights


def _counts(
    cells: Sequence[PlanningCell],
    baseline_counts: Sequence[int],
    candidate_counts: Sequence[int],
    parameters: PlanningParameters,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(parameters, PlanningParameters):
        raise ValueError("parameters must be PlanningParameters.")
    validated_baseline_counts, validated_candidate_counts = tuple(baseline_counts), tuple(candidate_counts)
    if len(validated_baseline_counts) != len(cells) or len(validated_candidate_counts) != len(cells):
        raise ValueError("Exactly one count per arm is required for every cell.")
    for cell, baseline_count, candidate_count in zip(
        cells, validated_baseline_counts, validated_candidate_counts, strict=True
    ):
        for count in (baseline_count, candidate_count):
            _integer(count, "Repeat count", minimum=2)
        if candidate_count % cell.repeat_step or candidate_count > parameters.max_repeats:
            raise ValueError("Candidate counts must follow repeat_step and not exceed max_repeats.")
        if parameters.baseline_mode == "existing":
            if baseline_count != cell.existing_baseline_repeats:
                raise ValueError("Baseline count must equal existing_baseline_repeats.")
        elif baseline_count % cell.repeat_step or baseline_count > parameters.max_repeats:
            raise ValueError("New baseline counts must follow repeat_step and not exceed max_repeats.")
    return tuple(int(repeat_count) for repeat_count in validated_baseline_counts), tuple(
        int(repeat_count) for repeat_count in validated_candidate_counts
    )


def power_for_counts(
    cells: Sequence[PlanningCell],
    baseline_counts: Sequence[int],
    candidate_counts: Sequence[int],
    *,
    parameters: PlanningParameters,
    weights: Sequence[float] | None = None,
) -> PlanningStatistics:
    """Evaluate fixed counts using Gym's Welch uncertainty and NCT power.

    Normalized one-pass variances are treated as fixed design inputs. The
    noncentrality is (expected delta + margin) / standard error. The NCT
    distribution approximates the future Welch statistic, whose estimated
    degrees of freedom vary between experiments.
    """
    cell_weights = _inputs(cells, weights)
    baseline_repeat_counts, candidate_repeat_counts = _counts(cells, baseline_counts, candidate_counts, parameters)
    benchmark_samples = tuple(
        BenchmarkSamples(
            baseline=SampleSummary(count=baseline_count, mean=0.0, sample_variance=cell.baseline_variance),
            candidate=SampleSummary(count=candidate_count, mean=0.0, sample_variance=cell.candidate_variance),
        )
        for cell, baseline_count, candidate_count in zip(
            cells, baseline_repeat_counts, candidate_repeat_counts, strict=True
        )
    )
    welch_statistics = aggregate_welch(
        benchmark_samples, weights=cell_weights, confidence_level=1 - 2 * parameters.alpha
    )
    if (
        welch_statistics.degrees_of_freedom is None
        or welch_statistics.ci_high is None
        or welch_statistics.standard_error == 0
    ):
        raise ValueError("Power is unavailable because the combined Welch variance is zero.")
    critical_value = welch_statistics.ci_high / welch_statistics.standard_error
    if not all(
        isfinite(value)
        for value in (
            welch_statistics.variance,
            welch_statistics.standard_error,
            welch_statistics.degrees_of_freedom,
            critical_value,
        )
    ):
        raise ValueError("Welch uncertainty is nonfinite for the supplied variances, alpha and counts.")
    noncentrality = (parameters.expected_delta_points + parameters.margin_points) / welch_statistics.standard_error
    power = float(nct.sf(critical_value, welch_statistics.degrees_of_freedom, noncentrality))
    if not isfinite(power):
        raise ValueError("Noncentral-t power is nonfinite for the supplied variances and counts.")
    return PlanningStatistics(
        variance=welch_statistics.variance,
        standard_error=welch_statistics.standard_error,
        degrees_of_freedom=welch_statistics.degrees_of_freedom,
        critical_value=critical_value,
        half_width=welch_statistics.ci_high,
        power=power,
    )


def _power(
    cells: Sequence[PlanningCell],
    repeat_counts: NDArray[np.float64],
    weights: tuple[float, ...],
    parameters: PlanningParameters,
) -> NDArray[np.float64]:
    """Evaluate one-sided non-inferiority power for one design or a batch.

    For cell i and arm a (baseline or candidate), let v_ia be its forecast
    one-pass variance, n_ia its complete-repeat count, and w_i its cell weight.
    The code evaluates the following equations, summing over both indices::

        q_ia = w_i**2 * v_ia / n_ia
        V = sum(q_ia)
        f_ia = q_ia / V
        df = 1 / sum(f_ia**2 / (n_ia - 1))
        noncentrality = (expected_delta_points + margin_points) / sqrt(V)
        critical = t.isf(alpha, df)  # t quantile at 1 - alpha
        power = nct.sf(critical, df, noncentrality)  # 1 - F_nct(critical; df, noncentrality)

    Delta means candidate minus baseline; the null boundary is -margin.
    The df expression equals V**2 / sum(q_ia**2 / (n_ia - 1)), using fractions
    for numerical stability. This is a prospective NCT approximation with
    fixed forecast variances; actual Welch decisions re-estimate them and df.

    Formula references:
    - NCSS, pp. 522-2 and 522-4: the null boundary and one-benchmark NCT formula.
      https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Two-Sample_T-Tests_for_Non-Inferiority_Allowing_Unequal_Variance.pdf#page=4
    - NIST: the weighted Welch-Satterthwaite generalization used across cells.
      https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/welchsat.htm
    - SciPy: the noncentral t parameterization and survival function (sf).
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html
    """
    pass_variances = np.array([(cell.baseline_variance, cell.candidate_variance) for cell in cells])
    counts_by_cell = repeat_counts.reshape(*repeat_counts.shape[:-1], len(cells), 2)
    variance_contributions = np.square(weights)[:, None] * pass_variances / counts_by_cell
    total_variance = variance_contributions.sum(axis=(-2, -1))
    variance_fractions = variance_contributions / total_variance[..., None, None]
    inverse_degrees_of_freedom = (variance_fractions**2 / (counts_by_cell - 1)).sum(axis=(-2, -1))
    degrees_of_freedom = 1 / inverse_degrees_of_freedom
    critical_value = t.isf(parameters.alpha, degrees_of_freedom)
    distance_from_null = parameters.expected_delta_points + parameters.margin_points
    power_values = np.asarray(nct.sf(critical_value, degrees_of_freedom, distance_from_null / np.sqrt(total_variance)))
    if not np.isfinite(power_values).all():
        raise ValueError("Nonfinite power during repeat search; check the variance scale.")
    return power_values


def _limits(cell: PlanningCell, parameters: PlanningParameters) -> tuple[int, int]:
    """Return scalar bounds for newly collected complete-repeat counts."""
    minimum_count = max(2, cell.repeat_step)
    maximum_count = parameters.max_repeats // cell.repeat_step * cell.repeat_step
    if maximum_count < minimum_count:
        raise ValueError("No legal new count fits repeat_step and max_repeats.")
    return minimum_count, maximum_count


def _scalar_counts(
    cells: Sequence[PlanningCell], parameters: PlanningParameters, weights: tuple[float, ...]
) -> tuple[NDArray[np.float64], str]:
    # Determine the legal repeat-count range and increment for this benchmark.
    cell = cells[0]
    repeat_step = cell.repeat_step
    minimum_count, maximum_count = _limits(cell, parameters)

    # Find the smallest candidate count when the baseline is already available.
    if parameters.baseline_mode == "existing":
        if cell.existing_baseline_repeats is None:
            raise ValueError("existing baseline mode requires existing_baseline_repeats for every cell.")
        candidate_counts = np.arange(minimum_count, maximum_count + 1, repeat_step, dtype=float)
        count_pairs = np.column_stack(
            (np.full(len(candidate_counts), cell.existing_baseline_repeats, dtype=float), candidate_counts)
        )
        meets_target_power = _power(cells, count_pairs, weights, parameters) >= parameters.target_power
        if not meets_target_power.any():
            raise ValueError(f"No feasible candidate count found within max_repeats={parameters.max_repeats}.")
        return count_pairs[meets_target_power.argmax()], "minimum candidate count within the configured bound"

    # Find a feasible starting allocation without exceeding the repeat-count cap.
    selected_counts = np.full(2, minimum_count, dtype=float)
    feasible_plan_found = True
    while _power(cells, selected_counts, weights, parameters) < parameters.target_power:
        increased_counts = np.minimum(float(maximum_count), selected_counts * 2)
        if np.array_equal(increased_counts, selected_counts):
            feasible_plan_found = False
            break
        selected_counts = increased_counts

    # Bound total work while retaining the whole grid if no feasible start was found.
    candidate_experiments = parameters.candidate_experiments
    best_total_passes = int(selected_counts[0] + candidate_experiments * selected_counts[1])
    if not feasible_plan_found:
        best_total_passes += 1

    # Check every cheaper allocation to certify the minimum total work.
    for candidate_count in range(
        minimum_count,
        min(maximum_count, (best_total_passes - minimum_count - 1) // candidate_experiments) + 1,
        repeat_step,
    ):
        maximum_baseline_count = min(maximum_count, best_total_passes - candidate_experiments * candidate_count - 1)
        baseline_counts = np.arange(minimum_count, maximum_baseline_count + 1, repeat_step, dtype=float)
        if not len(baseline_counts):
            continue
        count_pairs = np.column_stack((baseline_counts, np.full(len(baseline_counts), candidate_count)))
        meets_target_power = _power(cells, count_pairs, weights, parameters) >= parameters.target_power
        if meets_target_power.any():
            selected_counts = count_pairs[meets_target_power.argmax()]
            best_total_passes = int(selected_counts[0] + candidate_experiments * candidate_count)
            feasible_plan_found = True

    # Report infeasibility or return the cheapest feasible allocation.
    if not feasible_plan_found:
        raise ValueError(f"No feasible allocation found within max_repeats={parameters.max_repeats}.")
    return selected_counts, "exhaustive minimum of the bounded integer analytic-power model"


def _normal_start(
    parameters: PlanningParameters,
    variance_coefficients: NDArray[np.float64],
    minimum_counts: NDArray[np.float64],
    maximum_counts: NDArray[np.float64],
    repeat_steps: NDArray[np.float64],
    repeat_costs: NDArray[np.float64],
) -> NDArray[np.float64]:
    distance_from_null = parameters.expected_delta_points + parameters.margin_points
    variance_limit = (distance_from_null / (norm.isf(parameters.alpha) + norm.ppf(parameters.target_power))) ** 2
    adjustable_arms = (repeat_costs > 0) & (variance_coefficients > 0)
    if not adjustable_arms.any():
        return minimum_counts.copy()

    def continuous(multiplier: float) -> NDArray[np.float64]:
        counts = minimum_counts.copy()
        counts[adjustable_arms] = np.clip(
            np.sqrt(multiplier * variance_coefficients[adjustable_arms] / repeat_costs[adjustable_arms]),
            minimum_counts[adjustable_arms],
            maximum_counts[adjustable_arms],
        )
        return counts

    upper_multiplier = 1.0
    for _ in range(1024):
        proposed_counts = continuous(upper_multiplier)
        if np.sum(variance_coefficients / proposed_counts) <= variance_limit or np.all(
            proposed_counts[adjustable_arms] == maximum_counts[adjustable_arms]
        ):
            break
        if upper_multiplier > np.finfo(float).max / 4:
            return minimum_counts.copy()
        upper_multiplier *= 4
    lower_multiplier = 0.0
    for _ in range(60):
        midpoint_multiplier = (lower_multiplier + upper_multiplier) / 2
        if np.sum(variance_coefficients / continuous(midpoint_multiplier)) <= variance_limit:
            upper_multiplier = midpoint_multiplier
        else:
            lower_multiplier = midpoint_multiplier
    return np.maximum(minimum_counts, np.floor(continuous(upper_multiplier) / repeat_steps) * repeat_steps)


def _suite_counts(
    cells: Sequence[PlanningCell], parameters: PlanningParameters, weights: tuple[float, ...]
) -> tuple[NDArray[np.float64], str]:
    # Determine the legal count range, increment, and cost for both arms of every cell.
    repeat_limits = np.array([_limits(cell, parameters) for cell in cells], dtype=float)
    minimum_counts = np.repeat(repeat_limits[:, 0], 2)
    maximum_counts = np.repeat(repeat_limits[:, 1], 2)
    repeat_steps = np.repeat([cell.repeat_step for cell in cells], 2).astype(float)
    repeat_costs = np.array(
        [(cell.repeat_cost, cell.repeat_cost * parameters.candidate_experiments) for cell in cells]
    )

    # Keep existing baseline counts fixed and exclude their collection cost from new work.
    if parameters.baseline_mode == "existing":
        for index, cell in enumerate(cells):
            if cell.existing_baseline_repeats is None:
                raise ValueError("existing baseline mode requires existing_baseline_repeats for every cell.")
            minimum_counts[2 * index] = maximum_counts[2 * index] = cell.existing_baseline_repeats
        repeat_costs[:, 0] = 0.0
    repeat_costs = repeat_costs.ravel()

    # Start at the legal minimum, using a normal approximation if more repeats are needed.
    selected_counts = minimum_counts.copy()
    variance_coefficients = (
        np.square(weights)[:, None] * [(c.baseline_variance, c.candidate_variance) for c in cells]
    ).ravel()
    if _power(cells, selected_counts, weights, parameters) < parameters.target_power:
        if not np.any((repeat_costs > 0) & (variance_coefficients > 0)):
            raise ValueError(
                "No feasible suite allocation: new repeats cannot reduce the existing baseline uncertainty."
            )
        selected_counts = _normal_start(
            parameters, variance_coefficients, minimum_counts, maximum_counts, repeat_steps, repeat_costs
        )

    # Reach target power by adding the repeat increment with the greatest power gain per cost.
    for _ in range(20_000):
        current_power = _power(cells, selected_counts, weights, parameters)
        if current_power >= parameters.target_power:
            break
        indices = np.flatnonzero((selected_counts + repeat_steps <= maximum_counts) & (variance_coefficients > 0))
        if not len(indices):
            raise ValueError(f"No feasible suite allocation found within max_repeats={parameters.max_repeats}.")
        proposals = np.repeat(selected_counts[None, :], len(indices), axis=0)
        proposals[np.arange(len(indices)), indices] += repeat_steps[indices]
        power_gain_per_cost = (_power(cells, proposals, weights, parameters) - current_power) / (
            repeat_costs[indices] * repeat_steps[indices]
        )
        selected_counts = proposals[int(np.argmax(power_gain_per_cost))]
    else:
        raise ValueError("Suite allocation exceeded its finite repair search.")

    # Lower cost through bounded local removals and exchanges while preserving target power.
    for _ in range(1_000):
        best_counts, best_cost = selected_counts, float(np.dot(repeat_costs, selected_counts))

        # Look for cheaper allocations by removing one repeat increment at a time.
        for index in np.flatnonzero(selected_counts > minimum_counts):
            reduced_counts = selected_counts.copy()
            reduced_counts[index] -= repeat_steps[index]
            proposals = [reduced_counts]

            # Limit exchange partners to the top four other arms by variance reduction per cost.
            variance_reduction_per_cost = np.divide(
                variance_coefficients,
                reduced_counts * (reduced_counts + repeat_steps) * repeat_costs,
                out=np.zeros_like(repeat_costs),
                where=repeat_costs > 0,
            )
            exchange_indices = [
                arm_index
                for arm_index in np.argsort(-variance_reduction_per_cost, kind="stable")
                if arm_index != index and repeat_costs[arm_index] > 0
            ][:4]

            # Try cheaper exchanges with up to 20 increments in one other arm.
            for other_arm_index in exchange_indices:
                maximum_added_steps = min(
                    20,
                    int(
                        (maximum_counts[other_arm_index] - reduced_counts[other_arm_index])
                        / repeat_steps[other_arm_index]
                    ),
                )
                for added_steps in range(1, maximum_added_steps + 1):
                    proposed_counts = reduced_counts.copy()
                    proposed_counts[other_arm_index] += added_steps * repeat_steps[other_arm_index]
                    if np.dot(repeat_costs, proposed_counts) < best_cost:
                        proposals.append(proposed_counts)

            # Select the cheapest proposal whose power still meets the target.
            designs = np.stack(proposals)
            design_powers = _power(cells, designs, weights, parameters)
            design_costs = designs @ repeat_costs
            feasible_indices = np.flatnonzero((design_powers >= parameters.target_power) & (design_costs < best_cost))
            if len(feasible_indices):
                chosen_index = int(feasible_indices[np.argmin(design_costs[feasible_indices])])
                best_counts, best_cost = designs[chosen_index], float(design_costs[chosen_index])

        # Return once no tested local move can reduce cost while retaining target power.
        if np.array_equal(best_counts, selected_counts):
            return selected_counts, "feasible cost-aware local allocation; global minimum not established"
        selected_counts = best_counts

    # Fail explicitly if the bounded improvement search does not converge.
    raise ValueError("Suite allocation exceeded its finite improvement search.")


def plan_repeats(
    cells: Sequence[PlanningCell],
    *,
    parameters: PlanningParameters,
    weights: Sequence[float] | None = None,
) -> RepeatAllocation:
    """Minimize new cost for a fixed pre-run comparison design.

    One-cell searches enumerate legal integer counts and certify the returned
    analytic-model minimum. Suites use deterministic cost-aware local search.
    Existing baseline observations contribute uncertainty but no new cost.
    """
    cell_weights = _inputs(cells, weights)
    if not isinstance(parameters, PlanningParameters):
        raise ValueError("parameters must be PlanningParameters.")
    try:
        finite_costs = all(isfinite(cell.repeat_cost * parameters.candidate_experiments) for cell in cells)
    except OverflowError as error:
        raise ValueError("Repeat costs overflowed after multiplying by candidate_experiments.") from error
    if not finite_costs:
        raise ValueError("Repeat costs overflowed after multiplying by candidate_experiments.")
    selected_counts, allocation_method = (
        _scalar_counts(cells, parameters, cell_weights)
        if len(cells) == 1
        else _suite_counts(cells, parameters, cell_weights)
    )
    baseline_counts = tuple(int(count) for count in selected_counts[::2])
    candidate_counts = tuple(int(count) for count in selected_counts[1::2])
    planning_statistics = power_for_counts(
        cells, baseline_counts, candidate_counts, parameters=parameters, weights=cell_weights
    )
    if planning_statistics.power < parameters.target_power - 1e-12:
        raise ValueError("Returned allocation failed Gym's scalar power feasibility check.")
    additional_baseline_counts = (0,) * len(cells) if parameters.baseline_mode == "existing" else baseline_counts
    total_additional_passes = sum(additional_baseline_counts) + parameters.candidate_experiments * sum(
        candidate_counts
    )
    total_cost = fsum(
        cell.repeat_cost * (additional_baseline_count + parameters.candidate_experiments * candidate_count)
        for cell, additional_baseline_count, candidate_count in zip(
            cells, additional_baseline_counts, candidate_counts, strict=True
        )
    )
    if not isfinite(total_cost):
        raise ValueError("Total planned cost is nonfinite.")
    return RepeatAllocation(
        baseline_counts,
        candidate_counts,
        planning_statistics,
        additional_baseline_counts,
        total_additional_passes,
        total_cost,
        allocation_method,
    )


def _rate(successes: int, trials: int) -> SimulationRate:
    estimate = successes / trials
    normal_critical_value = float(norm.isf(0.025))
    denominator = 1 + normal_critical_value**2 / trials
    center = (estimate + normal_critical_value**2 / (2 * trials)) / denominator
    radius = (
        normal_critical_value
        * sqrt(estimate * (1 - estimate) / trials + normal_critical_value**2 / (4 * trials**2))
        / denominator
    )
    return SimulationRate(successes, trials, estimate, max(0.0, center - radius), min(1.0, center + radius))


def simulate_plan(
    cells: Sequence[PlanningCell],
    allocation: RepeatAllocation,
    *,
    parameters: PlanningParameters,
    weights: Sequence[float] | None = None,
    trials: int = 10_000,
    seed: int = 20_261_109,
) -> SimulationResult:
    """Evaluate actual Gym Welch decisions under independent normal pass scores.

    Draw sufficient statistics: normal sample means and independent chi-square
    sample variances. Each trial calls ``aggregate_welch`` with its estimated
    variances. Power uses the expected delta; false positives use the negative
    margin boundary. Wilson intervals quantify simulation sampling error only.
    Historical variance-estimation uncertainty is not resampled.

    Both arms' sample means and variances are redrawn in hypothetical
    experiments, including existing-baseline mode. This is an unconditional
    repeated-experiment diagnostic, not a calculation conditional on the
    already collected baseline observations.
    """
    cell_weights = _inputs(cells, weights)
    _integer(trials, "trials")
    _integer(seed, "seed", minimum=0)
    if not isinstance(allocation, RepeatAllocation):
        raise ValueError("allocation must be RepeatAllocation.")
    baseline_counts, candidate_counts = _counts(
        cells, allocation.baseline_counts, allocation.candidate_counts, parameters
    )
    random_generator = np.random.default_rng(seed)
    sample_means, sample_variances = [], []
    for counts, variance_field in ((baseline_counts, "baseline_variance"), (candidate_counts, "candidate_variance")):
        population_variances = np.array([getattr(cell, variance_field) for cell in cells])
        counts_array = np.array(counts)
        sample_means.append(
            random_generator.normal(size=(trials, len(cells))) * np.sqrt(population_variances / counts_array)
        )
        sample_variances.append(
            random_generator.chisquare(counts_array - 1, size=(trials, len(cells)))
            * population_variances
            / (counts_array - 1)
        )
    successes = false_positives = 0
    for trial in range(trials):
        comparisons = tuple(
            BenchmarkSamples(
                SampleSummary(
                    baseline_count, float(sample_means[0][trial, index]), float(sample_variances[0][trial, index])
                ),
                SampleSummary(
                    candidate_count, float(sample_means[1][trial, index]), float(sample_variances[1][trial, index])
                ),
            )
            for index, (baseline_count, candidate_count) in enumerate(
                zip(baseline_counts, candidate_counts, strict=True)
            )
        )
        comparison_statistics = aggregate_welch(
            comparisons, weights=cell_weights, confidence_level=1 - 2 * parameters.alpha
        )
        if comparison_statistics.ci_low is None:
            raise ValueError("A simulation trial had zero total variance; its Welch decision is unavailable.")
        successes += comparison_statistics.ci_low + parameters.expected_delta_points > -parameters.margin_points
        false_positives += comparison_statistics.ci_low > 0
    return SimulationResult(
        trials=trials,
        seed=seed,
        power=_rate(successes, trials),
        false_positive=_rate(false_positives, trials),
        coverage=_rate(trials - false_positives, trials),
    )
