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
    identities = [(cell.model, cell.benchmark) for cell in cells]
    if len(set(identities)) != len(identities):
        raise ValueError("Model/benchmark cells must be unique.")
    resolved = (1.0 / len(cells),) * len(cells) if weights is None else tuple(weights)
    if len(resolved) != len(cells):
        raise ValueError("Exactly one weight is required per cell.")
    for weight in resolved:
        _real(weight, "weight", positive=True)
    try:
        total_weight = fsum(resolved)
    except OverflowError as error:
        raise ValueError("Cell weights overflowed; they must sum to one.") from error
    if not isclose(total_weight, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Cell weights must sum to one.")
    if not any(cell.baseline_variance > 0 or cell.candidate_variance > 0 for cell in cells):
        raise ValueError("Power is unavailable when all one-pass variances are zero.")
    return resolved


def _counts(
    cells: Sequence[PlanningCell],
    baseline_counts: Sequence[int],
    candidate_counts: Sequence[int],
    parameters: PlanningParameters,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(parameters, PlanningParameters):
        raise ValueError("parameters must be PlanningParameters.")
    baseline, candidate = tuple(baseline_counts), tuple(candidate_counts)
    if len(baseline) != len(cells) or len(candidate) != len(cells):
        raise ValueError("Exactly one count per arm is required for every cell.")
    for cell, nb, nc in zip(cells, baseline, candidate, strict=True):
        for count in (nb, nc):
            _integer(count, "Repeat count", minimum=2)
        if nc % cell.repeat_step or nc > parameters.max_repeats:
            raise ValueError("Candidate counts must follow repeat_step and not exceed max_repeats.")
        if parameters.baseline_mode == "existing":
            if nb != cell.existing_baseline_repeats:
                raise ValueError("Baseline count must equal existing_baseline_repeats.")
        elif nb % cell.repeat_step or nb > parameters.max_repeats:
            raise ValueError("New baseline counts must follow repeat_step and not exceed max_repeats.")
    return tuple(int(n) for n in baseline), tuple(int(n) for n in candidate)


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
    resolved = _inputs(cells, weights)
    baseline, candidate = _counts(cells, baseline_counts, candidate_counts, parameters)
    comparisons = tuple(
        BenchmarkSamples(
            baseline=SampleSummary(count=nb, mean=0.0, sample_variance=cell.baseline_variance),
            candidate=SampleSummary(count=nc, mean=0.0, sample_variance=cell.candidate_variance),
        )
        for cell, nb, nc in zip(cells, baseline, candidate, strict=True)
    )
    result = aggregate_welch(comparisons, weights=resolved, confidence_level=1 - 2 * parameters.alpha)
    if result.degrees_of_freedom is None or result.ci_high is None or result.standard_error == 0:
        raise ValueError("Power is unavailable because the combined Welch variance is zero.")
    critical = result.ci_high / result.standard_error
    if not all(
        isfinite(value) for value in (result.variance, result.standard_error, result.degrees_of_freedom, critical)
    ):
        raise ValueError("Welch uncertainty is nonfinite for the supplied variances, alpha and counts.")
    noncentrality = (parameters.expected_delta_points + parameters.margin_points) / result.standard_error
    power = float(nct.sf(critical, result.degrees_of_freedom, noncentrality))
    if not isfinite(power):
        raise ValueError("Noncentral-t power is nonfinite for the supplied variances and counts.")
    return PlanningStatistics(
        variance=result.variance,
        standard_error=result.standard_error,
        degrees_of_freedom=result.degrees_of_freedom,
        critical_value=critical,
        half_width=result.ci_high,
        power=power,
    )


def _powers(
    cells: Sequence[PlanningCell],
    designs: NDArray[np.float64],
    weights: tuple[float, ...],
    parameters: PlanningParameters,
) -> NDArray[np.float64]:
    """Batch the Welch equations for search; returned designs are checked by Gym."""
    variances = np.array([(cell.baseline_variance, cell.candidate_variance) for cell in cells])
    contribution = np.square(weights)[None, :, None] * variances[None, :, :] / designs.reshape(-1, len(cells), 2)
    total = contribution.sum(axis=(1, 2))
    fractions = contribution / total[:, None, None]
    denominator = (fractions**2 / (designs.reshape(-1, len(cells), 2) - 1)).sum(axis=(1, 2))
    degrees = 1 / denominator
    critical = t.isf(parameters.alpha, degrees)
    distance = parameters.expected_delta_points + parameters.margin_points
    powers = np.asarray(nct.sf(critical, degrees, distance / np.sqrt(total)))
    if not np.isfinite(powers).all():
        raise ValueError("Nonfinite power during repeat search; check the variance scale.")
    return powers


def _limits(
    cells: Sequence[PlanningCell], parameters: PlanningParameters
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    steps = np.repeat([cell.repeat_step for cell in cells], 2).astype(float)
    lower = np.ceil(2 / steps) * steps
    upper = np.floor(parameters.max_repeats / steps) * steps
    try:
        prices = np.array([(cell.repeat_cost, cell.repeat_cost * parameters.candidate_experiments) for cell in cells])
    except OverflowError as error:
        raise ValueError("Repeat costs overflowed after multiplying by candidate_experiments.") from error
    if not np.isfinite(prices).all():
        raise ValueError("Repeat costs overflowed after multiplying by candidate_experiments.")
    if parameters.baseline_mode == "existing":
        for index, cell in enumerate(cells):
            if cell.existing_baseline_repeats is None:
                raise ValueError("existing baseline mode requires existing_baseline_repeats for every cell.")
            lower[2 * index] = upper[2 * index] = cell.existing_baseline_repeats
        prices[:, 0] = 0.0
    if np.any(upper < lower):
        raise ValueError("No legal new count fits repeat_step and max_repeats.")
    return lower, upper, steps, prices.ravel()


def _could_reach_target(
    cell: PlanningCell,
    designs: NDArray[np.float64],
    weight: float,
    parameters: PlanningParameters,
) -> NDArray[np.bool_]:
    """Retain scalar designs whose known-variance power bound permits the target.

    For T=(Z+lambda)/sqrt(U/df), the Student-t rejection rule has null size
    alpha. The Neyman-Pearson lemma makes the Z test most powerful when the
    variance is known, so NCT power cannot exceed Phi(lambda-z_(1-alpha)),
    whatever the degrees of freedom. This bounds the analytic planning model,
    not the actual Welch simulation's rejection rate. No monotonicity of
    Welch power in either arm count or degrees of freedom is assumed.

    References: https://stat210a.berkeley.edu/fall-2025/reader/hypothesis-testing.html
    and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html
    """
    variance = weight**2 * (cell.baseline_variance / designs[:, 0] + cell.candidate_variance / designs[:, 1])
    distance = parameters.expected_delta_points + parameters.margin_points
    with np.errstate(divide="ignore", over="ignore"):
        normal_power = norm.sf(norm.isf(parameters.alpha) - distance / np.sqrt(variance))
    # A conservative buffer preserves exhaustive checks near the floating-point boundary.
    return np.asarray(normal_power >= parameters.target_power - 1e-12)


def _scalar_counts(
    cells: Sequence[PlanningCell], parameters: PlanningParameters, weights: tuple[float, ...]
) -> tuple[NDArray[np.float64], str]:
    lower, upper, steps, _ = _limits(cells, parameters)
    step = int(steps[1])
    if not _could_reach_target(cells[0], upper[None, :], weights[0], parameters)[0]:
        subject = "candidate count" if parameters.baseline_mode == "existing" else "allocation"
        raise ValueError(
            f"No feasible {subject} found within max_repeats={parameters.max_repeats}: "
            "even the known-variance power upper bound is below the target."
        )
    if parameters.baseline_mode == "existing":
        candidates = np.arange(lower[1], upper[1] + 1, step)
        designs = np.column_stack((np.full(len(candidates), lower[0]), candidates))
        designs = designs[_could_reach_target(cells[0], designs, weights[0], parameters)]
        feasible = np.flatnonzero(_powers(cells, designs, weights, parameters) >= parameters.target_power)
        if not len(feasible):
            raise ValueError(f"No feasible candidate count found within max_repeats={parameters.max_repeats}.")
        return designs[feasible[0]], "minimum candidate count within the configured bound"
    incumbent = lower.copy()
    found = True
    while _powers(cells, incumbent[None, :], weights, parameters)[0] < parameters.target_power:
        enlarged = np.minimum(upper, incumbent * 2)
        if np.array_equal(enlarged, incumbent):
            found = False
            break
        incumbent = enlarged
    experiments = parameters.candidate_experiments
    best_total = int(incumbent[0] + experiments * incumbent[1])
    if not found:
        # This bound includes the whole legal rectangle until a feasible pair is found.
        best_total += 1
    # Every cheaper legal pair is evaluated or excluded by the known-variance
    # upper bound, including low-count nonmonotone power.
    for nc in range(int(lower[1]), min(int(upper[1]), (best_total - int(lower[0]) - 1) // experiments) + 1, step):
        maximum = min(int(upper[0]), best_total - experiments * nc - 1)
        baselines = np.arange(lower[0], maximum + 1, steps[0])
        if not len(baselines):
            continue
        designs = np.column_stack((baselines, np.full(len(baselines), nc)))
        designs = designs[_could_reach_target(cells[0], designs, weights[0], parameters)]
        if not len(designs):
            continue
        feasible = np.flatnonzero(_powers(cells, designs, weights, parameters) >= parameters.target_power)
        if len(feasible):
            incumbent = designs[feasible[0]]
            best_total = int(incumbent[0] + experiments * nc)
            found = True
    if not found:
        raise ValueError(f"No feasible allocation found within max_repeats={parameters.max_repeats}.")
    return incumbent, "exhaustive minimum of the bounded integer analytic-power model"


def _normal_start(
    cells: Sequence[PlanningCell],
    parameters: PlanningParameters,
    weights: tuple[float, ...],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    steps: NDArray[np.float64],
    prices: NDArray[np.float64],
) -> NDArray[np.float64]:
    coefficient = (np.square(weights)[:, None] * [(c.baseline_variance, c.candidate_variance) for c in cells]).ravel()
    distance = parameters.expected_delta_points + parameters.margin_points
    variance_limit = (distance / (norm.isf(parameters.alpha) + norm.ppf(parameters.target_power))) ** 2
    mutable = (prices > 0) & (coefficient > 0)
    if not mutable.any():
        return lower.copy()

    def continuous(multiplier: float) -> NDArray[np.float64]:
        counts = lower.copy()
        counts[mutable] = np.clip(
            np.sqrt(multiplier * coefficient[mutable] / prices[mutable]), lower[mutable], upper[mutable]
        )
        return counts

    high = 1.0
    for _ in range(1024):
        current = continuous(high)
        if np.sum(coefficient / current) <= variance_limit or np.all(current[mutable] == upper[mutable]):
            break
        if high > np.finfo(float).max / 4:
            return lower.copy()
        high *= 4
    low = 0.0
    for _ in range(60):
        middle = (low + high) / 2
        if np.sum(coefficient / continuous(middle)) <= variance_limit:
            high = middle
        else:
            low = middle
    return np.maximum(lower, np.floor(continuous(high) / steps) * steps)


def _suite_counts(
    cells: Sequence[PlanningCell], parameters: PlanningParameters, weights: tuple[float, ...]
) -> tuple[NDArray[np.float64], str]:
    lower, upper, steps, prices = _limits(cells, parameters)
    counts = lower.copy()
    coefficient = (np.square(weights)[:, None] * [(c.baseline_variance, c.candidate_variance) for c in cells]).ravel()
    if _powers(cells, counts[None, :], weights, parameters)[0] < parameters.target_power:
        if not np.any((prices > 0) & (coefficient > 0)):
            raise ValueError(
                "No feasible suite allocation: new repeats cannot reduce the existing baseline uncertainty."
            )
        counts = _normal_start(cells, parameters, weights, lower, upper, steps, prices)
    for _ in range(20_000):
        current_power = _powers(cells, counts[None, :], weights, parameters)[0]
        if current_power >= parameters.target_power:
            break
        indices = np.flatnonzero((counts + steps <= upper) & (coefficient > 0))
        if not len(indices):
            raise ValueError(f"No feasible suite allocation found within max_repeats={parameters.max_repeats}.")
        proposals = np.repeat(counts[None, :], len(indices), axis=0)
        proposals[np.arange(len(indices)), indices] += steps[indices]
        gain = (_powers(cells, proposals, weights, parameters) - current_power) / (prices[indices] * steps[indices])
        counts = proposals[int(np.argmax(gain))]
    else:
        raise ValueError("Suite allocation exceeded its finite repair search.")
    # Remove steps or exchange one expensive step for up to 20 cheaper steps.
    # Four promising partner coordinates bound the work; this is a local search.
    for _ in range(1_000):
        best, best_cost = counts, float(np.dot(prices, counts))
        for index in np.flatnonzero(counts > lower):
            reduced = counts.copy()
            reduced[index] -= steps[index]
            proposals = [reduced]
            benefit = np.divide(
                coefficient,
                reduced * (reduced + steps) * prices,
                out=np.zeros_like(prices),
                where=prices > 0,
            )
            partners = [j for j in np.argsort(-benefit, kind="stable") if j != index and prices[j] > 0][:4]
            for other in partners:
                maximum = min(20, int((upper[other] - reduced[other]) / steps[other]))
                for added in range(1, maximum + 1):
                    proposal = reduced.copy()
                    proposal[other] += added * steps[other]
                    if np.dot(prices, proposal) < best_cost:
                        proposals.append(proposal)
            designs = np.stack(proposals)
            powers = _powers(cells, designs, weights, parameters)
            costs = designs @ prices
            feasible = np.flatnonzero((powers >= parameters.target_power) & (costs < best_cost))
            if len(feasible):
                chosen = int(feasible[np.argmin(costs[feasible])])
                best, best_cost = designs[chosen], float(costs[chosen])
        if np.array_equal(best, counts):
            return counts, "feasible cost-aware local allocation; global minimum not established"
        counts = best
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
    resolved = _inputs(cells, weights)
    if not isinstance(parameters, PlanningParameters):
        raise ValueError("parameters must be PlanningParameters.")
    counts, method = (
        _scalar_counts(cells, parameters, resolved) if len(cells) == 1 else _suite_counts(cells, parameters, resolved)
    )
    baseline = tuple(int(count) for count in counts[::2])
    candidate = tuple(int(count) for count in counts[1::2])
    statistics = power_for_counts(cells, baseline, candidate, parameters=parameters, weights=resolved)
    if statistics.power < parameters.target_power - 1e-12:
        raise ValueError("Returned allocation failed Gym's scalar power feasibility check.")
    additional = (0,) * len(cells) if parameters.baseline_mode == "existing" else baseline
    total_passes = sum(additional) + parameters.candidate_experiments * sum(candidate)
    total_cost = fsum(
        cell.repeat_cost * (nb + parameters.candidate_experiments * nc)
        for cell, nb, nc in zip(cells, additional, candidate, strict=True)
    )
    if not isfinite(total_cost):
        raise ValueError("Total planned cost is nonfinite.")
    return RepeatAllocation(baseline, candidate, statistics, additional, total_passes, total_cost, method)


def _rate(successes: int, trials: int) -> SimulationRate:
    estimate = successes / trials
    z = float(norm.isf(0.025))
    denominator = 1 + z**2 / trials
    center = (estimate + z**2 / (2 * trials)) / denominator
    radius = z * sqrt(estimate * (1 - estimate) / trials + z**2 / (4 * trials**2)) / denominator
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
    resolved = _inputs(cells, weights)
    _integer(trials, "trials")
    _integer(seed, "seed", minimum=0)
    if not isinstance(allocation, RepeatAllocation):
        raise ValueError("allocation must be RepeatAllocation.")
    baseline, candidate = _counts(cells, allocation.baseline_counts, allocation.candidate_counts, parameters)
    rng = np.random.default_rng(seed)
    means, variances = [], []
    for counts, arm in ((baseline, "baseline_variance"), (candidate, "candidate_variance")):
        population_variances = np.array([getattr(cell, arm) for cell in cells])
        counts_array = np.array(counts)
        means.append(rng.normal(size=(trials, len(cells))) * np.sqrt(population_variances / counts_array))
        variances.append(
            rng.chisquare(counts_array - 1, size=(trials, len(cells))) * population_variances / (counts_array - 1)
        )
    successes = false_positives = 0
    for trial in range(trials):
        comparisons = tuple(
            BenchmarkSamples(
                SampleSummary(nb, float(means[0][trial, index]), float(variances[0][trial, index])),
                SampleSummary(nc, float(means[1][trial, index]), float(variances[1][trial, index])),
            )
            for index, (nb, nc) in enumerate(zip(baseline, candidate, strict=True))
        )
        result = aggregate_welch(comparisons, weights=resolved, confidence_level=1 - 2 * parameters.alpha)
        if result.ci_low is None:
            raise ValueError("A simulation trial had zero total variance; its Welch decision is unavailable.")
        successes += result.ci_low + parameters.expected_delta_points > -parameters.margin_points
        false_positives += result.ci_low > 0
    return SimulationResult(
        trials=trials,
        seed=seed,
        power=_rate(successes, trials),
        false_positive=_rate(false_positives, trials),
        coverage=_rate(trials - false_positives, trials),
    )
