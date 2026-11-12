# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bradley-Terry ratings fitted to fractional comparison outcomes, as EQ-Bench 4 reports them.

Each comparison gives model A a fraction f of a win over model B. The fit maximises
sum f*log(p) + (1-f)*log(1-p), with p = sigmoid(theta_A - theta_B), under a weak L2 prior that fixes the free offset,
and reports 1200 + theta * 400 / ln(10). Intervals come from resampling comparisons within each model pair and
measuring how far a model's average gap to its nearest-ranked neighbours moves. This follows the EQ-Bench 4 rating
method (https://github.com/EQ-bench/eqbench4, revision 93dcb7f5d430433ad49d501e672a98bcff2c9678) as an independent
implementation.
"""

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


BASE_RATING = 1200.0
RATING_SCALE = 400.0 / math.log(10.0)
L2_PRIOR = 1e-6
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_SEED = 12345
NEIGHBOUR_RADIUS = 3


@dataclass(frozen=True)
class Outcome:
    """Model ``model_a`` earned ``fraction_a`` of the comparison against ``model_b``."""

    model_a: str
    model_b: str
    fraction_a: float


def canonical(outcome: Outcome) -> Outcome:
    """The same outcome with the two models in sorted order."""
    if outcome.model_a <= outcome.model_b:
        return outcome
    return Outcome(outcome.model_b, outcome.model_a, 1.0 - outcome.fraction_a)


def fit_ratings(
    models: Sequence[str],
    outcomes: Sequence[Outcome],
    *,
    l2: float = L2_PRIOR,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> dict[str, float]:
    """Newton-fit Bradley-Terry ratings on the Elo scale; models without comparisons stay at the base rating."""
    names = list(dict.fromkeys(models))
    index = {name: position for position, name in enumerate(names)}
    totals: dict[tuple[int, int], list[float]] = {}
    for outcome in outcomes:
        if outcome.model_a in index and outcome.model_b in index and outcome.model_a != outcome.model_b:
            pair = totals.setdefault((index[outcome.model_a], index[outcome.model_b]), [0.0, 0.0])
            pair[0] += outcome.fraction_a
            pair[1] += 1.0
    if not names or not totals:
        return {name: BASE_RATING for name in names}

    first = np.array([pair[0] for pair in totals], dtype=int)
    second = np.array([pair[1] for pair in totals], dtype=int)
    wins = np.array([value[0] for value in totals.values()])
    counts = np.array([value[1] for value in totals.values()])
    size = len(names)
    theta = np.zeros(size)
    for _ in range(max_iterations):
        probability = 1.0 / (1.0 + np.exp(-(theta[first] - theta[second])))
        residual = counts * probability - wins
        gradient = 2.0 * l2 * theta
        np.add.at(gradient, first, residual)
        np.add.at(gradient, second, -residual)
        weight = counts * probability * (1.0 - probability)
        hessian = np.diag(np.full(size, 2.0 * l2))
        np.add.at(hessian, (first, first), weight)
        np.add.at(hessian, (second, second), weight)
        np.add.at(hessian, (first, second), -weight)
        np.add.at(hessian, (second, first), -weight)
        step = np.linalg.solve(hessian, gradient)
        theta = theta - step
        theta = theta - theta.mean()
        if np.max(np.abs(step)) < tolerance:
            break
    return {name: float(BASE_RATING + RATING_SCALE * theta[index[name]]) for name in names}


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class Rating:
    rating: float
    ci_low: float
    ci_high: float
    bootstrap_sd: float


def rate_with_intervals(
    models: Sequence[str],
    outcomes: Sequence[Outcome],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    neighbour_radius: int = NEIGHBOUR_RADIUS,
) -> dict[str, Rating]:
    """Ratings with 95% intervals from a pair-stratified bootstrap of the comparisons.

    A model's interval is its rating plus the 2.5th and 97.5th percentiles of how far its mean gap to the models within
    ``neighbour_radius`` ranks of it moves across resamples.
    """
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    names = list(dict.fromkeys(models))
    ordered_outcomes = [canonical(outcome) for outcome in outcomes]
    baseline = fit_ratings(names, ordered_outcomes)
    ladder = sorted(names, key=lambda name: baseline[name], reverse=True)

    strata: dict[tuple[str, str], list[Outcome]] = defaultdict(list)
    for outcome in ordered_outcomes:
        strata[(outcome.model_a, outcome.model_b)].append(outcome)
    samples = []
    for sample_index in range(resamples):
        rng = random.Random(seed + sample_index * 1009)
        drawn = [rng.choice(stratum) for pair in sorted(strata) for stratum in [strata[pair]] for _ in stratum]
        samples.append(fit_ratings(names, drawn))

    ratings: dict[str, Rating] = {}
    for rank, name in enumerate(ladder):
        neighbours = [
            ladder[position]
            for position in range(rank - neighbour_radius, rank + neighbour_radius + 1)
            if position != rank and 0 <= position < len(ladder)
        ]
        base = baseline[name]
        if neighbours:
            base_gap = sum(base - baseline[other] for other in neighbours) / len(neighbours)
            errors = [
                sum(sample[name] - sample[other] for other in neighbours) / len(neighbours) - base_gap
                for sample in samples
            ]
        else:
            errors = [0.0]
        mean_error = sum(errors) / len(errors)
        spread = (
            math.sqrt(sum((error - mean_error) ** 2 for error in errors) / (len(errors) - 1))
            if len(errors) > 1
            else 0.0
        )
        ratings[name] = Rating(
            rating=base,
            ci_low=base + _percentile(errors, 0.025),
            ci_high=base + _percentile(errors, 0.975),
            bootstrap_sd=spread,
        )
    return ratings
