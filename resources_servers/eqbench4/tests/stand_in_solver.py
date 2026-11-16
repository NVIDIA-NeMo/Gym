# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for the downloaded EQ-Bench 4 rating solver, for tests that cannot download it.

It has the solver's call surface (``outcomes_from_comparisons``, ``fit_soft_bt``,
``compute_local_delta_percentile_bootstrap_ratings``, and ``eqbench.ranking.config`` for its import) and a trivial
rating, the base plus 100 per net share won, so tests check how Gym loads and calls the solver, not the fit.
"""

import hashlib
from pathlib import Path

import pytest

from resources_servers.eqbench4 import upstream_rating


STAND_IN_FILES = {
    "src/eqbench/ranking/config.py": b"DEFAULT_ELO = 1200.0\n",
    "src/eqbench/ranking/soft_bt_solver.py": b"""
from dataclasses import dataclass

from eqbench.ranking.config import DEFAULT_ELO


@dataclass(frozen=True)
class SoftOutcome:
    model_a: str
    model_b: str
    fraction_a: float


@dataclass
class Fit:
    ratings: dict


def outcomes_from_comparisons(comparisons):
    return [
        SoftOutcome(c["pair"]["model_a"], c["pair"]["model_b"], float(c["aggregated"]["fraction_for_model_a"]))
        for c in comparisons
    ]


def fit_soft_bt(models, *, outcomes):
    ratings = {model: DEFAULT_ELO for model in models}
    for outcome in outcomes:
        if outcome.model_a in ratings and outcome.model_b in ratings:
            ratings[outcome.model_a] += 100 * (outcome.fraction_a - 0.5)
            ratings[outcome.model_b] -= 100 * (outcome.fraction_a - 0.5)
    return Fit(ratings)


def compute_local_delta_percentile_bootstrap_ratings(models, comparisons, *, n_bootstrap, bootstrap_jobs):
    ratings = fit_soft_bt(models, outcomes=outcomes_from_comparisons(comparisons)).ratings
    return {
        model: {"elo": rating, "ci_low": rating - n_bootstrap, "ci_high": rating + n_bootstrap, "bootstrap_sd": 1.0}
        for model, rating in ratings.items()
    }
""",
}


def install_stand_in_solver(directory: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the stand-in under ``directory`` and pin its hashes in place of the upstream files'."""
    pins = {path: hashlib.sha256(data).hexdigest() for path, data in STAND_IN_FILES.items()}
    monkeypatch.setattr(upstream_rating, "SOLVER_SHA256", pins)
    upstream_rating.write_solver(directory, STAND_IN_FILES)
    return directory


def stand_in_rating(models: list[str], outcomes: list[upstream_rating.Outcome]) -> dict[str, float]:
    """What the stand-in rates, for expected values."""
    ratings = {model: 1200.0 for model in models}
    for outcome in outcomes:
        if outcome.model_a in ratings and outcome.model_b in ratings:
            ratings[outcome.model_a] += 100 * (outcome.fraction_a - 0.5)
            ratings[outcome.model_b] -= 100 * (outcome.fraction_a - 0.5)
    return ratings
