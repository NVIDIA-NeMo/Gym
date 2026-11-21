# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for the downloaded EQ-Bench 4 rating solver, for tests that cannot download it.

It has the solver's call surface (``outcomes_from_comparisons``, ``fit_soft_bt``,
``compute_local_delta_percentile_bootstrap_ratings``, and ``eqbench.ranking.config`` for its import) and a trivial
rating, the base plus 100 per net share won, so tests check how Gym loads and calls the solver, not the fit.

Like upstream's settings, the stand-in ``config.py`` imports ``eqbench`` and reads the rubric files beside
``eqbench.__file__`` when it is imported, so the loader's ``eqbench`` package and the layout ``write_solver`` produces
are exercised without a download.
"""

import hashlib
from pathlib import Path

import pytest

from resources_servers.eqbench4 import upstream_rating


# The dimensions the stand-in rubric declares, as its ``config.py`` reads them.
STAND_IN_DIMENSIONS = ["bond_and_rapport", "attunement"]
STAND_IN_FILES = {
    "src/eqbench/ranking/config.py": b"""
from pathlib import Path

import eqbench

DEFAULT_ELO = 1200.0


def _rubric(name):
    # As upstream's config.py: the rubric beside the eqbench package, one "key: text" per line, # for comments.
    text = (Path(eqbench.__file__).parent / "rubrics" / name).read_text(encoding="utf-8")
    return [line.split(":", 1) for line in map(str.strip, text.splitlines()) if ":" in line and line[0] != "#"]


PAIRWISE_DIMENSIONS = [key.strip() for key, _ in _rubric("pairwise_dimensions.txt")]
DIMENSION_WEIGHTS = {key.strip(): float(value) for key, value in _rubric("pairwise_dimensions_weights.txt")}
""",
    "src/eqbench/rubrics/pairwise_dimensions.txt": b"# dims\nbond_and_rapport: Rapport.\n\nattunement: Reading.\n",
    "src/eqbench/rubrics/pairwise_dimensions_weights.txt": b"bond_and_rapport: 2.0\n",
    "src/eqbench/ranking/soft_bt_solver.py": b"""
from dataclasses import dataclass

from eqbench.ranking.config import DEFAULT_ELO, PAIRWISE_DIMENSIONS


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
