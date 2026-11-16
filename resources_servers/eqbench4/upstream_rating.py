# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EQ-Bench 4's own Bradley-Terry rating solver, run from the copy that preparation downloads.

Gym holds no rating code. The solver is ``src/eqbench/ranking/soft_bt_solver.py`` with its settings module
``src/eqbench/ranking/config.py`` from https://github.com/EQ-bench/eqbench4 at revision
93dcb7f5d430433ad49d501e672a98bcff2c9678, licensed under BSD-3-Clause with the Commons Clause condition (Licensor:
Samuel J. Paech). ``gym eval prepare --benchmark eqbench4`` downloads both, with the two rubric files the settings
module reads, checks each against a pinned SHA-256 and writes them to the solver directory (gitignored); this module
checks them again, imports them from there and calls
the solver's own functions: ``fit_soft_bt`` for a rating, and ``compute_local_delta_percentile_bootstrap_ratings`` for
a rating with its 95% interval, so the ratings are the leaderboard's by construction.
"""

import hashlib
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


SOLVER_SHA256 = {
    "src/eqbench/ranking/config.py": "de27fb4778ce39b2d1e3133f1e85c26fda5e578c9f4409e82fadef8d6e330277",
    "src/eqbench/ranking/soft_bt_solver.py": "79953bd642e7bab64cacff3cfb57ca82e97fde3e978a78ceda350f15f35c9695",
    # Read by config.py when it is imported, from beside the ``eqbench`` package.
    "src/eqbench/rubrics/pairwise_dimensions.txt": "a3da48ffcd5bf3902a944c3653fd6799dbdff8a5d0c523a6ef0cc437a623845f",
    "src/eqbench/rubrics/pairwise_dimensions_weights.txt": (
        "23c7166ca47128c4b12544cbed0b8cd14d386ea42ed04a381bab7c72768d7a83"
    ),
}
_PACKAGE_DIR = "src/eqbench"
# The solver imports its settings as ``eqbench.ranking.config``; both load under their upstream module names.
_MODULE_NAMES = {
    "src/eqbench/ranking/config.py": "eqbench.ranking.config",
    "src/eqbench/ranking/soft_bt_solver.py": "eqbench.ranking.soft_bt_solver",
}
_PACKAGE_NAMES = ("eqbench", "eqbench.ranking")


@dataclass(frozen=True)
class Outcome:
    """Model ``model_a`` earned ``fraction_a`` of the comparison against ``model_b``."""

    model_a: str
    model_b: str
    fraction_a: float


def canonical(outcome: Outcome) -> Outcome:
    """The same outcome with the two models in sorted order, as the published comparison graph stores it."""
    if outcome.model_a <= outcome.model_b:
        return outcome
    return Outcome(outcome.model_b, outcome.model_a, 1.0 - outcome.fraction_a)


@dataclass(frozen=True)
class Rating:
    rating: float
    ci_low: float
    ci_high: float
    bootstrap_sd: float


def write_solver(directory: Path, files: dict[str, bytes]) -> None:
    """Write the downloaded solver files, already checked against their pins, under ``directory``."""
    for path in SOLVER_SHA256:
        target = Path(directory) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[path])


def _comparison(outcome: Outcome) -> dict:
    """An outcome in the shape of a published comparison, which the solver's own reader takes."""
    return {
        "pair": {"model_a": outcome.model_a, "model_b": outcome.model_b},
        "aggregated": {"fraction_for_model_a": outcome.fraction_a},
    }


class UpstreamSolver:
    """The downloaded solver, imported from ``directory`` after checking each file against its pin."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        sources = {}
        for path, expected in SOLVER_SHA256.items():
            file = directory / path
            if not file.is_file():
                raise FileNotFoundError(
                    f"EQ-Bench 4 rating solver not found at {file}; run `gym eval prepare --benchmark eqbench4`"
                )
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"{file} does not match its pinned SHA-256 (got {digest}); prepare again")
            sources[path] = file
        self._solver = _import_solver(sources, directory / _PACKAGE_DIR)

    def fit(self, models: Sequence[str], outcomes: Sequence[Outcome]) -> dict[str, float]:
        """Ratings on the Elo scale; a model without comparisons stays at the base rating."""
        comparisons = [_comparison(outcome) for outcome in outcomes]
        solver_outcomes = self._solver.outcomes_from_comparisons(comparisons)
        return dict(self._solver.fit_soft_bt(list(models), outcomes=solver_outcomes).ratings)

    def rate_with_intervals(
        self, models: Sequence[str], outcomes: Sequence[Outcome], *, resamples: int
    ) -> dict[str, Rating]:
        """Ratings with the leaderboard's 95% intervals, from its pair-stratified bootstrap (one process)."""
        rated = self._solver.compute_local_delta_percentile_bootstrap_ratings(
            list(models), [_comparison(outcome) for outcome in outcomes], n_bootstrap=resamples, bootstrap_jobs=1
        )
        return {
            model: Rating(
                rating=values["elo"],
                ci_low=values["ci_low"],
                ci_high=values["ci_high"],
                bootstrap_sd=values["bootstrap_sd"],
            )
            for model, values in rated.items()
        }


def _import_solver(sources: dict[str, Path], package_dir: Path) -> ModuleType:
    """Import the solver and its settings under their upstream names, then take those names out of
    ``sys.modules`` again: the solver keeps its own references, and nothing else sees an ``eqbench`` package.

    The ``eqbench`` package is an empty stand-in located at ``package_dir`` (nothing of upstream's package
    ``__init__`` runs), where the settings module reads its rubric files."""
    names = (*_PACKAGE_NAMES, *_MODULE_NAMES.values())
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    try:
        for package in _PACKAGE_NAMES:
            module = ModuleType(package)
            module.__path__ = []
            sys.modules[package] = module
        sys.modules["eqbench"].__file__ = str(package_dir / "__init__.py")
        loaded = {}
        for path in ("src/eqbench/ranking/config.py", "src/eqbench/ranking/soft_bt_solver.py"):
            spec = importlib.util.spec_from_file_location(_MODULE_NAMES[path], sources[path])
            module = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE_NAMES[path]] = module
            spec.loader.exec_module(module)
            loaded[path] = module
        return loaded["src/eqbench/ranking/soft_bt_solver.py"]
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
