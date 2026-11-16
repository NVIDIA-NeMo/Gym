# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path

import pytest

from resources_servers.eqbench4 import upstream_rating
from resources_servers.eqbench4.tests.stand_in_solver import install_stand_in_solver, stand_in_rating
from resources_servers.eqbench4.upstream_rating import SOLVER_SHA256, Outcome, UpstreamSolver, canonical


# The upstream revision's pins, as imported (the conftest replaces them with the stand-in's during each test).
UPSTREAM_PINS = dict(SOLVER_SHA256)
OUTCOMES = [Outcome("a", "b", 0.7), Outcome("b", "c", 0.6), Outcome("a", "c", 0.8)]


def test_the_solver_is_called_through_its_own_functions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    solver = UpstreamSolver(install_stand_in_solver(tmp_path, monkeypatch))
    assert solver.fit(["a", "b", "c", "d"], OUTCOMES) == pytest.approx(stand_in_rating(["a", "b", "c", "d"], OUTCOMES))
    rated = solver.rate_with_intervals(["a", "b", "c"], OUTCOMES, resamples=7)
    expected = stand_in_rating(["a", "b", "c"], OUTCOMES)
    assert set(rated) == set(expected)
    for model, value in expected.items():
        assert (rated[model].rating, rated[model].ci_low, rated[model].ci_high) == pytest.approx(
            (value, value - 7, value + 7)
        )


def test_loading_the_solver_leaves_no_eqbench_package_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setitem(sys.modules, "eqbench", sentinel)
    UpstreamSolver(install_stand_in_solver(tmp_path, monkeypatch))
    assert sys.modules["eqbench"] is sentinel
    assert not [name for name in sys.modules if name.startswith("eqbench.")]


def test_a_missing_or_changed_solver_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(FileNotFoundError, match="gym eval prepare --benchmark eqbench4"):
        UpstreamSolver(tmp_path / "absent")
    directory = install_stand_in_solver(tmp_path, monkeypatch)
    (directory / "src/eqbench/ranking/soft_bt_solver.py").write_text("raise SystemExit('never run')\n")
    with pytest.raises(ValueError, match="does not match its pinned SHA-256"):
        UpstreamSolver(directory)


def test_canonical_orders_the_models() -> None:
    assert canonical(Outcome("b", "a", 0.3)) == Outcome("a", "b", pytest.approx(0.7))
    assert canonical(Outcome("a", "b", 0.3)) == Outcome("a", "b", 0.3)


# The downloaded solver itself, when a prepared copy is at hand (``gym eval prepare --benchmark eqbench4``, or
# EQBENCH4_RATING_SOLVER_DIR): the pins are the upstream revision's, and its ratings are on the leaderboard's scale.
_PREPARED = Path(
    os.environ.get("EQBENCH4_RATING_SOLVER_DIR", Path(__file__).parents[3] / "benchmarks/eqbench4/data/rating_solver")
)


@pytest.mark.skipif(
    not all((_PREPARED / path).is_file() for path in SOLVER_SHA256), reason="no prepared EQ-Bench 4 rating solver"
)
def test_the_prepared_upstream_solver_rates_on_the_leaderboard_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upstream_rating, "SOLVER_SHA256", UPSTREAM_PINS)  # conftest pins the stand-in's
    solver = UpstreamSolver(_PREPARED)
    ratings = solver.fit(["a", "b", "c"], OUTCOMES)
    assert sum(ratings.values()) / 3 == pytest.approx(1200.0, abs=1.0)
    assert ratings["a"] > ratings["b"] > ratings["c"]
    rated = solver.rate_with_intervals(["a", "b", "c"], OUTCOMES, resamples=20)
    assert all(r.ci_low <= r.rating <= r.ci_high for r in rated.values())
