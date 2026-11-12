# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import pytest

from resources_servers.eqbench4.rating import (
    BASE_RATING,
    RATING_SCALE,
    Outcome,
    canonical,
    fit_ratings,
    rate_with_intervals,
)


def test_canonical_orders_models_and_flips_fraction() -> None:
    assert canonical(Outcome("b", "a", 0.8)) == Outcome("a", "b", pytest.approx(0.2))
    assert canonical(Outcome("a", "b", 0.8)) == Outcome("a", "b", 0.8)


def test_two_model_fit_matches_the_closed_form() -> None:
    # With a negligible prior, p(a beats b) equals the mean fraction, so the gap is the logit of 0.75.
    ratings = fit_ratings(["a", "b"], [Outcome("a", "b", 1.0), Outcome("a", "b", 0.5), Outcome("b", "a", 0.25)])
    assert ratings["a"] - ratings["b"] == pytest.approx(RATING_SCALE * math.log(3.0), abs=1e-3)
    assert (ratings["a"] + ratings["b"]) / 2 == pytest.approx(BASE_RATING)


def test_models_without_comparisons_stay_at_base() -> None:
    assert fit_ratings(["a", "b"], []) == {"a": BASE_RATING, "b": BASE_RATING}
    ratings = fit_ratings(["a", "b", "c"], [Outcome("a", "b", 0.9), Outcome("a", "a", 1.0), Outcome("a", "x", 1.0)])
    assert ratings["c"] == pytest.approx(BASE_RATING)
    assert ratings["a"] > ratings["c"] > ratings["b"]


def test_intervals_are_reproducible_and_bracket_the_rating() -> None:
    outcomes = [Outcome("a", "b", fraction) for fraction in (0.9, 0.8, 0.7, 0.6, 0.55, 0.9, 0.95, 0.65)] + [
        Outcome("b", "c", fraction) for fraction in (0.7, 0.6, 0.8, 0.5, 0.75)
    ]
    first = rate_with_intervals(["a", "b", "c"], outcomes, resamples=40)
    assert first == rate_with_intervals(["a", "b", "c"], outcomes, resamples=40)
    for rating in first.values():
        assert rating.ci_low <= rating.rating <= rating.ci_high
        assert rating.bootstrap_sd >= 0
    assert first["a"].rating > first["b"].rating > first["c"].rating


def test_intervals_for_a_lone_model_and_bad_resample_count() -> None:
    lone = rate_with_intervals(["solo"], [], resamples=2)["solo"]
    assert (lone.rating, lone.ci_low, lone.ci_high, lone.bootstrap_sd) == (BASE_RATING,) * 3 + (0.0,)
    assert rate_with_intervals(["a", "b"], [Outcome("a", "b", 0.7)], resamples=1)["a"].bootstrap_sd == 0.0
    with pytest.raises(ValueError, match="resamples"):
        rate_with_intervals(["a"], [], resamples=0)
