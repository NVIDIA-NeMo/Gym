# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the chemistry_direct resources server."""

import math

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root

from resources_servers.chemistry_direct.app import (
    compute_reward,
    extract_predicted_value,
)


# ---------------------------------------------------------------------------
# extract_predicted_value
# ---------------------------------------------------------------------------

class TestExtractPredictedValue:

    def test_strict_integer(self):
        assert extract_predicted_value("42", "count") == 42.0

    def test_strict_float(self):
        assert extract_predicted_value("2.54", "float") == pytest.approx(2.54)

    def test_trailing_period(self):
        assert extract_predicted_value("3.", "count") == 3.0

    def test_negative_float(self):
        assert extract_predicted_value("-1.23", "float") == pytest.approx(-1.23)

    def test_scientific_notation(self):
        assert extract_predicted_value("1.5e-3", "float") == pytest.approx(1.5e-3)

    def test_permissive_last_number(self):
        assert extract_predicted_value("The logP is approximately -2.5.", "float") == pytest.approx(-2.5)

    def test_permissive_mixed_text(self):
        assert extract_predicted_value("Answer: 7", "count") == 7.0

    def test_bool_true_text(self):
        assert extract_predicted_value("yes", "presence") == 1.0

    def test_bool_false_text(self):
        assert extract_predicted_value("No, it does not.", "fragment") == 0.0

    def test_bool_text_ignored_for_float(self):
        # Boolean text fallback only applies to presence/fragment
        assert extract_predicted_value("yes", "float") is None

    def test_empty_string(self):
        assert extract_predicted_value("", "count") is None

    def test_non_string(self):
        assert extract_predicted_value(None, "float") is None


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------

_EMPTY_STATS: dict = {}
_DUMMY_STATS: dict = {
    "float_accuracy_threshold": 0.95,
    "n_quantiles": 5,
    "properties": {
        "MolLogP": {
            "quantiles": [-3.0, -1.0, 0.0, 1.0, 3.0],
            "min": -3.0,
            "max": 3.0,
        }
    },
}


class TestComputeReward:

    def test_exact_match_count(self):
        assert compute_reward(5.0, 5.0, "count", "RingCount", _EMPTY_STATS, 0.95) == 1.0

    def test_exact_match_count_wrong(self):
        assert compute_reward(4.0, 5.0, "count", "RingCount", _EMPTY_STATS, 0.95) == 0.0

    def test_exact_match_bool(self):
        assert compute_reward(1.0, 1.0, "bool", "PassLipinski", _EMPTY_STATS, 0.95) == 1.0

    def test_exact_match_presence(self):
        assert compute_reward(0.0, 0.0, "presence", "HasAmide", _EMPTY_STATS, 0.95) == 1.0

    def test_none_prediction(self):
        assert compute_reward(None, 5.0, "count", "RingCount", _EMPTY_STATS, 0.95) == 0.0

    def test_nan_prediction(self):
        assert compute_reward(float("nan"), 5.0, "count", "RingCount", _EMPTY_STATS, 0.95) == 0.0

    def test_float_perfect(self):
        # True value = 0.0; predicted = 0.0; error = 0 → beats all quantiles
        reward = compute_reward(0.0, 0.0, "float", "MolLogP", _DUMMY_STATS, 0.95)
        assert reward == 1.0

    def test_float_very_wrong(self):
        # True value = 0.0; predicted = 1000 → error is enormous
        reward = compute_reward(1000.0, 0.0, "float", "MolLogP", _DUMMY_STATS, 0.95)
        assert reward == 0.0

    def test_float_missing_property_fallback(self):
        # Property not in stats → fallback relative-error rule (< 5%)
        reward = compute_reward(1.0, 1.0, "float", "UnknownProp", _DUMMY_STATS, 0.95)
        assert reward == 1.0

    def test_float_missing_property_fallback_fail(self):
        reward = compute_reward(2.0, 1.0, "float", "UnknownProp", _DUMMY_STATS, 0.95)
        assert reward == 0.0
