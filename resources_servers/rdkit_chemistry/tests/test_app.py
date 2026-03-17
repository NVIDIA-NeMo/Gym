# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the rdkit_chemistry resources server."""

import math

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root

from resources_servers.rdkit_chemistry.app import (
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
        assert extract_predicted_value("yes", "float") is None

    def test_empty_string(self):
        assert extract_predicted_value("", "count") is None

    def test_non_string(self):
        assert extract_predicted_value(None, "float") is None


# ---------------------------------------------------------------------------
# compute_reward — discrete (exact-match) properties
# ---------------------------------------------------------------------------


class TestComputeRewardDiscrete:
    def test_count_correct(self):
        assert compute_reward(5.0, 5.0, "count") == 1.0

    def test_count_wrong(self):
        assert compute_reward(4.0, 5.0, "count") == 0.0

    def test_bool_correct(self):
        assert compute_reward(1.0, 1.0, "bool") == 1.0

    def test_bool_wrong(self):
        assert compute_reward(0.0, 1.0, "bool") == 0.0

    def test_presence_correct(self):
        assert compute_reward(0.0, 0.0, "presence") == 1.0

    def test_fragment_correct(self):
        assert compute_reward(3.0, 3.0, "fragment") == 1.0

    def test_none_prediction(self):
        assert compute_reward(None, 5.0, "count") == 0.0

    def test_nan_prediction(self):
        assert compute_reward(float("nan"), 5.0, "count") == 0.0


# ---------------------------------------------------------------------------
# compute_reward — float (negative absolute error)
# ---------------------------------------------------------------------------


class TestComputeRewardFloat:
    def test_perfect_prediction(self):
        assert compute_reward(2.5, 2.5, "float") == pytest.approx(0.0)

    def test_error_of_half(self):
        assert compute_reward(1.0, 1.5, "float") == pytest.approx(-0.5)

    def test_error_of_one(self):
        assert compute_reward(0.0, 1.0, "float") == pytest.approx(-1.0)

    def test_large_error(self):
        assert compute_reward(10.0, 0.0, "float") == pytest.approx(-10.0)

    def test_negative_values(self):
        assert compute_reward(-1.0, -2.5, "float") == pytest.approx(-1.5)

    def test_reward_is_nonpositive(self):
        reward = compute_reward(3.7, 2.1, "float")
        assert reward <= 0.0

    def test_none_prediction(self):
        assert compute_reward(None, 1.0, "float") == 0.0

    def test_nan_prediction(self):
        assert compute_reward(float("nan"), 1.0, "float") == 0.0
