# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the rdkit_chemistry resources server."""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root

from resources_servers.rdkit_chemistry.app import (
    compute_reward,
    extract_predicted_value,
)


# ---------------------------------------------------------------------------
# extract_predicted_value
# ---------------------------------------------------------------------------


class TestExtractPredictedValueStrict:
    """Non-boxed mode requires ((answer)) — bare text is rejected."""

    def test_bare_integer_rejected(self):
        assert extract_predicted_value("42", "count") is None

    def test_bare_float_rejected(self):
        assert extract_predicted_value("2.54", "float") is None

    def test_bare_text_with_number_rejected(self):
        assert extract_predicted_value("The logP is approximately -2.5.", "float") is None

    def test_bool_text_rejected(self):
        assert extract_predicted_value("yes", "presence") is None

    def test_empty_string(self):
        assert extract_predicted_value("", "count") is None

    def test_non_string(self):
        assert extract_predicted_value(None, "float") is None


# ---------------------------------------------------------------------------
# extract_predicted_value — boxed format
# ---------------------------------------------------------------------------


class TestExtractPredictedValueBoxed:
    def test_boxed_integer(self):
        assert extract_predicted_value(r"\boxed{42}", "count", use_box_format=True) == 42.0

    def test_boxed_float(self):
        assert extract_predicted_value(r"\boxed{0.83}", "float", use_box_format=True) == pytest.approx(0.83)

    def test_boxed_negative(self):
        assert extract_predicted_value(r"\boxed{-1.5}", "float", use_box_format=True) == pytest.approx(-1.5)

    def test_boxed_zero_or_one(self):
        assert extract_predicted_value(r"\boxed{1}", "bool", use_box_format=True) == 1.0
        assert extract_predicted_value(r"\boxed{0}", "bool", use_box_format=True) == 0.0

    def test_boxed_with_surrounding_text(self):
        text = r"The QED score is \boxed{0.83}."
        assert extract_predicted_value(text, "float", use_box_format=True) == pytest.approx(0.83)

    def test_boxed_last_occurrence_wins(self):
        text = r"First attempt: \boxed{1.0}. Correction: \boxed{2.5}"
        assert extract_predicted_value(text, "float", use_box_format=True) == pytest.approx(2.5)

    def test_boxed_scientific_notation(self):
        assert extract_predicted_value(r"\boxed{1.5e-3}", "float", use_box_format=True) == pytest.approx(1.5e-3)

    def test_boxed_missing_returns_none(self):
        assert extract_predicted_value("42", "count", use_box_format=True) is None

    def test_boxed_empty_braces_returns_none(self):
        assert extract_predicted_value(r"\boxed{}", "float", use_box_format=True) is None

    def test_boxed_non_numeric_returns_none(self):
        assert extract_predicted_value(r"\boxed{hello}", "float", use_box_format=True) is None

    def test_boxed_not_required_when_flag_false(self):
        assert extract_predicted_value("((42))", "count", use_box_format=False) == 42.0

    def test_bare_number_rejected_when_boxed_required(self):
        assert extract_predicted_value("The answer is 42", "count", use_box_format=True) is None

    def test_boxed_with_whitespace_inside(self):
        assert extract_predicted_value(r"\boxed{ 3.14 }", "float", use_box_format=True) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# extract_predicted_value — double-parentheses format (non-boxed)
# ---------------------------------------------------------------------------


class TestExtractPredictedValueDoubleParens:
    def test_double_parens_integer(self):
        assert extract_predicted_value("The answer is ((42))", "count") == 42.0

    def test_double_parens_float(self):
        assert extract_predicted_value("((0.83))", "float") == pytest.approx(0.83)

    def test_double_parens_negative(self):
        assert extract_predicted_value("((-1.5))", "float") == pytest.approx(-1.5)

    def test_double_parens_zero_or_one(self):
        assert extract_predicted_value("((1))", "bool") == 1.0
        assert extract_predicted_value("((0))", "bool") == 0.0

    def test_double_parens_with_surrounding_text(self):
        assert extract_predicted_value("After analysis, the count is ((8)).", "fragment") == 8.0

    def test_double_parens_last_occurrence_wins(self):
        text = "First ((3)), actually ((5))"
        assert extract_predicted_value(text, "count") == 5.0

    def test_double_parens_scientific_notation(self):
        assert extract_predicted_value("((1.5e-3))", "float") == pytest.approx(1.5e-3)

    def test_double_parens_whitespace_inside(self):
        assert extract_predicted_value("(( 3.14 ))", "float") == pytest.approx(3.14)

    def test_double_parens_empty_returns_none(self):
        assert extract_predicted_value("(())", "count") is None

    def test_double_parens_non_numeric_returns_none(self):
        assert extract_predicted_value("((hello))", "float") is None

    def test_double_parens_preferred_over_bare_number(self):
        text = "The value 99 is wrong, the correct answer is ((42))"
        assert extract_predicted_value(text, "count") == 42.0

    def test_bare_number_rejected_without_double_parens(self):
        assert extract_predicted_value("42", "count") is None


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
