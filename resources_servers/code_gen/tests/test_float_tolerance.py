# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for floating-point tolerance in stdout answer comparison."""

from __future__ import annotations

from lcb_integration.testing_util import convert_line_to_decimals, decimal_lines_match


def _match(expected: str, predicted: str) -> bool:
    expected_ok, expected_line = convert_line_to_decimals(expected)
    predicted_ok, predicted_line = convert_line_to_decimals(predicted)
    assert expected_ok and predicted_ok
    return decimal_lines_match(expected.split(), predicted.split(), expected_line, predicted_line)


class TestFloatsGetTolerance:
    """Judges accept 1e-6 absolute or relative error on floating-point output."""

    def test_exact_float_matches(self):
        assert _match("0.333333333333", "0.333333333333")

    def test_small_relative_error_matches(self):
        assert _match("0.333333333333", "0.3333333333")

    def test_six_decimal_places_matches_twelve(self):
        assert _match("0.333333333333", "0.333333")

    def test_relative_error_on_large_magnitude_matches(self):
        assert _match("1000000.0", "1000000.0000001")

    def test_scientific_notation_matches(self):
        assert _match("1e-7", "0.0000001")

    def test_error_above_tolerance_fails(self):
        assert not _match("0.333333333333", "0.334")

    def test_every_token_must_be_within_tolerance(self):
        assert _match("1.0 3.0 3.5", "1.0000001 2.9999999 3.5")
        assert not _match("1.0 3.0 3.5", "1.0000001 2.5 3.5")


class TestIntegersStayExact:
    """Tolerance must not resurrect the np.isclose large-integer false positive."""

    def test_large_integer_off_by_one_fails(self):
        assert not _match("50000000000000000", "50000000000000001")

    def test_small_integer_off_by_one_fails(self):
        assert not _match("41", "42")

    def test_equal_integers_match(self):
        assert _match("50000000000000000", "50000000000000000")


class TestStructuralMismatches:
    def test_token_count_mismatch_fails(self):
        assert not _match("1.0", "1.0 2.0")

    def test_non_finite_does_not_crash(self):
        assert not _match("1.0", "inf")
        assert not _match("1.0", "nan")
