# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for case-insensitive Yes/No verdicts in stdout comparison."""

from __future__ import annotations

import pytest
from lcb_integration.testing_util import is_case_insensitive_verdict


class TestYesNoIsCaseInsensitive:
    """Codeforces-style statements accept any capitalisation of Yes/No."""

    @pytest.mark.parametrize(
        "expected,predicted",
        [
            ("Yes", "YES"),
            ("YES", "yes"),
            ("yes", "Yes"),
            ("No", "NO"),
            ("NO", "no"),
            ("no", "nO"),
        ],
    )
    def test_capitalisation_variants_match(self, expected, predicted):
        assert is_case_insensitive_verdict(expected, predicted)

    def test_identical_verdicts_match(self):
        assert is_case_insensitive_verdict("Yes", "Yes")


class TestOnlyYesNoAreCaseInsensitive:
    """No other answer may become case-insensitive as a side effect."""

    @pytest.mark.parametrize(
        "expected,predicted",
        [
            ("Snuke", "SNUKE"),
            ("Alice", "alice"),
            ("First", "FIRST"),
            ("Fennec", "fennec"),
        ],
    )
    def test_other_words_stay_case_sensitive(self, expected, predicted):
        assert not is_case_insensitive_verdict(expected, predicted)

    def test_yes_does_not_match_no(self):
        assert not is_case_insensitive_verdict("No", "YES")
        assert not is_case_insensitive_verdict("Yes", "NO")

    def test_verdict_does_not_match_unrelated_answer(self):
        assert not is_case_insensitive_verdict("42", "yes")
        assert not is_case_insensitive_verdict("Yes", "42")

    def test_multi_token_line_is_not_a_verdict(self):
        assert not is_case_insensitive_verdict("Yes 1", "YES 1")
