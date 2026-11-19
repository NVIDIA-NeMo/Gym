# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A scorer failure must be flagged, not averaged in as a real low score."""

from resources_servers.gdpval.app import _is_invalid_judge_result
from resources_servers.gdpval.scoring import SCORING_ERROR_KEY


def test_missing_result_is_invalid():
    assert _is_invalid_judge_result(None) is True


def test_real_judgement_is_valid():
    assert _is_invalid_judge_result({"criteria_scores": [1, 0, 1], "score": 0.66}) is False


def test_every_scorer_failure_mode_is_flagged():
    for mode in ("no_valid_scores", "truncated_json", "no_score_in_response"):
        assert _is_invalid_judge_result({SCORING_ERROR_KEY: mode}) is True, mode


def test_a_judge_emitting_its_own_error_field_is_not_discarded():
    """The success-path metadata IS the judge's parsed JSON; 'error' may be its own."""
    assert _is_invalid_judge_result({"error": "criterion 3 unclear", "score": 0.5}) is False


def test_empty_scoring_error_is_not_a_failure():
    assert _is_invalid_judge_result({SCORING_ERROR_KEY: ""}) is False
