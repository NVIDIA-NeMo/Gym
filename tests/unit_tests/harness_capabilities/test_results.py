# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.harness_capabilities.results import Results, render_matrices


def test_missing_prerequisite_blocks_only_dependent_checks():
    result = Results()
    result.check("calls.present", "schema", False, evidence=("TE-1", "TE-2"))

    def must_not_run():
        raise AssertionError("blocked predicate evaluated")

    result.check("calls.tokens", "schema", must_not_run, depends_on=("calls.present",), evidence=("TE-2",))
    result.check("tools.present", "schema", True, evidence=("TE-5",))
    assert len(result.dump()) == 3
    assert result.rows["calls.tokens"].status == "not_assessed"
    assert result.rows["calls.tokens"].blocked_by == ["calls.present"]
    assert result.rows["tools.present"].status == "pass"


def test_multiple_items_cannot_hide_failure_or_missing_assessment():
    result = Results()
    for value in (True, False, True):
        result.check("call.id", "schema", value, location="/calls")
    assert result.rows["call.id"].status == "fail"
    result.check("tool.id", "schema", True)
    result.check("tool.id", "schema", True, available=False)
    assert result.rows["tool.id"].status == "not_assessed"


def test_matrix_reports_independent_results_and_absence():
    a = Results()
    a.check("call.id", "schema", True)
    a.check("tool.status", "behavioral", False)
    text = render_matrices(
        {"example": {"scenarios": [{"scenario": "normal", "checks": a.dump()}, {"scenario": "error", "checks": []}]}}
    )
    assert "Schema: `call.id` | Pass | Not assessed" in text
    assert "Behavioral: `tool.status` | Fail | Not assessed" in text
    assert "Gate" not in text
