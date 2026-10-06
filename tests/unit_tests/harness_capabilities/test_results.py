# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest

from nemo_gym.harness_capabilities.checks import BehavioralCheck, SchemaCheck, SemanticCheck
from nemo_gym.harness_capabilities.results import Results, gate_passes, render_matrices


def schema_check(check_id="call.id", value="call-1", **kwargs):
    return SchemaCheck(
        id=check_id,
        tier="P0",
        evidence=("TE-1",),
        location="$.ng_trajectory.model_calls",
        reason="expected a string",
        schema={"type": "string"},
        value=value,
        **kwargs,
    )


def test_missing_prerequisite_blocks_only_dependent_checks():
    result = Results()
    result.run(schema_check("calls.present", None))

    def must_not_run():
        raise AssertionError("blocked predicate evaluated")

    result.run(
        SemanticCheck(
            id="calls.owner",
            tier="P0",
            evidence=("TE-8",),
            location="$.ng_trajectory.invocations",
            reason="unresolved call owner",
            predicate=must_not_run,
            depends_on=("calls.present",),
        )
    )
    result.run(schema_check("tools.present"))
    assert len(result.dump()) == 3
    assert result.rows["calls.owner"].status == "fail"
    assert result.rows["calls.owner"].blocked_by == ["calls.present"]
    assert result.rows["calls.owner"].reasons == ["blocked by calls.present; unresolved call owner"]
    assert result.rows["tools.present"].status == "pass"


@pytest.mark.parametrize(
    "settings,status",
    [
        ({"applies": False}, "not_applicable"),
        ({"applies": False, "available": False, "depends_on": ("absent",)}, "not_applicable"),
        ({"available": False}, "fail"),
        ({"depends_on": ("absent",)}, "fail"),
    ],
)
def test_excluded_or_blocked_schema_does_not_evaluate(settings, status):
    # An invalid schema would raise if evaluated: prerequisites must be checked first.
    check = replace(schema_check(), schema={"type": "invalid-type"}, **settings)
    result = Results()
    result.run(check)
    assert result.rows[check.id].status == status


def test_multiple_items_cannot_hide_failure_or_unavailable_input():
    result = Results()
    for value in ("call", None, "call"):
        result.run(schema_check(value=value))
    assert result.rows["call.id"].status == "fail"
    result.run(schema_check("tool.id"))
    result.run(schema_check("tool.id", available=False))
    assert result.rows["tool.id"].status == "fail"
    assert result.rows["tool.id"].reasons == ["required input is unavailable; expected a string"]


def test_matrix_reports_independent_results_and_absence():
    a = Results()
    a.run(schema_check())
    a.run(
        BehavioralCheck(
            id="tool.status",
            tier="P0",
            evidence=("TE-5",),
            location="$.ng_trajectory.tool_calls",
            reason="status differs from tool witness",
            predicate=lambda: False,
        )
    )
    text = render_matrices(
        {"example": {"scenarios": [{"scenario": "normal", "checks": a.dump()}, {"scenario": "error", "checks": []}]}}
    )
    assert "| **Schema** | | |\n| `call.id` | ✓ | ✗ |" in text
    assert "| **Behavioral** | | |\n| `tool.status` | ✗ | ✗ |" in text
    assert "Gate" not in text
    assert "?" not in text


@pytest.mark.parametrize("kind", [SemanticCheck, BehavioralCheck])
def test_predicate_subclass_supplies_kind_and_serializes_priority(kind):
    results = Results()
    check = kind(id="test", tier="P1", evidence=(), location="$", reason="mismatch", predicate=lambda: False)
    results.run(check)
    saved = json.loads(json.dumps(results.dump()))[0]
    assert saved["kind"] == ("semantic" if kind is SemanticCheck else "behavioral")
    assert saved["tier"] == "P1"
    assert saved["status"] == "fail"
    assert "predicate" not in saved


def test_priority_is_required_and_validated():
    with pytest.raises(TypeError, match="tier"):
        SchemaCheck(id="missing", evidence=(), location="$", reason="invalid", schema={}, value={})
    with pytest.raises(ValueError, match="invalid priority tier"):
        replace(schema_check(), tier="P3")


def test_repeated_id_cannot_change_priority():
    results = Results()
    check = schema_check()
    results.run(check)
    with pytest.raises(ValueError, match="conflicting definition"):
        results.run(replace(check, tier="P1"))


def test_gate_selects_individual_priorities_not_evidence_labels():
    results = Results()
    results.run(replace(schema_check("required"), evidence=()))
    results.run(replace(schema_check("later", None), evidence=("TE-1", "TE-8", "TE-9"), tier="P1"))
    results.run(replace(schema_check("future", None), tier="P2"))
    assert gate_passes(results.dump())
    assert not gate_passes(results.dump(), tier="P1")
    assert not gate_passes(results.dump(), tier="P2")
    # A failed P0 check cannot be exempted by another check sharing its TE label.
    results.run(replace(schema_check("required-owner", None), evidence=("TE-8",)))
    results.run(replace(schema_check("step-owner"), evidence=("TE-9",)))
    assert not gate_passes(results.dump())


@pytest.mark.parametrize("available,applies,passed", [(False, True, False), (True, False, True)])
def test_gate_handles_blocked_and_excluded_checks(available, applies, passed):
    results = Results()
    results.run(schema_check("present"))
    results.run(schema_check("optional", available=available, applies=applies))
    assert gate_passes(results.dump()) is passed
    assert not gate_passes([])
    assert not gate_passes([row for row in results.dump() if row["status"] == "not_applicable"])


def test_schema_failure_reports_field_path_without_payload_or_definition():
    results = Results()
    check = replace(
        schema_check(),
        schema={"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}},
        value=[{"id": "private-payload-canary"}],
    )
    results.run(check)
    row = results.dump()[0]
    assert row["locations"] == ["$.ng_trajectory.model_calls", "$.ng_trajectory.model_calls[0].id"]
    assert "private-payload-canary" not in json.dumps(row)
    assert "schema" not in row and "value" not in row
