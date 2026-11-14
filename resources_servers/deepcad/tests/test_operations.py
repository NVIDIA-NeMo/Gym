# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from resources_servers.deepcad.operations import (
    OperationTrace,
    add_operation_hint,
    extract_operation_candidates,
    operation_bonus,
    operation_hint,
)


def test_candidates_ignore_comments_and_strings_and_keep_aliases():
    code = """
import cadquery as cq
# cq.Workplane().shell(1)
text = "part.sweep(path)"
part = cq.Workplane("XY").box(4, 4, 4)
finish = part.edges().fillet
result = finish(.2).faces(">Z").workplane().hole(1)
"""
    assert extract_operation_candidates(code) == {"box": 1, "fillet": 1, "hole": 1}


def test_trace_real_cadquery_excludes_dead_code_fake_methods_and_internal_calls():
    cq = pytest.importorskip("cadquery")
    code = """
import cadquery as cq
class Fake:
    def sweep(self): return self
Fake().sweep()
if False:
    cq.Workplane("XY").sphere(3)
def make_part():
    part = cq.Workplane("XY").box(4,4,4)
    return part.edges().fillet(.2)
result = make_part()
"""
    previous = sys.getprofile()
    with OperationTrace(cq) as trace:
        namespace = {"__name__": "__main__"}
        exec(compile(code, "model.py", "exec"), namespace)
    assert trace.executed == {"box", "fillet"}
    assert namespace["result"].val().isValid()
    assert sys.getprofile() is previous


def test_trace_does_not_supply_cq_and_restores_on_error():
    cq = pytest.importorskip("cadquery")
    previous = sys.getprofile()
    with pytest.raises(NameError), OperationTrace(cq):
        exec(compile("result = cq.Workplane('XY').box(1,1,1)", "model.py", "exec"), {})
    assert sys.getprofile() is previous


def test_bonus_is_distinct_and_normalized_per_reference():
    bonus, matches = operation_bonus(0.4, ["loft", "fillet"], ["fillet"] * 50 + ["shell"])
    assert bonus == pytest.approx(0.02)
    assert matches == ["fillet"]
    assert operation_bonus(0.4, ["loft", "fillet"], ["loft", "fillet"])[0] == pytest.approx(0.04)
    assert operation_bonus(0.4, ["fillet"] * 10, ["fillet"])[0] == pytest.approx(0.04)
    assert operation_bonus(0.4, ["loft", "fillet", "shell", "extrude"], ["loft", "fillet"])[0] == pytest.approx(0.02)
    assert operation_bonus(0.4, ["loft"], ["loft"], max_fraction=0.05)[0] == pytest.approx(0.02)
    assert operation_bonus(0, ["loft"], ["loft"])[0] == 0
    assert operation_bonus(-1, ["loft"], ["loft"])[0] == 0
    assert operation_bonus(0.4, [], ["loft"])[0] == 0
    assert operation_bonus(0.4, ["not_cad"], ["not_cad"]) == (0, [])
    assert operation_bonus(0.4, ["loft", "not_cad"], ["loft"])[0] == pytest.approx(0.04)


@pytest.mark.parametrize("fraction", [-1, 0.21, float("nan"), float("inf")])
def test_invalid_bonus_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        operation_bonus(0.4, ["loft"], ["loft"], max_fraction=fraction)


def test_hint_determinism_boundaries_and_sampling():
    assert operation_hint("a", ["loft"], probability=0) == ""
    assert operation_hint("a", ["loft", "fillet", "loft"], probability=1) == "Try to use fillet/loft operations."
    assert operation_hint("a", [], probability=1) == ""
    hints = [operation_hint(str(i), ["loft"], probability=0.5) for i in range(1000)]
    assert 430 < sum(bool(x) for x in hints) < 570
    for index in range(20):
        assert len({operation_hint(str(index), ["loft"], probability=0.5) for _ in range(16)}) == 1


@pytest.mark.parametrize(
    "content", ["Require explicit imports.", [{"type": "input_text", "text": "Require explicit imports."}]]
)
def test_hint_goes_in_system_message_and_targets_stay_in_metadata(content):
    row = {
        "verifier_metadata": {"task_id": "part-1", "reference_step": "private.step"},
        "responses_create_params": {
            "input": [{"role": "system", "content": content}, {"role": "user", "content": "Render views"}]
        },
    }
    result = add_operation_hint(row, ["shell", "loft"], probability=1)
    assert result["verifier_metadata"]["reference_operations"] == ["loft", "shell"]
    assert result["verifier_metadata"]["operation_hint_applied"] is True
    assert "Try to use loft/shell operations." in str(result["responses_create_params"]["input"][0]["content"])
    assert "private.step" not in str(result["responses_create_params"])
    assert result["responses_create_params"]["input"][1]["content"] == "Render views"


def test_twenty_percent_bonus():
    assert operation_bonus(0.8, ["box", "fillet"], ["box", "fillet"], max_fraction=0.2)[0] == pytest.approx(0.16)
    assert operation_bonus(0.8, ["box", "fillet"], ["box"], max_fraction=0.2)[0] == pytest.approx(0.08)
