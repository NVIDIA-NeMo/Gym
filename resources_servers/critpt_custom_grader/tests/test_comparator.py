# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-safe comparator regressions on synthetic values and inert hostile text.

The comparator is a pure function with no IO or execution. Tests using the symbolic_runtime fixture skip
without SymPy. None needs a sandbox or a provider.
"""

import math
from decimal import Decimal
from fractions import Fraction

import pytest

from resources_servers.critpt_custom_grader import comparator
from resources_servers.critpt_custom_grader.codec import (
    ComplexValue,
    LegacyText,
    RepresentationError,
    SetValue,
    SymbolicText,
    encode_value,
)
from resources_servers.critpt_custom_grader.comparator import statement_promises
from resources_servers.critpt_custom_grader.task_data import (
    MAX_DEPTH,
    MAX_INTEGER_DIGITS,
    MAX_STATEMENT_CHARS,
    WIRE_FORMAT,
    TaskData,
)


def value_outcome(value):
    return {"kind": "value", "value": encode_value(value)}


def request(observed, expected, *, role="candidate", policy=None):
    return {
        "version": 1,
        "observed_role": role,
        "observed": observed,
        "expected": expected,
        "policy": {} if policy is None else policy,
    }


def compare(observed, expected, *, role="candidate", policy=None):
    payload = request(value_outcome(observed), value_outcome(expected), role=role, policy=policy)
    return comparator.compare_request(payload)


NO_PROMISE = {"clauses": [], "blanket": None, "named": {}}


def clause_shapes(promises):
    # What grading reads from a clause: its value, its kind and whether it is honoured, never where it sits.
    return [(clause["value"], clause["kind"], clause["honoured"]) for clause in promises["clauses"]]


@pytest.fixture
def symbolic_runtime():
    pytest.importorskip("sympy")
    from resources_servers.critpt_custom_grader import symbolic

    return symbolic


def test_silent_leaf_default_mirrors_the_official_harness():
    # A silent numeric leaf resolves to rtol 1e-5, atol 1e-8, not the strict 5e-12/0.
    assert compare(Decimal("1.000009"), Decimal("1"))["status"] == "equal"
    assert compare(Decimal("1.00002"), Decimal("1"))["status"] == "mismatch"
    # The 1e-8 absolute floor admits a near-zero expected value. An integer expectation stays exact.
    assert compare(Decimal("1e-9"), Decimal("0"))["status"] == "equal"
    assert compare(Decimal("1e-7"), Decimal("0"))["status"] == "mismatch"


def test_configured_silent_default_overrides_the_strict_fallback():
    # A leaf 1e-6 off fails at the strict 5e-12 fallback but passes at the 1e-5 server default.
    observed, expected = Decimal("1.000001"), Decimal("1")
    assert compare(observed, expected)["status"] == "equal"
    assert compare(observed, expected, policy={"default_rtol": "5e-12", "default_atol": "0"})["status"] == "mismatch"
    assert compare(observed, expected, policy={"default_rtol": "1e-5", "default_atol": "1e-8"})["status"] == "equal"


def test_comparison_policy_transports_the_server_default_only_when_supplied():
    task = TaskData.model_validate(
        {
            "problem_id": "synthetic",
            "reference_source": "def solve():\n    return 1\n",
            "entrypoint": "solve",
            "test_cases": [{"input": [], "output": 1}],
        }
    )
    bare = task.comparison_policy(0)
    assert "default_rtol" not in bare and "default_atol" not in bare
    configured = task.comparison_policy(0, default_rtol="2e-4", default_atol="3e-9")
    assert configured["default_rtol"] == "2e-4"
    assert configured["default_atol"] == "3e-9"
    # The transported default resolves a silent leaf, never re-read after a resolved verdict.
    assert (
        compare(Decimal("1.0002"), Decimal("1"), policy={"default_rtol": "2e-4", "default_atol": "3e-9"})["status"]
        == "equal"
    )
    assert (
        compare(Decimal("1.0003"), Decimal("1"), policy={"default_rtol": "2e-4", "default_atol": "3e-9"})["status"]
        == "mismatch"
    )


def test_task_and_statement_tolerances_win_over_the_server_default():
    observed, expected = Decimal("1.001"), Decimal("1")  # 1e-3 off
    server_default = {"default_rtol": "1e-5", "default_atol": "1e-8"}
    assert compare(observed, expected, policy=server_default)["status"] == "mismatch"
    # An explicit task-level tolerance still wins over the server default.
    assert compare(observed, expected, policy={**server_default, "rtol": "1e-2"})["status"] == "equal"
    # A statement promise still wins over the server default, tighter or looser.
    promise = {**server_default, "statement": "Relative error below 2e-3."}
    assert compare(observed, expected, policy=promise)["status"] == "equal"
    assert compare(Decimal("1.0021"), Decimal("1"), policy=promise)["status"] == "mismatch"


def test_default_integers_are_exact_even_above_binary64_and_float_range():
    large = 10**400 + 17
    assert compare(large, large)["status"] == "equal"
    assert compare(large + 1, large)["status"] == "mismatch"
    assert compare(float(2**53), 2**53 + 1)["status"] == "mismatch"
    assert compare(7.0, 7)["status"] == "equal"
    assert compare(101, 100, policy={"rtol": "0.01"})["status"] == "equal"


def test_fraction_and_decimal_comparison_never_rounds_through_float():
    exact = {"rtol": "0", "atol": "0"}
    number = Fraction(10**300 + 17, 23)
    assert compare(number, number, policy=exact)["status"] == "equal"
    assert compare(number + Fraction(1, 23), number, policy=exact)["status"] == "mismatch"
    assert compare(Fraction(1, 10), Decimal("0.1"), policy=exact)["status"] == "equal"
    assert compare(0.1, Decimal("0.1"), policy=exact)["status"] == "mismatch"
    assert compare(1 / 3, Fraction(1, 3))["status"] == "equal"
    assert compare(1 / 3, Fraction(1, 3), policy=exact)["status"] == "mismatch"


def test_tolerance_boundary_is_exact_and_expected_relative():
    policy = {"rtol": "0.1", "atol": "0.01"}
    assert compare(Decimal("1.11"), Decimal("1"), policy=policy)["status"] == "equal"
    assert compare(Decimal("1.1100000000000000001"), Decimal("1"), policy=policy)["status"] == "mismatch"
    assert compare(Decimal("1.115"), Decimal("1"), policy=policy)["status"] == "mismatch"


@pytest.mark.parametrize("observed, expected", [(True, 1), (1, True), (False, 0.0), ("1", True), (True, Fraction(1))])
def test_boolean_never_matches_numeric(observed, expected):
    assert compare(observed, expected)["status"] == "mismatch"
    assert compare(True, True)["status"] == "equal"


def test_per_leaf_overrides_do_not_widen_other_quantities():
    expected = {"large": Decimal("1"), "tiny": Decimal("1e-40")}
    policy = {"tolerances": [{"path": "large", "rtol": "0", "atol": "0.02"}]}
    # The tiny leaf has no entry, so it takes the silent default, never the large leaf's 0.02 window.
    result = compare({"large": Decimal("1.01"), "tiny": Decimal("0.01")}, expected, policy=policy)
    assert result["status"] == "mismatch"
    assert result["path"] == "/tiny"
    assert compare({"large": Decimal("1.01"), "tiny": Decimal("1e-40")}, expected, policy=policy)["status"] == "equal"
    assert (
        compare(Decimal("1.000000000004"), Decimal("1"), policy={"tolerances": [{"path": "root", "atol": "0"}]})[
            "status"
        ]
        == "equal"
    )


def test_tolerance_pointer_escaping_and_nested_positions():
    expected = {"a/b~c": [Decimal("1")]}
    policy = {"tolerances": [{"path": "/a~1b~0c/0", "rtol": "0", "atol": "0.1"}]}
    assert compare({"a/b~c": (Decimal("1.05"),)}, expected, policy=policy)["status"] == "equal"


@pytest.mark.parametrize(
    "policy",
    [
        {"rtol": True},
        {"atol": "-1"},
        {"rtol": "NaN"},
        {"rtol": "1e999999"},
        # A well-formed entry whose path is not consumed is ignored, not a defect
        # (see test_unused_tolerance_path_is_ignored_not_a_defect). Only a malformed entry stays
        # invalid: a path with no tolerance value, or an unknown field.
        {"tolerances": [{"path": "/"}, {"path": "root"}]},
        {"tolerances": [{"quantity": "not a path", "atol": "1"}]},
        {"timeout": 10},
    ],
)
def test_invalid_or_unmatched_tolerance_is_expected_defect(policy):
    result = compare(1, 1, policy=policy)
    assert result["status"] == "invalid_expected"
    assert result["equal"] is None


def test_composites_require_all_elements_exact_keys_and_lengths():
    assert compare({"b": [True, None], "a": (1, 2)}, {"a": [1, 2], "b": [True, None]})["status"] == "equal"
    assert compare([1], [1, 2])["status"] == "mismatch"
    assert compare([], [1])["status"] == "mismatch"
    assert compare({"x": 1, "extra": 9}, {"x": 1})["status"] == "mismatch"
    assert compare({"x": 1}, [1])["status"] == "mismatch"
    assert compare([1, False], [1, 0])["status"] == "mismatch"
    assert compare(None, None)["status"] == "equal"


def test_complex_is_a_typed_two_component_leaf():
    expected = ComplexValue(Fraction(1, 2), Decimal("0.25"))
    assert compare(0.5 + 0.25j, expected)["status"] == "equal"
    assert compare(0.5 + 0.26j, expected)["status"] == "mismatch"
    assert compare([0.5, 0.25], expected)["status"] == "mismatch"
    assert compare({"__complex__": [0.5, 0.25]}, expected)["status"] == "mismatch"
    assert (
        compare(
            ComplexValue(Decimal("0.51"), Decimal("0.26")),
            expected,
            policy={"tolerances": [{"path": "", "atol": "0.02"}]},
        )["status"]
        == "equal"
    )


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf, Decimal("NaN"), Decimal("-Infinity")])
def test_nonfinite_candidate_mismatches_but_expected_is_unscorable(number):
    assert compare(number, 1)["status"] == "mismatch"
    assert compare(number, number)["status"] == "invalid_expected"
    assert compare({}, {"nested": [number]})["status"] == "invalid_expected"
    assert compare(0j, ComplexValue(0, number))["status"] == "invalid_expected"


def test_categorical_strings_do_not_become_expressions_or_numbers():
    assert compare("upper-band", "upper-band")["status"] == "equal"
    assert compare("upper-band+0", LegacyText("upper-band"))["status"] == "mismatch"
    assert compare("category+0", "category")["status"] == "mismatch"
    assert compare("2*pi", "pi")["status"] == "mismatch"
    assert compare(Decimal("1.5"), "1.5")["status"] == "mismatch"
    assert compare("nan", "nan")["status"] == "equal"
    value = {"__nonfinite__": "inf", "value": ["symbolic", "x+"]}
    assert compare(value, value)["status"] == "equal"


def test_equation_and_assignment_grade_as_categorical_and_infinity_is_rejected():
    # An equation or assignment is refused by the grammar and compared as an exact string, categorical: an
    # identical reproduction is equal and a different one mismatches. A bare infinity atom is nonfinite, so a
    # symbolic-carrier expectation of it is an invalid expectation (nonfinite_expected), never a verdict on the
    # candidate.
    equation = "a^2 - 4*b*c = 0"
    assignment = "y0 = k/(2*L^2)"
    for carrier in (SymbolicText, LegacyText):
        assert compare(carrier(equation), carrier(equation))["status"] == "equal"
        assert compare(carrier(assignment), carrier(assignment))["status"] == "equal"
        rejected = compare(carrier("oo"), carrier("oo"))
        assert rejected["status"] == "invalid_expected" and rejected["code"] == "nonfinite_expected"
        assert compare(carrier(equation), carrier("a = b"))["status"] == "mismatch"
    # The expectation side is accepted, not refused as an invalid reference expression.
    assert compare(SymbolicText(equation), SymbolicText(equation), role="reference")["status"] == "equal"
    # A plain value against an equation expectation is a mismatch, never a parse of the equation.
    assert compare(5, SymbolicText(equation))["status"] == "mismatch"


def test_numeric_looking_strings_follow_expected_type_without_precision_loss():
    assert compare("7/19", Fraction(7, 19), policy={"rtol": "0"})["status"] == "equal"
    assert compare(Decimal("1.5"), LegacyText("1.5"))["status"] == "equal"
    assert compare("9007199254740993", 9007199254740992)["status"] == "mismatch"
    assert compare("0.10000000000000000001", Decimal("0.1"), policy={"rtol": "0"})["status"] == "mismatch"
    assert compare("1e999999", Decimal("1"))["status"] == "invalid_candidate"
    assert compare(1, LegacyText("1e999999"))["status"] == "invalid_expected"
    assert compare(0, LegacyText("1/0"))["status"] == "invalid_expected"


@pytest.mark.parametrize("text", ["[1, 2]", "{'x': 1}", "'green'", "True", "None", "1_000"])
def test_ambiguous_legacy_literal_expectations_are_not_silently_rewritten(text):
    result = compare(text, LegacyText(text))
    assert result["status"] == "invalid_expected"
    assert result["code"] == "ambiguous_legacy_literal"
    assert compare(text, text)["status"] == "equal"


def test_expected_exception_checks_outcome_not_message_or_returned_error():
    expected = {"kind": "exception", "message": "a diagnostic, not a criterion"}
    observed = {"kind": "exception", "type": "SomeRejection", "message": "different diagnostic"}
    assert comparator.compare_request(request(observed, expected))["status"] == "equal"
    returned_text = comparator.compare_request(request(value_outcome("different diagnostic"), expected))
    assert returned_text["code"] == "did_not_raise"
    assert comparator.compare_request(request(value_outcome({"kind": "exception"}), expected))["status"] == "mismatch"
    assert comparator.compare_request(request(observed, value_outcome("different diagnostic")))["code"] == "raised"
    result = comparator.compare_request(request({"kind": "encoding_error", "code": "invalid_type"}, expected))
    assert result["status"] == "invalid_candidate"


@pytest.mark.parametrize(
    "observed, expected, role, status, code",
    [
        # An expected-error case passes only when the call raises, type and message are never compared.
        # The reference role never yields "mismatch": a bad reference is invalid_reference.
        ("exception", "exception", "reference", "equal", "expected_exception"),
        ("value", "exception", "reference", "invalid_reference", "did_not_raise"),
        ("exception", "exception", "candidate", "equal", "expected_exception"),
        ("value", "exception", "candidate", "mismatch", "did_not_raise"),
        ("exception", "value", "reference", "invalid_reference", "raised"),
        ("exception", "value", "candidate", "mismatch", "raised"),
    ],
)
def test_exception_kind_verdicts_match_the_execution_contract(observed, expected, role, status, code):
    def outcome(kind):
        return {"kind": "exception"} if kind == "exception" else value_outcome(1)

    result = comparator.compare_request(request(outcome(observed), outcome(expected), role=role))
    assert result["status"] == status
    assert result["code"] == code


def test_reference_mismatch_and_representation_failure_never_grade_candidate():
    result = compare(8, 7, role="reference")
    assert result["status"] == "invalid_reference"
    assert result["equal"] is None
    assert result["side"] == "reference"
    expected = {"kind": "exception"}
    no_exception = comparator.compare_request(request(value_outcome(1), expected, role="reference"))
    assert no_exception["status"] == "invalid_reference"


@pytest.mark.parametrize("role, status", [("candidate", "invalid_candidate"), ("reference", "invalid_reference")])
def test_unknown_tags_are_attributed_to_the_correct_side(role, status):
    invalid = {"kind": "value", "value": {"format": WIRE_FORMAT, "value": ["unapproved", "x"]}}
    assert comparator.compare_request(request(invalid, value_outcome(1), role=role))["status"] == status
    result = comparator.compare_request(request(invalid, invalid, role=role))
    assert result["status"] == "invalid_expected"
    assert result["code"] == "unknown_tag"


def test_deep_and_oversized_representations_have_side_specific_outcomes():
    node = ["null"]
    for _ in range(MAX_DEPTH + 1):
        node = ["list", [node]]
    for invalid_node in (node, ["int", "9" * (MAX_INTEGER_DIGITS + 1)], ["decimal", "1e999999"]):
        invalid = {"kind": "value", "value": {"format": WIRE_FORMAT, "value": invalid_node}}
        assert comparator.compare_request(request(invalid, value_outcome(1)))["status"] == "invalid_candidate"
        assert comparator.compare_request(request(value_outcome(1), invalid))["status"] == "invalid_expected"


@pytest.mark.parametrize(
    "observed, expected, status",
    [
        ("(z+2)^2", "z^2+4*z+4", "equal"),
        ("I*I", "-1", "equal"),
        ("i+I", "2*I", "equal"),
        ("7/10 + 1/5", "9/10", "equal"),
        ("Rational(7, 10)", "7/10", "equal"),
        ("sin(z)^2+cos(z)^2", "1", "equal"),
        ("ln(E)", "1", "equal"),
        ("(2+3*i)*(2-3*I)", "13", "equal"),
        ("z+2", "z+3", "mismatch"),
        ("z", "w", "mismatch"),
        ("0.10000000000000000001", "1/10", "mismatch"),
    ],
)
def test_symbolic_arithmetic_and_approved_functions(symbolic_runtime, observed, expected, status):
    assert compare(SymbolicText(observed), SymbolicText(expected))["status"] == status


@pytest.mark.parametrize("name", ["pi", "E", "I"])
def test_bare_and_parenthesized_reserved_names_are_the_same_constant(symbolic_runtime, name):
    # A reserved name reads as its constant, bare or parenthesized
    # (see test_runner.test_bare_reserved_symbolic_name_materializes_as_the_constant).
    assert compare(SymbolicText(name), SymbolicText(f"({name})"))["status"] == "equal"
    assert compare(SymbolicText(name), SymbolicText(f"2*{name}"))["status"] == "mismatch"


def test_symbolic_numeric_bridge_and_legacy_formula_detection(symbolic_runtime):
    assert compare("3+3", 6)["status"] == "equal"
    assert compare("3+4", 6)["status"] == "mismatch"
    assert compare(6, LegacyText("3+3"))["status"] == "equal"
    assert compare("-I", LegacyText("-i"))["status"] == "equal"
    assert compare(SymbolicText("sqrt(z^2)"), SymbolicText("z"))["status"] != "equal"


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os')",
        "open('marker')",
        "(1).__class__",
        "sqrt.__globals__",
        "x[0]",
        "[x for x in (1, 2)]",
        "(lambda x: x)(1)",
        "sin(**{})",
        "sin(x=0)",
        "Symbol('x', real=True)",
        "2x",
        "[x+1]",
        "x**x",
        "x+",
    ],
)
def test_hostile_and_unsupported_expressions_are_side_specific(symbolic_runtime, text):
    assert compare(SymbolicText(text), SymbolicText("x+1"))["status"] == "invalid_candidate"
    result = compare(SymbolicText(text), SymbolicText(text))
    assert result["status"] == "invalid_expected"
    assert result["equal"] is None


@pytest.mark.parametrize("text", ["tr(rho)", "unapproved(x)", "f(x, y)"])
def test_undefined_function_application_parses_and_compares(symbolic_runtime, text):
    # A name not in the function table is a fresh sympy.Function: it parses on both sides and grades by
    # equivalence, not as a defect.
    assert compare(SymbolicText(text), SymbolicText(text))["status"] == "equal"
    # A different undefined-function argument is a genuine, undecidable mismatch, never invalid data.
    assert compare(SymbolicText("g(a)"), SymbolicText("g(b)"))["status"] in ("mismatch", "uncertain")


@pytest.mark.parametrize(
    "text",
    ["9" * 129, "1e129", "x**33", "2**(2**8)", "(x+y+z)**8", "sin(sin(sin(sin(sin(x)))))", "x" * 4097],
)
def test_symbolic_size_exponent_and_growth_limits(symbolic_runtime, text):
    assert compare(SymbolicText(text), SymbolicText("x"))["status"] == "invalid_candidate"
    assert compare(SymbolicText("x"), SymbolicText(text))["status"] == "invalid_expected"


def test_invalid_expected_expression_is_found_before_shape_or_exception_checks(symbolic_runtime):
    expected = {"valid": 1, "bad": [SymbolicText("x+")]}
    result = comparator.compare_request(request({"kind": "exception"}, value_outcome(expected)))
    assert result["status"] == "invalid_expected"
    assert result["path"] == "/bad/0"
    assert compare({}, expected)["status"] == "invalid_expected"


@pytest.mark.parametrize(
    "failure, code",
    [
        (TimeoutError(), "comparator_timeout"),
        (MemoryError(), "comparator_memory"),
        (RuntimeError(), "comparator_failure"),
    ],
)
def test_comparator_failures_are_uncertainty_not_wrong_answers(monkeypatch, failure, code):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(comparator, "_parse", fail)
    result = compare(SymbolicText("x"), SymbolicText("x"))
    assert result["status"] == "uncertain"
    assert result["equal"] is None
    assert result["code"] == code


def test_undecided_algebra_remains_uncertain(symbolic_runtime, monkeypatch):
    monkeypatch.setattr(symbolic_runtime, "equivalent", lambda left, right: None)
    result = compare(SymbolicText("sin(x)"), SymbolicText("x"))
    assert result["status"] == "uncertain"
    assert result["code"] == "symbolic_undecided"


def test_symbolic_expression_against_a_number_is_decided_never_undecided(symbolic_runtime):
    # zeta binds to the SymPy library function, so an expression built from it reduces to a real number and
    # grades definitively against a float expectation. 2.5 is deliberately off the expression's value (about
    # 2.75), so the pair is a mismatch. The code is symbolic_comparison, so zeta resolved.
    decided = compare(SymbolicText("2*zeta(3) + zeta(5)/3"), 2.5)
    assert decided["status"] == "mismatch"
    assert decided["code"] == "symbolic_comparison"
    # A float never equals a symbolic constant, even its rounding: 1.4142135623730951 is the double nearest
    # sqrt(2), and the pair is still a decided mismatch, with the float on either side.
    rounded = compare(SymbolicText("sqrt(2)"), 1.4142135623730951)
    assert rounded["status"] == "mismatch"
    assert rounded["code"] == "symbolic_comparison"
    flipped = compare(1.4142135623730951, SymbolicText("sqrt(2)"))
    assert flipped["status"] == "mismatch"
    assert flipped["code"] == "symbolic_comparison"
    # A candidate that cannot reduce to a number is undecided by algebra. Against a concrete numeric
    # expectation it collapses to mismatch. Two SymbolicText sides stay uncertain
    # (test_undecided_algebra_remains_uncertain).
    collapsed = compare(SymbolicText("tr(x)"), 5)
    assert collapsed["status"] == "mismatch"
    assert collapsed["code"] == "symbolic_number_mismatch"


@pytest.mark.parametrize(
    "failure, code, status, side",
    [
        (TimeoutError(), "comparator_timeout", "uncertain", "comparator"),
        (MemoryError(), "comparator_memory", "invalid_candidate", "candidate"),
    ],
)
def test_candidate_memory_exhaustion_is_scored_wrong_but_a_timeout_stays_uncertain(
    monkeypatch, failure, code, status, side
):
    def exhaust(*args, **kwargs):
        raise failure

    # The expectation resolves within the caps, so a memory cap is in the candidate value and scored wrong. A
    # per-case timeout bounds the comparator's own proof work, so it stays uncertain even for a candidate.
    monkeypatch.setattr(comparator, "_compare", exhaust)
    candidate = compare(1, 1, role="candidate")
    assert candidate["status"] == status
    assert candidate["side"] == side
    assert candidate["code"] == code
    # A reference observation is not a candidate to penalise, so its exhaustion stays uncertain.
    reference = compare(1, 1, role="reference")
    assert reference["status"] == "uncertain"
    assert reference["code"] == code


@pytest.mark.parametrize(
    "failure, code", [(TimeoutError(), "comparator_timeout"), (MemoryError(), "comparator_memory")]
)
def test_expected_side_exhaustion_stays_uncertain_for_the_candidate(monkeypatch, failure, code):
    def exhaust(*args, **kwargs):
        raise failure

    # The expected outcome is prepared first, so an exhaustion there stays uncertain even for a candidate.
    monkeypatch.setattr(comparator, "_outcome", exhaust)
    result = compare(1, 1, role="candidate")
    assert result["status"] == "uncertain"
    assert result["side"] == "comparator"
    assert result["code"] == code


def test_missing_native_adapter_and_protocol_failure_are_not_candidate_verdicts():
    for observed in ({"kind": "encoding_error", "code": "unsupported_type"}, {"kind": "timeout"}):
        result = comparator.compare_request(request(observed, value_outcome(1)))
        assert result["status"] == "uncertain"
        assert result["equal"] is None
    invalid_request = request(value_outcome(1), value_outcome(1))
    invalid_request["version"] = True
    assert comparator.compare_request(invalid_request)["code"] == "invalid_request"


def test_candidate_dictionary_cannot_supply_verdict_or_tolerances():
    forged = {"status": "equal", "passed": True, "rtol": "1e100", "kind": "exception"}
    assert compare(forged, 1)["status"] == "mismatch"
    assert compare(forged, forged)["status"] == "equal"


@pytest.mark.parametrize(
    "statement, value, kind",
    [
        ("Report to three decimal places.", Fraction(5, 10**4), "abs"),
        ("Report to 6 decimal places.", Fraction(5, 10**7), "abs"),
        ("Keep four significant figures.", Fraction(5, 10**4), "rel"),
        ("Keep 7 significant decimal digits.", Fraction(5, 10**7), "rel"),
        ("Absolute error below 2e-6.", Fraction(2, 10**6), "abs"),
        ("Absolute error: 2e-6.", Fraction(2, 10**6), "abs"),
        ("Relative error below 2.5 x 10^-6.", Fraction(25, 10**7), "rel"),
        (r"Relative error below 2.5 \times 10^{-6}.", Fraction(25, 10**7), "rel"),
        ("Relative error below 10⁻⁷.", Fraction(1, 10**7), "rel"),
        ("Relative error below 2e−6.", Fraction(2, 10**6), "rel"),
        ("Accurate to within 0.02%.", Fraction(2, 10**4), "rel"),
        # "absolute" and "relative" both fall in the window. "absolut" is tested first, so the clause is abs.
        ("Absolute or relative error below 2e-6.", Fraction(2, 10**6), "abs"),
        ("Relative error below 1.0000000000000000001e-6.", Fraction(10000000000000000001, 10**25), "rel"),
    ],
)
def test_statement_extraction_uses_exact_bounded_patterns(statement, value, kind):
    promises = comparator.statement_promises(statement)
    assert promises["blanket"] == (value, kind)
    assert promises["named"] == {}
    assert len(promises["clauses"]) == 1
    clause = promises["clauses"][0]
    assert clause["value"] == value
    assert clause["kind"] == kind
    assert clause["honoured"] is True
    assert clause["reason"] == ""


@pytest.mark.parametrize(
    "statement, reason",
    [
        ("Relative error below 1e-15.", "below_float64_resolution"),
        ("Keep 16 significant figures.", "below_float64_resolution"),
        ("Relative error below 20%.", "too_loose_to_be_an_accuracy_claim"),
        ("Report to 2048 decimal places.", "unsupported_number"),
        ("Relative error below 1e-2048.", "unsupported_number"),
        # "within" borrows the claim of "Accurate" in its sentence, but no kind is stated for a bare number.
        ("Accurate to within 2e-6.", "no_kind"),
    ],
)
def test_refused_statement_clauses_remain_visible_and_do_not_erase_author(statement, reason):
    promises = comparator.statement_promises(statement, ["rate"])
    assert promises["blanket"] is None
    assert promises["named"] == {}
    assert len(promises["clauses"]) == 1
    assert promises["clauses"][0]["honoured"] is False
    assert promises["clauses"][0]["reason"] == reason
    policy = {"statement": statement, "rtol": "0", "atol": "0.01"}
    assert compare(Decimal("1.01"), Decimal("1"), policy=policy)["status"] == "equal"
    assert compare(Decimal("1.011"), Decimal("1"), policy=policy)["status"] == "mismatch"


@pytest.mark.parametrize("statement", ["Relative error below 2e-15.", "Relative error below 10%."])
def test_statement_honour_interval_keeps_supported_boundaries(statement):
    promises = comparator.statement_promises(statement)
    assert promises["blanket"] is not None
    assert promises["clauses"][0]["honoured"] is True


@pytest.mark.parametrize(
    "statement",
    [
        "Keep a reasonable number of digits.",
        "Absolute error below 0.0002.",
        "Report to the nearest tenth.",
        "The fixture constant is 2e-6.",
        # A keyword's window stops at its sentence break, so it never pulls a number out of the next sentence.
        "Accurate values are required. The fixture constant is 2e-6.",
        "```text\nRelative error below 2e-6.\n```",
        "```text\nRelative error below 2e-6.",
    ],
)
def test_unrecognized_prose_and_fenced_code_do_not_create_promises(statement):
    assert comparator.statement_promises(statement) == NO_PROMISE
    policy = {"statement": statement}
    assert compare(Decimal("1.000009"), Decimal("1"), policy=policy)["status"] == "equal"
    assert compare(Decimal("1.00002"), Decimal("1"), policy=policy)["status"] == "mismatch"


@pytest.mark.parametrize("separator", [". ", "; ", "\n", "! ", "? "])
def test_error_kinds_and_accuracy_windows_do_not_cross_sentences(separator):
    statement = (
        "Report lift with absolute error below 2e-6" + separator + "Report drift with relative error below 4e-5."
    )
    promises = comparator.statement_promises(statement, ["lift", "drift"])
    # Kind is now sentence-bounded, the same bounds quantity attribution uses. Across a real sentence break (. ; or a
    # newline that is not a soft wrap, as here, where the next line starts with a capital) drift keeps its own
    # "relative" kind. "!" and "?" followed by a space do not end a sentence, so the one sentence carries "absolute"
    # first and both leaves read abs.
    if separator in (". ", "; ", "\n"):
        assert promises["named"] == {"lift": (Fraction(2, 10**6), "abs"), "drift": (Fraction(4, 10**5), "rel")}
        assert promises["blanket"] == (Fraction(4, 10**5), "rel")
    else:
        assert promises["named"] == {"lift": (Fraction(2, 10**6), "abs"), "drift": (Fraction(4, 10**5), "abs")}
        assert promises["blanket"] == (Fraction(4, 10**5), "abs")
    assert len(promises["clauses"]) == 2


@pytest.mark.parametrize("separator", [". ", "; ", ": ", "\n", "! ", "? "])
def test_a_prior_sentence_cannot_claim_the_next_quantity(separator):
    statement = (
        "Report omitted with relative error below 2e-9" + separator + "Report residue with relative error below 4e-5."
    )
    promises = comparator.statement_promises(statement, ["residue"])
    # Only [.;:] + whitespace and a newline that is not a soft wrap (here the next line starts with a capital) end a
    # sentence. After those the two sentences split, so the earlier 2e-9 clause cannot reach residue. After "!" or "?"
    # and a space the text is one sentence, so that clause claims residue at the tighter 2e-9 and rejects a 3e-5 miss.
    if separator in ("! ", "? "):
        assert promises["named"] == {"residue": (Fraction(2, 10**9), "rel")}
        expected_status = "mismatch"
    else:
        assert promises["named"] == {"residue": (Fraction(4, 10**5), "rel")}
        expected_status = "equal"
    assert (
        compare({"residue": Decimal("1.00003")}, {"residue": Decimal("1")}, policy={"statement": statement})["status"]
        == expected_status
    )


@pytest.mark.parametrize(
    "statement",
    [
        "Report span and shift with relative error below 2e-8.",
        "Require relative error below 2e-8 for span and shift.",
    ],
)
def test_a_multi_name_clause_binds_its_margin_to_every_named_quantity(statement):
    statement += " Other outputs permit relative error below 4e-4."
    promises = comparator.statement_promises(statement, ["span", "shift"])
    assert promises["named"] == {"span": (Fraction(2, 10**8), "rel"), "shift": (Fraction(2, 10**8), "rel")}
    assert promises["blanket"] == (Fraction(4, 10**4), "rel")
    assert (
        compare(
            {"span": Decimal("1.0002"), "shift": Decimal("1.0002")},
            {"span": Decimal("1"), "shift": Decimal("1")},
            policy={"statement": statement},
        )["status"]
        == "mismatch"
    )


def test_a_multi_name_clause_leaves_the_blanket_for_an_unnamed_third_leaf():
    statement = "Report span and shift with relative error below 2e-8. Other outputs permit relative error below 4e-4."
    promises = comparator.statement_promises(statement, ["span", "shift", "drift"])
    assert promises["named"] == {"span": (Fraction(2, 10**8), "rel"), "shift": (Fraction(2, 10**8), "rel")}
    assert promises["blanket"] == (Fraction(4, 10**4), "rel")
    expected = {"span": Decimal("1"), "shift": Decimal("1"), "drift": Decimal("1")}
    # The two named leaves hold 2e-8. The unnamed leaf keeps the 4e-4 blanket, accepting a 2e-4 error.
    passing = {"span": Decimal("1.00000001"), "shift": Decimal("1.00000001"), "drift": Decimal("1.0002")}
    assert compare(passing, expected, policy={"statement": statement})["status"] == "equal"
    over_blanket = compare({**passing, "drift": Decimal("1.0005")}, expected, policy={"statement": statement})
    assert over_blanket["status"] == "mismatch"
    assert over_blanket["path"] == "/drift"
    over_named = compare({**passing, "span": Decimal("1.0002")}, expected, policy={"statement": statement})
    assert over_named["status"] == "mismatch"
    assert over_named["path"] == "/span"


def test_only_known_whole_quantity_names_are_claimed():
    statement = "Use rate_extra with relative error below 2e-8. Require relative error below 4e-4 for radius."
    promises = comparator.statement_promises(statement, ["rate", "radius"])
    assert promises["named"] == {"radius": (Fraction(4, 10**4), "rel")}
    assert comparator.statement_promises(statement)["named"] == {}
    vocabulary = "Report relative error below 2e-8."
    assert comparator.statement_promises(vocabulary, ["relative", "error"])["named"] == {}


def test_first_named_promise_is_not_replaced_by_a_looser_blanket():
    statement = "Report rate with relative error below 2e-8. Report rate with relative error below 4e-4."
    promises = comparator.statement_promises(statement, ["rate"])
    assert promises["named"] == {"rate": (Fraction(2, 10**8), "rel")}
    assert promises["blanket"] == (Fraction(4, 10**4), "rel")
    assert (
        compare({"rate": Decimal("1.000001")}, {"rate": Decimal("1")}, policy={"statement": statement})["status"]
        == "mismatch"
    )


def test_fenced_code_is_blank_and_normalization_preserves_clause_offsets():
    statement = "```text\nRelative error below 9e-3.\n```\nReport rate with relative error below 2e−6."
    promises = comparator.statement_promises(statement, ["rate"])
    assert promises["named"] == {"rate": (Fraction(2, 10**6), "rel")}
    assert len(promises["clauses"]) == 1
    clause = promises["clauses"][0]
    assert clause["at"] == statement.index("2e−6")
    assert clause["text"] == "2e-6"


@pytest.mark.parametrize("author_rtol", ["1e-3", "1e-9"])
@pytest.mark.parametrize("per_leaf", [False, True])
def test_author_governs_whether_statement_is_tighter_or_looser(author_rtol, per_leaf):
    authored = {"rtol": author_rtol, "atol": "0"}
    policy = {"tolerances": [{"path": "root", **authored}]} if per_leaf else authored
    policy["statement"] = "Return output with relative error below 2e-6."
    # The author governs either way: the statement's 2e-6 is tighter than an author 1e-3, looser than 1e-9.
    status = "equal" if author_rtol == "1e-3" else "mismatch"
    assert compare(Decimal("100.0002"), Decimal("100"), policy=policy)["status"] == status
    assert compare(Decimal("100.0002000000000000001"), Decimal("100"), policy=policy)["status"] == status


def test_named_promises_resolve_nested_leaves_without_a_task_wide_maximum():
    expected = {"band": {"width": Decimal("2"), "depth": Decimal("4")}, "other": Decimal("1")}
    # No leaf is authored, so each named promise governs its own nested leaf and the blanket governs the rest.
    policy = {"statement": "Grade width with relative error below 2e-8. Grade depth with relative error below 3e-4."}
    passing = {"band": {"width": Decimal("2.00000004"), "depth": Decimal("4.0012")}, "other": Decimal("1.0003")}
    assert compare(passing, expected, policy=policy)["status"] == "equal"
    for name, value in (("width", Decimal("2.000000041")), ("depth", Decimal("4.0012001"))):
        observed = {**passing, "band": {**passing["band"], name: value}}
        result = compare(observed, expected, policy=policy)
        assert result["status"] == "mismatch"
        assert result["path"] == "/band/" + name
    result = compare({**passing, "other": Decimal("1.0003001")}, expected, policy=policy)
    assert result["status"] == "mismatch"
    assert result["path"] == "/other"


@pytest.mark.parametrize(
    "entry, boundary_status",
    [
        ({"atol": "0"}, "equal"),
        ({"rtol": None, "atol": "0"}, "equal"),
        ({"rtol": "0", "atol": "0"}, "mismatch"),
        ({"rtol": "0"}, "mismatch"),
    ],
)
def test_silent_statement_uses_whole_authored_leaf_pair_not_task_components(entry, boundary_status):
    expected = 10**12
    policy = {
        "statement": "Return the requested value.",
        "rtol": "0.1",
        "atol": "100",
        "tolerances": [{"path": "root", **entry}],
    }
    assert compare(expected + 5, expected, policy=policy)["status"] == boundary_status
    assert compare(expected + 6, expected, policy=policy)["status"] == "mismatch"


@pytest.mark.parametrize(
    "expected, status",
    [
        (10**12, "mismatch"),
        (LegacyText("1000000000000"), "mismatch"),
        (Decimal("1000000000000"), "equal"),
        (Fraction(10**12, 1), "equal"),
    ],
)
def test_exact_integer_default_depends_on_expected_type_not_integer_valued_magnitude(expected, status):
    assert compare(10**12 + 1, expected, policy={"statement": "Compute the requested count."})["status"] == status


@pytest.mark.parametrize("expected", [10**400 + 19, Fraction(10**300 + 37, 29)])
def test_statement_margin_preserves_exact_integer_and_fraction_arithmetic(expected):
    policy = {"statement": "Relative error below 2.5 x 10^-6."}
    boundary = expected + expected * Fraction(25, 10**7)
    assert compare(boundary, expected, policy=policy)["status"] == "equal"
    assert compare(boundary + Fraction(1, 29), expected, policy=policy)["status"] == "mismatch"


def test_absolute_scale_guard_is_per_leaf_and_preserves_deliberate_author_tolerance():
    tiny, margin = Fraction(1, 10**30), Fraction(1, 10**44)
    expected = {"coarse": Decimal("2"), "tiny": tiny, "pinned": tiny}
    # The coarse leaf keeps the 5e-4 promise. The guard empties it for the unauthored tiny leaf, which falls to the
    # strict 5e-12 relative floor. The pinned leaf keeps its author's deliberate 1e-44 window, tighter than the floor.
    policy = {
        "statement": "Report values to three decimal places.",
        "tolerances": [{"path": "/pinned", "rtol": "0", "atol": "1e-44"}],
    }
    passing = {"coarse": Decimal("2.0005"), "tiny": tiny * (1 + Fraction(4, 10**12)), "pinned": tiny + margin}
    assert compare(passing, expected, policy=policy)["status"] == "equal"
    wrong = (
        ("tiny", 0),
        ("tiny", tiny * (1 + Fraction(6, 10**12))),
        ("pinned", 0),
        ("pinned", tiny * (1 + Fraction(4, 10**12))),
        ("pinned", tiny + 2 * margin),
    )
    for leaf, value in wrong:
        result = compare({**passing, leaf: value}, expected, policy=policy)
        assert result["status"] == "mismatch"
        assert result["path"] == "/" + leaf
    result = compare({**passing, "coarse": Decimal("2.000500000000001")}, expected, policy=policy)
    assert result["status"] == "mismatch"
    assert result["path"] == "/coarse"


@pytest.mark.parametrize(
    "expected, status",
    [
        (Decimal("0.0005"), "equal"),
        (Decimal("-0.0005"), "equal"),
        (Decimal("0.000499"), "mismatch"),
        (Decimal("-0.000499"), "mismatch"),
    ],
)
def test_absolute_scale_guard_uses_strict_magnitude_boundary(expected, status):
    assert compare(0, expected, policy={"statement": "Report to three decimal places."})["status"] == status


@pytest.mark.parametrize("expected", [0, Decimal("0"), Fraction(1, 10**30)])
def test_emptied_absolute_promise_never_erases_authored_window(expected):
    policy = {
        "statement": "Absolute error below 2e-6.",
        "tolerances": [{"path": "", "rtol": "0", "atol": "2e-8"}],
    }
    assert compare(Fraction(expected) + Fraction(2, 10**8), expected, policy=policy)["status"] == "equal"
    assert compare(Fraction(expected) + Fraction(21, 10**9), expected, policy=policy)["status"] == "mismatch"


def test_emptied_named_promise_does_not_switch_to_another_quantities_blanket():
    expected = {"speck": Fraction(1, 10**30)}
    # The guard empties speck's own absolute promise. The unauthored leaf then falls to the strict 5e-12 relative
    # floor, never to the 4e-4 relative blanket that governs other outputs.
    policy = {
        "statement": "Report speck with absolute error below 2e-6. Other outputs permit relative error below 4e-4.",
    }
    observed = {"speck": expected["speck"] * (1 + Fraction(3, 10**4))}
    result = compare(observed, expected, policy=policy)
    assert result["status"] == "mismatch"
    assert result["path"] == "/speck"
    floor = {"speck": expected["speck"] * (1 + Fraction(4, 10**12))}
    assert compare(floor, expected, policy=policy)["status"] == "equal"


@pytest.mark.parametrize(
    "authored, status",
    [({}, "equal"), ({"rtol": "0"}, "mismatch"), ({"tolerances": [{"path": "", "rtol": "0"}]}, "mismatch")],
)
def test_relative_promise_is_floored_only_where_no_author_spoke(authored, status):
    policy = {**authored, "statement": "Relative error below 2e-13."}
    # Where an author spoke, the author's exact rtol 0 governs outright, so even the 2e-13 miss is rejected.
    assert compare(Decimal("1.000000000004"), Decimal("1"), policy=policy)["status"] == status
    assert compare(Decimal("1.0000000000002"), Decimal("1"), policy=policy)["status"] == status
    assert compare(Decimal("1.000000000006"), Decimal("1"), policy=policy)["status"] == "mismatch"


def test_absolute_promise_floor_uses_leaf_margin_not_componentwise_maxima():
    policy = {"statement": "Report noninteger outputs to twelve decimal places."}
    expected = Decimal("1000000000000")
    assert compare(10**12 + 5, expected, policy=policy)["status"] == "equal"
    assert compare(Fraction(10**12 + 5) + Fraction(1, 10**13), expected, policy=policy)["status"] == "mismatch"
    # An authored entry governs its leaf outright: its own 1e-28 relative margin (1e-16 here), neither the
    # promise's 5e-13 nor the strict default's 5.
    policy["tolerances"] = [{"path": "", "rtol": "1e-28", "atol": "0"}]
    assert compare(Fraction(expected) + Fraction(1, 10**16), expected, policy=policy)["status"] == "equal"
    assert compare(Fraction(expected) + Fraction(5, 10**13), expected, policy=policy)["status"] == "mismatch"


def test_looser_absolute_promise_does_not_also_add_the_default_relative_margin():
    policy = {"statement": "Report to six decimal places."}
    assert compare(Decimal("2.0000005"), Decimal("2"), policy=policy)["status"] == "equal"
    assert compare(Decimal("2.000000500005"), Decimal("2"), policy=policy)["status"] == "mismatch"


def test_emptied_promise_without_author_falls_to_the_strict_default():
    # The "three decimal places" promise is emptied by the scale guard for an expected value this tiny.
    # With no author entry, an emptied promise floors to the strict default (rtol 5e-12, atol 0). So a
    # 4e-12 relative miss is admitted, but a zero candidate against a nonzero expected is not.
    policy = {"statement": "Report to three decimal places."}
    expected = Fraction(1, 10**30)
    assert compare(expected * (1 + Fraction(4, 10**12)), expected, policy=policy)["status"] == "equal"
    assert compare(0, expected, policy=policy)["status"] == "mismatch"
    assert compare(Decimal("1e-7"), Decimal("0"), policy=policy)["status"] == "mismatch"
    assert compare(Decimal("1e-3"), Decimal("0"), policy=policy)["status"] == "mismatch"


def test_unknown_kind_clause_is_refused_rather_than_honoured_as_both_components():
    # A number under an accuracy keyword with no stated kind is refused (no_kind) instead of binding both
    # components, so a silent leaf keeps the server default (rtol 1e-5, atol 1e-8) and an author still governs.
    statement = "The result must have an error below 2e-6."
    assert [clause["reason"] for clause in statement_promises(statement)["clauses"]] == ["no_kind"]
    silent = {"statement": statement}
    assert compare(Decimal("1.000006"), Decimal("1"), policy=silent)["status"] == "equal"
    assert compare(Decimal("1.0000101"), Decimal("1"), policy=silent)["status"] == "mismatch"
    authored = {**silent, "rtol": "0", "atol": "0"}
    assert compare(Decimal("1.000004"), Decimal("1"), policy=authored)["status"] == "mismatch"


@pytest.mark.parametrize("role, status", [("candidate", "mismatch"), ("reference", "invalid_reference")])
def test_statement_mismatch_preserves_reference_preflight_classification(role, status):
    policy = {"statement": "Report rate with relative error below 2e-8."}
    result = compare({"rate": Decimal("1.000001")}, {"rate": Decimal("1")}, role=role, policy=policy)
    assert result["status"] == status
    assert result["side"] == role
    assert result["path"] == "/rate"
    assert result["equal"] is (False if role == "candidate" else None)
    result = compare({"rate": Decimal("1.00000002")}, {"rate": Decimal("1")}, role=role, policy=policy)
    assert result["status"] == "equal"


@pytest.mark.parametrize("role", ["candidate", "reference"])
@pytest.mark.parametrize(
    "policy, code",
    [
        ({"statement": True}, "invalid_policy"),
        ({"statement": ["Relative error below 2e-6."]}, "invalid_policy"),
        ({"statement": "x" * (MAX_STATEMENT_CHARS + 1)}, "statement_limit"),
        ({"statement": "\ud800"}, "invalid_policy"),
        ({"statement": "Relative error below 2e-6.", "promises": {"rtol": "1"}}, "invalid_policy"),
    ],
)
def test_invalid_statement_policy_is_expected_defect_before_observation(policy, code, role):
    result = comparator.compare_request(request({"kind": "timeout"}, value_outcome(1), role=role, policy=policy))
    assert result["status"] == "invalid_expected"
    assert result["side"] == "expected"
    assert result["equal"] is None
    assert result["code"] == code


@pytest.mark.parametrize("statement", [None, 17, "x" * (MAX_STATEMENT_CHARS + 1)])
def test_statement_inspection_rejects_unbounded_or_nontext_input(statement):
    with pytest.raises(RepresentationError, match="statement_limit"):
        comparator.statement_promises(statement)


@pytest.mark.parametrize("character", ["\U0001f7e2", "\x00"])
def test_protected_comparator_accepts_full_statement_byte_allowance(character):
    policy = {"statement": character * MAX_STATEMENT_CHARS}
    assert compare(6, 6, policy=policy)["status"] == "equal"


def test_statement_at_character_limit_is_not_truncated_before_extraction():
    ending = "Relative error below 2e-6."
    policy = {"statement": " " * (MAX_STATEMENT_CHARS - len(ending)) + ending}
    assert compare(Decimal("1.000002"), Decimal("1"), policy=policy)["status"] == "equal"
    assert compare(Decimal("1.0000021"), Decimal("1"), policy=policy)["status"] == "mismatch"


def test_exception_outcomes_do_not_extract_numeric_promises(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("an exception expectation has no numeric leaf")

    monkeypatch.setattr(comparator, "statement_promises", fail)
    policy = {"statement": "Relative error below 2e-6.", "tolerances": [{"path": "/unused", "rtol": "0"}]}
    expected = {"kind": "exception", "message": "diagnostic only"}
    observed = {"kind": "exception", "message": "another diagnostic"}
    assert comparator.compare_request(request(observed, expected, policy=policy))["status"] == "equal"
    result = comparator.compare_request(request(value_outcome(1), expected, policy=policy))
    assert result["status"] == "mismatch"
    assert result["code"] == "did_not_raise"


@pytest.mark.parametrize(
    "failure, code",
    [
        (TimeoutError(), "comparator_timeout"),
        (MemoryError(), "comparator_memory"),
        (RuntimeError(), "comparator_failure"),
    ],
)
def test_statement_extraction_failure_remains_uncertainty(monkeypatch, failure, code):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(comparator, "statement_promises", fail)
    result = compare(1, 1, policy={"statement": "Relative error below 2e-6."})
    assert result["status"] == "uncertain"
    assert result["side"] == "comparator"
    assert result["equal"] is None
    assert result["code"] == code


def test_quantity_metadata_is_not_a_statement_name_or_positional_path_alias():
    policy = {
        "statement": "Report pulse with relative error below 2e-8. Other outputs permit relative error below 4e-4.",
        "tolerances": [{"path": "/0", "quantity": "pulse", "rtol": "0"}],
    }
    # The entry at "/0" is authored, so its exact rtol governs that leaf whatever its quantity label says.
    assert compare([Decimal("1.0002")], [Decimal("1")], policy=policy)["status"] == "mismatch"
    # "/pulse" addresses no leaf and "quantity" is not a path alias, so the entry is ignored and the
    # blanket 4e-4 promise still governs "/0".
    policy["tolerances"] = [{"path": "/pulse", "quantity": "pulse", "rtol": "0"}]
    result = compare([Decimal("1.0002")], [Decimal("1")], policy=policy)
    assert result["status"] == "equal"


@pytest.mark.parametrize("name", ["peak-width", "2nd", "cross section"])
def test_unsupported_quantity_name_shapes_keep_explicit_blanket_only_result(name):
    # A leaf key that is not a single identifier token never attributes, so the clause survives only as the
    # blanket. Single-character identifiers do attribute (test_single_character_leaf_names_attribute_as_whole_tokens).
    promises = comparator.statement_promises(f"Report {name} with relative error below 2e-8.", [name])
    assert promises["named"] == {}
    assert promises["blanket"] == (Fraction(2, 10**8), "rel")


def test_statement_tolerance_keeps_complex_values_atomic_and_component_scaled():
    expected = {"pair": ComplexValue(Decimal("2"), Decimal("3"))}
    policy = {"statement": "Report pair with relative error below 2e-6."}
    # Each component is judged against its own reference part: real tolerates 2e-6*|2|, imaginary 2e-6*|3|.
    observed = {"pair": ComplexValue(Decimal("2.000004"), Decimal("3.000006"))}
    assert compare(observed, expected, policy=policy)["status"] == "equal"
    # A real-part error above its own 4e-6 bound rejects, though it sits well inside the imaginary bound.
    observed["pair"] = ComplexValue(Decimal("2.0000041"), Decimal("3.000006"))
    result = compare(observed, expected, policy=policy)
    assert result["status"] == "mismatch"
    assert result["path"] == "/pair"
    # A complex leaf is atomic: "/pair/real" addresses no sub-leaf, so the entry is ignored.
    policy["tolerances"] = [{"path": "/pair/real", "rtol": "0"}]
    result = compare(observed, expected, policy=policy)
    assert result["status"] == "mismatch"
    assert result["path"] == "/pair"


def test_complex_component_scaling_holds_a_near_zero_part_strictly():
    expected = ComplexValue(Decimal("1"), Decimal("0"))
    # rtol only, atol 0. The imaginary part scales by its own reference |0| = 0, so its bound is 0.
    strict = {"tolerances": [{"path": "", "rtol": "2e-6", "atol": "0"}]}
    # Per-component scaling rejects this residual: the imaginary bound is |0| = 0, so 1e-9 > 0.
    residual = ComplexValue(Decimal("1"), Decimal("1e-9"))
    assert compare(residual, expected, policy=strict)["status"] == "mismatch"

    # Each component within its own bound passes: real 5e-7 <= 2e-6*|1|, imag 1e-6 <= 2e-6*|2|.
    both = ComplexValue(Decimal("1"), Decimal("2"))
    good = ComplexValue(Decimal("1.0000005"), Decimal("2.000001"))
    assert compare(good, both, policy=strict)["status"] == "equal"

    # A nonzero absolute tolerance admits the small component: its bound is atol, not zero.
    admit = {"tolerances": [{"path": "", "rtol": "0", "atol": "1e-3"}]}
    assert compare(residual, expected, policy=admit)["status"] == "equal"


def test_statement_does_not_turn_categories_or_booleans_into_numeric_leaves():
    policy = {"statement": "Relative error below 2e-6."}
    assert compare("1.000001", "1", policy=policy)["status"] == "mismatch"
    assert compare(1, True, policy=policy)["status"] == "mismatch"
    assert compare("1", "1", policy=policy)["status"] == "equal"


def test_statement_does_not_make_symbolic_equivalence_tolerant(symbolic_runtime):
    policy = {"statement": "Absolute error below 10%."}
    assert compare(SymbolicText("z+1"), SymbolicText("z+1"), policy=policy)["status"] == "equal"
    assert compare(SymbolicText("z+1.01"), SymbolicText("z+1"), policy=policy)["status"] == "mismatch"


def test_float_leaf_carries_its_exact_binary64_value_not_its_decimal_rendering():
    # A binary64 0.1 is not the decimal 0.1: at zero tolerance they must not match. A codec float goes
    # through numeric tolerance like any other real leaf.
    assert compare(0.1, Decimal("0.1"), policy={"rtol": "0", "atol": "0"})["status"] == "mismatch"
    assert compare(0.0999999, Decimal("0.1"), policy={"rtol": "1e-5", "atol": "0"})["status"] == "equal"


def test_spelled_zero_binds_a_mixed_clause_to_one_component_only():
    # The clause states both components, so 1e-3 binds to absolute alone, never also as rtol.
    policy = {"statement": "Absolute error at most 1e-3 and relative tolerance zero."}
    assert compare(Decimal("1.0005"), Decimal("1"), policy=policy)["status"] == "equal"
    assert compare(Decimal("1.0015"), Decimal("1"), policy=policy)["status"] == "mismatch"


@pytest.mark.parametrize(
    "statement",
    [
        "Relative error below 1e-4 and absolute error below 1e-6.",
        "Absolute error below 1e-6 and relative error below 1e-4.",
    ],
)
def test_two_component_clause_binds_each_bound_to_its_nearest_component(statement):
    # Each bound stays with its own component keyword regardless of order, so the loosest honoured clause
    # (relative 1e-4) is the blanket and the absolute 1e-6 never inflates the relative margin.
    policy = {"statement": statement}
    assert compare(Decimal("1.00005"), Decimal("1"), policy=policy)["status"] == "equal"
    assert compare(Decimal("1.0002"), Decimal("1"), policy=policy)["status"] == "mismatch"


def test_single_character_leaf_names_attribute_as_whole_tokens():
    # A one-character leaf key attributes its own clause, so a tight named promise is not lost to the
    # loosest blanket. The letter must appear as a whole token: prose that merely contains it inside a
    # word ("axis") attributes nothing.
    named = comparator.statement_promises(
        "Report x with relative error below 2e-8. Report y with relative error below 4e-4.", ["x", "y"]
    )["named"]
    assert named == {"x": (Fraction(2, 10**8), "rel"), "y": (Fraction(4, 10**4), "rel")}
    control = comparator.statement_promises("Report the axis value with relative error below 4e-4.", ["x"])
    assert control["named"] == {}
    assert control["blanket"] == (Fraction(4, 10**4), "rel")

    statement = "Report x with relative error below 2e-8. Report y with relative error below 4e-4."
    expected = {"x": Decimal("1"), "y": Decimal("1")}
    off_x = {"x": Decimal("1.0002"), "y": Decimal("1")}
    # x is held to its own 2e-8, not widened to y's 4e-4.
    assert compare(off_x, expected, policy={"statement": statement})["status"] == "mismatch"
    tight_x = {"statement": statement, "tolerances": [{"path": "/x", "rtol": "0", "atol": "0"}]}
    assert compare(off_x, expected, policy=tight_x)["status"] == "mismatch"
    # y still tolerates its own 4e-4 and rejects beyond it.
    assert (
        compare({"x": Decimal("1"), "y": Decimal("1.0003")}, expected, policy={"statement": statement})["status"]
        == "equal"
    )
    assert (
        compare({"x": Decimal("1"), "y": Decimal("1.0005")}, expected, policy={"statement": statement})["status"]
        == "mismatch"
    )


def test_named_statement_blanket_does_not_widen_an_authored_leaf():
    # When a statement names some leaves, its blanket is a catch-all for the rest. It widens a leaf that
    # has no author entry, but never a leaf the author gave its own tolerance.
    statement = "Report x with relative error below 2e-8. Other outputs permit relative error below 4e-4."
    expected = {"x": Decimal("1"), "z": Decimal("1")}
    off_z = {"x": Decimal("1"), "z": Decimal("1.0002")}
    # z has no author entry, so the catch-all blanket 4e-4 governs and accepts a 2e-4 deviation.
    assert compare(off_z, expected, policy={"statement": statement})["status"] == "equal"
    # An explicit tight author entry on z must not be widened by that catch-all blanket.
    tight_z = {"statement": statement, "tolerances": [{"path": "/z", "rtol": "0", "atol": "0"}]}
    assert compare(off_z, expected, policy=tight_z)["status"] == "mismatch"


# --- set-valued answers: unordered one-to-one matching ------------------------------------------ #


def test_a_set_matches_a_plain_list_expectation_unordered():
    # The candidate returns a set. The stored expectation is a plain list in a different order.
    result = compare({3, 1, 2}, [1, 2, 3])
    assert result["status"] == "equal"
    assert result["code"] == "unordered_comparison"


def test_a_set_matches_a_set_carrier_expectation():
    # Either side may be a set carrier.
    assert compare({1, 2, 3}, SetValue((3, 2, 1)))["status"] == "equal"
    assert compare(SetValue((1, 2)), {2, 1})["status"] == "equal"


def test_a_set_of_tuples_and_strings_matches_unordered():
    # Members may be tuples and categorical strings. String members keep the categorical rule.
    assert compare({(1, "x"), (2, "y")}, [(2, "y"), (1, "x")])["status"] == "equal"
    assert compare({"red", "green", "blue"}, ["blue", "red", "green"])["status"] == "equal"


def test_an_unparseable_cross_pair_does_not_defeat_a_valid_matching(symbolic_runtime):
    # "x+" is expression-shaped, so pairing it against the numeric 1 raises a parse error. That cross-pair
    # is not a match, but the valid one-to-one matching (1 to 1, "x+" to "x+") still passes.
    assert compare({1, "x+"}, [1, "x+"])["status"] == "equal"
    # A genuine member difference is still a mismatch, not an accidental pass.
    assert compare({2, "x+"}, [1, "x+"])["status"] == "mismatch"


def test_an_empty_set_matches_an_empty_list():
    assert compare(set(), [])["status"] == "equal"
    assert compare(SetValue(()), SetValue(()))["status"] == "equal"


def test_a_set_that_differs_in_a_member_is_a_mismatch():
    result = compare({1, 2, 3}, [1, 2, 4])
    assert result["status"] == "mismatch"
    assert result["code"] == "unordered_comparison"


def test_a_set_of_a_different_size_is_a_structure_mismatch():
    result = compare({1, 2}, [1, 2, 3])
    assert result["status"] == "mismatch"
    assert result["code"] == "structure_mismatch"


def test_a_set_against_a_scalar_or_dict_or_string_expectation_is_a_type_mismatch():
    assert compare({1, 2}, 5)["code"] == "type_mismatch"
    assert compare({1, 2}, {"a": 1})["code"] == "type_mismatch"
    # A set observed against a categorical string expectation is a mismatch, never an accidental equal.
    assert compare({1, 2}, "answer")["status"] == "mismatch"


def test_set_matching_is_one_to_one_under_tolerance_overlap():
    # With a loose blanket tolerance one observed member can match both expected slots, but the matching
    # stays one-to-one: 13 matches only 12, so 11 must take 10. The augmenting search must reassign.
    policy = {"default_rtol": "0.1", "default_atol": "0"}
    assert compare({11.0, 13.0}, [12.0, 10.0], policy=policy)["status"] == "equal"
    # 20.0 matches neither expected slot, so no perfect matching exists: a mismatch.
    assert compare({11.0, 20.0}, [12.0, 10.0], policy=policy)["status"] == "mismatch"


def test_a_per_member_tolerance_path_resolves_for_a_set_expectation():
    # A tolerance entry addressed at a member's positional path resolves, because a set expectation
    # enumerates its members positionally exactly as a list does.
    policy = {"tolerances": [{"path": "/0", "rtol": "0.2", "atol": "0"}]}
    # Slot /0 holds 10.0 and admits 11.5 (15% < 20%). Slot /1 holds 100.0 at the strict default.
    assert compare({11.5, 100.0}, SetValue((10.0, 100.0)), policy=policy)["status"] == "equal"


def test_an_exhausted_unordered_budget_is_uncertain_not_a_guess(monkeypatch):
    # When the step budget is spent the comparator reports uncertain, never a guessed match or mismatch.
    monkeypatch.setattr(comparator, "_UNORDERED_STEP_BUDGET", 1)
    result = compare({1, 2, 3}, [1, 2, 3])
    assert result["status"] == "uncertain"
    assert result["code"] == "unordered_budget"


def test_a_set_of_sympy_expressions_matches_a_list_of_symbolic_notation(symbolic_runtime):
    # A set of SymPy answers (as SymbolicText) matches the stored list of the same notation, unordered.
    observed = {SymbolicText("pi"), SymbolicText("sqrt(2)")}
    expected = [SymbolicText("sqrt(2)"), SymbolicText("pi")]
    assert compare(observed, expected)["status"] == "equal"


def test_unordered_matching_over_undecided_pairs_is_uncertain_not_a_guess(symbolic_runtime, monkeypatch):
    # Every member comparison is undecided, so a one-to-one matching exists over possible pairs but never over
    # certain pairs. The comparator reports the undecided verdict, never a guessed equal or mismatch.
    monkeypatch.setattr(symbolic_runtime, "equivalent", lambda left, right: None)
    result = compare({SymbolicText("a"), SymbolicText("b")}, [SymbolicText("c"), SymbolicText("d")])
    assert result["status"] == "uncertain"
    assert result["equal"] is None
    assert result["side"] == "comparator"
    assert result["code"] == "unordered_undecided"
    assert result["path"] == ""


# --- invalid_reference regressions: notations and tolerance shapes a reference value can carry -- #
#
# Each test feeds a synthetic expectation and an identical reference-shaped observation and asserts equal, so a
# notation or tolerance shape the comparator wrongly refused would surface as invalid_expected or
# invalid_reference. No reference or candidate source is executed. The behaviour each test pins is noted in its
# first comment.


def test_unused_tolerance_path_is_ignored_not_a_defect(symbolic_runtime):
    # A tolerance entry whose path addresses no numeric or complex leaf is ignored, never a
    # defect. A missing path, a symbolic leaf, and a set container all have no numeric margin to set.
    assert compare(1, 1, policy={"tolerances": [{"path": "/missing", "rtol": "0"}]})["status"] == "equal"
    symbolic = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    assert compare(SymbolicText("exp(I*pi/4)"), SymbolicText("exp(I*pi/4)"), policy=symbolic)["status"] == "equal"
    a_set = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    assert compare({1.0, 2.0}, [1.0, 2.0], policy=a_set)["status"] == "equal"


def test_consumed_tolerance_entry_still_governs_its_leaf():
    # The rule above must not throw the baby out: an entry whose path IS a numeric leaf still governs it.
    strict = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    assert compare(Decimal("1.0000000001"), Decimal("1"), policy=strict)["status"] == "equal"
    assert compare(Decimal("1.001"), Decimal("1"), policy=strict)["status"] == "mismatch"
    nested = {"tolerances": [{"path": "/1", "rtol": "0.2", "atol": "0"}]}
    # Slot /1 admits 15% error. Slot /0 keeps the strict default and rejects the same error.
    assert compare([Decimal("1"), Decimal("1.15")], [Decimal("1"), Decimal("1")], policy=nested)["status"] == "equal"
    off = compare([Decimal("1.15"), Decimal("1")], [Decimal("1"), Decimal("1")], policy=nested)
    assert off["status"] == "mismatch" and off["path"] == "/0"


def test_invalid_reference_regression_unused_tolerance_on_a_root_of_unity_leaf(symbolic_runtime):
    # A root tolerance entry on a symbolic root-of-unity answer is unused, not a defect.
    policy = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    value = SymbolicText("exp(2*I*pi/7)")
    assert compare(value, value, role="reference", policy=policy)["status"] == "equal"


def test_invalid_reference_regression_unused_tolerance_on_a_complex_valued_symbolic_leaf(symbolic_runtime):
    # A root tolerance entry on a complex-valued symbolic answer with decimal coefficients is unused, not a defect.
    policy = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    value = SymbolicText(
        "0.437290516873842*sqrt(3)*(0.0123456789012345 + 0.612*I)*(0.6 - 0.2*I)**2*exp((-0.6 - 0.2*I)*(0.6 - 0.2*I))"
    )
    assert compare(value, value, role="reference", policy=policy)["status"] == "equal"


def test_invalid_reference_regression_unused_tolerance_paths_on_symbolic_list_members(symbolic_runtime):
    # Per-member tolerance paths "/0/1" and "/1/1" name symbolic leaves. Both are ignored,
    # while the integer members at "/0/0" and "/1/0" still compare exactly.
    policy = {
        "tolerances": [
            {"path": "/0/1", "rtol": "1e-9", "atol": "0"},
            {"path": "/1/1", "rtol": "1e-9", "atol": "0"},
        ]
    }
    value = [[7, SymbolicText("-0.0612345678901234*pi")], [3, SymbolicText("0.0456789012345678*pi")]]
    assert compare(value, value, role="reference", policy=policy)["status"] == "equal"


def test_invalid_reference_regression_piecewise_and_undefined_function_notation(symbolic_runtime):
    # Piecewise((value, condition), ...), Eq(a, b) and the boolean catch-all condition parse
    # as undefined-function and boolean notation instead of raising symbolic_function.
    value = [
        SymbolicText(
            "Piecewise((0, Eq(w, 0)), (sin(w)*atan(w/Abs(w - 1))/sqrt(1 + w**2), True))"
            " + (w**3 - 2)*log(1 + w)/(1 + w)"
        ),
        SymbolicText(
            "-w + Piecewise((0, Eq(w, 0)), (sin(w)*atan(w/Abs(w - 1))/sqrt(1 + w**2), True)) + 5"
            " + (w**3 - 2)*(log(3*k_z**2*w*(1 + w)/m_0**2) - 1/eps_UV)/(1 + w)"
        ),
    ]
    assert compare(value, value, role="reference")["status"] == "equal"


def test_invalid_reference_regression_undefined_function_application(symbolic_runtime):
    # 'moment(rho)' is an undefined function application, built as sympy.Function('moment').
    value = [SymbolicText("moment(rho)"), SymbolicText("moment(rho**3)")]
    assert compare(value, value, role="reference")["status"] == "equal"


def test_invalid_reference_regression_interval_constructor_notation(symbolic_runtime):
    # The member-access operator "." parses for the Interval.open(...) set constructor. The
    # numeric siblings resolve under the ignored root tolerance entry.
    policy = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    case0 = [0.317254961830921, SymbolicText("Interval.open(0, pi/3)"), 1.30571382164927]
    assert compare(case0, case0, role="reference", policy=policy)["status"] == "equal"
    case1 = [0.682041925301384, SymbolicText("Interval.open(0, 0.785398163397448)"), 0.591738062153084]
    assert compare(case1, case1, role="reference", policy=policy)["status"] == "equal"


def test_invalid_reference_regression_float_bearing_expression_past_growth_bound(symbolic_runtime, monkeypatch):
    # A float-bearing constant expression whose simplify explodes past the growth bound is kept comparable (parse
    # falls back to the bounded raw expression) instead of raising symbolic_growth_limit. The unused root tolerance
    # entry is ignored. Simplify is forced past the node bound here, so the fallback runs whatever SymPy would do.
    exploded = symbolic_runtime.sp.Add(*symbolic_runtime.sp.symbols(f"s0:{symbolic_runtime.MAX_RESULT_NODES + 1}"))
    monkeypatch.setattr(symbolic_runtime.sp, "simplify", lambda expression, *args, **kwargs: exploded)
    policy = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    value = SymbolicText(
        "-1800*log(81)/log(3) + (2 + log(81)/log(3))*log(exp(3)/(2*pi))/(6*log(3)) + 1500*log(E*pi)/log(3) + 1250000.0"
    )
    assert compare(value, value, role="reference", policy=policy)["status"] == "equal"


# --- regressions: authored tolerances, prose claims, legacy text ---------------------------------- #


def test_incidental_percentage_does_not_widen_an_authored_tolerance():
    statement = (
        "The trap frequency is stable to within 2% during the measurement. "
        "Report the ground-state energy with relative error below 1e-6."
    )
    policy = {"rtol": "1e-9", "atol": "0", "statement": statement}
    assert compare(Decimal("1.015"), Decimal("1"), policy=policy)["status"] == "mismatch"
    promises = statement_promises(statement)
    by_text = {clause["text"]: clause for clause in promises["clauses"]}
    assert by_text["2%"]["honoured"] is False and by_text["2%"]["reason"] == "no_accuracy_keyword"
    assert promises["blanket"] == (Fraction(1, 10**6), "rel")


@pytest.mark.parametrize("per_leaf", [False, True])
def test_authored_tolerances_beat_statement_promises(per_leaf):
    def policy_for(author_rtol):
        authored = {"rtol": author_rtol, "atol": "0"}
        policy = {"tolerances": [{"path": "", **authored}]} if per_leaf else dict(authored)
        policy["statement"] = "Return output with relative error below 2e-6."
        return policy

    assert (
        compare(Decimal("100.00015"), Decimal("100"), policy=policy_for("1e-9"))["status"] == "mismatch"
    )  # author tighter
    assert compare(Decimal("100.05"), Decimal("100"), policy=policy_for("1e-3"))["status"] == "equal"  # author looser


def test_accuracy_clauses_still_honoured_after_narrowing():
    assert (
        compare(
            Decimal("1.00049"), Decimal("1"), policy={"statement": "Values are compared to three decimal places."}
        )["status"]
        == "equal"
    )
    rel = {"statement": "Report the result with relative error below 2e-6."}
    assert compare(Decimal("1.0000015"), Decimal("1"), policy=rel)["status"] == "equal"
    assert compare(Decimal("1.000003"), Decimal("1"), policy=rel)["status"] == "mismatch"
    assert (
        compare(Decimal("1.004"), Decimal("1"), policy={"statement": "Answers are correct to 3 significant figures."})[
            "status"
        ]
        == "equal"
    )
    bare = statement_promises("The result must have an error below 1e-6.")
    assert bare["blanket"] is None and [clause["reason"] for clause in bare["clauses"]] == ["no_kind"]
    within = statement_promises("Agree to within 1e-3 of the reference.")
    assert within["blanket"] is None and within["clauses"][0]["reason"] == "no_accuracy_keyword"


def test_typed_symbolic_observation_against_legacy_text_expectation(symbolic_runtime):
    assert compare(SymbolicText("-x"), LegacyText("-x"), role="reference")["status"] == "equal"
    assert compare(SymbolicText("x - 1"), LegacyText("x-1"))["status"] == "equal"
    assert compare(SymbolicText("y"), LegacyText("x"))["status"] == "mismatch"
    assert compare("x-1", LegacyText("x-1"))["status"] == "equal"  # text against text stays categorical
    assert compare("upper-band+0", LegacyText("upper-band"))["status"] == "mismatch"


@pytest.mark.parametrize(
    "statement, value, kind",
    [
        # "Accurate to within 0.02%." is covered in test_statement_extraction_uses_exact_bounded_patterns.
        ("Relative error within 1e-6.", Fraction(1, 10**6), "rel"),
        ("Accuracy within 0.5%.", Fraction(5, 10**3), "rel"),
        ("Report the energy to within 1e-6 relative error.", Fraction(1, 10**6), "rel"),
        ("Absolute error within 2e-6.", Fraction(2, 10**6), "abs"),
    ],
)
def test_within_borrows_the_claim_and_kind_of_an_accuracy_keyword_in_its_sentence(statement, value, kind):
    promises = statement_promises(statement)
    assert [(clause["value"], clause["kind"], clause["reason"]) for clause in promises["clauses"]] == [
        (value, kind, "")
    ]
    assert promises["blanket"] == (value, kind)


def test_within_takes_the_kind_of_its_nearest_lender_when_a_sentence_carries_both_kinds():
    promises = statement_promises("Relative error within 1e-6 and absolute error below 1e-9.")
    assert [(clause["value"], clause["kind"], clause["honoured"]) for clause in promises["clauses"]] == [
        (Fraction(1, 10**6), "rel", True),
        (Fraction(1, 10**9), "abs", True),
    ]


@pytest.mark.parametrize(
    "statement", ["The trap frequency is stable to within 1e-6 over the run.", "Give the answer to within 1%."]
)
def test_bare_within_without_an_accuracy_keyword_in_its_sentence_stays_refused(statement):
    promises = statement_promises(statement)
    assert promises["blanket"] is None
    assert [clause["reason"] for clause in promises["clauses"]] == ["no_accuracy_keyword"]


@pytest.mark.parametrize(
    "statement",
    [
        # The only accuracy keyword already claims its own number, so it lends nothing to "within".
        "The trap is stable to within 2% and the energy has relative error below 1e-6.",
        # A keyword's window stops at its sentence break, so the 2% in the next sentence is never pulled in.
        "Report the energy with relative error below 1e-6. The trap frequency drifts by 2% during the measurement.",
    ],
)
def test_an_incidental_percentage_beside_a_claimed_accuracy_keyword_is_not_a_promise(statement):
    promises = statement_promises(statement)
    assert promises["blanket"] == (Fraction(1, 10**6), "rel")
    assert not any(clause["honoured"] and clause["value"] == Fraction(1, 50) for clause in promises["clauses"])


def test_an_error_or_failure_rate_is_a_measured_quantity_not_a_tolerance():
    for statement in (
        "Consider a surface code with physical error rate 0.1%. Compute the logical error rate at distance 5.",
        "Each gate fails with error probability 2e-3 per cycle.",
        "The detector failure rate is 0.5% per hour.",
    ):
        promises = statement_promises(statement)
        assert promises["blanket"] is None, statement
        assert [clause["reason"] for clause in promises["clauses"]] == ["no_accuracy_keyword"], statement
    # An "error" that states a tolerance keeps its meaning.
    assert statement_promises("Report the energy with relative error 1e-6.")["blanket"] == (Fraction(1, 10**6), "rel")
    bare = statement_promises("The result must have an error below 1e-6.")
    assert [clause["reason"] for clause in bare["clauses"]] == ["no_kind"]


@pytest.mark.parametrize(
    "statement, value",
    [("Tolerance: within 1%.", Fraction(1, 100)), ("Relative error: within 1e-6.", Fraction(1, 10**6))],
)
def test_a_colon_right_after_an_accuracy_keyword_does_not_block_borrowing(statement, value):
    promises = statement_promises(statement)
    assert [(clause["value"], clause["kind"], clause["reason"]) for clause in promises["clauses"]] == [
        (value, "rel", "")
    ]


def test_a_sentence_break_still_blocks_borrowing():
    promises = statement_promises("Relative error below 1e-6. Within 2%.")
    assert promises["blanket"] == (Fraction(1, 10**6), "rel")
    assert [(clause["value"], clause["reason"]) for clause in promises["clauses"]] == [
        (Fraction(1, 10**6), ""),
        (Fraction(1, 50), "no_accuracy_keyword"),
    ]


def test_a_borrowed_kind_is_a_whole_word_inside_the_lenders_window():
    # "relativistic" is not the word "relative", so the lender states no kind and the bare number is refused.
    relativistic = "Use the relativistic dispersion relation and give energies accurate to within 1e-6."
    assert [clause["reason"] for clause in statement_promises(relativistic)["clauses"]] == ["no_kind"]
    # A "relative" far before the lender, outside the lender's own window, states nothing about it either.
    far = (
        "Relative motion of the probe frame is included, "
        + "and the setup continues at length, " * 4
        + "giving energies accurate to within 1e-6."
    )
    assert [clause["reason"] for clause in statement_promises(far)["clauses"]] == ["no_kind"]


# Statements are wrapped Markdown, so a single newline inside a sentence is a soft line wrap, never a sentence break:
# wrapping a statement must not change what it promises. Each test sets a wrapped statement beside the same words on
# one line, and a line break that is not a soft wrap beside the words joined, to show the break still separates them.
def test_a_soft_line_wrap_between_a_count_and_its_keyword_does_not_change_the_promise():
    wrapped = statement_promises("Return the ratio to 6\nsignificant figures.")
    unwrapped = statement_promises("Return the ratio to 6 significant figures.")
    assert clause_shapes(wrapped) == [(Fraction(5, 10**6), "rel", True)]
    assert clause_shapes(wrapped) == clause_shapes(unwrapped)
    assert wrapped["blanket"] == unwrapped["blanket"] == (Fraction(5, 10**6), "rel")
    assert wrapped["named"] == unwrapped["named"] == {}


@pytest.mark.parametrize(
    "statement",
    [
        "Results are graded to a relative tolerance of\n1e-8.",
        "Results are graded to a relative tolerance of\n$1e-8$.",
        "Results are graded to a relative tolerance of\n\\(1e-8\\).",
        "Results are graded to a relative tolerance of\n(1e-8).",
        # The wrap may also fall between the kind word and the keyword.
        "Results are graded to a relative\ntolerance of 1e-8.",
    ],
)
def test_a_soft_line_wrap_inside_a_scaled_promise_keeps_its_value_and_kind(statement):
    promises = statement_promises(statement)
    assert clause_shapes(promises) == [(Fraction(1, 10**8), "rel", True)]
    assert promises["blanket"] == (Fraction(1, 10**8), "rel")


@pytest.mark.parametrize(
    "statement",
    [
        "Return the ratio to 6  \nsignificant figures.",
        "Return the ratio to 6\t\nsignificant figures.",
        "Return the ratio to 6\n    significant figures.",
        "Return the ratio to 6\n\tsignificant figures.",
        "Return the ratio to 6 \t \n \t significant figures.",
        "Return the ratio to 6\xa0\nsignificant figures.",
        "Return the ratio to 6\n\xa0significant figures.",
    ],
)
def test_blanks_around_a_soft_line_wrap_do_not_make_it_a_break(statement):
    promises = statement_promises(statement)
    assert clause_shapes(promises) == [(Fraction(5, 10**6), "rel", True)]
    assert promises["blanket"] == (Fraction(5, 10**6), "rel")


@pytest.mark.parametrize(
    "statement, value",
    [
        ("relative tolerance of\n1.5e-6.", Fraction(15, 10**7)),
        ("Results are graded to a relative tolerance of\n2.5 x 10^-6.", Fraction(25, 10**7)),
        ("Results are graded to a relative tolerance of\n0.5%.", Fraction(5, 10**3)),
        ("Report the energy with relative error within\n0.5%.", Fraction(5, 10**3)),
        ("Report the energy to within\n0.5% relative error.", Fraction(5, 10**3)),
    ],
)
def test_a_wrapped_line_that_starts_with_a_decimal_number_is_a_continuation_not_a_list_item(statement, value):
    # A list marker is digits, then "." or ")", then a blank: "0.5%" and "1.5e-6" are numbers, so the wrap holds.
    wrapped = statement_promises(statement)
    unwrapped = statement_promises(statement.replace("\n", " "))
    assert clause_shapes(wrapped) == [(value, "rel", True)]
    assert clause_shapes(wrapped) == clause_shapes(unwrapped)
    assert wrapped["blanket"] == unwrapped["blanket"] == (value, "rel")


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("Return the ratio to 6\nsignificant figures.", [(Fraction(5, 10**6), "rel", True)]),
        ("Return the ratio to 6  \nsignificant figures.", [(Fraction(5, 10**6), "rel", True)]),
        ("Results are graded to a relative tolerance of\n1e-8.", [(Fraction(1, 10**8), "rel", True)]),
        ("Results are graded to a relative tolerance of\n0.5%.", [(Fraction(5, 10**3), "rel", True)]),
        # "error" and "rate" meet across the wrap: a measured rate, so refused and never a promise.
        ("The physical error\nrate is 0.1%.", [(Fraction(1, 10**3), "rel", False)]),
        # "correct to" meets across the wrap and borrows the claim of the "error" that owns no number.
        ("Give the energy with relative error, correct\nto 0.5%.", [(Fraction(5, 10**3), "rel", True)]),
    ],
)
def test_a_crlf_wrap_reads_like_its_lf_twin(statement, expected):
    assert clause_shapes(statement_promises(statement)) == expected
    assert clause_shapes(statement_promises(statement.replace("\n", "\r\n"))) == expected


@pytest.mark.parametrize("lead", ["# ", "## ", "  ### ", "| "])
def test_a_heading_or_a_table_row_never_joins_the_next_line(lead):
    statement = f"{lead}Part 6\nsignificant figures are required."
    # The 6 belongs to the heading or the row, so it is no count of the keyword on the next line ...
    assert statement_promises(statement) == NO_PROMISE
    # ... where the same two lines without the mark are one sentence.
    joined = statement_promises(statement.removeprefix(lead))
    assert clause_shapes(joined) == [(Fraction(5, 10**6), "rel", True)]


def test_a_part_label_or_a_closing_question_stops_a_kind_word_reaching_the_next_part():
    # Part (a) ends in a question mark, so the "absolute" of part (b) is another sentence and part (a) stays relative.
    question = statement_promises(
        "(a) State the energy to a relative accuracy of 1e-4?\n(b) State the length in absolute units, scaled by 1e-6."
    )
    assert [(clause["value"], clause["kind"], clause["reason"]) for clause in question["clauses"]] == [
        (Fraction(1, 10**4), "rel", "")
    ]
    # Part (2) opens with a list marker and states no kind of its own: its number is refused, not given the "absolute"
    # of part (1).
    ratio = statement_promises(
        "(1) Give the offset with absolute error below 1e-6\n(2) Give the ratio with error below 1e-3"
    )
    assert [(clause["value"], clause["kind"], clause["reason"]) for clause in ratio["clauses"]] == [
        (Fraction(1, 10**6), "abs", ""),
        (Fraction(1, 10**3), "unknown", "no_kind"),
    ]


@pytest.mark.parametrize(
    "statement, expected",
    [
        # An indent from a list item's continuation line, a trailing blank, and a double space on one line.
        ("- Return the ratio to 6 significant\n  figures.", [(Fraction(5, 10**6), "rel", True)]),
        ("Report the answer to 3 decimal \nplaces.", [(Fraction(5, 10**4), "abs", True)]),
        ("Report the answer to 3 decimal  places.", [(Fraction(5, 10**4), "abs", True)]),
        # "agree to" borrows the claim of the accuracy keyword that owns no number, however it is split.
        ("Give the energy with relative error, agreeing\n  to 1e-6.", [(Fraction(1, 10**6), "rel", True)]),
    ],
)
def test_the_words_of_a_keyword_may_be_separated_by_any_blanks(statement, expected):
    assert clause_shapes(statement_promises(statement)) == expected


@pytest.mark.parametrize("wrap", ["\n", "\r\n", "\n  ", " \n"])
def test_a_borrower_split_by_a_wrap_is_refused_like_its_one_line_twin(wrap):
    # "error" owns its own number, so "correct to" has no accuracy keyword to borrow from and its 0.5% stays refused.
    promises = statement_promises(f"The result must have a relative error below 1e-6 and be correct{wrap}to 0.5%.")
    assert [(clause["value"], clause["kind"], clause["reason"]) for clause in promises["clauses"]] == [
        (Fraction(1, 10**6), "rel", ""),
        (Fraction(5, 10**3), "rel", "no_accuracy_keyword"),
    ]
    assert promises["blanket"] == (Fraction(1, 10**6), "rel")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_a_bare_part_label_ends_a_wrapped_line_whatever_the_line_endings(newline):
    statement = newline.join(
        [
            "Answer the following",
            "(a)",
            "the energy, with relative error below 1e-4",
            "(b)",
            "the length, with absolute error below 1e-6",
        ]
    )
    # A label alone on its line is a list marker, so it starts its part; the "absolute" of (b) stays out of part (a).
    promises = statement_promises(statement)
    assert [(clause["value"], clause["kind"], clause["reason"]) for clause in promises["clauses"]] == [
        (Fraction(1, 10**4), "rel", ""),
        (Fraction(1, 10**6), "abs", ""),
    ]


@pytest.mark.parametrize("blank", [" ", "\t", "\xa0", "\N{EM SPACE}"])
def test_a_colon_ending_a_line_is_a_break_whatever_whitespace_follows_it(blank):
    # The colon ends the line however it is padded, so the 1% is not read as the value the colon introduces.
    assert statement_promises(f"Tolerance:{blank}\n1%") == NO_PROMISE


def test_a_blank_line_stays_a_sentence_break():
    statement = "The value is 6\n\nsignificant deviations are rejected with 1% error."
    promises = statement_promises(statement)
    # The second paragraph keeps its own percentage, and the 6 before the blank line is no count.
    assert [(clause["text"], clause["at"]) for clause in promises["clauses"]] == [("1%", statement.index("1%"))]
    assert clause_shapes(promises) == [(Fraction(1, 100), "rel", True)]
    # Only a soft wrap joins two lines. Split by a blank line, with blanks on it or with CRLF endings, a count and its
    # keyword never meet, while on one line they are a promise.
    assert statement_promises("Return the ratio to 6\n\nsignificant figures.") == NO_PROMISE
    assert statement_promises("Return the ratio to 6\n \nsignificant figures.") == NO_PROMISE
    assert statement_promises("Return the ratio to 6\r\n\r\nsignificant figures.") == NO_PROMISE
    one_line = statement_promises("Return the ratio to 6 significant figures.")
    assert clause_shapes(one_line) == [(Fraction(5, 10**6), "rel", True)]


def test_a_markdown_item_after_a_line_break_is_read_as_after_a_blank_line():
    # The count and its keyword both sit inside the item, so it reads the same however the break before it is made.
    item = statement_promises("compute to\n- 6 significant figures")
    after_blank = statement_promises("compute to\n\n- 6 significant figures")
    assert clause_shapes(item) == clause_shapes(after_blank) == [(Fraction(5, 10**6), "rel", True)]
    assert item["blanket"] == after_blank["blanket"] == (Fraction(5, 10**6), "rel")
    # A numbered item starts its own sentence: no clause comes from it.
    assert statement_promises("accuracy of\n1. one part") == NO_PROMISE


@pytest.mark.parametrize(
    "item",
    ["# 1e-6", "- 1e-6", "* 1e-6", "+ 1e-6", "| 1e-6 |", "> 1e-6", "`1e-6`", "$$1e-6$$", "1) 1e-6", "12) 1e-6"],
)
def test_a_line_that_starts_a_markdown_block_stays_a_sentence_break(item):
    statement = f"Give the relative error\n{item}"
    # The keyword's window ends at the break, so the number on the next line is nobody's promise ...
    assert statement_promises(statement) == NO_PROMISE
    # ... where the same two lines joined into one sentence would promise it.
    joined = statement_promises(statement.replace("\n", " "))
    assert clause_shapes(joined) == [(Fraction(1, 10**6), "rel", True)]


def test_a_line_ending_in_a_colon_stays_a_break_and_colon_lending_is_unchanged():
    # On one line, a colon right after an accuracy keyword introduces its value.
    assert clause_shapes(statement_promises("Tolerance: 1%")) == [(Fraction(1, 100), "rel", True)]
    # Ending the line, it does not: the next line starts a new sentence, as it did before soft wraps were read.
    assert statement_promises("Tolerance:\n1%") == NO_PROMISE
    # A blank after the colon changes nothing, and a line ending in a period or semicolon is a break as ever.
    for ending in (": ", ".", ";"):
        assert statement_promises(f"Tolerance{ending}\n1%") == NO_PROMISE


@pytest.mark.parametrize(
    "statement, clauses",
    [
        ("Return the ratio to 6\nsignificant figures.", 1),
        ("Results are graded to a relative tolerance of\n1e-8.", 1),
        ("Return the ratio to 6  \nsignificant figures.", 1),
        ("Relative error below\n2e−6.", 1),
        ("Results are graded to a relative tolerance of\n0.5%.", 1),
        ("Return the ratio to 6\r\nsignificant figures.", 1),
        ("Return the ratio to 6\r\n\r\nsignificant figures.", 0),
        ("The value is 6\n\nsignificant deviations are rejected with 1% error.", 1),
        ("compute to\n- 6 significant figures", 1),
        ("compute to\n\n- 6 significant figures", 1),
        ("accuracy of\n1. one part", 0),
        ("Tolerance:\n1%", 0),
        ("The grid holds 6\nSignificant figures matter.", 0),
    ],
)
def test_reading_soft_wraps_keeps_the_text_length_and_every_clause_offset(statement, clauses):
    assert len(comparator._normalized(statement)) == len(statement)
    found = statement_promises(statement)["clauses"]
    assert len(found) == clauses
    for clause in found:
        at, text = clause["at"], clause["text"]
        # `at` is an offset into the statement as written, at the clause's first character.
        assert statement[at] == text[0]
        assert statement[at : at + len(text)].replace("\n", " ").replace("−", "-") == text


def test_a_capitalised_continuation_stays_a_sentence_break():
    # A line that starts with a capital begins a sentence, so the 6 before it is no count of the keyword after it ...
    assert statement_promises("The grid holds 6\nSignificant figures matter.") == NO_PROMISE
    # ... where the same words on one line are a promise.
    one_line = statement_promises("The grid holds 6 Significant figures matter.")
    assert clause_shapes(one_line) == [(Fraction(5, 10**6), "rel", True)]


@pytest.mark.parametrize(
    "wrapped, within, beyond",
    [
        ("Return the ratio to 6\nsignificant figures.", "1.000004", "1.000009"),
        ("Results are graded to a relative tolerance of\n1e-8.", "1.000000005", "1.00000002"),
        ("Results are graded to a relative tolerance of\n0.5%.", "1.004", "1.006"),
        ("Return the ratio to 6\r\nsignificant figures.", "1.000004", "1.000009"),
    ],
)
def test_a_soft_wrapped_statement_grades_like_the_same_statement_on_one_line(wrapped, within, beyond):
    # Each pair is chosen so that the silent default (rtol 1e-5) would get one of the values wrong: only the
    # statement's own promise answers equal for `within` and mismatch for `beyond`.
    for statement in (wrapped, wrapped.replace("\n", " ")):
        policy = {"statement": statement}
        assert compare(Decimal(within), Decimal("1"), policy=policy)["status"] == "equal"
        assert compare(Decimal(beyond), Decimal("1"), policy=policy)["status"] == "mismatch"


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("to 6\nsignificant", "to 6 significant"),
        ("to 6  \nsignificant", "to 6   significant"),
        ("to 6\t\n\tsignificant", "to 6\t \tsignificant"),
        ("to\n$x$", "to $x$"),
        ("to\n\\(x\\)", "to \\(x\\)"),
        ("to\n(x)", "to (x)"),
        ("to\n6 places", "to 6 places"),
        ("to\n15% more", "to 15% more"),
        ("to\n1e-8", "to 1e-8"),
        ("a\nb\nc", "a b c"),
        ("to\n0.5% more", "to 0.5% more"),
        ("to\n1.5e-6", "to 1.5e-6"),
        ("to\n2.5 x 10^-6", "to 2.5 x 10^-6"),
        ("to\n10.5%", "to 10.5%"),
        ("to 6\r\nsignificant", "to 6\r significant"),
        ("to 6 \r\nsignificant", "to 6 \r significant"),
        ("to\n(1e-8)", "to (1e-8)"),
        ("to\n(abcde) items", "to (abcde) items"),
        ("to\ni.e. items", "to i.e. items"),
        ("to\ne.g. items", "to e.g. items"),
        (
            "to 6\nsignificant figures.\nAnother sentence\nof words\n\nnew paragraph\n- item\ncontinued",
            "to 6 significant figures.\nAnother sentence of words\n\nnew paragraph\n- item continued",
        ),
        ("to\n(a)\nitems", "to\n(a) items"),
        ("to\niv)\nitems", "to\niv) items"),
        ("to\n12)\nitems", "to\n12) items"),
        ("to\r\n(a)\r\nitems", "to\r\n(a)\r items"),
        ("to 6\xa0\nsignificant", "to 6\xa0 significant"),
        ("to 6\n\xa0significant", "to 6 \xa0significant"),
    ],
)
def test_normalization_turns_each_soft_line_wrap_into_a_space(statement, expected):
    assert comparator._normalized(statement) == expected
    assert len(comparator._normalized(statement)) == len(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "to 6\n\nsignificant",
        "to 6\n \nsignificant",
        "to 6.\nsignificant",
        "to 6;\nsignificant",
        "to 6:\nsignificant",
        "to 6: \nsignificant",
        "to 6\nSignificant",
        'to 6\n"significant"',
        "to 6\n1. significant",
        "to 6\n1) significant",
        "to 6\n12. significant",
        "to 6\n12) significant",
        "\nsignificant",
        "to 6\n",
        "to\n12) items",
        "accuracy of\n1. one part",
        "to\n6.",
        "to\n6)",
        "to 6\nécart",
        "to 6\r\n\r\nsignificant",
        "to 6.\r\nsignificant",
        "to 6:\r\nsignificant",
        "to 6?\nsignificant",
        "to 6!\nsignificant",
        "to 6? \nsignificant",
        "to 6!\r\nsignificant",
        "to\n(a) items",
        "to\n(iv) items",
        "to\n(12) items",
        "to\n(abcd) items",
        "to\n(x) items",
        "to\na) items",
        "to\nb. items",
        "to\niv) items",
        "to\nii. items",
        "to\nx) items",
        "to\nv. items",
        "to\nviii) items",
        "## Part 6\nsignificant",
        "# Part 6\nsignificant",
        "  ### Part 6\nsignificant",
        "| quantity | 6 |\nsignificant",
        "~~~ 6\nsignificant",
        "to\nb.\nitems",
        "to\n1.\nitems",
        "to\n(a)\n\nitems",
        "to\r\n(a)\r\n\r\nitems",
        "to 6:\xa0\nsignificant",
        "to 6\xa0\n\xa0\nsignificant",
    ],
)
def test_normalization_keeps_every_other_line_break(statement):
    assert comparator._normalized(statement) == statement


def test_a_fenced_block_is_blanked_before_wraps_are_read():
    # The blanked fence leaves a blank line between the two lines, so neither newline is a soft wrap.
    statement = "to 6\n```\nx\n```\nsignificant"
    assert comparator._normalized(statement) == "to 6\n" + " " * 9 + "\nsignificant"


def test_typed_string_expectation_is_never_compared_symbolically(symbolic_runtime):
    # Authored typing wins: a typed "str" expectation is a string, so a symbolic observation of the same text is a
    # categorical mismatch, while the same text stored as legacy text is re-read in the grammar.
    typed = compare(SymbolicText("x - 1"), "x-1")
    assert (typed["status"], typed["code"]) == ("mismatch", "categorical_comparison")
    assert compare(SymbolicText("x - 1"), LegacyText("x-1"))["status"] == "equal"
