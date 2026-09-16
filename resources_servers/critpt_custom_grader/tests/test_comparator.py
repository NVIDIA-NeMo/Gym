# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-safe comparator regressions on synthetic values and inert hostile text.

The comparator is a pure function with no IO or execution. Tests using the symbolic_runtime fixture skip
without SymPy. None needs Daytona.
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
    # symbolic-carrier expectation of it is unscorable (README: a nonfinite expectation is unscorable).
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


@pytest.mark.parametrize("text", ["tr(psi)", "unapproved(x)", "f(x, y)"])
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
    # grades definitively against a float expectation. The code is symbolic_comparison, so zeta resolved.
    decided = compare(SymbolicText("-3/2 * zeta(3)/zeta(5)"), -1.738872689824075)
    assert decided["status"] == "mismatch"
    assert decided["code"] == "symbolic_comparison"
    # A candidate that cannot reduce to a number is undecided by algebra. Against a concrete numeric
    # expectation it collapses to mismatch. Two SymbolicText sides stay uncertain
    # (test_undecided_algebra_remains_uncertain).
    collapsed = compare(SymbolicText("tr(x)"), 5)
    assert collapsed["status"] == "mismatch"
    assert collapsed["code"] == "symbolic_number_mismatch"


@pytest.mark.parametrize(
    "failure, code", [(TimeoutError(), "comparator_timeout"), (MemoryError(), "comparator_memory")]
)
def test_candidate_side_exhaustion_after_tractable_expected_is_scored_wrong(monkeypatch, failure, code):
    def exhaust(*args, **kwargs):
        raise failure

    # The expectation resolves within the caps, so the exhaustion is in the candidate value and scored wrong.
    monkeypatch.setattr(comparator, "_compare", exhaust)
    candidate = compare(1, 1, role="candidate")
    assert candidate["status"] == "invalid_candidate"
    assert candidate["side"] == "candidate"
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
        ("Accurate to within 2e-6.", Fraction(2, 10**6), "unknown"),
        # "absolute" and "relative" both fall in the window. "absolut" is tested first, so the clause is abs.
        ("Absolute or relative error below 2e-6.", Fraction(2, 10**6), "abs"),
        # A keyword in one sentence pulls a bare number out of the next through the fixed 60/160-char window.
        ("Accurate values are required. The fixture constant is 2e-6.", Fraction(2, 10**6), "unknown"),
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
        "```text\nRelative error below 2e-6.\n```",
        "```text\nRelative error below 2e-6.",
    ],
)
def test_unrecognized_prose_and_fenced_code_do_not_create_promises(statement):
    assert comparator.statement_promises(statement) == {"clauses": [], "blanket": None, "named": {}}
    policy = {"statement": statement}
    assert compare(Decimal("1.000009"), Decimal("1"), policy=policy)["status"] == "equal"
    assert compare(Decimal("1.00002"), Decimal("1"), policy=policy)["status"] == "mismatch"


@pytest.mark.parametrize("separator", [". ", "; ", "\n", "! ", "? "])
def test_error_kinds_and_accuracy_windows_do_not_cross_sentences(separator):
    statement = (
        "Report lift with absolute error below 2e-6" + separator + "Report drift with relative error below 4e-5."
    )
    promises = comparator.statement_promises(statement, ["lift", "drift"])
    # Kind is now sentence-bounded, the same bounds quantity attribution uses. Across a real sentence break
    # (. ; \n) drift keeps its own "relative" kind. "!" and "?" do not end a sentence here, so the one
    # sentence carries "absolute" first and both leaves read abs.
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
    # Only [.;:] + whitespace and newline end a sentence. After those the two sentences split, so the
    # earlier 2e-9 clause cannot reach residue. After "!" or "?" the text is one sentence, so that clause
    # claims residue at the tighter 2e-9 and rejects a 3e-5 miss.
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
def test_statement_governs_whether_tighter_or_looser_than_author(author_rtol, per_leaf):
    authored = {"rtol": author_rtol, "atol": "0"}
    policy = {"tolerances": [{"path": "root", **authored}]} if per_leaf else authored
    policy["statement"] = "Return output with relative error below 2e-6."
    assert compare(Decimal("100.0002"), Decimal("100"), policy=policy)["status"] == "equal"
    assert compare(Decimal("100.0002000000000000001"), Decimal("100"), policy=policy)["status"] == "mismatch"


def test_named_promises_resolve_nested_leaves_without_a_task_wide_maximum():
    expected = {"band": {"width": Decimal("2"), "depth": Decimal("4")}, "other": Decimal("1")}
    policy = {
        "statement": "Grade width with relative error below 2e-8. Grade depth with relative error below 3e-4.",
        "tolerances": [{"path": "/band/width", "rtol": "1e-3"}, {"path": "/band/depth", "rtol": "1e-10"}],
    }
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
    expected = {"coarse": Decimal("2"), "tiny": tiny}
    policy = {
        "statement": "Report values to three decimal places.",
        "tolerances": [
            {"path": "/coarse", "rtol": "0", "atol": "1e-20"},
            {"path": "/tiny", "rtol": "0", "atol": "1e-44"},
        ],
    }
    passing = {"coarse": Decimal("2.0005"), "tiny": tiny + margin}
    assert compare(passing, expected, policy=policy)["status"] == "equal"
    for wrong in (0, tiny + 2 * margin):
        result = compare({**passing, "tiny": wrong}, expected, policy=policy)
        assert result["status"] == "mismatch"
        assert result["path"] == "/tiny"
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
    policy = {
        "statement": "Report speck with absolute error below 2e-6. Other outputs permit relative error below 4e-4.",
        "tolerances": [{"path": "/speck", "rtol": "0", "atol": "1e-44"}],
    }
    observed = {"speck": expected["speck"] * (1 + Fraction(3, 10**4))}
    result = compare(observed, expected, policy=policy)
    assert result["status"] == "mismatch"
    assert result["path"] == "/speck"
    assert compare({"speck": expected["speck"] + Fraction(1, 10**44)}, expected, policy=policy)["status"] == "equal"


@pytest.mark.parametrize(
    "authored, status",
    [({}, "equal"), ({"rtol": "0"}, "mismatch"), ({"tolerances": [{"path": "", "rtol": "0"}]}, "mismatch")],
)
def test_relative_promise_is_floored_only_where_no_author_spoke(authored, status):
    policy = {**authored, "statement": "Relative error below 2e-13."}
    assert compare(Decimal("1.000000000004"), Decimal("1"), policy=policy)["status"] == status
    assert compare(Decimal("1.0000000000002"), Decimal("1"), policy=policy)["status"] == "equal"
    assert compare(Decimal("1.000000000006"), Decimal("1"), policy=policy)["status"] == "mismatch"


def test_absolute_promise_floor_uses_leaf_margin_not_componentwise_maxima():
    policy = {"statement": "Report noninteger outputs to twelve decimal places."}
    expected = Decimal("1000000000000")
    assert compare(10**12 + 5, expected, policy=policy)["status"] == "equal"
    assert compare(Fraction(10**12 + 5) + Fraction(1, 10**13), expected, policy=policy)["status"] == "mismatch"
    policy["tolerances"] = [{"path": "", "rtol": "1e-28", "atol": "0"}]
    assert compare(Fraction(expected) + Fraction(5, 10**13), expected, policy=policy)["status"] == "equal"
    assert compare(Fraction(expected) + Fraction(6, 10**13), expected, policy=policy)["status"] == "mismatch"


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


def test_unknown_kind_honours_both_components_and_scale_guard_retains_relative():
    policy = {"statement": "Accurate to within 2e-6.", "rtol": "0", "atol": "0"}
    assert compare(Decimal("1.000004"), Decimal("1"), policy=policy)["status"] == "equal"
    assert compare(Decimal("1.000004000000001"), Decimal("1"), policy=policy)["status"] == "mismatch"
    expected = Fraction(1, 10**20)
    boundary = expected * (1 + Fraction(2, 10**6))
    assert compare(boundary, expected, policy=policy)["status"] == "equal"
    assert compare(boundary + Fraction(1, 10**27), expected, policy=policy)["status"] == "mismatch"


@pytest.mark.parametrize("role, status", [("candidate", "mismatch"), ("reference", "invalid_reference")])
def test_statement_mismatch_preserves_reference_preflight_classification(role, status):
    policy = {"statement": "Report rate with relative error below 2e-8.", "rtol": "0.1"}
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
    assert compare([Decimal("1.0002")], [Decimal("1")], policy=policy)["status"] == "equal"
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


# --- set-valued answers: unordered one-to-one matching (Gap 1) --------------------------------- #


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


# --- parity: reference_invalid regressions on the public-70 tasks --------------------------------- #
#
# Each test feeds a stored public-70 expectation and an identical reference-shaped observation and
# asserts equal. Public data only. No reference or candidate source is executed. The fixed code is noted per test.


def test_unused_tolerance_path_is_ignored_not_a_defect(symbolic_runtime):
    # Code 1: a tolerance entry whose path addresses no numeric or complex leaf is ignored, never a
    # defect. A missing path, a symbolic leaf, and a set container all have no numeric margin to set.
    assert compare(1, 1, policy={"tolerances": [{"path": "/missing", "rtol": "0"}]})["status"] == "equal"
    symbolic = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    assert compare(SymbolicText("exp(I*pi/4)"), SymbolicText("exp(I*pi/4)"), policy=symbolic)["status"] == "equal"
    a_set = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    assert compare({1.0, 2.0}, [1.0, 2.0], policy=a_set)["status"] == "equal"


def test_consumed_tolerance_entry_still_governs_its_leaf():
    # Code 1 must not throw the baby out: an entry whose path IS a numeric leaf still governs it.
    strict = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    assert compare(Decimal("1.0000000001"), Decimal("1"), policy=strict)["status"] == "equal"
    assert compare(Decimal("1.001"), Decimal("1"), policy=strict)["status"] == "mismatch"
    nested = {"tolerances": [{"path": "/1", "rtol": "0.2", "atol": "0"}]}
    # Slot /1 admits 15% error. Slot /0 keeps the strict default and rejects the same error.
    assert compare([Decimal("1"), Decimal("1.15")], [Decimal("1"), Decimal("1")], policy=nested)["status"] == "equal"
    off = compare([Decimal("1.15"), Decimal("1")], [Decimal("1"), Decimal("1")], policy=nested)
    assert off["status"] == "mismatch" and off["path"] == "/0"


def test_public70_main_12_unused_tolerance_on_symbolic_leaf(symbolic_runtime):
    # Code 1. A root tolerance entry on a symbolic root-of-unity answer is unused, not a defect.
    policy = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    value = SymbolicText("exp(-2*I*pi/5)")
    assert compare(value, value, role="reference", policy=policy)["status"] == "equal"


def test_public70_main_39_unused_tolerance_on_complex_symbolic_leaf(symbolic_runtime):
    # Code 1. A root tolerance entry on a complex-valued symbolic answer is unused, not a defect.
    policy = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    value = SymbolicText(
        "0.142857142857143*sqrt(2)*(0.00699999999999995 + 0.524*I)*(0.7 - 0.4*I)**2*exp((-0.7 - 0.4*I)*(0.7 - 0.4*I))"
    )
    assert compare(value, value, role="reference", policy=policy)["status"] == "equal"


def test_public70_main_59_unused_tolerance_paths_on_symbolic_list_members(symbolic_runtime):
    # Code 1. Per-member tolerance paths "/0/1" and "/1/1" name symbolic leaves. Both are ignored,
    # while the integer members at "/0/0" and "/1/0" still compare exactly.
    policy = {
        "tolerances": [
            {"path": "/0/1", "rtol": "1e-9", "atol": "0"},
            {"path": "/1/1", "rtol": "1e-9", "atol": "0"},
        ]
    }
    value = [[11, SymbolicText("-0.0987591240875912*pi")], [9, SymbolicText("0.0808029197080292*pi")]]
    assert compare(value, value, role="reference", policy=policy)["status"] == "equal"


def test_public70_main_24_piecewise_and_undefined_function_notation(symbolic_runtime):
    # Code 2. Piecewise((value, condition), ...), Eq(a, b) and the boolean catch-all condition parse
    # as undefined-function and boolean notation instead of raising symbolic_function.
    value = [
        SymbolicText(
            "Piecewise((-2, Eq(y, 1/2)), ((3*y - 1)*atan(sqrt(1 - 2*y)/Abs(y))"
            "/(sqrt(1 - 2*y)*(y - 1)), True)) - 1 + (y**2 + 1)*log(-(1 - y)/y)/(1 - y) - 1/(1 - y)"
        ),
        SymbolicText(
            "-y + Piecewise((-2, Eq(y, 1/2)), ((3*y - 1)*atan(sqrt(1 - 2*y)/Abs(y))"
            "/(sqrt(1 - 2*y)*(y - 1)), True)) + 2 + (y**2 + 1)*(log(4*p_z**2*y*(1 - y)/mu**2)"
            " - 1/epsilon_IR)/(1 - y) - 1/(1 - y)"
        ),
    ]
    assert compare(value, value, role="reference")["status"] == "equal"


def test_public70_main_65_undefined_trace_function(symbolic_runtime):
    # Code 2. 'tr(psi)' is an undefined function application, built as sympy.Function('tr').
    value = [SymbolicText("tr(psi)"), SymbolicText("tr(psi**3)")]
    assert compare(value, value, role="reference")["status"] == "equal"


def test_public70_main_62_interval_constructor_notation(symbolic_runtime):
    # Code 3. The member-access operator "." parses for the Interval.open(...) set constructor. The
    # numeric siblings resolve under the ignored root tolerance entry.
    policy = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    case0 = [0.228290381792673, SymbolicText("Interval.open(0, pi/2)"), 1.06089029087253]
    assert compare(case0, case0, role="reference", policy=policy)["status"] == "equal"
    case1 = [0.621454279011698, SymbolicText("Interval.open(0, 1.04719755119660)"), 0.724450598842256]
    assert compare(case1, case1, role="reference", policy=policy)["status"] == "equal"


def test_public70_main_49_float_bearing_expression_past_growth_bound(symbolic_runtime):
    # Code 4. A float-bearing constant expression whose simplify explodes past the growth bound is
    # kept comparable (parse falls back to the bounded raw expression) instead of raising
    # symbolic_growth_limit. The unused root tolerance entry is ignored.
    policy = {"tolerances": [{"path": "", "rtol": "1e-9", "atol": "0"}]}
    value = SymbolicText(
        "-2500*log(625)/log(2) + (4 + log(625)/log(2))*log(exp(2)/(4*pi))/(8*log(2))"
        " + (4 + log(625)/log(2))**2/16 + 2500*log(E*pi)/log(2) + 24990000.0"
    )
    assert compare(value, value, role="reference", policy=policy)["status"] == "equal"
