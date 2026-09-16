# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the bounded symbolic grammar and the numeric-equivalence fallback.

Skips without SymPy. No Daytona, and no reference or candidate source is executed.
"""

import pytest

from resources_servers.critpt_custom_grader.codec import RepresentationError


sympy = pytest.importorskip("sympy")

from resources_servers.critpt_custom_grader import symbolic  # noqa: E402


def parse(text):
    return symbolic.parse_expression(text)


# --- code 2: undefined function applications and the named function table ------------------------- #


@pytest.mark.parametrize("text", ["tr(psi)", "unapproved(x)", "f(x, y)", "g(a*b + 1)"])
def test_undefined_function_application_builds_a_fresh_function(text):
    expr = parse(text)
    assert isinstance(expr, sympy.Basic)
    # It is a genuine undefined function, never a resolved library object.
    assert expr.atoms(sympy.core.function.AppliedUndef)


def test_undefined_function_requires_at_least_one_argument():
    with pytest.raises(RepresentationError, match="symbolic_arity"):
        parse("f()")


def test_undefined_function_equivalence_and_undecidability():
    assert symbolic.equivalent(parse("tr(psi)"), parse("tr(psi)")) is True
    assert symbolic.equivalent(parse("tr(x)"), parse("tr(y)")) is None
    assert symbolic.equivalent(parse("tr(psi**3)"), parse("tr(psi**3)")) is True


@pytest.mark.parametrize(
    "text",
    ["gamma(x)", "loggamma(x)", "factorial(x)", "besselj(0, x)", "bessely(1, x)", "factorial(6)"],
)
def test_named_function_table_additions_parse(text):
    assert isinstance(parse(text), sympy.Basic)


def test_named_function_semantics_are_the_library_function():
    # gamma(x + 1) == x*gamma(x) is a real identity, decided here.
    assert symbolic.equivalent(parse("gamma(x + 1)"), parse("x*gamma(x)")) is True
    assert symbolic.equivalent(parse("factorial(6)"), parse("720")) is True


def test_zeta_binds_to_the_library_function_and_grades_definitively():
    # zeta binds to sympy.zeta: zeta(2) == pi^2/6 is a decided identity, not an undecided free symbol.
    assert symbolic.equivalent(parse("zeta(2)"), parse("pi**2/6")) is True
    assert symbolic.equivalent(parse("zeta(4)/zeta(2)"), parse("zeta(2)/zeta(4)")) is False


def test_long_written_out_polynomial_parses_within_the_raised_node_limit():
    # A written-out polynomial of a few hundred tokens must parse. Sum of 1..119 times x collapses to 7140*x.
    text = " + ".join(f"{i}*x" for i in range(1, 120))
    assert len(text) > 700
    expr = parse(text)
    assert isinstance(expr, sympy.Basic)
    assert symbolic.equivalent(expr, parse("7140*x")) is True


def test_piecewise_tuple_and_boolean_literal_notation_parses():
    # Piecewise is undefined here, but the tuple and boolean-literal notation must parse rather than raise.
    expr = parse("Piecewise((-2, Eq(y, 1/2)), (3*y - 1, True))")
    assert isinstance(expr, sympy.Basic)
    assert symbolic.equivalent(expr, parse("Piecewise((-2, Eq(y, 1/2)), (3*y - 1, True))")) is True


def test_empty_tuple_is_rejected():
    with pytest.raises(RepresentationError):
        parse("Piecewise(())")


# --- code 3: the Interval set constructors and the bounded "." operator --------------------------- #


@pytest.mark.parametrize(
    "text, bounds",
    [
        ("Interval.open(0, pi/2)", (True, True)),
        ("Interval.closed(0, 1)", (False, False)),
        ("Interval.Lopen(0, 1)", (True, False)),
        ("Interval.Ropen(0, 1)", (False, True)),
        ("Interval(0, 1)", (False, False)),
    ],
)
def test_interval_constructors_parse_with_correct_open_flags(text, bounds):
    expr = parse(text)
    assert isinstance(expr, sympy.Interval)
    assert (bool(expr.left_open), bool(expr.right_open)) == bounds


def test_equal_intervals_are_equivalent_and_unequal_are_undecided_not_a_crash():
    assert symbolic.equivalent(parse("Interval.open(0, pi/2)"), parse("Interval.open(0, pi/2)")) is True
    # Different intervals return None from the fallback, never raise.
    assert symbolic.equivalent(parse("Interval.open(0, pi/2)"), parse("Interval.open(0, pi/3)")) is None


@pytest.mark.parametrize("text", ["Foo.bar(1)", "Interval.unknown(0, 1)", "sqrt.__globals__", "x.y"])
def test_member_access_other_than_interval_constructors_is_rejected(text):
    with pytest.raises(RepresentationError):
        parse(text)


# --- code 4: the parse-time growth relaxation and the numeric-equivalence fallback ---------------- #


def test_float_bearing_expression_past_simplify_growth_bound_still_parses():
    # An expression whose simplify explodes past the growth bound must still parse.
    # An identical reference reproduction then compares equal.
    text = (
        "-2500*log(625)/log(2) + (4 + log(625)/log(2))*log(exp(2)/(4*pi))/(8*log(2))"
        " + (4 + log(625)/log(2))**2/16 + 2500*log(E*pi)/log(2) + 24990000.0"
    )
    expr = parse(text)
    assert isinstance(expr, sympy.Basic)
    assert symbolic.equivalent(expr, parse(text)) is True


@pytest.mark.parametrize("text", ["(x + y + z)**8", "x**33", "2**(2**8)", "9" * 129])
def test_build_time_growth_and_size_limits_still_raise(text):
    # Build-time and literal limits stay hard errors even with the growth relaxation.
    with pytest.raises(RepresentationError):
        parse(text)


def test_numeric_equivalent_decides_equal_false_and_undecided():
    assert symbolic._numeric_equivalent(parse("(x + 1)**2"), parse("x**2 + 2*x + 1")) is True
    assert symbolic._numeric_equivalent(parse("x + 1"), parse("x + 2")) is False
    # A complex-valued sample is undecidable by this real fallback, never guessed.
    assert symbolic._numeric_equivalent(parse("I*x"), parse("I*x")) is None
    # An undefined function that does not reduce to a number is undecidable.
    assert symbolic._numeric_equivalent(parse("tr(x)"), parse("tr(y)")) is None


def test_undecided_branch_function_difference_is_not_accepted_on_samples():
    # Abs(x - 4) and 4 - x agree at every fixed sample below 4, but differ past the branch point at x = 4.
    # An undecided difference that carries a branch function stays None, never True on samples.
    assert symbolic.equivalent(parse("Abs(x - 4)"), parse("4 - x")) is None
    # A sign mismatch with no branch function is still caught by sampling.
    assert symbolic.equivalent(parse("sqrt(z**2)"), parse("z")) is False


def test_numeric_equivalent_is_deterministic():
    left, right = parse("(x + 1)**3"), parse("x**3 + 3*x**2 + 3*x + 1")
    first = symbolic._numeric_equivalent(left, right)
    second = symbolic._numeric_equivalent(left, right)
    assert first is True and second is True


def _raise_polynomial_error(*args, **kwargs):
    raise sympy.PolynomialError("synthetic: not a polynomial in these generators")


def _raise_attribute_error(*args, **kwargs):
    raise AttributeError("synthetic: is_polynomial cannot handle this expression")


def test_polynomial_check_that_raises_is_treated_as_not_a_polynomial(monkeypatch):
    # SymPy's polynomial fast path can raise on an expression it will not treat as a polynomial: an Add that
    # carries a Poly object makes is_polynomial raise today.
    x = sympy.Symbol("x")
    with pytest.raises(Exception):
        (x + sympy.Poly(x, x)).is_polynomial(x)
    # The guard turns any raise there into the not-a-polynomial result (the numeric fallback), never an
    # unhandled exception. The difference x*(x - 1) has is_zero None, so equivalence reaches the polynomial branch.
    left, right = parse("x**2"), parse("x")
    expected = symbolic._numeric_equivalent(left, right)

    monkeypatch.setattr(symbolic.sp, "Poly", _raise_polynomial_error)
    assert symbolic.equivalent(left, right) == expected
    monkeypatch.undo()

    monkeypatch.setattr(sympy.Expr, "is_polynomial", _raise_attribute_error)
    assert symbolic.equivalent(left, right) == expected


def test_numeric_equivalent_respects_a_tighter_case_rtol():
    from fractions import Fraction

    left, right = parse("x + 1"), parse("x + 1 + 1/1000000")
    # A near miss passes at a loose rtol but fails at a tight one, showing the case tolerance is used.
    assert symbolic._numeric_equivalent(left, right, rtol=Fraction(1, 100)) is True
    assert symbolic._numeric_equivalent(left, right, rtol=Fraction(1, 10**12)) is False
