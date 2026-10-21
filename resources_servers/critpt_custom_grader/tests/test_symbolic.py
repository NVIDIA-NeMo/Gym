# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the bounded symbolic grammar and the numeric-equivalence fallback.

Skips without SymPy. No sandbox is created, and no reference or candidate source is executed.
"""

import pytest

from resources_servers.critpt_custom_grader.codec import RepresentationError


sympy = pytest.importorskip("sympy")

from resources_servers.critpt_custom_grader import symbolic  # noqa: E402


def parse(text):
    return symbolic.parse_expression(text)


# --- undefined function applications and the named function table ------------------------------- #


@pytest.mark.parametrize("text", ["tr(rho)", "unapproved(x)", "f(x, y)", "g(a*b + 1)"])
def test_undefined_function_application_builds_a_fresh_function(text):
    expr = parse(text)
    assert isinstance(expr, sympy.Basic)
    # It is a genuine undefined function, never a resolved library object.
    assert expr.atoms(sympy.core.function.AppliedUndef)


def test_undefined_function_requires_at_least_one_argument():
    with pytest.raises(RepresentationError, match="symbolic_arity"):
        parse("f()")


def test_undefined_function_equivalence_and_undecidability():
    assert symbolic.equivalent(parse("tr(rho)"), parse("tr(rho)")) is True
    assert symbolic.equivalent(parse("tr(x)"), parse("tr(y)")) is None
    assert symbolic.equivalent(parse("tr(rho**3)"), parse("tr(rho**3)")) is True


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
    expr = parse("Piecewise((-3, Eq(t, 1/4)), (2*t - 1, True))")
    assert isinstance(expr, sympy.Basic)
    assert symbolic.equivalent(expr, parse("Piecewise((-3, Eq(t, 1/4)), (2*t - 1, True))")) is True


def test_empty_tuple_is_rejected():
    with pytest.raises(RepresentationError):
        parse("Piecewise(())")


# --- the Interval set constructors and the bounded "." operator --------------------------------- #


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


# --- the parse-time growth relaxation and the numeric-equivalence fallback ---------------------- #


def test_float_bearing_expression_past_simplify_growth_bound_still_parses(monkeypatch):
    # An expression whose simplify explodes past the growth bound must still parse: the bounded raw
    # expression is kept. Simplify is forced past the node bound here, so the fallback runs whatever SymPy
    # would do. An identical reference reproduction then compares equal.
    exploded = sympy.Add(*sympy.symbols(f"s0:{symbolic.MAX_RESULT_NODES + 1}"))
    monkeypatch.setattr(symbolic.sp, "simplify", lambda expression, *args, **kwargs: exploded)
    text = (
        "-1800*log(81)/log(3) + (2 + log(81)/log(3))*log(exp(3)/(2*pi))/(6*log(3)) + 1500*log(E*pi)/log(3) + 1250000.0"
    )
    expr = parse(text)
    assert isinstance(expr, sympy.Basic)
    assert not expr.has(sympy.Symbol("s0"))  # the raw expression was kept, not the exploded result
    assert symbolic.equivalent(expr, parse(text)) is True


@pytest.mark.parametrize("text", ["(x + y + z)**8", "x**33", "2**(2**8)", "9" * 129])
def test_build_time_growth_and_size_limits_still_raise(text):
    # Build-time and literal limits stay hard errors even with the growth relaxation.
    with pytest.raises(RepresentationError):
        parse(text)


def test_numeric_equivalent_decides_equal_false_and_undecided():
    # The constants-only helper still decides on samples.
    assert symbolic._numeric_equivalent(parse("(x + 1)**2"), parse("x**2 + 2*x + 1")) is True
    assert symbolic._numeric_equivalent(parse("x + 1"), parse("x + 2")) is False
    # A complex-valued sample is undecidable by this real fallback, never guessed.
    assert symbolic._numeric_equivalent(parse("I*x"), parse("I*x")) is None
    # An undefined function that does not reduce to a number is undecidable.
    assert symbolic._numeric_equivalent(parse("tr(x)"), parse("tr(y)")) is None


def test_undecided_branch_function_difference_is_not_accepted_on_samples():
    # Abs(x - 4) and 4 - x agree at every disproof point (each lies in (0, 4)) but differ past the branch point x = 4.
    # Samples never accept a difference that carries a branch function, so the pair stays undecided.
    assert symbolic.equivalent(parse("Abs(x - 4)"), parse("4 - x")) is None
    # A disagreement only a negative value shows is no disproof: sqrt(z**2) equals z on the positive reals.
    assert symbolic.equivalent(parse("sqrt(z**2)"), parse("z")) is None


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


def test_sampling_never_certifies_equality_for_free_symbols():
    assert symbolic.equivalent(parse("x + sin(840*pi*x)"), parse("x")) is not True
    assert symbolic.equivalent(parse("conjugate(x)"), parse("x")) is not True
    assert sympy.conjugate in symbolic._BRANCH_FUNCTIONS


def test_algebraic_identities_are_proved():
    assert symbolic.equivalent(parse("(x + 1)**2"), parse("x**2 + 2*x + 1")) is True
    assert symbolic.equivalent(parse("sin(x)**2 + cos(x)**2"), parse("1")) is True
    assert symbolic.equivalent(parse("(x**2 - 1)/(x - 1)"), parse("x + 1")) is True
    assert symbolic.equivalent(parse("exp(2*x)"), parse("exp(x)**2")) is True


def test_disproof_still_comes_from_samples_and_undecided_stays_none():
    # A disagreement only a negative value shows is no disproof: sqrt(z**2) equals z for positive z.
    assert symbolic.equivalent(parse("sqrt(z**2)"), parse("z")) is None
    assert symbolic.equivalent(parse("x + 1"), parse("x + 2")) is False
    assert symbolic.equivalent(parse("tr(x)"), parse("tr(y)")) is None
    assert (
        symbolic.equivalent(parse("Abs(x - 4)"), parse("4 - x")) is None
        or symbolic.equivalent(parse("Abs(x - 4)"), parse("4 - x")) is False
    )


def test_constants_keep_numeric_agreement_and_results_are_deterministic():
    assert symbolic.equivalent(parse("2*sqrt(2)"), parse("sqrt(8)")) is True
    left, right = parse("x*exp(x) + sin(840*pi*x)"), parse("x*exp(x)")
    assert symbolic.equivalent(left, right) == symbolic.equivalent(left, right)


def test_a_timeout_propagates_and_memory_exhaustion_is_a_failed_attempt(monkeypatch):
    # The per-case timer belongs to the caller and propagates out of a proof rewrite and parse-time simplify.
    # Memory exhaustion there is only a failed attempt: the comparison goes on and never certifies from it.
    left, right = parse("tr(x)"), parse("tr(y)")  # no disproof point decides this pair, so the rewrites run

    def raising(error):
        def attempt(*args, **kwargs):
            raise error("synthetic")

        return attempt

    monkeypatch.setattr(symbolic, "_PROOF_REWRITES", (raising(TimeoutError),))
    with pytest.raises(TimeoutError):
        symbolic.equivalent(left, right)
    monkeypatch.setattr(symbolic, "_PROOF_REWRITES", (raising(MemoryError),))
    assert symbolic.equivalent(left, right) is None
    monkeypatch.setattr(symbolic.sp, "simplify", raising(TimeoutError))
    with pytest.raises(TimeoutError):
        parse("x + 1")
    monkeypatch.setattr(symbolic.sp, "simplify", raising(MemoryError))
    assert isinstance(parse("x + 1"), sympy.Basic)  # the bounded raw expression is kept


@pytest.mark.parametrize(
    "left, right, verdict",
    [
        # A disagreement only a negative coordinate or a complex point would show leaves the pair undecided.
        ("sqrt(z**2)", "z", None),
        ("conjugate(x)", "x", None),
        ("Abs(A*exp(I*k*x))**2", "Abs(A)**2", None),
        ("re(exp(I*x))", "cos(x)", None),
        ("Abs(x)**2", "x**2", None),
        ("sqrt(a*b)", "sqrt(a)*sqrt(b)", None),
        # Both values must be finite and real: at every disproof point one side here is not.
        ("sqrt(1 - x**2)", "sqrt(x**2 - 1)", None),
        # Two different finite real values at a positive point are a disproof.
        ("sqrt(k/m)", "sqrt(m/k)", False),
        ("log(V2/V1)", "log(V1/V2)", False),
        ("sqrt(2*g*h)", "sqrt(g*h)", False),
        ("1/sqrt(1 - beta**2)", "sqrt(1 - beta**2)", False),  # decided at the point inside (0, 1)
        ("x", "-x", False),
        ("x*sin(x)", "-x*sin(x)", False),
    ],
)
def test_disproof_needs_two_different_real_values_at_a_positive_point(left, right, verdict):
    assert symbolic.equivalent(parse(left), parse(right)) is verdict


@pytest.mark.parametrize(
    "left, right",
    [
        ("sqrt(m**2*c**4 + p**2*c**2)", "c*sqrt(m**2*c**2 + p**2)"),
        ("hbar*k/sqrt(2*m)", "sqrt(hbar**2*k**2/(2*m))"),
    ],
)
def test_identities_that_hold_for_positive_quantities_are_never_disproved(left, right):
    assert symbolic.equivalent(parse(left), parse(right)) is not False


def test_a_symbol_declared_non_real_enables_the_complex_points():
    # Parsed symbols carry no assumptions; only an explicit non-real declaration lets the complex points disprove.
    for z in (sympy.Symbol("z", complex=True), sympy.Symbol("z", real=False)):
        assert symbolic.equivalent(sympy.Abs(z) ** 2, z**2) is False


def test_polynomial_verdict_needs_an_exact_coefficient_domain():
    # atan(1/2) + atan(1/3) == pi/4, but a polynomial over pi and the two atans cannot see it, so no False from it.
    assert symbolic.equivalent(parse("x*(atan(1/2) + atan(1/3))"), parse("pi*x/4")) is not False
    # The constant identity is accepted on 50-digit agreement: the one path that returns True from evaluation.
    assert symbolic.equivalent(parse("atan(1/2) + atan(1/3)"), parse("pi/4")) is True


def test_expand_trig_rewrite_skips_a_large_integer_multiple_angle(monkeypatch):
    # expand_trig writes sin(N*x) as a degree-N polynomial before any bound applies, so a multiple above 32 skips it.
    calls = []
    original = sympy.expand_trig
    monkeypatch.setattr(
        symbolic.sp, "expand_trig", lambda expression: calls.append(expression) or original(expression)
    )
    x = sympy.Symbol("x")
    assert symbolic._bounded_expand_trig(sympy.sin(33 * x)) == sympy.sin(33 * x)
    assert calls == []
    assert symbolic._bounded_expand_trig(sympy.sin(32 * x)) != sympy.sin(32 * x)
    assert calls == [sympy.sin(32 * x)]
    assert symbolic._bounded_expand_trig(sympy.sin(2 * x)) == 2 * sympy.sin(x) * sympy.cos(x)
