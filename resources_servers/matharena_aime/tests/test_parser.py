# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe-wrapper tests and fixtures measured against the immutable MathArena parser."""

from pathlib import Path

import pytest
import sympy

from resources_servers.matharena_aime.parser import (
    UPSTREAM_REVISION,
    UnsafeMathExpression,
    official_parser,
    parse_result,
    safe_sympify,
)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("1+2", sympy.Integer(3)),
        ("5-2", sympy.Integer(3)),
        ("2*3", sympy.Integer(6)),
        ("3/2", sympy.Rational(3, 2)),
        ("2**3", sympy.Integer(8)),
        ("x**y", sympy.Symbol("x") ** sympy.Symbol("y")),
        ("+3", sympy.Integer(3)),
        ("-3", sympy.Integer(-3)),
        ("1.25", sympy.Float(1.25)),
        ("sqrt(9)", sympy.Integer(3)),
        ("factorial(5)", sympy.Integer(120)),
        ("binomial(5,2)", sympy.Integer(10)),
        ("sin(pi/2)", sympy.Integer(1)),
        ("Abs(-3)", sympy.Integer(3)),
        ("x", sympy.Symbol("x")),
        ("pi", sympy.pi),
        ("E", sympy.E),
        ("I", sympy.I),
        ("(1,2)", [sympy.Integer(1), sympy.Integer(2)]),
        ("[3,4]", [sympy.Integer(3), sympy.Integer(4)]),
        (sympy.Integer(4), sympy.Integer(4)),
        (3, 3),
        (3.5, 3.5),
    ],
)
def test_safe_arithmetic_constructs_expected_values_without_string_eval(expression, expected):
    assert safe_sympify(expression) == expected


@pytest.mark.parametrize(
    "expression,reason",
    [
        (None, "Only strings"),
        ({"x": 1}, "Only strings"),
        pytest.param("x" * 8193, "8192 characters", id="oversized-expression"),
        ("__import__(x)", "private names"),
        ('"text"', "Python strings"),
        ("x;y", "statements"),
        ("`x`", "Python strings"),
        pytest.param("+".join(["1"] * 200), "512 syntax nodes", id="too-many-ast-nodes"),
        ("1e999", "Nonfinite"),
        pytest.param(str(2**4100), "4096 bits", id="oversized-integer"),
        ("2**10001", "Exponent"),
        ("factorial(10001)", "Combinatorial"),
        ("binomial(10001,2)", "Combinatorial"),
        ("sqrt(x=1)", "positional"),
        ("Max(1,2,3,4,5,6,7,8,9)", "bounded positional"),
        ("open(1)", "Unsupported"),
        ("x.real", "Unsupported"),
        ("x[0]", "Unsupported"),
        ("[x for x in y]", "Unsupported"),
        ("lambda:1", "Unsupported"),
        ("1//2", "Unsupported"),
        ("True", "Unsupported"),
        ("x_y", "Unsupported"),
    ],
)
def test_unsafe_or_unbounded_expressions_are_not_silently_reinterpreted(expression, reason):
    with pytest.raises(UnsafeMathExpression, match=reason):
        safe_sympify(expression)
    assert not issubclass(UnsafeMathExpression, Exception)


def test_latex_syntax_reaches_only_the_upstream_latex_parser():
    with pytest.raises(ValueError, match="LaTeX parser"):
        safe_sympify(r"\frac{1}{2}")


# Measured with the unmodified upstream parser at this revision in its isolated
# SymPy 1.14 / ANTLR 4.11 runtime. These expressions do not need ANTLR fallback.
# Each tuple is text, gold, strict extraction, loose extraction, warning, loose correctness.
UPSTREAM_FIXTURES = [
    (r"\boxed{123}", 123, "123", "123", 0, True),
    (r"\boxed{124}", 123, "124", "124", 0, False),
    (r"\boxed{None}", 123, "None", "None", 0, False),
    (r"\boxed{6/2}", 3, "3", "3", 0, True),
    (r"\boxed{\frac{6}{2}}", 3, "3", "3", 0, True),
    (r"\boxed{\sqrt{9}}", 3, "3", "3", 0, True),
    (r"\boxed{2^3}", 8, "8", "8", 0, True),
    (r"\boxed{0}", 0, "0", "0", 0, True),
    (r"\boxed{001}", 1, "1", "1", 0, True),
    (r"\boxed{1.0}", 1, "1", "1", 0, True),
    (r"\boxed{1,234}", 234, "1234", "1234", 0, False),
    ("My final answer is 123", 123, None, "123", 3, True),
    ("No answer", 123, None, None, 3, False),
    ("", 123, None, None, 3, False),
    (r"\boxed{2} then \boxed{3}", 3, "3", "3", 0, True),
    (r"\fbox{3}", 3, "3", "3", 0, True),
    (r"\boxed{2+3}", 5, "5", "5", 0, True),
    (r"\boxed{-1}", 1, "-1", "-1", 0, False),
]


@pytest.mark.parametrize("text,gold,strict_answer,loose_answer,warning,correct", UPSTREAM_FIXTURES)
def test_official_extraction_and_score_fixtures(text, gold, strict_answer, loose_answer, warning, correct):
    assert UPSTREAM_REVISION == "b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b"
    first = parse_result(text, strict=True)
    assert first["extracted_answer"] == strict_answer
    assert first["parser_warning"] == (3 if strict_answer is None else warning)
    assert first["needs_format_retry"] == (strict_answer is None)
    final = parse_result(text, strict=False, expected_answer=gold)
    assert final["extracted_answer"] == loose_answer
    assert final["reward"] == float(correct)
    assert final["parser_warning"] == warning


@pytest.mark.parametrize("output_tokens", [1000, 2000, 1024, 16384, 50000])
def test_official_suspicious_length_warning_for_wrong_answers(output_tokens):
    assert (
        parse_result(r"\boxed{2}", strict=False, expected_answer=3, output_tokens=output_tokens)["parser_warning"] == 1
    )
    assert (
        parse_result(r"\boxed{3}", strict=False, expected_answer=3, output_tokens=output_tokens)["parser_warning"] == 0
    )


def test_warning_retains_unboxed_gold_mentions_and_empty_answers():
    assert parse_result("123 occurs above but final is 456", strict=False, expected_answer=123)["parser_warning"] == 3
    assert parse_result("", strict=False, expected_answer=123, output_tokens=1234)["parser_warning"] == 3
    assert parse_result("", strict=False, expected_answer=123, output_tokens=1000)["parser_warning"] == 1


@pytest.mark.parametrize("filename", ["parser.py.txt", "parse_manual.py.txt"])
def test_vendor_integrity_is_checked_before_any_source_execution(monkeypatch, filename):
    original = Path.read_bytes
    official_parser.cache_clear()
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"tampered" if path.name == filename else original(path))
    try:
        with pytest.raises(RuntimeError, match="source hash mismatch"):
            official_parser()
    finally:
        official_parser.cache_clear()


def test_parser_module_cache_and_sympy_proxy_do_not_modify_global_sympy():
    first = official_parser()
    assert official_parser() is first
    assert first.sympy.sympify is safe_sympify
    assert first.sympy.Integer is sympy.Integer
    assert sympy.sympify is not safe_sympify


def test_untrusted_python_answer_is_rejected_before_last_integer_fallback():
    with pytest.raises(UnsafeMathExpression):
        parse_result(r"\boxed{__import__(x)}", strict=False, expected_answer=1)
