# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-safe runner tests: input materialization only, no candidate code and no subprocess.

No submission is loaded or run and no worker is forked. Hostile subprocess tests are in test_runner_remote.py.
"""

from decimal import Decimal
from fractions import Fraction

import pytest

from resources_servers.critpt_custom_grader.codec import (
    ComplexValue,
    RepresentationError,
    SetValue,
    SymbolicText,
    decode_value,
    encode_value,
)
from resources_servers.critpt_custom_grader.runner import RunnerError, adapt_output, materialize_input


@pytest.mark.parametrize(
    "component",
    [
        9007199254740993,  # first odd integer a binary64 double cannot hold
        Fraction(1, 3),  # a non-binary fraction has no exact double
        Decimal("1e-700"),  # a decimal that a double narrows to zero on underflow
    ],
)
def test_complex_component_that_a_double_cannot_hold_is_rejected_not_narrowed(component):
    for value in (ComplexValue(component, 0), ComplexValue(0, component)):
        decoded = decode_value(encode_value(value))
        assert decoded == value  # the codec keeps the component exact, only materialization would narrow it
        with pytest.raises(RunnerError) as raised:
            materialize_input(decoded)
        assert raised.value.code == "invalid_job"


@pytest.mark.parametrize(
    "value, expected",
    [
        (2 + 3j, complex(2.0, 3.0)),
        (ComplexValue(2, 3), complex(2.0, 3.0)),
        (ComplexValue(Decimal("0.5"), Fraction(1, 4)), complex(0.5, 0.25)),
    ],
)
def test_complex_components_a_double_holds_exactly_still_materialize(value, expected):
    materialized = materialize_input(decode_value(encode_value(value)))
    assert type(materialized) is complex
    assert materialized == expected


def test_adapt_output_emits_a_finite_sympy_float_as_exact_binary64():
    # A finite SymPy Float at double precision has the same bits as a Python float, so it is emitted as binary64.
    sympy = pytest.importorskip("sympy")
    result = adapt_output(sympy.Float("0.1"))
    assert type(result) is float
    assert result == float("0.1")
    assert Fraction(result) == Fraction(0.1)  # the binary64 rational, not Fraction(1, 10)


def test_adapt_output_rejects_a_higher_precision_sympy_float_rather_than_rounding_it():
    # A Float storing more precision than a double is rejected, never narrowed.
    sympy = pytest.importorskip("sympy")
    with pytest.raises(RepresentationError) as raised:
        adapt_output(sympy.Float("0.1", 30))
    assert raised.value.code == "unsupported_type"


def test_adapt_output_keeps_a_float_atom_inside_a_larger_symbolic_expression_as_symbolic_text():
    # A Float buried in an expression stays symbolic: str(expr) renders a decimal the grammar reads back exactly.
    sympy = pytest.importorskip("sympy")
    from resources_servers.critpt_custom_grader.symbolic import equivalent, parse_expression

    expression = sympy.Symbol("x") + sympy.Float("0.5")
    result = adapt_output(expression)
    assert type(result) is SymbolicText
    # The emitted notation reads back to the same expression it came from.
    assert equivalent(parse_expression(result.text), parse_expression("x + 0.5")) is True


@pytest.mark.parametrize(
    "text",
    [
        "-0.00613030100852602/pi",
        "0.0362277324545884*2**(3/4)*sqrt(pi)",
        "(-3.06138153035693 + 0.765345382589233*pi**2)*exp(-0.932557711424465*pi)",
        "0.142857142857143*sqrt(2)*(0.00699999999999995 + 0.524*I)*(0.7 - 0.4*I)**2*exp((-0.7 - 0.4*I)*(0.7 - 0.4*I))",
    ],
)
def test_float_bearing_symbolic_notation_parses_and_compares_equal_to_itself(text):
    # A Float-bearing symbolic answer in str(expr) shape parses through the grammar and two equal copies compare equal.
    pytest.importorskip("sympy")
    from resources_servers.critpt_custom_grader.symbolic import equivalent, parse_expression

    assert equivalent(parse_expression(text), parse_expression(text)) is True


def test_adapt_output_carries_a_set_of_ints_as_an_unordered_set_value():
    # A set-valued return becomes an inert SetValue carrier holding its adapted members.
    result = adapt_output({3, 1, 2})
    assert type(result) is SetValue
    assert set(result.items) == {1, 2, 3}


def test_materialize_input_gives_task_code_a_real_set_not_the_carrier():
    # A canonical set argument reaches task code as a Python set, so the set API works.
    result = materialize_input(SetValue((1, 2)))
    assert type(result) is set
    assert result == {1, 2}
    assert result.copy() == {1, 2}


def test_adapt_output_carries_a_frozenset_and_adapts_sympy_members():
    sympy = pytest.importorskip("sympy")
    result = adapt_output(frozenset({sympy.sqrt(2), sympy.Integer(3)}))
    assert type(result) is SetValue
    kinds = sorted(type(member).__name__ for member in result.items)
    assert kinds == ["SymbolicText", "int"]


def test_adapt_output_rejects_a_set_over_the_element_budget():
    from resources_servers.critpt_custom_grader.task_data import MAX_ELEMENTS

    with pytest.raises(RepresentationError) as raised:
        adapt_output(set(range(MAX_ELEMENTS + 1)))
    assert raised.value.code == "structure_limit"


def test_adapt_output_carries_a_numpy_string_scalar_as_a_python_str():
    # A numpy text scalar is a str subclass, normalized to a base str so it round-trips through encode.
    numpy = pytest.importorskip("numpy")
    result = adapt_output({"choice": numpy.str_("L")})
    assert type(result["choice"]) is str
    assert result["choice"] == "L"
    assert decode_value(encode_value(result)) == {"choice": "L"}


def test_adapt_output_refuses_an_oversized_numpy_string_scalar_like_an_oversized_str():
    # The converted str carries no bound of its own, so the codec's text bound applies as to any str.
    numpy = pytest.importorskip("numpy")
    from resources_servers.critpt_custom_grader.task_data import MAX_TEXT_BYTES

    oversized = "x" * (MAX_TEXT_BYTES + 1)
    with pytest.raises(RepresentationError) as from_str:
        encode_value(adapt_output({"k": oversized}))
    with pytest.raises(RepresentationError) as from_numpy:
        encode_value(adapt_output({"k": numpy.str_(oversized)}))
    assert from_numpy.value.code == from_str.value.code == "text_limit"


def test_adapt_output_refuses_a_numpy_bytes_scalar():
    # A numpy bytes scalar has no codec carrier, so it stays refused rather than coerced to text.
    numpy = pytest.importorskip("numpy")
    with pytest.raises(RepresentationError) as raised:
        adapt_output(numpy.bytes_(b"ab"))
    assert raised.value.code == "unsupported_type"


def test_adapt_output_refuses_a_numpy_datetime_scalar():
    # A numpy datetime scalar yields a date the codec cannot carry, so it stays refused.
    numpy = pytest.importorskip("numpy")
    with pytest.raises(RepresentationError) as raised:
        adapt_output(numpy.datetime64("2020-01-01"))
    assert raised.value.code == "unsupported_type"


def test_adapt_output_refuses_a_numpy_structured_void_scalar():
    # A structured (void) scalar has no codec carrier, so it stays refused.
    numpy = pytest.importorskip("numpy")
    scalar = numpy.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")])[0]
    with pytest.raises(RepresentationError) as raised:
        adapt_output(scalar)
    assert raised.value.code == "unsupported_type"


@pytest.mark.parametrize(
    "output, decoded",
    [
        ({(0, 0): 1, (1, 2): 3}, {"(0, 0)": 1, "(1, 2)": 3}),  # reference keyed by tuples
        ({0: "a", 1: "b"}, {"0": "a", "1": "b"}),  # reference keyed by ints
    ],
)
def test_output_dict_with_non_string_keys_transports_with_stringified_keys(output, decoded):
    # A dict keyed by tuples or ints serializes with str() keys: adapt_output leaves keys alone, the codec stringifies.
    result = decode_value(encode_value(adapt_output(output)))
    assert result == decoded


def _run_job(conversions):
    return {
        "version": 1,
        "mode": "run",
        "nonce": "abc123",
        "entrypoint": "f",
        "source_path": "/tmp/source.py",
        "cases": [{"args": [], "kwargs": {}}],
        "input_conversions": conversions,
        "limits": {"suite_deadline_s": 1, "memory_mib": 1, "cpu_time_s": 1, "processes": 1},
    }


def test_run_job_requires_the_input_conversions_key():
    from resources_servers.critpt_custom_grader import runner

    job = _run_job([])
    assert runner.validate_job(job, "run") is None
    del job["input_conversions"]
    assert runner.validate_job(job, "run") == "invalid_job"


@pytest.mark.parametrize("conversions", [[], [None], ["symbol"], ["function"], [None, "symbol", "function"]])
def test_run_job_accepts_well_shaped_input_conversions(conversions):
    from resources_servers.critpt_custom_grader import runner

    assert runner.validate_job(_run_job(conversions), "run") is None


@pytest.mark.parametrize(
    "conversions",
    ["symbol", {"0": "symbol"}, ["Symbol"], [True], [1], ["symbol", "operator"], [None] * 129],
)
def test_run_job_rejects_malformed_input_conversions(conversions):
    from resources_servers.critpt_custom_grader import runner

    assert runner.validate_job(_run_job(conversions), "run") == "invalid_job"


def test_input_conversions_turn_string_positional_args_into_symbol_and_function():
    sympy = pytest.importorskip("sympy")
    from resources_servers.critpt_custom_grader.runner import _apply_input_conversions

    cases = [(["x", "g", "1.0", "left"], {"k": "v"})]
    converted = _apply_input_conversions(cases, ["symbol", "function", None])
    args, kwargs = converted[0]
    assert args[0] == sympy.Symbol("x")
    assert args[1] == sympy.Function("g")
    assert args[2] == "1.0"  # null position leaves the argument alone
    assert args[3] == "left"  # a missing position leaves the argument alone
    assert kwargs == {"k": "v"}  # keyword arguments are never converted


def test_input_conversions_skip_a_non_string_argument_and_an_empty_list_is_a_noop():
    pytest.importorskip("sympy")
    from resources_servers.critpt_custom_grader.runner import _apply_input_conversions

    cases = [([7, "x"], {})]
    # A "symbol" at a position holding a non-string leaves it unchanged.
    converted = _apply_input_conversions(cases, ["symbol", "symbol"])
    assert converted[0][0][0] == 7
    # An empty conversions list returns the cases object unchanged.
    assert _apply_input_conversions(cases, []) is cases


def test_adapt_output_keeps_a_float_free_symbolic_expression_as_symbolic_text():
    # An expression with no Float atom still takes the symbolic path unchanged.
    sympy = pytest.importorskip("sympy")
    result = adapt_output(sympy.Symbol("x") + sympy.Integer(1))
    assert type(result) is SymbolicText


def _materialize_symbolic(text):
    return materialize_input(decode_value(encode_value(SymbolicText(text))))


@pytest.mark.parametrize("name", ["pi", "E", "i", "I"])
def test_bare_reserved_symbolic_name_materializes_as_the_constant(name):
    # A symbolic input uses the comparator's parser, so a bare reserved name is the constant, not a free symbol.
    sympy = pytest.importorskip("sympy")
    from resources_servers.critpt_custom_grader.symbolic import parse_expression

    materialized = _materialize_symbolic(name)
    assert materialized == parse_expression(name)  # the comparator's reading of the same text
    assert materialized == _materialize_symbolic(f"({name})")  # bare and parenthesized agree
    assert not materialized.free_symbols  # a constant, not a free symbol
    assert materialized != sympy.Symbol(name)


def test_bare_symbolic_pi_input_is_the_numeric_constant():
    # A bare "pi" input is sympy.pi, so float(pi) succeeds where float(Symbol("pi")) raised.
    sympy = pytest.importorskip("sympy")
    assert _materialize_symbolic("pi") == sympy.pi
    assert float(_materialize_symbolic("pi")) == float(sympy.pi)


def test_ordinary_symbolic_name_stays_a_free_symbol():
    sympy = pytest.importorskip("sympy")
    assert _materialize_symbolic("x") == sympy.Symbol("x")
    assert _materialize_symbolic("y_1") == sympy.Symbol("y_1")


def test_nonfinite_and_grammar_rejected_symbolic_names_are_invalid_jobs():
    # A name the restricted grammar rejects is an invalid job, not a silently accepted free symbol.
    pytest.importorskip("sympy")
    for text in ("nan", "oo", "_x", "a__b"):
        with pytest.raises(RunnerError) as raised:
            _materialize_symbolic(text)
        assert raised.value.code == "invalid_job"
