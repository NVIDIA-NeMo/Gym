# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-safe codec unit tests: encode, decode and the wire bounds, on values built in-process.

No candidate code, worker subprocess or external service runs. Production codec execution runs inside
Daytona, but these tests do not require it. Hostile subprocess tests are in test_runner_remote.py.
"""

import json
import math
import sys
from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from resources_servers.critpt_custom_grader.codec import (
    ComplexValue,
    LegacyText,
    RepresentationError,
    SetValue,
    SymbolicText,
    decode_value,
    dump_value,
    encode_value,
    load_value,
)
from resources_servers.critpt_custom_grader.task_data import (
    MAX_CASES,
    MAX_DECIMAL_EXPONENT,
    MAX_DEPTH,
    MAX_ELEMENTS,
    MAX_INTEGER_DIGITS,
    MAX_POLICY_BYTES,
    MAX_STATEMENT_CHARS,
    MAX_TEXT_BYTES,
    MAX_WIRE_BYTES,
    WIRE_FORMAT,
    CasePolicy,
    TaskData,
    bounded_json,
)
from resources_servers.critpt_custom_grader.task_data import TestCase as CaseData


def wire(node):
    return {"format": WIRE_FORMAT, "value": node}


def task(**updates):
    data = {
        "problem_id": "synthetic-polynomial",
        "reference_source": "def solve():\n    return 6\n",
        "entrypoint": "solve",
        "test_cases": [{"input": [], "output": 6}],
    }
    data.update(updates)
    return data


@pytest.mark.parametrize("number", [0, -(10**400 + 73), 2**53 + 1, 10 ** (MAX_INTEGER_DIGITS - 1)])
def test_large_integers_are_string_tagged_before_json(number):
    encoded = encode_value(number)
    assert encoded["value"] == ["int", str(number)]
    assert decode_value(json.loads(json.dumps(encoded))) == number
    assert type(load_value(dump_value(number))) is int


@pytest.mark.parametrize(
    "kind,parts",
    [
        ("fraction", (10**300 + 9, 10**250 + 11)),
        ("fraction", (-7, 19)),
        ("decimal", ("12.3400",)),
        ("decimal", ("-0.000",)),
        ("decimal", ("1e-800",)),
    ],
)
def test_exact_fraction_and_decimal_roundtrip(kind, parts):
    value = Fraction(*parts) if kind == "fraction" else Decimal(*parts)
    decoded = load_value(dump_value(value))
    assert type(decoded) is type(value)
    assert decoded == value
    if type(value) is Decimal:
        assert decoded.as_tuple() == value.as_tuple()


@pytest.mark.parametrize("number", [-0.0, 0.1, "0x0.0000000000001p-1022", 1.7976931348623157e308])
def test_binary64_roundtrip_uses_exact_hex(number):
    if type(number) is str:
        number = float.fromhex(number)
    decoded = load_value(dump_value(number))
    assert decoded.hex() == number.hex()


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf, "NaN", "Infinity"])
def test_nonfinite_representation_is_lossless_and_json_safe(number):
    if type(number) is str:
        number = Decimal(number)
    encoded = encode_value(number)
    json.dumps(encoded, allow_nan=False)
    decoded = decode_value(encoded)
    assert type(decoded) is type(number)
    if type(number) is Decimal:
        assert str(decoded) == str(number)
    elif math.isnan(number):
        assert math.isnan(decoded)
    else:
        assert decoded == number


def test_complex_parts_do_not_narrow_exact_numbers():
    value = ComplexValue(Fraction(17, 31), Decimal("1e-700"))
    assert load_value(dump_value(value)) == value
    assert decode_value(encode_value(2 + 3j)) == ComplexValue(2.0, 3.0)


def test_composites_and_tag_looking_categories_roundtrip():
    value = {
        "format": WIRE_FORMAT,
        "value": ["int", "17"],
        "__complex__": [1, 2],
        "__nonfinite__": "inf",
        "category": '["symbolic", "x+1"]',
        "pair": (True, None),
    }
    assert decode_value(encode_value(value)) == value
    assert encode_value(value)["value"][0] == "map"
    assert decode_value(encode_value("pi")) == "pi"
    assert decode_value(encode_value(SymbolicText("pi"))) == SymbolicText("pi")
    assert decode_value(encode_value(LegacyText("1/7"))) == LegacyText("1/7")


def test_boolean_is_not_integer_one():
    assert encode_value(True)["value"] == ["bool", True]
    assert encode_value(1)["value"] == ["int", "1"]
    with pytest.raises(RepresentationError, match="invalid_boolean"):
        decode_value(wire(["bool", 1]))


@pytest.mark.parametrize(
    "node",
    [
        ["pickle", "anything"],
        ["call", "sqrt", "9"],
        ["int", "1", "extra"],
        ["int", "+1"],
        ["int", "-0"],
        ["fraction", "3", "0"],
        ["fraction", "3", "-4"],
        ["float", "1.0"],
        ["float", "0x1.0p+9999"],
        ["complex", ["bool", True], ["int", "0"]],
        ["nonfinite", "unknown"],
        ["map", [["a", ["null"]], ["a", ["null"]]]],
        ["map", [[1, ["null"]]]],
        ["list", {}],
        ["decimal", "sNaN"],
        ["str", "\ud800"],
    ],
)
def test_malformed_and_unknown_nodes_are_typed_errors(node):
    with pytest.raises(RepresentationError) as error:
        decode_value(wire(node))
    assert error.value.attributable


def test_wire_envelope_is_exact():
    with pytest.raises(RepresentationError, match="invalid_envelope"):
        decode_value({**wire(["null"]), "passed": True})
    with pytest.raises(RepresentationError, match="invalid_envelope"):
        decode_value({"format": "future-format", "value": ["null"]})


@pytest.mark.parametrize("depth", [MAX_DEPTH + 1, MAX_DEPTH + 50])
def test_deep_values_are_bounded_on_both_sides(depth):
    value, node = None, ["null"]
    for _ in range(depth):
        value, node = [value], ["list", [node]]
    with pytest.raises(RepresentationError):
        encode_value(value)
    with pytest.raises(RepresentationError):
        decode_value(wire(node))


def test_cycles_element_count_and_total_bytes_are_bounded():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(RepresentationError, match="structure_limit"):
        encode_value(cycle)
    assert len(decode_value(encode_value([None] * (MAX_ELEMENTS - 1)))) == MAX_ELEMENTS - 1
    with pytest.raises(RepresentationError, match="structure_limit"):
        encode_value([None] * MAX_ELEMENTS)
    with pytest.raises(RepresentationError, match="json_bounds"):
        encode_value(["z" * MAX_TEXT_BYTES] * (MAX_WIRE_BYTES // MAX_TEXT_BYTES + 1))
    with pytest.raises(RepresentationError, match="text_limit"):
        encode_value("z" * (MAX_TEXT_BYTES + 1))


def test_integer_and_exponent_limits_apply_before_conversion():
    with pytest.raises(RepresentationError, match="integer_limit"):
        encode_value(10**MAX_INTEGER_DIGITS)
    with pytest.raises(RepresentationError):
        encode_value(Fraction(10 ** (MAX_INTEGER_DIGITS * 5), 3))
    with pytest.raises(RepresentationError):
        decode_value(wire(["int", "9" * (MAX_INTEGER_DIGITS + 1)]))
    with pytest.raises(RepresentationError, match="invalid_decimal"):
        decode_value(wire(["decimal", f"1e{MAX_DECIMAL_EXPONENT + 1}"]))
    with pytest.raises(RepresentationError, match="invalid_decimal"):
        decode_value(wire(["decimal", "1e" + "9" * 200]))


def test_unrecognized_objects_are_not_inspected_or_coerced():
    class HostileMeta(type):
        def __eq__(cls, other):
            raise AssertionError("must not call user metaclass equality")

    class Hostile(metaclass=HostileMeta):
        def __getattribute__(self, name):
            raise AssertionError("must not inspect user attributes")

        def __str__(self):
            raise AssertionError("must not stringify user objects")

    with pytest.raises(RepresentationError, match="unsupported_type") as error:
        encode_value(Hostile())
    assert not error.value.attributable
    with pytest.raises(RepresentationError, match="invalid_type") as error:
        encode_value(b"raw bytes")
    assert error.value.attributable

    # An unsupported key type is refused without its __str__ being run. Whitelisted key types are
    # stringified (see test_non_string_keys_stringify_like_the_historical_grader).
    class HostileKey:
        def __str__(self):
            raise AssertionError("must not stringify user object keys")

    with pytest.raises(RepresentationError, match="invalid_text") as error:
        encode_value({HostileKey(): "value"})
    assert error.value.attributable


@pytest.mark.parametrize(
    "value, decoded",
    [
        ({0: "a", 1: "b"}, {"0": "a", "1": "b"}),  # int keys
        ({(0, 0): 1, (0, 1): 2}, {"(0, 0)": 1, "(0, 1)": 2}),  # tuple keys
        ({1.5: "x"}, {"1.5": "x"}),
        ({True: "t", None: "n"}, {"True": "t", "None": "n"}),
        ({Fraction(1, 3): "f"}, {"1/3": "f"}),
        ({Decimal("2.50"): "d"}, {"2.50": "d"}),
        ({complex(1, 2): "c"}, {"(1+2j)": "c"}),
    ],
)
def test_non_string_keys_stringify_like_the_historical_grader(value, decoded):
    # A dict keyed by tuples or ints transports with str() keys, and decode returns a plain dict of string keys.
    wire_value = encode_value(value)
    assert wire_value["value"][0] == "map"
    assert decode_value(wire_value) == decoded
    keys = [entry[0] for entry in wire_value["value"][1]]
    assert keys == list(decoded)


def test_int_keyed_maps_that_stringify_alike_compare_alike_and_apart():
    # Equal structures compare equal and different keys compare different.
    assert encode_value({0: "a", 1: "b"}) == encode_value({0: "a", 1: "b"})
    assert encode_value({0: "a", 1: "b"}) != encode_value({0: "a", 2: "b"})


def test_keys_that_collapse_to_one_wire_string_are_refused():
    # 1 and "1" are distinct Python keys but one wire string. encode refuses rather than emit a wire
    # decode would reject as a duplicate map key.
    with pytest.raises(RepresentationError, match="invalid_text") as error:
        encode_value({1: "a", "1": "b"})
    assert error.value.attributable


def test_a_frozenset_key_is_refused_not_stringified():
    # frozenset is outside the stringifiable-key whitelist: its str() order is not a stable comparison basis.
    with pytest.raises(RepresentationError, match="invalid_text"):
        encode_value({frozenset({1, 2}): "x"})


@pytest.mark.parametrize(
    "payload",
    [
        b'{"format":"critpt-value-v1","format":"critpt-value-v1","value":["null"]}',
        b'{"format":"critpt-value-v1","value":["int",1]}',
        b'{"format":"critpt-value-v1","value":["float",NaN]}',
        b"[" * 200 + b"]" * 200,
        b"\xff",
        b" " * (MAX_WIRE_BYTES + 1),
    ],
)
def test_json_loader_rejects_duplicate_keys_raw_numbers_depth_and_bytes(payload):
    with pytest.raises(RepresentationError):
        load_value(payload)


@pytest.fixture
def schema_transport_only(monkeypatch):
    blocked = MagicMock(side_effect=AssertionError("schema transport check attempted runtime work"))
    schema = sys.modules["resources_servers.critpt_custom_grader.task_data"]
    assert not hasattr(schema, "ast") and not hasattr(schema, "_input_literal")
    # Do not import runtime modules just to install guards.
    for name, functions in (
        (__name__, ("encode_value", "decode_value", "load_value", "dump_value")),
        ("resources_servers.critpt_custom_grader.codec", ("encode_value", "decode_value", "load_value", "dump_value")),
        (
            "resources_servers.critpt_custom_grader.runner",
            (
                "parse_source",
                "_input_literal",
                "_decode_input",
                "_decode_cases",
                "materialize_input",
                "_symbolic_input",
                "adapt_output",
                "_invoke",
                "_run_worker",
                "_compare_worker",
                "execute_run",
                "execute_compare",
            ),
        ),
        ("resources_servers.critpt_custom_grader.comparator", ("compare_request",)),
        ("resources_servers.critpt_custom_grader.symbolic", ("parse_expression",)),
    ):
        module = sys.modules.get(name)
        if module is not None:
            for function in functions:
                monkeypatch.setattr(module, function, blocked)
            if hasattr(module, "ast"):
                # Guard grader lookups, including saved function aliases, without breaking inspect or pytest.
                grader_ast = SimpleNamespace(**vars(module.ast))
                grader_ast.parse = grader_ast.literal_eval = blocked
                monkeypatch.setattr(module, "ast", grader_ast)
    yield
    blocked.assert_not_called()


def test_delivered_positional_and_expected_exception_shapes(schema_transport_only):
    positional = CaseData.model_validate({"input": [10**400], "output": {"a": "7/19"}})
    assert positional.args[0].value == ["int", str(10**400)]
    assert positional.expected.value.value == ["map", [["a", ["legacy", "7/19"]]]]
    exception = CaseData.model_validate({"input": [-1], "expected_error": "any rejection"})
    assert exception.args[0].value == ["int", "-1"]
    assert exception.expected.kind == "exception"
    assert exception.expected.message == "any rejection"


def test_keyword_literal_inputs_do_not_rewrite_stored_expectations(schema_transport_only):
    inputs = {"amount": "1.25", "flags": "[True, False]", "label": "'green'", "pair": "(2, 3)"}
    case = CaseData.model_validate({"inputs": inputs, "expected_output": "[1, 2]"})
    assert case.args == []
    assert list(case.kwargs) == list(inputs)
    assert {key: value.value for key, value in case.kwargs.items()} == {
        key: ["legacy", text] for key, text in inputs.items()
    }
    assert case.expected.value.value == ["legacy", "[1, 2]"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  pi\n",
        "sqrt(4)",
        "x+1",
        "'green'",
        "True",
        "False",
        "None",
        "-0.0",
        "9007199254740993",
        "[x for x in (1, 2)]",
        "(lambda: 1)()",
        "{'x': 1, 'x': 2}",
        "1e99999",
        "1e-999",
        "[0] * 999",
        "[",
        "(sqrt(4))",
        "1/7",
        "1_000",
        "'\\ud800'",
        "9" * (MAX_INTEGER_DIGITS + 1),
        "[" * (MAX_DEPTH + 1) + "0" + "]" * (MAX_DEPTH + 1),
        "[" + ",".join(["0"] * ((MAX_TEXT_BYTES - 1) // 2)) + "]",
    ],
)
def test_bounded_keyword_input_text_is_preserved_for_remote_validation(schema_transport_only, text):
    case = CaseData.model_validate({"inputs": {"x": text}, "expected_output": text})
    assert case.kwargs["x"].value == ["legacy", text]
    assert case.expected.value.value == ["legacy", text]
    positional = CaseData.model_validate({"input": [text], "output": text})
    assert positional.args[0].value == ["str", text]
    canonical = CaseData.model_validate(
        {"args": [wire(["str", text])], "kwargs": {"x": wire(["str", text])}, "expected": {"kind": "exception"}}
    )
    assert canonical.args[0].value == canonical.kwargs["x"].value == ["str", text]


def test_delivered_nested_strings_and_exact_builtins_are_transport_only(schema_transport_only):
    large = 10**400 + 73
    value = {"pair": (large, -0.0), "items": ["[1, 2]", True, None, 0.1], "label": "'green'"}
    node = [
        "map",
        [
            ["pair", ["tuple", [["int", str(large)], ["float", "-0x0.0p+0"]]]],
            ["items", ["list", [["str", "[1, 2]"], ["bool", True], ["null"], ["float", "0x1.999999999999ap-4"]]]],
            ["label", ["str", "'green'"]],
        ],
    ]
    positional = CaseData.model_validate({"input": [value], "output": "unchanged"})
    keyword = CaseData.model_validate({"inputs": {"x": value}, "expected_output": "unchanged"})
    assert positional.args[0].value == keyword.kwargs["x"].value == node
    assert positional.expected.value.value == keyword.expected.value.value == ["legacy", "unchanged"]


@pytest.mark.parametrize(
    "node",
    [["fraction", "7", "19"], ["decimal", "1.00000000000000000000001"], ["decimal", "1e-800"]],
)
def test_canonical_exact_numeric_tags_are_opaque(schema_transport_only, node):
    case = CaseData.model_validate(
        {"args": [wire(node)], "kwargs": {"x": wire(node)}, "expected": {"kind": "value", "value": wire(node)}}
    )
    assert case.args[0].value == case.kwargs["x"].value == case.expected.value.value == node


@pytest.mark.parametrize("name", ["", "class", "for", "x-y", "obj.x", "1x", "_x", "x" * 129, 1])
def test_invalid_delivered_keyword_names_are_rejected(schema_transport_only, name):
    with pytest.raises(ValidationError):
        CaseData.model_validate({"inputs": {name: "["}, "expected_output": "unchanged"})


@pytest.mark.parametrize("value", [b"1", {1}, {1: "x"}, object()])
def test_delivered_unsupported_native_types_are_rejected(schema_transport_only, value):
    for case in ({"inputs": {"x": value}, "expected_output": 1}, {"input": [value], "output": 1}):
        with pytest.raises(ValidationError, match="plain values|bounded string keys"):
            CaseData.model_validate(case)


def test_delivered_subclasses_are_not_coerced(schema_transport_only):
    class Text(str):
        def __str__(self):
            raise AssertionError("must not coerce delivered subclasses")

    with pytest.raises(ValidationError, match="plain values"):
        CaseData.model_validate({"inputs": {"x": Text("1.25")}, "expected_output": 1})


def test_delivered_transport_bounds_remain_enforced(schema_transport_only):
    for text in ("x" * MAX_TEXT_BYTES, "\U0001f642" * (MAX_TEXT_BYTES // 4)):
        case = CaseData.model_validate({"inputs": {"x": text}, "expected_output": text})
        assert case.kwargs["x"].value == case.expected.value.value == ["legacy", text]
    for text in ("x" * (MAX_TEXT_BYTES + 1), "\U0001f642" * (MAX_TEXT_BYTES // 4 + 1)):
        with pytest.raises(ValidationError, match="string size limit"):
            CaseData.model_validate({"inputs": {"x": text}, "expected_output": 1})
    deep = None
    for _ in range(MAX_DEPTH + 1):
        deep = [deep]
    for value in (deep, [None] * MAX_ELEMENTS, 10**MAX_INTEGER_DIGITS):
        with pytest.raises(ValidationError, match="structure limit|integer size limit"):
            CaseData.model_validate({"inputs": {"x": value}, "expected_output": 1})
    with pytest.raises(ValidationError, match="JSON byte"):
        CaseData.model_validate(
            {"inputs": {"x": ["x" * MAX_TEXT_BYTES] * (MAX_WIRE_BYTES // MAX_TEXT_BYTES + 1)}, "expected_output": 1}
        )
    with pytest.raises(ValidationError, match="bounded keyword map"):
        CaseData.model_validate({"inputs": {f"x{index}": "1" for index in range(129)}, "expected_output": 1})
    with pytest.raises(ValidationError, match="bounded positional list"):
        CaseData.model_validate({"input": [0] * 129, "output": 1})


@pytest.mark.parametrize(
    "case",
    [
        {"input": [], "output": 1, "expected_error": "bad"},
        {"inputs": {}, "input": [], "expected_output": 1},
        {"input": [], "expected_output": 1},
        {"args": [], "expected_error": "bad"},
        {"input": []},
    ],
)
def test_ambiguous_or_missing_expectations_are_rejected(schema_transport_only, case):
    with pytest.raises(ValidationError):
        CaseData.model_validate(case)


def test_task_aliases_and_explicit_reference_entrypoint(schema_transport_only):
    data = task()
    data["reference_code"] = data.pop("reference_source")
    data["entry"] = data.pop("entrypoint")
    data["testcases"] = data.pop("test_cases")
    validated = TaskData.model_validate(data)
    assert validated.effective_reference_entrypoint == "solve"
    assert "reference_code" not in validated.model_dump()
    separate_entry = TaskData.model_validate(task(reference_entrypoint="reference_solve"))
    assert separate_entry.effective_reference_entrypoint == "reference_solve"
    data["entrypoint"] = "solve"
    with pytest.raises(ValidationError):
        TaskData.model_validate(data)


@pytest.mark.parametrize("extra", [{"network": True}, {"image": "chosen"}, {"timeout": 10}, {"schema_version": True}])
def test_tasks_cannot_select_operator_runtime_policy(extra):
    with pytest.raises(ValidationError):
        TaskData.model_validate(task(**extra))


def test_entrypoint_is_never_inferred_and_case_count_is_bounded():
    data = task(code_template="def other():\n    pass\n")
    del data["entrypoint"]
    with pytest.raises(ValidationError):
        TaskData.model_validate(data)
    with pytest.raises(ValidationError):
        TaskData.model_validate(task(entrypoint="obj.solve"))
    with pytest.raises(ValidationError):
        TaskData.model_validate(task(test_cases=[]))
    with pytest.raises(ValidationError):
        TaskData.model_validate(task(test_cases=[{"input": [], "output": 1}] * (MAX_CASES + 1)))


def test_tolerance_paths_and_case_overrides_are_explicit():
    validated = TaskData.model_validate(
        task(
            rtol="1e-9",
            tolerances=[{"path": "root", "rtol": "1e-8"}],
            test_cases=[{"input": [], "output": 1, "tolerances": [{"path": ".", "rtol": 0, "atol": "0.01"}]}],
        )
    )
    assert validated.comparison_policy(0) == {
        "rtol": "1e-9",
        "atol": None,
        "tolerances": [{"path": "", "rtol": "0", "atol": "0.01", "quantity": None}],
        "statement": None,
    }
    with pytest.raises(ValidationError):
        TaskData.model_validate(task(tolerances=[{"path": "root"}, {"path": "/"}]))
    with pytest.raises(ValidationError):
        TaskData.model_validate(task(tolerances=[{"path": "/bad~2escape"}]))


def test_case_policy_transports_verbatim_statement_and_decimal_text_only():
    statement = "  Report slope to 2048 decimal places.\nThe notation is sqrt(.\n"
    validated = TaskData.model_validate(
        task(
            problem=statement,
            rtol="1.230000000000000000000001e-20",
            atol="0.000",
            tolerances=[{"path": "/slope", "quantity": "a descriptive label", "atol": "1.000e-700"}],
            test_cases=[{"input": [], "output": {"slope": "1/13"}}],
        )
    )
    policy = validated.comparison_policy(0)
    assert policy == {
        "rtol": "1.230000000000000000000001e-20",
        "atol": "0.000",
        "tolerances": [{"path": "/slope", "quantity": "a descriptive label", "rtol": None, "atol": "1.000e-700"}],
        "statement": statement,
    }
    # comparison_policy omits the server-default keys, filled with their None default on a round trip.
    assert CasePolicy.model_validate(policy).model_dump(mode="json") == {
        **policy,
        "default_rtol": None,
        "default_atol": None,
    }
    assert validated.test_cases[0].expected.value.value == ["map", [["slope", ["legacy", "1/13"]]]]


@pytest.mark.parametrize("character", ["a", "\U0001f7e2", "\x00"])
def test_maximum_statement_is_transportable_including_json_escaping(character):
    statement = character * MAX_STATEMENT_CHARS
    validated = TaskData.model_validate(task(problem=statement))
    policy = validated.comparison_policy(0)
    payload = json.dumps(policy, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(payload) <= MAX_POLICY_BYTES
    bounded_json(policy, max_bytes=MAX_POLICY_BYTES)
    restored = CasePolicy.model_validate(json.loads(payload))
    assert restored.statement == statement
    assert restored.rtol is None
    assert restored.atol is None
    assert restored.tolerances == []


@pytest.mark.parametrize("statement", ["x" * (MAX_STATEMENT_CHARS + 1), True, 12, {"text": "accuracy"}])
def test_statement_transport_rejects_oversize_and_nontext_values(statement):
    with pytest.raises(ValidationError):
        TaskData.model_validate(task(problem=statement))
    with pytest.raises(ValidationError):
        CasePolicy.model_validate({"statement": statement})


def test_case_overrides_replace_whole_entries_without_resolving_or_flattening():
    statement = "Report rate with relative error below 2e-8."
    validated = TaskData.model_validate(
        task(
            problem=statement,
            rtol="0.1",
            atol="2",
            tolerances=[{"path": "/rate", "rtol": "1e-9"}, {"path": "/bias", "rtol": "0", "atol": "1e-30"}],
            test_cases=[
                {"input": [], "output": {"rate": 1, "bias": 2}, "tolerances": [{"path": "rate", "atol": "0"}]},
                {"input": [], "output": {"rate": 1, "bias": 2}},
            ],
        )
    )
    first, second = validated.comparison_policy(0), validated.comparison_policy(1)
    assert first["tolerances"] == [
        {"path": "/rate", "rtol": None, "atol": "0", "quantity": None},
        {"path": "/bias", "rtol": "0", "atol": "1e-30", "quantity": None},
    ]
    assert second["tolerances"] == [
        {"path": "/rate", "rtol": "1e-9", "atol": None, "quantity": None},
        {"path": "/bias", "rtol": "0", "atol": "1e-30", "quantity": None},
    ]
    for policy in (first, second):
        assert policy["rtol"] == "0.1"
        assert policy["atol"] == "2"
        assert policy["statement"] == statement
        assert CasePolicy.model_validate(policy).model_dump(mode="json") == {
            **policy,
            "default_rtol": None,
            "default_atol": None,
        }


@pytest.mark.parametrize("rtol", [None, "0", 0, "0.000", "5.000e-12"])
def test_leaf_policy_transport_keeps_omitted_and_explicit_zero_distinct(rtol):
    entry = {"path": "/", "atol": "0"}
    if rtol is not None:
        entry["rtol"] = rtol
    validated = TaskData.model_validate(task(tolerances=[entry]))
    transported = validated.comparison_policy(0)["tolerances"][0]
    assert transported["rtol"] == (None if rtol is None else str(rtol))
    assert transported["atol"] == "0"
    assert transported["path"] == ""


@pytest.mark.parametrize(
    "policy",
    [
        {"named": {"rate": "1e-8"}},
        {"statement": {"rtol": "1e-8"}},
        {"tolerances": [{"quantity": "rate", "rtol": "1e-8"}]},
        {"tolerances": [{"path": "/rate", "problem": "main", "rtol": "1e-8"}]},
        {"tolerances": [{"path": "root", "rtol": "0"}, {"path": "/", "atol": "0"}]},
    ],
)
def test_unsupported_policy_metadata_is_not_silently_transported(policy):
    with pytest.raises(ValidationError):
        CasePolicy.model_validate(policy)


@pytest.mark.parametrize(
    "value",
    [
        set(),
        {1, 2, 3},
        frozenset({"a", "b", "c"}),
        {(1, "x"), (2, "y")},
        SetValue((1, 2, 3)),
    ],
)
def test_set_values_roundtrip_as_a_set_carrier(value):
    # A set, frozenset and SetValue all encode to the "set" node. Member order carries no meaning.
    encoded = encode_value(value)
    assert encoded["value"][0] == "set"
    decoded = decode_value(encoded)
    assert type(decoded) is SetValue
    members = value.items if type(value) is SetValue else value
    assert set(decoded.items) == set(members)
    assert len(decoded.items) == len(members)


def test_nested_set_inside_a_list_roundtrips():
    decoded = decode_value(encode_value([{1, 2}, 3]))
    assert type(decoded[0]) is SetValue
    assert set(decoded[0].items) == {1, 2}
    assert decoded[1] == 3


def test_set_with_unhashable_carrier_members_still_encodes():
    # A SetValue may hold members a plain set could not, such as ComplexValue carriers, since it holds a tuple.
    value = SetValue((ComplexValue(1, 2), ComplexValue(3, 4)))
    decoded = decode_value(encode_value(value))
    assert type(decoded) is SetValue
    assert set(decoded.items) == {ComplexValue(1, 2), ComplexValue(3, 4)}


def test_a_set_over_the_element_budget_is_a_structure_limit():
    with pytest.raises(RepresentationError, match="structure_limit"):
        encode_value(set(range(MAX_ELEMENTS + 1)))


def test_a_set_node_over_the_element_budget_is_rejected_on_decode():
    wire = {"format": WIRE_FORMAT, "value": ["set", [["int", str(i)] for i in range(MAX_ELEMENTS + 1)]]}
    with pytest.raises(RepresentationError):
        decode_value(wire)


def test_set_node_arity_is_enforced():
    wire = {"format": WIRE_FORMAT, "value": ["set", [["int", "1"]], "extra"]}
    with pytest.raises(RepresentationError, match="invalid_arity"):
        decode_value(wire)
