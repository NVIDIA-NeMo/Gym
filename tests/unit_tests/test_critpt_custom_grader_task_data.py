# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core-suite tests for resources_servers/critpt_custom_grader/task_data.py.

tests/unit_tests/test_task_data.py imports every server's task_data.py, so the core job measures the schema module
against its coverage gate ([tool.coverage.report] fail_under). The server's own tests under
resources_servers/critpt_custom_grader/tests/ are not collected there (testpaths = ["tests"]), so this file covers
the schema's bounds, delivered row shapes and validators in the core suite, on synthetic values.
"""

import math
from collections import OrderedDict
from decimal import Decimal

import pytest
from pydantic import ValidationError

from resources_servers.critpt_custom_grader import task_data as schema
from resources_servers.critpt_custom_grader.task_data import (
    MAX_CASES,
    MAX_DECIMAL_EXPONENT,
    MAX_DEPTH,
    MAX_ELEMENTS,
    MAX_INTEGER_DIGITS,
    MAX_STATEMENT_CHARS,
    MAX_TASK_BYTES,
    MAX_TEXT_BYTES,
    MAX_WIRE_BYTES,
    WIRE_FORMAT,
    CasePolicy,
    ComparisonPolicy,
    ExceptionExpectation,
    LeafTolerance,
    TaskData,
    Tolerance,
    ValueExpectation,
    WireValue,
    bounded_json,
    decimal_text,
    normalize_path,
    tolerance_text,
)
from resources_servers.critpt_custom_grader.task_data import TestCase as CaseData


class Label(str):
    """A str subclass: delivered values must be exact built-ins."""


class Count(int):
    """An int subclass: delivered values must be exact built-ins."""


def wire(node):
    return {"format": WIRE_FORMAT, "value": node}


def canonical_case(**updates):
    case = {"args": [wire(["int", "1"])], "expected": {"kind": "value", "value": wire(["int", "1"])}}
    case.update(updates)
    return case


def task(**updates):
    row = {
        "problem_id": "synthetic-identity",
        "reference_source": "def solve(x):\n    return x\n",
        "entrypoint": "solve",
        "test_cases": [canonical_case()],
    }
    row.update(updates)
    return row


def nested_lists(depth):
    value = None
    for _ in range(depth):
        value = [value]
    return value


class TestBoundedJson:
    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            0,
            2**63 - 1,
            -(2**63 - 1),
            1.5,
            -0.0,
            "",
            "plain text",
            "é\U0001f642",
            [],
            {},
            {"key": [1, 2.5, None, False, "text", {"inner": []}]},
        ],
    )
    def test_accepts_exact_json_values(self, value):
        assert bounded_json(value) is None

    @pytest.mark.parametrize(
        "value, limits, message",
        [
            pytest.param(nested_lists(3), {"max_depth": 2}, "JSON structure limit", id="depth"),
            pytest.param("x" * 11, {"max_bytes": 10}, "JSON byte limit", id="long-string"),
            pytest.param({1: "one"}, {}, "JSON keys must be strings", id="int-key"),
            pytest.param({"k" * 11: 0}, {"max_bytes": 10}, "JSON byte limit", id="long-key"),
            pytest.param({"abcd": 0, "efgh": 0}, {"max_bytes": 10}, "JSON byte limit", id="keys-add-up"),
            pytest.param(2**63, {}, "large JSON integers require a wire tag", id="large-int"),
            pytest.param(-(2**63), {}, "large JSON integers require a wire tag", id="large-negative-int"),
            pytest.param(math.nan, {}, "nonfinite JSON numbers require a wire tag", id="nan"),
            pytest.param(-math.inf, {}, "nonfinite JSON numbers require a wire tag", id="negative-infinity"),
            pytest.param((1, 2), {}, "unsupported JSON type", id="tuple"),
            pytest.param(b"raw", {}, "unsupported JSON type", id="bytes"),
            pytest.param(Decimal("1.5"), {}, "unsupported JSON type", id="decimal"),
            pytest.param([1, 2], {"max_bytes": 30}, "JSON byte or structure limit", id="scalars-add-up"),
            pytest.param("é" * 6, {"max_bytes": 10}, "JSON byte or structure limit", id="multibyte-text"),
            pytest.param("abcd", {"max_bytes": 5}, "JSON byte limit", id="serialized-quotes"),
            pytest.param("\x00" * 3, {"max_bytes": 10}, "JSON byte limit", id="serialized-escapes"),
        ],
    )
    def test_rejects_values_beyond_a_limit(self, value, limits, message):
        with pytest.raises(ValueError, match=message):
            bounded_json(value, **limits)

    @pytest.mark.parametrize(
        "value, message",
        [
            pytest.param([1, 2, 3], "JSON structure limit", id="node-count"),
            pytest.param([0, 0, 0, 0], "JSON structure limit", id="list-length"),
            pytest.param(dict.fromkeys("abcd", 0), "JSON structure limit", id="map-size"),
            pytest.param([[1, 2, 3], [4, 5, 6]], "JSON byte or structure limit", id="pending-nodes"),
        ],
    )
    def test_node_caps(self, monkeypatch, value, message):
        # The real cap, MAX_JSON_NODES, is about a million nodes; a cap of three reaches the same checks cheaply.
        monkeypatch.setattr(schema, "MAX_JSON_NODES", 3)
        assert bounded_json([1, 2]) is None
        with pytest.raises(ValueError, match=message):
            bounded_json(value)


class TestDecimalText:
    @pytest.mark.parametrize(
        "text",
        [
            "0",
            "-0",
            "+7",
            "12.500",
            "1.",
            ".5",
            "-.5e-3",
            "1e5",
            "2.5E+10",
            "1e+0001",
            f"1e{MAX_DECIMAL_EXPONENT}",
            f"1e-{MAX_DECIMAL_EXPONENT}",
            f"1.5e{MAX_DECIMAL_EXPONENT}",
            pytest.param("9" * MAX_INTEGER_DIGITS, id="max-digits"),
            pytest.param("." + "1" * MAX_INTEGER_DIGITS, id="max-places"),
        ],
    )
    def test_accepts_bounded_decimal_text_unchanged(self, text):
        assert decimal_text(text) == text

    @pytest.mark.parametrize(
        "text",
        [
            "",
            " 1",
            "1 ",
            "1\n",
            ".",
            "+",
            "1.2.3",
            "e5",
            "1e",
            "1e+",
            "0x10",
            "1_000",
            "1,5",
            "nan",
            "-inf",
            "١",
            pytest.param("1" * (MAX_INTEGER_DIGITS + 17), id="too-long"),
            pytest.param(1, id="int"),
            pytest.param(1.5, id="float"),
            pytest.param(None, id="none"),
            pytest.param(b"1", id="bytes"),
            pytest.param(Decimal("1"), id="decimal"),
        ],
    )
    def test_rejects_malformed_text_and_non_text(self, text):
        with pytest.raises(ValueError, match="invalid decimal"):
            decimal_text(text)

    @pytest.mark.parametrize(
        "text",
        [pytest.param("9" * (MAX_INTEGER_DIGITS + 1), id="too-many-digits"), "1e00001", "1e+10000", "-1E-00001"],
    )
    def test_rejects_too_many_digits_or_exponent_digits(self, text):
        with pytest.raises(ValueError, match="decimal size limit"):
            decimal_text(text)

    @pytest.mark.parametrize(
        "text",
        [
            f"1e{MAX_DECIMAL_EXPONENT + 1}",
            f"1e-{MAX_DECIMAL_EXPONENT + 1}",
            "1e9999",
            f"1.5e-{MAX_DECIMAL_EXPONENT}",
            pytest.param("." + "1" * 1000 + "e-25", id="places-plus-exponent"),
        ],
    )
    def test_rejects_scales_beyond_the_exponent_cap(self, text):
        with pytest.raises(ValueError, match="decimal exponent limit"):
            decimal_text(text)


class TestToleranceText:
    def test_none_means_unset(self):
        assert tolerance_text(None) is None

    @pytest.mark.parametrize(
        "value, text",
        [
            pytest.param("0", "0", id="str-zero"),
            ("0.000", "0.000"),
            ("5.000e-12", "5.000e-12"),
            (".5", ".5"),
            pytest.param(0, "0", id="int-zero"),
            (3, "3"),
            pytest.param(10**1000, "1" + "0" * 1000, id="big-int"),
            (0.25, "0.25"),
            (1e-5, "1e-05"),
            (1e300, "1e+300"),
        ],
    )
    def test_accepts_finite_nonnegative_decimals_as_exact_text(self, value, text):
        assert tolerance_text(value) == text

    @pytest.mark.parametrize(
        "value, message",
        [
            (True, "finite nonnegative decimal"),
            (False, "finite nonnegative decimal"),
            (Decimal("0.1"), "finite nonnegative decimal"),
            ([1], "finite nonnegative decimal"),
            pytest.param(b"1", "finite nonnegative decimal", id="bytes"),
            pytest.param(2 ** (MAX_INTEGER_DIGITS * 4), "tolerance size limit", id="int-bits"),
            pytest.param(10 ** (MAX_INTEGER_DIGITS + 100), "invalid decimal", id="int-text-length"),
            pytest.param("-1", "negative tolerance", id="str-negative"),
            pytest.param(-1, "negative tolerance", id="int-negative"),
            (-0.5, "negative tolerance"),
            ("-1e-9", "negative tolerance"),
            (math.inf, "invalid decimal"),
            (-math.inf, "invalid decimal"),
            (math.nan, "invalid decimal"),
            ("NaN", "invalid decimal"),
            ("1e-2000", "decimal exponent limit"),
        ],
    )
    def test_rejects_other_types_large_negative_and_nonfinite_values(self, value, message):
        with pytest.raises(ValueError, match=message):
            tolerance_text(value)


class TestNormalizePath:
    @pytest.mark.parametrize(
        "path, normalized",
        [
            ("", ""),
            ("/", ""),
            (".", ""),
            ("root", ""),
            ("  root\n", ""),
            (" / ", ""),
            ("rate", "/rate"),
            ("/rate", "/rate"),
            ("a/0/b", "/a/0/b"),
            (" /x ", "/x"),
            ("/a~0b/c~1d", "/a~0b/c~1d"),
        ],
    )
    def test_root_aliases_and_relative_paths(self, path, normalized):
        assert normalize_path(path) == normalized

    @pytest.mark.parametrize("path", ["/a~2b", "/a~", "~", "x~y"])
    def test_rejects_invalid_pointer_escapes(self, path):
        with pytest.raises(ValueError, match="invalid JSON Pointer escape"):
            normalize_path(path)


class TestLegacyWire:
    @pytest.mark.parametrize(
        "value, node",
        [
            (None, ["null"]),
            (True, ["bool", True]),
            (False, ["bool", False]),
            (0, ["int", "0"]),
            (-42, ["int", "-42"]),
            pytest.param(
                -(10 ** (MAX_INTEGER_DIGITS - 1)), ["int", "-1" + "0" * (MAX_INTEGER_DIGITS - 1)], id="max-digits"
            ),
            (0.5, ["float", "0x1.0000000000000p-1"]),
            (-0.0, ["float", "-0x0.0p+0"]),
            (math.nan, ["nonfinite", "nan"]),
            (math.inf, ["nonfinite", "inf"]),
            (-math.inf, ["nonfinite", "-inf"]),
            ("text", ["str", "text"]),
            ([1, "a"], ["list", [["int", "1"], ["str", "a"]]]),
            ((1, None), ["tuple", [["int", "1"], ["null"]]]),
            ({"a": 1, "b": [True]}, ["map", [["a", ["int", "1"]], ["b", ["list", [["bool", True]]]]]]),
            ({"__complex__": [1, 2.5]}, ["complex", ["int", "1"], ["float", "0x1.4000000000000p+1"]]),
            ({"__complex__": [-0.5, 0]}, ["complex", ["float", "-0x1.0000000000000p-1"], ["int", "0"]]),
        ],
    )
    def test_encodes_each_exact_builtin(self, value, node):
        assert schema._legacy_wire(value) == {"format": WIRE_FORMAT, "value": node}

    def test_legacy_strings_tag_every_nested_string_but_not_map_keys(self):
        encoded = schema._legacy_wire(["a", {"k": "v"}, ("t",)], legacy_strings=True)
        assert encoded["value"] == [
            "list",
            [["legacy", "a"], ["map", [["k", ["legacy", "v"]]]], ["tuple", [["legacy", "t"]]]],
        ]

    @pytest.mark.parametrize(
        "parts, node",
        [
            ([True, 1], ["list", [["bool", True], ["int", "1"]]]),
            ([1.0, math.nan], ["list", [["float", "0x1.0000000000000p+0"], ["nonfinite", "nan"]]]),
            ([1, 2, 3], ["list", [["int", "1"], ["int", "2"], ["int", "3"]]]),
            ((1, 2), ["tuple", [["int", "1"], ["int", "2"]]]),
            (["1", "2"], ["list", [["str", "1"], ["str", "2"]]]),
        ],
    )
    def test_complex_carrier_needs_a_list_of_two_finite_reals(self, parts, node):
        assert schema._legacy_wire({"__complex__": parts})["value"] == ["map", [["__complex__", node]]]

    def test_complex_carrier_beside_another_key_stays_a_map(self):
        encoded = schema._legacy_wire({"__complex__": [1, 2], "tag": "x"})["value"]
        assert encoded == ["map", [["__complex__", ["list", [["int", "1"], ["int", "2"]]]], ["tag", ["str", "x"]]]]

    def test_largest_accepted_depth_and_element_count(self):
        expected = ["null"]
        for _ in range(MAX_DEPTH):
            expected = ["list", [expected]]
        assert schema._legacy_wire(nested_lists(MAX_DEPTH))["value"] == expected
        encoded = schema._legacy_wire([None] * (MAX_ELEMENTS - 1))["value"]
        assert encoded[0] == "list"
        assert len(encoded[1]) == MAX_ELEMENTS - 1

    @pytest.mark.parametrize(
        "value, message",
        [
            pytest.param(2 ** (MAX_INTEGER_DIGITS * 4), "integer size limit", id="int-bits"),
            pytest.param(10**MAX_INTEGER_DIGITS, "integer size limit", id="int-digits"),
            pytest.param(-(10**MAX_INTEGER_DIGITS), "integer size limit", id="negative-int-digits"),
            pytest.param("x" * (MAX_TEXT_BYTES + 1), "string size limit", id="text-chars"),
            pytest.param("é" * (MAX_TEXT_BYTES // 2 + 1), "string size limit", id="text-bytes"),
            pytest.param(nested_lists(MAX_DEPTH + 1), "value structure limit", id="depth"),
            pytest.param([None] * MAX_ELEMENTS, "value structure limit", id="element-count"),
            pytest.param([None] * (MAX_ELEMENTS + 1), "value structure limit", id="list-length"),
            pytest.param((None,) * (MAX_ELEMENTS + 1), "value structure limit", id="tuple-length"),
            pytest.param(dict.fromkeys(map(str, range(MAX_ELEMENTS + 1))), "value structure limit", id="map-size"),
            pytest.param({1: "one"}, "bounded string keys", id="int-key"),
            pytest.param({("a",): 1}, "bounded string keys", id="tuple-key"),
            pytest.param({"k" * (MAX_TEXT_BYTES + 1): 0}, "bounded string keys", id="key-chars"),
            pytest.param({"é" * (MAX_TEXT_BYTES // 2 + 1): 0}, "bounded string keys", id="key-bytes"),
            pytest.param(b"raw", "plain values", id="bytes"),
            pytest.param({1, 2}, "plain values", id="set"),
            pytest.param(Decimal("1.5"), "plain values", id="decimal"),
            pytest.param(complex(1, 2), "plain values", id="complex"),
            pytest.param(object(), "plain values", id="object"),
            pytest.param(Label("text"), "plain values", id="str-subclass"),
            pytest.param(Count(3), "plain values", id="int-subclass"),
            pytest.param(OrderedDict(a=1), "plain values", id="dict-subclass"),
            pytest.param(
                ["x" * MAX_TEXT_BYTES] * (MAX_WIRE_BYTES // MAX_TEXT_BYTES + 1), "JSON byte", id="wire-bytes"
            ),
        ],
    )
    def test_rejects_values_beyond_the_transport_bounds(self, value, message):
        with pytest.raises(ValueError, match=message):
            schema._legacy_wire(value)


class TestWireValue:
    def test_validates_a_canonical_envelope(self):
        value = WireValue.model_validate(wire(["map", [["k", ["int", "1"]]]]))
        assert value.format == WIRE_FORMAT == "critpt-value-v1"
        assert value.value == ["map", [["k", ["int", "1"]]]]
        assert WireValue.model_validate({"value": ["null"]}).format == WIRE_FORMAT

    def test_an_existing_instance_is_used_as_is(self):
        existing = WireValue(value=["bool", True])
        assert WireValue.model_validate(existing) is existing
        assert ValueExpectation(kind="value", value=existing).value is existing

    @pytest.mark.parametrize(
        "payload, message",
        [
            pytest.param({"format": "critpt-value-v2", "value": ["null"]}, None, id="other-format"),
            pytest.param({"value": ["null"], "note": "x"}, None, id="extra-key"),
            pytest.param({"value": "null"}, None, id="value-not-a-list"),
            pytest.param({}, None, id="missing-value"),
            pytest.param(["int", "1"], None, id="not-an-envelope"),
            pytest.param(wire(("int", "1")), "unsupported JSON type", id="tuple-node"),
            pytest.param(wire(["int", 2**64]), "wire tag", id="raw-large-int"),
            pytest.param(wire(["float", math.inf]), "wire tag", id="raw-infinity"),
            pytest.param(wire(["str", "x" * MAX_WIRE_BYTES]), "JSON byte", id="oversized"),
        ],
    )
    def test_rejects_malformed_or_unbounded_envelopes(self, payload, message):
        with pytest.raises(ValidationError, match=message):
            WireValue.model_validate(payload)


class TestToleranceModels:
    def test_tolerance_knobs_are_exact_text(self):
        tolerance = Tolerance.model_validate({"rtol": 1e-5, "atol": 0})
        assert (tolerance.rtol, tolerance.atol) == ("1e-05", "0")
        assert Tolerance.model_validate({"rtol": None, "atol": "0.000"}).model_dump() == {
            "rtol": None,
            "atol": "0.000",
        }
        assert Tolerance().model_dump() == {"rtol": None, "atol": None}

    @pytest.mark.parametrize(
        "payload, message",
        [
            pytest.param({"rtol": True}, "finite nonnegative decimal", id="bool"),
            pytest.param({"atol": "-1e-9"}, "negative tolerance", id="negative"),
            pytest.param({"rtol": "1e-5", "unit": "m"}, None, id="extra-key"),
        ],
    )
    def test_tolerance_rejects_other_types_negatives_and_extra_keys(self, payload, message):
        with pytest.raises(ValidationError, match=message):
            Tolerance.model_validate(payload)

    @pytest.mark.parametrize(
        "payload, path",
        [
            ({"path": "rate", "rtol": "1e-3"}, "/rate"),
            ({"path": " root ", "atol": 0}, ""),
            ({"path": "/a~1b/0", "rtol": "0", "atol": "0", "quantity": "a synthetic label"}, "/a~1b/0"),
        ],
    )
    def test_leaf_paths_are_canonical_json_pointers(self, payload, path):
        assert LeafTolerance.model_validate(payload).path == path

    @pytest.mark.parametrize(
        "payload, message",
        [
            pytest.param({"path": "/rate"}, "requires rtol or atol", id="no-override"),
            pytest.param({"path": "/rate", "rtol": None, "atol": None}, "requires rtol or atol", id="explicit-nones"),
            pytest.param({"path": "/rate~2", "rtol": "0"}, "invalid JSON Pointer escape", id="bad-escape"),
            pytest.param({"path": "/" + "x" * 1024, "rtol": "0"}, None, id="long-path"),
            pytest.param({"path": "/rate", "rtol": "0", "quantity": "q" * 257}, None, id="long-quantity"),
            pytest.param({"quantity": "rate", "rtol": "0"}, None, id="quantity-is-not-a-path"),
        ],
    )
    def test_leaf_entries_need_a_path_and_an_explicit_override(self, payload, message):
        with pytest.raises(ValidationError, match=message):
            LeafTolerance.model_validate(payload)

    def test_policy_paths_are_unique_after_normalization(self):
        policy = ComparisonPolicy.model_validate(
            {"rtol": "0.1", "tolerances": [{"path": "/a", "atol": "0"}, {"path": "b", "rtol": "0"}]}
        )
        assert [entry.path for entry in policy.tolerances] == ["/a", "/b"]
        with pytest.raises(ValidationError, match="duplicate tolerance path"):
            ComparisonPolicy.model_validate(
                {"tolerances": [{"path": "root", "rtol": "0"}, {"path": "/", "atol": "0"}]}
            )


class TestCasePolicy:
    def test_transports_the_statement_and_operator_defaults(self):
        policy = CasePolicy.model_validate(
            {
                "rtol": "0",
                "tolerances": [{"path": "/x", "atol": "0.5"}],
                "statement": "A synthetic statement.",
                "default_rtol": 1e-5,
                "default_atol": "1e-8",
            }
        )
        assert policy.statement == "A synthetic statement."
        assert (policy.default_rtol, policy.default_atol) == ("1e-05", "1e-8")
        assert CasePolicy.model_validate({"default_rtol": None}).default_rtol is None
        assert CasePolicy().model_dump() == {
            "rtol": None,
            "atol": None,
            "tolerances": [],
            "statement": None,
            "default_rtol": None,
            "default_atol": None,
        }

    @pytest.mark.parametrize(
        "statement", [pytest.param("", id="empty"), pytest.param("x" * MAX_STATEMENT_CHARS, id="max-length")]
    )
    def test_statement_up_to_the_bound(self, statement):
        assert CasePolicy.model_validate({"statement": statement}).statement == statement

    @pytest.mark.parametrize(
        "payload, message",
        [
            pytest.param({"default_rtol": "-1"}, "negative tolerance", id="negative-default"),
            pytest.param({"default_atol": False}, "finite nonnegative decimal", id="bool-default"),
            pytest.param({"default_rtol": math.nan}, "invalid decimal", id="nan-default"),
            pytest.param({"statement": "x" * (MAX_STATEMENT_CHARS + 1)}, None, id="long-statement"),
            pytest.param({"statement": 12}, None, id="number-statement"),
            pytest.param({"statement": {"text": "synthetic"}}, None, id="map-statement"),
        ],
    )
    def test_rejects_bad_defaults_and_statements(self, payload, message):
        with pytest.raises(ValidationError, match=message):
            CasePolicy.model_validate(payload)


class TestExpectations:
    def test_value_expectation_wraps_a_wire_value(self):
        expectation = ValueExpectation.model_validate({"kind": "value", "value": wire(["null"])})
        assert expectation.value.value == ["null"]

    @pytest.mark.parametrize("message", [None, "", pytest.param("m" * MAX_TEXT_BYTES, id="max-length")])
    def test_exception_message_is_optional_bounded_provenance(self, message):
        assert ExceptionExpectation.model_validate({"kind": "exception", "message": message}).message == message
        assert ExceptionExpectation.model_validate({"kind": "exception"}).message is None

    @pytest.mark.parametrize(
        "expected",
        [
            pytest.param({"kind": "exception", "message": "m" * (MAX_TEXT_BYTES + 1)}, id="long-message"),
            pytest.param({"kind": "exception", "message": 7}, id="number-message"),
            pytest.param({"kind": "value"}, id="missing-value"),
            pytest.param({"kind": "value", "value": wire(["null"]), "message": "m"}, id="value-with-message"),
            pytest.param({"kind": "other"}, id="unknown-kind"),
            pytest.param({"value": wire(["null"])}, id="missing-kind"),
        ],
    )
    def test_expectations_are_discriminated_by_kind(self, expected):
        with pytest.raises(ValidationError):
            CaseData.model_validate({"expected": expected})


class TestCanonicalCases:
    def test_arguments_keywords_expectation_and_tolerances(self):
        case = CaseData.model_validate(
            {
                "args": [wire(["int", "2"]), wire(["str", "two"])],
                "kwargs": {"scale": wire(["float", "0x1.0000000000000p-1"]), "mode_2": wire(["null"])},
                "expected": {"kind": "value", "value": wire(["fraction", "1", "3"])},
                "tolerances": [{"path": "/", "rtol": "1e-9"}],
            }
        )
        assert [arg.value for arg in case.args] == [["int", "2"], ["str", "two"]]
        assert {key: value.value for key, value in case.kwargs.items()} == {
            "scale": ["float", "0x1.0000000000000p-1"],
            "mode_2": ["null"],
        }
        assert case.expected.value.value == ["fraction", "1", "3"]
        assert [(entry.path, entry.rtol) for entry in case.tolerances] == [("", "1e-9")]

    def test_an_expected_exception_needs_no_arguments(self):
        case = CaseData.model_validate({"expected": {"kind": "exception"}})
        assert (case.args, case.kwargs, case.tolerances) == ([], {}, [])
        assert case.expected.kind == "exception"
        assert case.expected.message is None

    @pytest.mark.parametrize("name", ["", "2x", "_x", "x-y", "a.b", "class", "lambda", "None", "xé"])
    def test_keyword_names_must_be_explicit_identifiers(self, name):
        with pytest.raises(ValidationError, match="non-keyword Python identifier"):
            CaseData.model_validate(canonical_case(kwargs={name: wire(["null"])}))

    def test_case_tolerance_paths_are_unique(self):
        with pytest.raises(ValidationError, match="duplicate tolerance path"):
            CaseData.model_validate(
                canonical_case(tolerances=[{"path": "/a", "rtol": "0"}, {"path": "a", "atol": "0"}])
            )

    @pytest.mark.parametrize(
        "updates",
        [
            pytest.param({"args": [wire(["null"])] * 129}, id="too-many-args"),
            pytest.param({"kwargs": {f"k{index}": wire(["null"]) for index in range(129)}}, id="too-many-kwargs"),
            pytest.param({"kwargs": {1: wire(["null"])}}, id="non-text-keyword"),
            pytest.param({"note": "x"}, id="extra-key"),
        ],
    )
    def test_rejects_unbounded_or_unknown_fields(self, updates):
        with pytest.raises(ValidationError):
            CaseData.model_validate(canonical_case(**updates))

    def test_rejects_a_case_that_is_not_a_mapping(self):
        with pytest.raises(ValidationError):
            CaseData.model_validate([wire(["null"])])


class TestDeliveredCases:
    def test_positional_input_and_output(self):
        case = CaseData.model_validate({"input": [3, "label", [1.5]], "output": {"ratio": "7/19"}})
        assert [arg.value for arg in case.args] == [
            ["int", "3"],
            ["str", "label"],
            ["list", [["float", "0x1.8000000000000p+0"]]],
        ]
        assert case.kwargs == {}
        assert case.expected.value.value == ["map", [["ratio", ["legacy", "7/19"]]]]

    @pytest.mark.parametrize("message", ["a synthetic rejection", None])
    def test_positional_input_and_expected_error(self, message):
        case = CaseData.model_validate({"input": [None], "expected_error": message})
        assert case.args[0].value == ["null"]
        assert case.expected.kind == "exception"
        assert case.expected.message == message

    def test_keyword_inputs_tag_only_top_level_text_as_legacy(self):
        case = CaseData.model_validate(
            {"inputs": {"amount": "1.25", "pair": ["a", 2], "flag": True}, "expected_output": ["done", 1]}
        )
        assert case.args == []
        assert {key: value.value for key, value in case.kwargs.items()} == {
            "amount": ["legacy", "1.25"],
            "pair": ["list", [["str", "a"], ["int", "2"]]],
            "flag": ["bool", True],
        }
        assert case.expected.value.value == ["list", [["legacy", "done"], ["int", "1"]]]

    @pytest.mark.parametrize("comparison_type", [None, "exact", pytest.param("c" * 64, id="64-bytes")])
    def test_keyword_shape_accepts_and_drops_a_short_comparison_type(self, comparison_type):
        case = CaseData.model_validate({"inputs": {}, "expected_output": 0, "comparison_type": comparison_type})
        assert case.model_dump(mode="json") == {
            "args": [],
            "kwargs": {},
            "expected": {"kind": "value", "value": {"format": schema.WIRE_FORMAT, "value": ["int", "0"]}},
            "tolerances": [],
        }

    def test_delivered_tolerances_ride_along(self):
        case = CaseData.model_validate({"input": [], "output": 1.0, "tolerances": [{"path": "root", "atol": "0.5"}]})
        assert [(entry.path, entry.atol) for entry in case.tolerances] == [("", "0.5")]

    @pytest.mark.parametrize(
        "case, message",
        [
            pytest.param({"input": (1,), "output": 1}, "bounded positional list", id="tuple-input"),
            pytest.param({"input": {"x": 1}, "output": 1}, "bounded positional list", id="map-input"),
            pytest.param({"input": [0] * 129, "output": 1}, "bounded positional list", id="long-input"),
            pytest.param({"inputs": [1], "expected_output": 1}, "bounded keyword map", id="list-inputs"),
            pytest.param(
                {"inputs": {f"k{index}": index for index in range(129)}, "expected_output": 1},
                "bounded keyword map",
                id="large-inputs",
            ),
            pytest.param(
                {"inputs": {}, "expected_output": 1, "comparison_type": 1},
                "comparison_type must be a string",
                id="number-comparison-type",
            ),
            pytest.param(
                {"inputs": {}, "expected_output": 1, "comparison_type": "c" * 65},
                "comparison_type must be a string",
                id="long-comparison-type",
            ),
            pytest.param(
                {"inputs": {}, "expected_output": 1, "comparison_type": "é" * 33},
                "comparison_type must be a string",
                id="multibyte-comparison-type",
            ),
            pytest.param(
                {"input": [], "output": 1, "expected_error": "both"}, "conflicting or unsupported", id="two-outcomes"
            ),
            pytest.param({"input": [], "inputs": {}, "output": 1}, "conflicting or unsupported", id="two-inputs"),
            pytest.param({"input": [], "expected_output": 1}, "conflicting or unsupported", id="mixed-shapes"),
            pytest.param({"inputs": {}, "output": 1}, "conflicting or unsupported", id="mixed-keyword-shape"),
            pytest.param(
                {"input": [], "output": 1, "comparison_type": "exact"},
                "conflicting or unsupported",
                id="positional-comparison-type",
            ),
            pytest.param({"input": [], "output": 1, "args": []}, "conflicting or unsupported", id="canonical-key"),
            pytest.param({"input": []}, "conflicting or unsupported", id="no-outcome"),
            pytest.param({"input": [b"raw"], "output": 1}, "plain values", id="unsupported-input"),
            pytest.param({"inputs": {"class": 1}, "expected_output": 1}, "identifier", id="keyword-name"),
        ],
    )
    def test_rejects_ambiguous_or_unbounded_delivered_cases(self, case, message):
        with pytest.raises(ValidationError, match=message):
            CaseData.model_validate(case)


class TestTaskData:
    def test_minimal_task_and_its_defaults(self):
        validated = TaskData.model_validate(task())
        assert validated.schema_version == 1
        assert (validated.rtol, validated.atol, validated.tolerances) == (None, None, [])
        assert validated.reference_entrypoint is None
        assert validated.effective_reference_entrypoint == "solve"
        assert validated.input_conversions is None

    @pytest.mark.parametrize(
        "reference_entrypoint, effective", [(None, "solve"), ("reference_solve", "reference_solve")]
    )
    def test_reference_entrypoint_is_explicit(self, reference_entrypoint, effective):
        validated = TaskData.model_validate(task(reference_entrypoint=reference_entrypoint))
        assert validated.effective_reference_entrypoint == effective

    @pytest.mark.parametrize("entrypoint", ["solve", "Solve2", "f_x", pytest.param("x" * 128, id="max-length")])
    def test_accepts_identifier_entrypoints(self, entrypoint):
        assert TaskData.model_validate(task(entrypoint=entrypoint)).entrypoint == entrypoint

    @pytest.mark.parametrize(
        "field, name",
        [
            ("entrypoint", "_solve"),
            ("entrypoint", "2solve"),
            ("entrypoint", "solve-it"),
            ("entrypoint", "obj.solve"),
            ("entrypoint", "class"),
            ("entrypoint", "None"),
            ("entrypoint", "solvé"),
            ("entrypoint", "solve "),
            ("reference_entrypoint", "def"),
            ("reference_entrypoint", "reference.solve"),
        ],
    )
    def test_rejects_entrypoints_that_are_not_identifiers(self, field, name):
        with pytest.raises(ValidationError, match="non-keyword Python identifier"):
            TaskData.model_validate(task(**{field: name}))

    def test_aliases_are_renamed_without_touching_the_row(self):
        row = {
            "problem_id": "synthetic-alias",
            "reference_code": "def solve():\n    return 1\n",
            "entry": "solve",
            "testcases": [{"input": [], "output": 1}],
        }
        validated = TaskData.model_validate(row)
        assert validated.reference_source == row["reference_code"]
        assert validated.entrypoint == "solve"
        assert validated.test_cases[0].expected.value.value == ["int", "1"]
        assert set(row) == {"problem_id", "reference_code", "entry", "testcases"}

    @pytest.mark.parametrize(
        "alias, canonical",
        [("reference_code", "reference_source"), ("entry", "entrypoint"), ("testcases", "test_cases")],
    )
    def test_an_alias_beside_its_canonical_field_is_refused(self, alias, canonical):
        row = task()
        row[alias] = row[canonical]
        with pytest.raises(ValidationError, match="duplicate task field alias"):
            TaskData.model_validate(row)

    def test_an_instance_passes_and_a_non_mapping_is_refused(self):
        validated = TaskData.model_validate(task())
        assert TaskData.model_validate(validated) is validated
        with pytest.raises(ValidationError):
            TaskData.model_validate(["problem_id", "synthetic-identity"])

    def test_one_to_max_cases(self):
        assert len(TaskData.model_validate(task(test_cases=[canonical_case()] * MAX_CASES)).test_cases) == MAX_CASES
        with pytest.raises(ValidationError):
            TaskData.model_validate(task(test_cases=[]))

    @pytest.mark.parametrize("key", ["test_cases", "testcases"])
    def test_more_than_max_cases_are_refused_before_validation(self, key):
        row = task()
        del row["test_cases"]
        row[key] = [canonical_case()] * (MAX_CASES + 1)
        with pytest.raises(ValidationError, match="testcase count limit"):
            TaskData.model_validate(row)

    @pytest.mark.parametrize("version", [True, "1", 1.0, None])
    def test_schema_version_must_be_an_integer(self, version):
        with pytest.raises(ValidationError, match="schema_version must be an integer"):
            TaskData.model_validate(task(schema_version=version))

    def test_schema_version_is_one(self):
        assert TaskData.model_validate(task(schema_version=1)).schema_version == 1
        with pytest.raises(ValidationError):
            TaskData.model_validate(task(schema_version=2))

    @pytest.mark.parametrize("extra", [{"network": True}, {"image": "synthetic"}, {"timeout": 10}])
    def test_operator_runtime_settings_are_not_task_fields(self, extra):
        with pytest.raises(ValidationError):
            TaskData.model_validate(task(**extra))

    @pytest.mark.parametrize(
        "field, accepted, refused",
        [
            pytest.param("problem_id", "p" * 256, "p" * 257, id="problem_id-max"),
            pytest.param("problem_id", "p", "", id="problem_id-min"),
            pytest.param("reference_source", "s" * 262_144, "s" * 262_145, id="reference_source-max"),
            pytest.param("reference_source", "s", "", id="reference_source-min"),
            pytest.param("entrypoint", "e" * 128, "e" * 129, id="entrypoint-max"),
            pytest.param("reference_entrypoint", "r" * 128, "r" * 129, id="reference_entrypoint-max"),
            pytest.param("problem", "x" * MAX_STATEMENT_CHARS, "x" * (MAX_STATEMENT_CHARS + 1), id="problem-max"),
            pytest.param("code_template", "c" * 131_072, "c" * 131_073, id="code_template-max"),
            pytest.param("uuid", "u" * 256, "u" * 257, id="uuid-max"),
            pytest.param("input_conversions", ["symbol", None, "function"], ["matrix"], id="conversion-kinds"),
            pytest.param("input_conversions", [None] * 128, [None] * 129, id="conversions-max"),
            pytest.param("input_conversions", None, "symbol", id="conversions-list"),
        ],
    )
    def test_field_bounds(self, field, accepted, refused):
        assert getattr(TaskData.model_validate(task(**{field: accepted})), field) == accepted
        with pytest.raises(ValidationError) as excinfo:
            TaskData.model_validate(task(**{field: refused}))
        assert {error["loc"][0] for error in excinfo.value.errors()} == {field}

    def test_serialized_task_stays_within_the_task_byte_budget(self):
        text = "x" * 500_000
        assert len(text) < MAX_WIRE_BYTES
        case = canonical_case(expected={"kind": "value", "value": wire(["str", text])})
        fitting = MAX_TASK_BYTES // len(text)
        assert len(TaskData.model_validate(task(test_cases=[case] * fitting)).test_cases) == fitting
        with pytest.raises(ValidationError, match="JSON byte"):
            TaskData.model_validate(task(test_cases=[case] * (fitting + 1)))


class TestTaskComparisonPolicy:
    def test_case_entries_replace_task_entries_by_path(self):
        statement = "A synthetic statement with no precision clause."
        validated = TaskData.model_validate(
            task(
                rtol="1e-6",
                atol=0,
                problem=statement,
                tolerances=[{"path": "/a", "rtol": "1e-3"}, {"path": "/b", "atol": "0.5", "quantity": "a label"}],
                test_cases=[
                    {
                        "input": [],
                        "output": {"a": 1, "b": 2, "c": 3},
                        "tolerances": [{"path": "a", "atol": "0.1"}, {"path": "/c", "rtol": 0.2}],
                    },
                    {"input": [], "output": 1},
                ],
            )
        )
        first = validated.comparison_policy(0)
        assert first == {
            "rtol": "1e-6",
            "atol": "0",
            "tolerances": [
                {"path": "/a", "rtol": None, "atol": "0.1", "quantity": None},
                {"path": "/b", "rtol": None, "atol": "0.5", "quantity": "a label"},
                {"path": "/c", "rtol": "0.2", "atol": None, "quantity": None},
            ],
            "statement": statement,
        }
        second = validated.comparison_policy(1)
        assert second["tolerances"] == [
            {"path": "/a", "rtol": "1e-3", "atol": None, "quantity": None},
            {"path": "/b", "rtol": None, "atol": "0.5", "quantity": "a label"},
        ]
        for policy in (first, second):
            assert CasePolicy.model_validate(policy).model_dump(mode="json") == {
                **policy,
                "default_rtol": None,
                "default_atol": None,
            }

    @pytest.mark.parametrize(
        "defaults",
        [{}, {"default_rtol": "1e-5"}, {"default_atol": "1e-8"}, {"default_rtol": "1e-5", "default_atol": "0"}],
    )
    def test_operator_defaults_are_transported_only_when_given(self, defaults):
        policy = TaskData.model_validate(task()).comparison_policy(0, **defaults)
        assert {key: policy[key] for key in ("default_rtol", "default_atol") if key in policy} == defaults
        restored = CasePolicy.model_validate(policy)
        assert (restored.default_rtol, restored.default_atol) == (
            defaults.get("default_rtol"),
            defaults.get("default_atol"),
        )
