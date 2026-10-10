# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light, fail-closed task schema. Importing it never imports the comparator.

Canonical values use {"format": "critpt-value-v1", "value": <tagged node>}. Delivered case aliases are
{input, output}, {input, expected_error}, and {inputs, expected_output}. Duplicate top-level spellings are
rejected, even when equal. This module performs no prose extraction, comparison, or symbolic parsing. Only
the comparator, inside its protected sandbox, reads precision clauses from the statement, which travels here
as opaque bounded text.
"""

import json
import keyword
import math
import re
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


WIRE_FORMAT = "critpt-value-v1"
# Per-value wire byte cap, above the largest value legal under MAX_ELEMENTS. Single source of truth for
# codec.py and runner.py. Independent of MAX_TEXT_BYTES, which bounds one string, not the whole value.
MAX_WIRE_BYTES = 524_288
MAX_TASK_BYTES = 4_194_304
MAX_DEPTH = 32
# Per-value element cap for a delivered payload, above the largest delivered value with margin. The official
# CritPt grader has no such bound. Single source of truth for codec.py and runner.py.
MAX_ELEMENTS = 32_768
# Aggregate JSON-node cap enforced by bounded_json. The wire form nests every scalar as a tagged list, which
# multiplies the node count. The cap sits above the largest delivered task with margin.
MAX_JSON_NODES = 1_048_576
MAX_TEXT_BYTES = 16_384
MAX_INTEGER_DIGITS = 1024
MAX_DECIMAL_EXPONENT = 1024
MAX_CASES = 64
# The statement bound equals the TaskData.problem bound. The byte allowance covers JSON escaping.
MAX_STATEMENT_CHARS = 131_072
MAX_STATEMENT_BYTES = 6 * MAX_STATEMENT_CHARS
MAX_POLICY_BYTES = MAX_WIRE_BYTES + MAX_STATEMENT_BYTES
DEFAULT_RTOL = "5e-12"
DEFAULT_ATOL = "0"
# The default a genuinely silent numeric leaf resolves to when no promise and no author speaks. These mirror
# the official CritPt harness defaults. DEFAULT_RTOL/DEFAULT_ATOL stay the strict floor for a harvested promise.
SILENT_DEFAULT_RTOL = "1e-5"
SILENT_DEFAULT_ATOL = "1e-8"
DECIMAL_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")


def bounded_json(value: Any, *, max_bytes: int = MAX_WIRE_BYTES, max_depth: int = 3 * MAX_DEPTH + 8) -> None:
    """Check exact JSON types and limits before allocating a serialized copy."""
    pending = [(value, 0)]
    nodes = size = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > max_depth:
            raise ValueError("JSON structure limit")
        kind = type(item)
        if kind is str:
            if len(item) > max_bytes:
                raise ValueError("JSON byte limit")
            size += len(item.encode("utf-8"))
        elif kind is dict:
            if len(item) > MAX_JSON_NODES:
                raise ValueError("JSON structure limit")
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("JSON keys must be strings")
                if len(key) > max_bytes:
                    raise ValueError("JSON byte limit")
                size += len(key.encode("utf-8")) + 3
                if size > max_bytes:
                    raise ValueError("JSON byte limit")
                pending.append((child, depth + 1))
        elif kind is list:
            if len(item) > MAX_JSON_NODES:
                raise ValueError("JSON structure limit")
            pending.extend((child, depth + 1) for child in item)
        elif kind is int:
            if item.bit_length() > 63:
                raise ValueError("large JSON integers require a wire tag")
            size += 20
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite JSON numbers require a wire tag")
            size += 24
        elif kind is bool or item is None:
            size += 5
        else:
            raise ValueError("unsupported JSON type")
        if size > max_bytes or len(pending) > MAX_JSON_NODES:
            raise ValueError("JSON byte or structure limit")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")) > max_bytes:
        raise ValueError("JSON byte limit")


def decimal_text(text: str) -> str:
    """Validate decimal syntax and scale without binary floating-point coercion."""
    if type(text) is not str or len(text) > MAX_INTEGER_DIGITS + 16 or not DECIMAL_PATTERN.fullmatch(text):
        raise ValueError("invalid decimal")
    mantissa, *exponents = re.split("[eE]", text)
    digits = sum(char.isdigit() for char in mantissa)
    exponent_text = exponents[0] if exponents else "0"
    if digits > MAX_INTEGER_DIGITS or len(exponent_text.lstrip("+-")) > 4:
        raise ValueError("decimal size limit")
    exponent = int(exponent_text)
    places = len(mantissa.partition(".")[2])
    if abs(exponent) > MAX_DECIMAL_EXPONENT or abs(exponent - places) > MAX_DECIMAL_EXPONENT:
        raise ValueError("decimal exponent limit")
    return text


def tolerance_text(value: Any) -> str | None:
    """A finite, nonnegative decimal as exact text, or None. The one gate for every tolerance knob."""
    if value is None:
        return None
    if not any(type(value) is allowed for allowed in (str, int, float)):
        raise ValueError("tolerance must be a finite nonnegative decimal")
    if type(value) is int and value.bit_length() > MAX_INTEGER_DIGITS * 4:
        raise ValueError("tolerance size limit")
    text = decimal_text(str(value))
    if Decimal(text) < 0:
        raise ValueError("negative tolerance")
    return text


def normalize_path(path: str) -> str:
    """Legacy root aliases plus escaped JSON Pointer. No wildcard or subtree rules."""
    path = path.strip()
    if path in ("", "/", ".", "root"):
        return ""
    path = path if path.startswith("/") else "/" + path
    if re.search(r"~(?![01])", path):
        raise ValueError("invalid JSON Pointer escape")
    return path


def _identifier(value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value) or keyword.iskeyword(value):
        raise ValueError("expected an explicit non-keyword Python identifier")
    return value


def _legacy_wire(value: Any, *, legacy_strings: bool = False) -> dict:
    """Bounded exact-built-in transport normalization, not a scientific codec or text parser.

    Native scientific types are unsupported. Callers must supply canonical wire values instead.
    """
    count = 0

    def node(item: Any, depth: int) -> list:
        nonlocal count
        count += 1
        if count > MAX_ELEMENTS or depth > MAX_DEPTH:
            raise ValueError("value structure limit")
        kind = type(item)
        if item is None:
            return ["null"]
        if kind is bool:
            return ["bool", item]
        if kind is int:
            if item.bit_length() > MAX_INTEGER_DIGITS * 4:
                raise ValueError("integer size limit")
            text = str(item)
            if len(text.lstrip("-")) > MAX_INTEGER_DIGITS:
                raise ValueError("integer size limit")
            return ["int", text]
        if kind is float:
            if not math.isfinite(item):
                return ["nonfinite", "nan" if math.isnan(item) else ("inf" if item > 0 else "-inf")]
            return ["float", item.hex()]
        if kind is str:
            if len(item) > MAX_TEXT_BYTES or len(item.encode("utf-8")) > MAX_TEXT_BYTES:
                raise ValueError("string size limit")
            return ["legacy" if legacy_strings else "str", item]
        if kind is list or kind is tuple:
            if len(item) > MAX_ELEMENTS:
                raise ValueError("value structure limit")
            return ["tuple" if kind is tuple else "list", [node(child, depth + 1) for child in item]]
        if kind is dict:
            if len(item) > MAX_ELEMENTS:
                raise ValueError("value structure limit")
            # The official grader's complex carrier: a dict {"__complex__": [re, im]} of two finite reals becomes
            # a complex wire node. bool and nonfinite floats are excluded, so any other shape stays a plain map.
            if len(item) == 1 and "__complex__" in item:
                parts = item["__complex__"]
                if (
                    type(parts) is list
                    and len(parts) == 2
                    and all(type(part) is int or (type(part) is float and math.isfinite(part)) for part in parts)
                ):
                    return ["complex", node(parts[0], depth + 1), node(parts[1], depth + 1)]
            for key in item:
                if type(key) is not str or len(key) > MAX_TEXT_BYTES or len(key.encode("utf-8")) > MAX_TEXT_BYTES:
                    raise ValueError("maps require bounded string keys")
            return ["map", [[key, node(child, depth + 1)] for key, child in item.items()]]
        raise ValueError("delivered metadata must contain plain values")

    result = {"format": WIRE_FORMAT, "value": node(value, 0)}
    bounded_json(result)
    return result


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class WireValue(StrictModel):
    format: Literal["critpt-value-v1"] = WIRE_FORMAT
    value: list[Any]

    @model_validator(mode="before")
    @classmethod
    def bounded(cls, value: Any) -> Any:
        if type(value) is cls:
            return value
        bounded_json(value)
        return value


class Tolerance(StrictModel):
    rtol: str | None = None
    atol: str | None = None

    @field_validator("rtol", "atol", mode="before")
    @classmethod
    def numeric_text(cls, value: Any) -> str | None:
        return tolerance_text(value)


class LeafTolerance(Tolerance):
    path: str = Field(max_length=1024)
    quantity: str | None = Field(default=None, max_length=256, description="Informational; never a path alias.")

    @field_validator("path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        return normalize_path(value)

    @model_validator(mode="after")
    def explicit_override(self) -> "LeafTolerance":
        if self.rtol is None and self.atol is None:
            raise ValueError("leaf tolerance requires rtol or atol")
        return self


class ComparisonPolicy(Tolerance):
    tolerances: list[LeafTolerance] = Field(default_factory=list, max_length=MAX_ELEMENTS)

    @field_validator("tolerances")
    @classmethod
    def unique_paths(cls, entries: list[LeafTolerance]) -> list[LeafTolerance]:
        paths = [entry.path for entry in entries]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate tolerance path")
        return entries


class CasePolicy(ComparisonPolicy):
    """Per-case comparator input: task settings, leaf entries, and the statement as bounded text.

    The statement is opaque here. Only the comparator reads precision clauses from it, so a host cannot
    pre-resolve a promise on the task's behalf. ``default_rtol``/``default_atol`` are the operator-owned
    server default for a silent numeric leaf, and never override a statement promise, a leaf entry, or a
    task setting.
    """

    statement: str | None = Field(default=None, max_length=MAX_STATEMENT_CHARS)
    default_rtol: str | None = None
    default_atol: str | None = None

    @field_validator("default_rtol", "default_atol", mode="before")
    @classmethod
    def numeric_default(cls, value: Any) -> str | None:
        return tolerance_text(value)


class ValueExpectation(StrictModel):
    kind: Literal["value"]
    value: WireValue


class ExceptionExpectation(StrictModel):
    kind: Literal["exception"]
    message: str | None = Field(
        default=None, max_length=MAX_TEXT_BYTES, description="Provenance only; never compared."
    )


Expectation = Annotated[ValueExpectation | ExceptionExpectation, Field(discriminator="kind")]


class TestCase(StrictModel):
    args: list[WireValue] = Field(default_factory=list, max_length=128)
    kwargs: dict[str, WireValue] = Field(default_factory=dict, max_length=128)
    expected: Expectation
    tolerances: list[LeafTolerance] = Field(default_factory=list, max_length=MAX_ELEMENTS)

    @model_validator(mode="before")
    @classmethod
    def delivered_shapes(cls, value: Any) -> Any:
        if type(value) is not dict or not ({"input", "inputs"} & value.keys()):
            return value
        data = dict(value)
        overrides = data.pop("tolerances", [])
        keys = set(data)
        if keys in ({"input", "output"}, {"input", "expected_error"}):
            if type(data["input"]) is not list or len(data["input"]) > 128:
                raise ValueError("input must be a bounded positional list")
            args = [_legacy_wire(item) for item in data["input"]]
            kwargs = {}
            expected = (
                {"kind": "exception", "message": data["expected_error"]}
                if "expected_error" in data
                else {"kind": "value", "value": _legacy_wire(data["output"], legacy_strings=True)}
            )
        elif keys in ({"inputs", "expected_output"}, {"inputs", "expected_output", "comparison_type"}):
            # The keyword shape may carry an extra comparison_type key. It is accepted only on this shape,
            # validated as a short string, then dropped. No other unknown key is accepted.
            comparison_type = data.pop("comparison_type", None)
            if comparison_type is not None and (
                type(comparison_type) is not str or len(comparison_type.encode("utf-8")) > 64
            ):
                raise ValueError("comparison_type must be a string of at most 64 bytes")
            if type(data["inputs"]) is not dict or len(data["inputs"]) > 128:
                raise ValueError("inputs must be a bounded keyword map")
            args = []
            kwargs = {
                key: _legacy_wire(item, legacy_strings=True) if type(item) is str else _legacy_wire(item)
                for key, item in data["inputs"].items()
            }
            expected = {"kind": "value", "value": _legacy_wire(data["expected_output"], legacy_strings=True)}
        else:
            raise ValueError("conflicting or unsupported delivered testcase fields")
        return {"args": args, "kwargs": kwargs, "expected": expected, "tolerances": overrides}

    @field_validator("kwargs")
    @classmethod
    def keyword_names(cls, values: dict) -> dict:
        for key in values:
            _identifier(key)
        return values

    @field_validator("tolerances")
    @classmethod
    def unique_paths(cls, values: list[LeafTolerance]) -> list[LeafTolerance]:
        return ComparisonPolicy.unique_paths(values)


class TaskData(ComparisonPolicy):
    schema_version: Literal[1] = 1
    problem_id: str = Field(min_length=1, max_length=256)
    reference_source: str = Field(min_length=1, max_length=262_144)
    entrypoint: str = Field(min_length=1, max_length=128)
    reference_entrypoint: str | None = Field(default=None, max_length=128)
    test_cases: list[TestCase] = Field(min_length=1, max_length=MAX_CASES)
    problem: str | None = Field(default=None, max_length=MAX_STATEMENT_CHARS)
    code_template: str | None = Field(default=None, max_length=131_072)
    uuid: str | None = Field(default=None, max_length=256)
    # Per-position input conversions aligned with the positional inputs: null (or a missing position) leaves the
    # argument alone, "symbol" makes a sympy Symbol, "function" a sympy Function. Mirrors the official grader's
    # symbolic_testcases.input_conversions. The runner applies it. This schema only transports it.
    input_conversions: list[Literal["symbol", "function"] | None] | None = Field(default=None, max_length=128)

    @model_validator(mode="before")
    @classmethod
    def explicit_aliases(cls, value: Any) -> Any:
        if type(value) is not dict:
            return value
        data = dict(value)
        aliases = (("reference_code", "reference_source"), ("entry", "entrypoint"), ("testcases", "test_cases"))
        for alias, canonical in aliases:
            if alias in data:
                if canonical in data:
                    raise ValueError("duplicate task field alias")
                data[canonical] = data.pop(alias)
        if type(data.get("test_cases")) is list and len(data["test_cases"]) > MAX_CASES:
            raise ValueError("testcase count limit")
        return data

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("entrypoint", "reference_entrypoint")
    @classmethod
    def explicit_entry(cls, value: str | None) -> str | None:
        return _identifier(value) if value is not None else None

    @model_validator(mode="after")
    def bounded_task(self) -> "TaskData":
        bounded_json(self.model_dump(mode="json"), max_bytes=MAX_TASK_BYTES, max_depth=3 * MAX_DEPTH + 16)
        return self

    def comparison_policy(
        self, case_index: int, default_rtol: str | None = None, default_atol: str | None = None
    ) -> dict:
        """Task defaults, case-specific leaf overrides, and the verbatim statement. Nothing is resolved here.

        The result validates as CasePolicy. The statement is bounded text, never a pre-extracted promise.
        ``default_rtol``/``default_atol`` are transported verbatim and applied only by the comparator.
        """
        entries = {entry.path: entry for entry in self.tolerances}
        entries.update((entry.path, entry) for entry in self.test_cases[case_index].tolerances)
        policy = {
            "rtol": self.rtol,
            "atol": self.atol,
            "tolerances": [entry.model_dump(mode="json") for entry in entries.values()],
            "statement": self.problem,
        }
        if default_rtol is not None:
            policy["default_rtol"] = default_rtol
        if default_atol is not None:
            policy["default_atol"] = default_atol
        return policy

    @property
    def effective_reference_entrypoint(self) -> str:
        return self.reference_entrypoint if self.reference_entrypoint is not None else self.entrypoint
