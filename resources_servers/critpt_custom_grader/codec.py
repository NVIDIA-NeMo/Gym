# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration-owned value transport. Encode before JSON/orjson dataset boundaries.

Nodes: ["null"], ["bool", b], ["int", int_text], ["float", float_hex], ["fraction", num, den],
["decimal", decimal_text], ["nonfinite", "nan" | "inf" | "-inf"], ["complex", real_node, imag_node],
["list" | "tuple", [nodes...]], ["set", [nodes...]], ["map", [[string_key, node]...]],
["str" | "symbolic" | "legacy", text]. Every node is tagged. Strings are always data, never an
executable tag, and native third-party objects are never inspected.

These functions do not deserialize Python objects. decode_value returns only exact built-in/stdlib types
and the inert carriers above. The executor and comparator, including decoding, belong in Daytona, not the host.
"""

import json
import math
import re
import types
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

from .task_data import (
    MAX_DEPTH,
    MAX_ELEMENTS,
    MAX_INTEGER_DIGITS,
    MAX_TEXT_BYTES,
    MAX_WIRE_BYTES,
    WIRE_FORMAT,
    bounded_json,
    decimal_text,
)


@dataclass(frozen=True, slots=True)
class SymbolicText:
    text: str


@dataclass(frozen=True, slots=True)
class LegacyText:
    text: str


@dataclass(frozen=True, slots=True)
class ComplexValue:
    real: Any
    imag: Any


@dataclass(frozen=True, slots=True)
class SetValue:
    """Inert carrier for a set-valued answer. Members are held as a tuple in arbitrary order.
    The comparator matches it UNORDERED, so member order carries no meaning."""

    items: tuple


class RepresentationError(ValueError):
    def __init__(self, code: str, path: str = ""):
        super().__init__(code)
        self.code = code
        self.path = path

    @property
    def attributable(self) -> bool:
        """Unknown native types can be a missing scientific-library adapter."""
        return self.code != "unsupported_type"


REAL_TYPES = (int, float, Fraction, Decimal)
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_HEX_FLOAT = re.compile(r"-?0x[01]\.[0-9a-f]+p[+-][0-9]{1,4}\Z")


def pointer_child(path: str, key: str | int) -> str:
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


# A non-string map key of a fixed safe type is stringified with str(), the way the historical grader keyed
# maps, so a dict keyed by tuples or ints compares as it did there. An arbitrary object key is refused, never
# stringified, so the codec never runs an unknown object's __str__.
_STRINGIFIABLE_KEY_TYPES = (bool, int, float, Fraction, Decimal, complex, tuple, type(None))


def _encode_key(key: Any, path: str) -> str:
    if type(key) is str:
        return _text(key, path)
    if type(key) in _STRINGIFIABLE_KEY_TYPES:
        return _text(str(key), path)
    raise RepresentationError("invalid_text", path)


def _text(value: Any, path: str, maximum: int = MAX_TEXT_BYTES) -> str:
    if type(value) is not str:
        raise RepresentationError("invalid_text", path)
    if len(value) > maximum:
        raise RepresentationError("text_limit", path)
    try:
        if len(value.encode("utf-8")) > maximum:
            raise RepresentationError("text_limit", path)
    except UnicodeError as exc:
        raise RepresentationError("invalid_unicode", path) from exc
    return value


def _integer(value: Any, path: str) -> int:
    text = _text(value, path, MAX_INTEGER_DIGITS + 1)
    if len(text.lstrip("-")) > MAX_INTEGER_DIGITS:
        raise RepresentationError("integer_limit", path)
    if not _INTEGER.fullmatch(text) or text == "-0":
        raise RepresentationError("invalid_integer", path)
    return int(text)


def _decimal(value: Any, path: str) -> Decimal:
    text = _text(value, path, MAX_INTEGER_DIGITS + 16)
    if text not in ("NaN", "Infinity", "-Infinity"):
        try:
            decimal_text(text)
        except ValueError as exc:
            raise RepresentationError("invalid_decimal", path) from exc
    return Decimal(text)


class _Budget:
    def __init__(self):
        self.nodes = 0

    def visit(self, depth: int, path: str) -> None:
        self.nodes += 1
        if self.nodes > MAX_ELEMENTS or depth > MAX_DEPTH:
            raise RepresentationError("structure_limit", path)


def _check_wire(wire: Any) -> None:
    try:
        bounded_json(wire)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RepresentationError("json_bounds") from exc
    if type(wire) is not dict or set(wire) != {"format", "value"} or wire["format"] != WIRE_FORMAT:
        raise RepresentationError("invalid_envelope")


def encode_value(value: Any) -> dict:
    """Return a JSON-safe envelope, or raise a typed RepresentationError.

    Supported exact types: None, bool, int, binary64 float, Fraction, Decimal,
    complex, list, tuple, set, frozenset, string-keyed dict, and the inert carriers
    above. Subclasses, object arrays, iterators and arbitrary objects are not traversed.
    """
    budget = _Budget()

    def encode(item: Any, path: str, depth: int) -> list:
        budget.visit(depth, path)
        kind = type(item)
        if item is None:
            return ["null"]
        if kind is bool:
            return ["bool", item]
        if kind is int:
            if item.bit_length() > MAX_INTEGER_DIGITS * 4:
                raise RepresentationError("integer_limit", path)
            text = str(item)
            _integer(text, path)
            return ["int", text]
        if kind is float:
            if not math.isfinite(item):
                return ["nonfinite", "nan" if math.isnan(item) else ("inf" if item > 0 else "-inf")]
            return ["float", item.hex()]
        if kind is Fraction:
            if max(item.numerator.bit_length(), item.denominator.bit_length()) > MAX_INTEGER_DIGITS * 4:
                raise RepresentationError("integer_limit", path)
            numerator, denominator = str(item.numerator), str(item.denominator)
            _integer(numerator, path)
            _integer(denominator, path)
            return ["fraction", numerator, denominator]
        if kind is Decimal:
            if len(item.as_tuple().digits) > MAX_INTEGER_DIGITS:
                raise RepresentationError("invalid_decimal", path)
            text = str(item)
            _decimal(text, path)
            return ["decimal", text]
        if kind is str:
            return ["str", _text(item, path)]
        if kind is SymbolicText or kind is LegacyText:
            return ["symbolic" if kind is SymbolicText else "legacy", _text(item.text, path)]
        if kind is complex or kind is ComplexValue:
            real, imag = item.real, item.imag
            if not any(type(real) is allowed for allowed in REAL_TYPES):
                raise RepresentationError("invalid_complex", path)
            if not any(type(imag) is allowed for allowed in REAL_TYPES):
                raise RepresentationError("invalid_complex", path)
            return ["complex", encode(real, path, depth + 1), encode(imag, path, depth + 1)]
        if kind is list or kind is tuple:
            if len(item) > MAX_ELEMENTS:
                raise RepresentationError("structure_limit", path)
            return [
                "list" if kind is list else "tuple",
                [encode(child, pointer_child(path, index), depth + 1) for index, child in enumerate(item)],
            ]
        if kind is set or kind is frozenset or kind is SetValue:
            members = item.items if kind is SetValue else item
            if len(members) > MAX_ELEMENTS:
                raise RepresentationError("structure_limit", path)
            # Order carries no meaning. The comparator matches a set unordered.
            return [
                "set",
                [encode(child, pointer_child(path, index), depth + 1) for index, child in enumerate(members)],
            ]
        if kind is dict:
            if len(item) > MAX_ELEMENTS:
                raise RepresentationError("structure_limit", path)
            entries = []
            seen: set[str] = set()
            for key, child in item.items():
                text = _encode_key(key, path)
                if text in seen:
                    # Two distinct Python keys collapsed to one wire string. decode forbids duplicate map keys,
                    # so refuse here rather than emit a wire decode would reject.
                    raise RepresentationError("invalid_text", path)
                seen.add(text)
                entries.append([text, encode(child, pointer_child(path, text), depth + 1)])
            return ["map", entries]
        invalid_types = (bytes, bytearray, types.FunctionType, types.ModuleType, types.GeneratorType)
        if any(kind is invalid for invalid in invalid_types):
            raise RepresentationError("invalid_type", path)
        raise RepresentationError("unsupported_type", path)

    wire = {"format": WIRE_FORMAT, "value": encode(value, "", 0)}
    _check_wire(wire)
    return wire


def decode_value(wire: dict) -> Any:
    """Validate all tags and limits, then decode in the protected remote process."""
    _check_wire(wire)
    budget = _Budget()

    def decode(node: Any, path: str, depth: int) -> Any:
        budget.visit(depth, path)
        if type(node) is not list or not node or type(node[0]) is not str:
            raise RepresentationError("invalid_node", path)
        tag = node[0]
        arities = {
            "null": 1,
            "bool": 2,
            "int": 2,
            "float": 2,
            "fraction": 3,
            "decimal": 2,
            "nonfinite": 2,
            "complex": 3,
            "str": 2,
            "symbolic": 2,
            "legacy": 2,
            "list": 2,
            "tuple": 2,
            "set": 2,
            "map": 2,
        }
        if tag not in arities:
            raise RepresentationError("unknown_tag", path)
        if len(node) != arities[tag]:
            raise RepresentationError("invalid_arity", path)
        if tag == "null":
            return None
        value = node[1]
        if tag == "bool":
            if type(value) is not bool:
                raise RepresentationError("invalid_boolean", path)
            return value
        if tag == "int":
            return _integer(value, path)
        if tag == "fraction":
            numerator, denominator = _integer(value, path), _integer(node[2], path)
            if denominator <= 0:
                raise RepresentationError("invalid_denominator", path)
            return Fraction(numerator, denominator)
        if tag == "decimal":
            return _decimal(value, path)
        if tag == "float":
            text = _text(value, path, 64)
            if not _HEX_FLOAT.fullmatch(text):
                raise RepresentationError("invalid_float", path)
            try:
                number = float.fromhex(text)
            except (ValueError, OverflowError) as exc:
                raise RepresentationError("invalid_float", path) from exc
            if not math.isfinite(number) or number.hex() != text:
                raise RepresentationError("invalid_float", path)
            return number
        if tag == "nonfinite":
            if type(value) is not str or value not in ("nan", "inf", "-inf"):
                raise RepresentationError("invalid_nonfinite", path)
            return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[value]
        if tag in ("str", "symbolic", "legacy"):
            text = _text(value, path)
            return SymbolicText(text) if tag == "symbolic" else LegacyText(text) if tag == "legacy" else text
        if tag == "complex":
            real, imag = decode(value, path, depth + 1), decode(node[2], path, depth + 1)
            if type(real) not in REAL_TYPES or type(imag) not in REAL_TYPES:
                raise RepresentationError("invalid_complex", path)
            return ComplexValue(real, imag)
        if type(value) is not list or len(value) > MAX_ELEMENTS:
            raise RepresentationError("invalid_composite", path)
        if tag in ("list", "tuple"):
            values = [decode(child, pointer_child(path, index), depth + 1) for index, child in enumerate(value)]
            return tuple(values) if tag == "tuple" else values
        if tag == "set":
            return SetValue(
                tuple(decode(child, pointer_child(path, index), depth + 1) for index, child in enumerate(value))
            )
        result = {}
        for entry in value:
            if type(entry) is not list or len(entry) != 2:
                raise RepresentationError("invalid_map_entry", path)
            key = _text(entry[0], path)
            if key in result:
                raise RepresentationError("duplicate_map_key", path)
            result[key] = decode(entry[1], pointer_child(path, key), depth + 1)
        return result

    return decode(wire["value"], "", 0)


def dump_value(value: Any) -> bytes:
    return json.dumps(encode_value(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def load_value(payload: bytes | str) -> Any:
    """Bound JSON syntax before loading. Duplicate keys and raw numbers are forbidden."""
    if (type(payload) is not bytes and type(payload) is not str) or len(payload) > MAX_WIRE_BYTES:
        raise RepresentationError("json_bounds")
    try:
        text = payload.decode("utf-8") if type(payload) is bytes else payload
        if len(text.encode("utf-8")) > MAX_WIRE_BYTES:
            raise RepresentationError("json_bounds")
        depth, quoted, escaped = 0, False, False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > 3 * MAX_DEPTH + 8:
                    raise RepresentationError("structure_limit")
            elif char in "]}":
                depth -= 1

        def pairs(entries: list) -> dict:
            result = {}
            for key, value in entries:
                if key in result:
                    raise RepresentationError("duplicate_json_key")
                result[key] = value
            return result

        def reject_number(text: str) -> None:
            raise RepresentationError("untagged_number")

        wire = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_int=reject_number,
            parse_float=reject_number,
            parse_constant=reject_number,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, RepresentationError):
            raise
        raise RepresentationError("invalid_json") from exc
    return decode_value(wire)
