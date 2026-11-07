# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JSON encoding and decoding that keep every value exact or raise.

``orjson`` handles almost every body Gym sends or stores, but it cannot write an integer wider than 64 bits or a
string holding a lone surrogate, and it reads such an integer back as a float without an error. Both can come from a
model: tool call arguments are model-written JSON. These helpers keep ``orjson`` for the common case and use the
standard library only for the values ``orjson`` would reject or change.
"""

import json
import logging
import math
from collections.abc import Callable
from typing import Any

import orjson


LOG = logging.getLogger(__name__)

# The ``orjson`` options the standard library fallback can reproduce.
_SUPPORTED_OPTIONS = orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE

# ``orjson`` reads integers in [-2**63, 2**64) exactly and anything wider as a float, so a lossy literal has at least
# 19 digits. The check maps each byte to its class, "0" for a digit, "." for itself and "x" for anything else, then
# looks for 19 digits that start a number: a run that follows "." is a fraction, which ``orjson`` reads exactly.
# Shortest float text has at most 17 significant digits and switches to an exponent from 1e16, so floats written by
# ``orjson``, ``json`` or Pydantic never match; a 19-digit integer that fits, or a long digit string inside a JSON
# string, does, and costs only a standard library parse of that body.
_BYTE_CLASSES = bytes(
    ord("0") if byte in b"0123456789" else byte if byte == ord(".") else ord("x") for byte in range(256)
)
_INTEGER_START = b"x" + b"0" * 19


def dumps_json_bytes(
    value: object, *, default: Callable[[Any], Any] | None = None, option: int | None = None
) -> bytes:
    """Serialize ``value`` like ``orjson.dumps``, falling back to the standard library for what ``orjson`` rejects.

    A value ``orjson`` accepts keeps its exact ``orjson`` bytes, so existing records and hashes do not change. The
    fallback covers integers wider than 64 bits, lone surrogates, and nesting deeper than ``orjson`` allows. It writes
    NaN and infinity as ``null``, as ``orjson`` does, so a value means the same whether or not the body also holds a
    wide integer. It escapes non-ASCII text only when the text holds a lone surrogate, so the bytes stay valid UTF-8.

    ``option`` takes ``orjson`` flags; the fallback supports ``OPT_SORT_KEYS``, ``OPT_INDENT_2`` and
    ``OPT_APPEND_NEWLINE``. Object keys must be strings: the fallback refuses other keys rather than coerce them,
    which could merge two keys into one.

    Raises:
        orjson.JSONEncodeError: ``value`` holds something neither encoder can write exactly.
    """
    try:
        return orjson.dumps(value, default=default, option=option)
    except orjson.JSONEncodeError as error:
        if option and option & ~_SUPPORTED_OPTIONS:
            raise
        try:
            return _dumps_stdlib(value, default=default, option=option or 0)
        except (TypeError, ValueError, RecursionError) as fallback_error:
            raise orjson.JSONEncodeError(f"{error}; standard library fallback: {fallback_error}") from fallback_error


def loads_json(data: bytes | bytearray | memoryview | str) -> Any:
    """Parse JSON like ``orjson.loads``, but read integers wider than 64 bits and lone surrogates exactly.

    The common case costs one byte translation and a substring search on top of ``orjson``. A body that holds a run
    of digits long enough to overflow ``orjson`` is parsed with the standard library instead, as is a body ``orjson``
    rejects (such as an escaped lone surrogate). The fallback refuses ``NaN``, ``Infinity`` and numbers that overflow
    a float, as ``orjson`` does.

    Raises:
        orjson.JSONDecodeError: ``data`` is not valid JSON. A body both parsers reject raises ``orjson``'s error.
    """
    if _may_hold_wide_integer(data):
        return _loads_stdlib(data, None)
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError as error:
        return _loads_stdlib(data, error)


def _may_hold_wide_integer(data: bytes | bytearray | memoryview | str) -> bool:
    """Whether ``data`` holds an integer literal ``orjson`` would read as a float, allowing false positives."""
    if not isinstance(data, (bytes, bytearray, memoryview, str)):
        return False  # ``orjson`` raises its own error for the type
    if isinstance(data, str):
        data = data.encode("utf-8", "surrogatepass")
    classes = (bytes(data) if isinstance(data, memoryview) else data).translate(_BYTE_CLASSES)
    # ``rfind`` rather than ``find``: CPython's forward search barely skips for a needle that starts with a common
    # byte, while its reverse search keys on that first byte and skips the whole needle on a miss.
    return classes.startswith(_INTEGER_START[1:]) or classes.rfind(_INTEGER_START) >= 0


def _loads_stdlib(data: bytes | bytearray | memoryview | str, orjson_error: orjson.JSONDecodeError | None) -> Any:
    try:
        text = data if isinstance(data, str) else bytes(data).decode("utf-8")
        return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
    except (ValueError, TypeError, RecursionError) as error:
        if orjson_error is not None:
            raise orjson_error from None
        if isinstance(error, json.JSONDecodeError):
            raise orjson.JSONDecodeError(error.msg, error.doc, error.pos) from None
        raise orjson.JSONDecodeError(str(error), "", 0) from error


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not valid JSON")


def _finite_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        raise ValueError(f"number {literal} is too large for a float")
    return value


def _dumps_stdlib(value: object, *, default: Callable[[Any], Any] | None, option: int) -> bytes:
    def fallback_default(item: Any) -> Any:
        # A type the standard library cannot write (a datetime, a dataclass) is written as ``orjson`` would write it.
        return orjson.loads(orjson.dumps(item, default=default))

    indent = 2 if option & orjson.OPT_INDENT_2 else None
    kwargs: dict[str, Any] = dict(
        sort_keys=bool(option & orjson.OPT_SORT_KEYS),
        indent=indent,
        separators=(",", ": ") if indent else (",", ":"),
        allow_nan=False,
        default=fallback_default,
    )
    plain = _json_ready(value)
    try:
        encoded = json.dumps(plain, ensure_ascii=False, **kwargs).encode("utf-8")
    except UnicodeEncodeError:
        encoded = json.dumps(plain, ensure_ascii=True, **kwargs).encode("utf-8")
    LOG.debug("Encoded a JSON value orjson rejects with the standard library")
    return encoded + b"\n" if option & orjson.OPT_APPEND_NEWLINE else encoded


def _json_ready(value: Any) -> Any:
    """Copy ``value`` with non-finite floats as ``None``, refusing object keys that are not strings."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Dict key must be str")
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
