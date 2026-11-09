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

One exception to "exact": NaN and infinity are written as ``null``, as ``orjson`` writes them. Values ``orjson``
accepts must keep their bytes, so the fallback follows the same rule rather than make a body's meaning depend on
whether it also holds a wide integer.
"""

import dataclasses
import enum
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
# 19 digits. Each check maps bytes to classes and searches for 19 digits that start a number.
#
# The first check is cheap: "0" for a digit, "." for itself and "x" for anything else. A run that follows "." is a
# fraction, which ``orjson`` reads exactly, and shortest float text from Python writers has at most 17 significant
# digits before switching to an exponent, so logprobs and token IDs never match.
_DIGIT_RUN_CLASSES = bytes(
    ord("0") if byte in b"0123456789" else byte if byte == ord(".") else ord("x") for byte in range(256)
)
_DIGIT_RUN = b"x" + b"0" * 19
# The second runs only when the first matches, since spaces make its search slower: a JSON number can only follow
# the start of the input, whitespace, "[", ",", ":" or "-", marked "s". It drops most matches inside strings, such as
# IDs and timestamps in paths. What still matches, such as a 19-digit integer that fits, costs a standard library
# parse of that body, which gives the same values as ``orjson`` would.
_NUMBER_START_CLASSES = bytes(
    ord("0") if byte in b"0123456789" else ord("s") if byte in b" \t\n\r[,:-" else ord("x") for byte in range(256)
)
_NUMBER_START = b"s" + b"0" * 19


def dumps_json(value: object, *, default: Callable[[Any], Any] | None = None, option: int | None = None) -> bytes:
    """Serialize ``value`` like ``orjson.dumps``, falling back to the standard library for what ``orjson`` rejects.

    A value ``orjson`` accepts keeps its exact ``orjson`` bytes, so existing records and hashes do not change. The
    fallback covers integers wider than 64 bits, lone surrogates, and nesting deeper than ``orjson`` allows. It writes
    NaN and infinity as ``null``, as ``orjson`` does, so a value means the same whether or not the body also holds a
    wide integer. It escapes non-ASCII text only when the text holds a lone surrogate, so the bytes stay valid UTF-8.

    ``option`` takes ``orjson`` flags; the fallback supports ``OPT_SORT_KEYS``, ``OPT_INDENT_2`` and
    ``OPT_APPEND_NEWLINE``. Object keys must be strings: the fallback refuses other keys rather than coerce them,
    which could merge two keys into one. The fallback accepts only the types ``orjson`` accepts, and for the others
    reuses what ``default`` returned during the ``orjson`` pass, so a ``default`` that consumes its argument, such as
    ``list`` on a generator, still gives the same output.

    Raises:
        orjson.JSONEncodeError: ``value`` holds something neither encoder can write exactly, or ``orjson`` rejects
            it while ``option`` holds a flag the fallback cannot honor.
    """
    # Each object ``default`` saw, kept alive so its id is not reused, with what ``default`` returned for it.
    results: dict[int, tuple[object, object]] = {}

    def recording_default(obj: Any) -> Any:
        result = default(obj)
        results.setdefault(id(obj), (obj, result))
        return result

    def replaying_default(obj: Any) -> Any:
        seen = results.get(id(obj))
        return seen[1] if seen is not None and seen[0] is obj else default(obj)

    try:
        return orjson.dumps(value, default=recording_default if default else None, option=option)
    except orjson.JSONEncodeError as error:
        if option and option & ~_SUPPORTED_OPTIONS:
            raise
        try:
            return _dumps_stdlib(value, default=replaying_default if default else None, option=option or 0)
        except (TypeError, ValueError, RecursionError) as fallback_error:
            raise orjson.JSONEncodeError(f"{error}; standard library fallback: {fallback_error}") from fallback_error


def loads_json(data: bytes | bytearray | memoryview | str) -> Any:
    """Parse JSON like ``orjson.loads``, but read integers wider than 64 bits and lone surrogates exactly.

    The common case costs one byte translation and a substring search on top of ``orjson``. A body that may hold an
    integer literal ``orjson`` would read as a float is parsed with the standard library instead, as is a body
    ``orjson`` rejects but the standard library reads exactly: an escaped lone surrogate, or nesting deeper than
    ``orjson`` allows. The fallback refuses ``NaN``, ``Infinity`` and numbers that overflow a float, as ``orjson``
    does. A whole number of 19 or more digits with no fraction or exponent reads as an ``int``, as the standard
    library reads it, even where a JavaScript or Go writer meant a float.

    Raises:
        orjson.JSONDecodeError: ``data`` is not valid JSON, or not a contiguous bytes-like object or ``str``. A body
            both parsers reject raises ``orjson``'s error.
    """
    if type(data) is bytes:
        # Replies and record lines arrive as bytes: check them inline, without the type dispatch below.
        wide = data.translate(_DIGIT_RUN_CLASSES).rfind(_DIGIT_RUN) >= 0 or data[:19].isdigit()
        wide = wide and _starts_a_wide_number(data)
    elif isinstance(data, (bytearray, str)) or (isinstance(data, memoryview) and data.contiguous):
        wide = _starts_a_wide_number(data.encode("utf-8", "surrogatepass") if isinstance(data, str) else bytes(data))
    else:
        return orjson.loads(data)  # ``orjson`` raises its own error, before ``bytes()`` copies or allocates anything
    if wide:
        return _loads_stdlib(data, None)
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError as error:
        return _loads_stdlib(data, error)


def _starts_a_wide_number(data: bytes | bytearray) -> bool:
    """Whether ``data`` may hold an integer literal ``orjson`` would read as a float, allowing false positives."""
    classes = data.translate(_NUMBER_START_CLASSES)
    # ``rfind`` rather than ``find``: CPython's forward search barely skips for a needle that starts with a common
    # byte, while its reverse search keys on that first byte and skips the whole needle on a miss.
    return classes.startswith(_NUMBER_START[1:]) or classes.rfind(_NUMBER_START) >= 0


def _loads_stdlib(data: bytes | bytearray | memoryview | str, orjson_error: orjson.JSONDecodeError | None) -> Any:
    try:
        text = data if isinstance(data, str) else bytes(data).decode("utf-8")
        value = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
        LOG.debug("Decoded a JSON body with the standard library: orjson would change or reject it")
        return value
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
    indent = 2 if option & orjson.OPT_INDENT_2 else None
    kwargs: dict[str, Any] = dict(
        sort_keys=bool(option & orjson.OPT_SORT_KEYS),
        indent=indent,
        separators=(",", ": ") if indent else (",", ":"),
        allow_nan=False,
    )
    plain = _json_ready(value, default)
    try:
        encoded = json.dumps(plain, ensure_ascii=False, **kwargs).encode("utf-8")
    except UnicodeEncodeError:
        encoded = json.dumps(plain, ensure_ascii=True, **kwargs).encode("utf-8")
    LOG.debug("Encoded a JSON value orjson rejects with the standard library")
    return encoded + b"\n" if option & orjson.OPT_APPEND_NEWLINE else encoded


def _json_ready(value: Any, default: Callable[[Any], Any] | None) -> Any:
    """Copy ``value`` as plain JSON types, accepting only the types ``orjson`` accepts.

    The standard library also writes namedtuples, float and tuple subclasses and NumPy floats, which ``orjson``
    refuses; they must not start encoding just because a wide integer sent the body here. Non-finite floats become
    ``None`` and object keys that are not strings are refused.
    """
    if value is None or isinstance(value, (str, int)):  # ``orjson`` accepts str and int subclasses; bool is an int
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):  # ``orjson`` refuses str subclasses as keys too
            raise TypeError("Dict key must be str")
        return {key: _json_ready(item, default) for key, item in value.items()}
    if isinstance(value, list) or type(value) is tuple:
        return [_json_ready(item, default) for item in value]
    # Containers ``orjson`` writes natively are unpacked here, so a wide integer inside one is written, not passed
    # to ``default`` with the whole container.
    # ``orjson`` writes an instance only of a class decorated itself, not of an undecorated subclass of one.
    if "__dataclass_fields__" in type(value).__dict__:
        # ``orjson`` leaves out fields whose names start with an underscore.
        return {
            field.name: _json_ready(getattr(value, field.name), default)
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, enum.Enum):
        return _json_ready(value.value, default)
    try:
        # Other types ``orjson`` writes natively, such as datetime and UUID, are written as it writes them.
        return orjson.loads(orjson.dumps(value))
    except TypeError:
        if default is None:
            raise
    return _json_ready(default(value), default)
