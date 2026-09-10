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
"""Encode token metadata as versioned base64 strings.

The wire format is ``ngtok1:<dtype>:<base64>``.
Token IDs use little-endian ``i32`` values.
Log probabilities use little-endian ``f64`` values.
Decoders also accept plain lists.
"""

from __future__ import annotations

import base64
import sys
from array import array
from typing import Any, Dict, Iterable, List, Union


# Change the prefix when the byte layout changes.
TOKEN_ENVELOPE_PREFIX = "ngtok1:"
I32_DTYPE = "i32"
F64_DTYPE = "f64"


def is_token_envelope(value: Any) -> bool:
    """Return whether ``value`` has a supported envelope header."""
    if not isinstance(value, str):
        return False
    try:
        token_envelope_dtype(value)
    except ValueError:
        return False
    return True


def token_envelope_dtype(value: str, expected: Iterable[str] | None = None) -> str:
    """Return the dtype from a supported envelope header.

    This validates the header without decoding the payload.
    """
    if not value.startswith(TOKEN_ENVELOPE_PREFIX):
        raise ValueError(f"expected an {TOKEN_ENVELOPE_PREFIX!r} token-metadata envelope")
    dtype, separator, _ = value[len(TOKEN_ENVELOPE_PREFIX) :].partition(":")
    if not separator:
        raise ValueError("token-metadata envelope header must end with ':'")
    _typecode_for(dtype)
    if expected is not None:
        expected_dtypes = tuple(expected)
        if dtype not in expected_dtypes:
            raise ValueError(f"token-metadata dtype must be one of {expected_dtypes}, got {dtype!r}")
    return dtype


def encode_token_list(values: List[Union[int, float]], dtype: str) -> str:
    """Pack a numeric list into a token-metadata envelope."""
    typecode = _typecode_for(dtype)
    try:
        packed = array(typecode, values)
    except (TypeError, OverflowError) as error:
        raise ValueError(f"cannot encode token metadata as {dtype}: {error}") from error
    if sys.byteorder == "big":  # pragma: no cover - little-endian everywhere we run
        packed.byteswap()
    return f"{TOKEN_ENVELOPE_PREFIX}{dtype}:{base64.b64encode(packed.tobytes()).decode('ascii')}"


def decode_token_list(value: Union[str, list], expected_dtypes: Iterable[str] | None = None) -> list:
    """Decode an envelope or return an existing list unchanged."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError(f"expected a list or an {TOKEN_ENVELOPE_PREFIX!r} envelope, got {type(value).__name__}")
    if not value.startswith(TOKEN_ENVELOPE_PREFIX):
        raise ValueError(f"expected a list or an {TOKEN_ENVELOPE_PREFIX!r} envelope, got str")
    dtype = token_envelope_dtype(value, expected_dtypes)
    payload = value.split(":", 2)[2]
    typecode = _typecode_for(dtype)
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError(f"malformed base64 payload in token-metadata envelope: {error}") from error
    unpacked = array(typecode)
    itemsize = unpacked.itemsize
    if len(raw) % itemsize:
        raise ValueError(f"token-metadata envelope payload of {len(raw)} bytes is not a multiple of {itemsize}")
    unpacked.frombytes(raw)
    if sys.byteorder == "big":  # pragma: no cover - little-endian everywhere we run
        unpacked.byteswap()
    return unpacked.tolist()


def encode_output_item_token_fields(item_dict: Dict[str, Any]) -> None:
    """Encode token-metadata lists on an output item in place."""
    for field in ("prompt_token_ids", "generation_token_ids"):
        value = item_dict.get(field)
        if isinstance(value, list):
            item_dict[field] = encode_token_list(value, I32_DTYPE)
    value = item_dict.get("generation_log_probs")
    if isinstance(value, list):
        item_dict["generation_log_probs"] = encode_token_list(value, F64_DTYPE)


def decode_output_item_token_fields(item_dict: Dict[str, Any]) -> None:
    """Decode token-metadata envelopes on an output item in place."""
    for field, expected_dtypes in (
        ("prompt_token_ids", (I32_DTYPE,)),
        ("generation_token_ids", (I32_DTYPE,)),
        ("generation_log_probs", (F64_DTYPE,)),
    ):
        value = item_dict.get(field)
        if isinstance(value, str):
            item_dict[field] = decode_token_list(value, expected_dtypes)


def _typecode_for(dtype: str) -> str:
    try:
        return _TYPECODES[dtype]
    except KeyError:
        raise ValueError(f"unsupported token-metadata dtype {dtype!r}; expected one of {sorted(_TYPECODES)}") from None


def _int32_typecode() -> str:
    # Resolve a four-byte signed typecode for the fixed wire layout.
    for code in ("i", "l"):
        if array(code).itemsize == 4:
            return code
    raise RuntimeError("no 4-byte signed integer array typecode on this platform")  # pragma: no cover


_TYPECODES = {I32_DTYPE: _int32_typecode(), F64_DTYPE: "d"}
