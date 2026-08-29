# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical integrity encoding for framework-owned token staging.

The call digest binds every custody-critical scalar and token column. Optional
per-token material is encoded separately and its digest is bound into the call
digest. This permits a metadata-only finalizer to verify the base row while a
later consumer verifies deferred extras against the same receipt.

All encodings are length-delimited and versioned. Token masks and log
probabilities are encoded as IEEE-754 float32 bit patterns because framework
staging columns use float32. Extras use a small typed binary encoding rather
than implementation-specific JSON serialization.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any


STAGING_SCHEMA_VERSION = 2
STAGING_DIGEST_VERSION = 2
EXTRAS_DIGEST_VERSION = 1

_CALL_DIGEST_DOMAIN = b"nemo-gym-staging-call-v2"
_EXTRAS_DIGEST_DOMAIN = b"nemo-gym-staging-extras-v1"
_TOKEN_DIGEST_DOMAIN = b"nemo-gym-staging-prefix-v2"
_HEX_DIGEST_LENGTH = 64


def _encode_bytes(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _encode_text(value: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"expected text, got {type(value).__name__}")
    return _encode_bytes(value.encode("utf-8"))


def _encode_optional_text(value: str | None) -> bytes:
    return b"\x00" if value is None else b"\x01" + _encode_text(value)


def _encode_uint(value: int, *, field: str) -> bytes:
    if type(value) is not int or not 0 <= value <= (2**64 - 1):
        raise ValueError(f"{field} must be an unsigned 64-bit integer, got {value!r}")
    return struct.pack(">Q", value)


def _validate_digest(value: str, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _HEX_DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _encode_optional_digest(value: str | None, *, field: str) -> bytes:
    if value is None:
        return b"\x00"
    _validate_digest(value, field=field)
    return b"\x01" + bytes.fromhex(value)


def encode_token_ids(token_ids: Sequence[int]) -> bytes:
    """Encode token IDs as a length followed by unsigned big-endian values."""
    encoded = bytearray(_encode_uint(len(token_ids), field="token_ids length"))
    for token_id in token_ids:
        encoded.extend(_encode_uint(token_id, field="token_id"))
    return bytes(encoded)


def hash_token_ids(token_ids: Sequence[int]) -> str:
    """Hash one exact cumulative token prefix."""
    return hashlib.sha256(_TOKEN_DIGEST_DOMAIN + encode_token_ids(token_ids)).hexdigest()


def _encode_float32_values(values: Sequence[float], *, field: str) -> bytes:
    encoded = bytearray(_encode_uint(len(values), field=f"{field} length"))
    for value in values:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{field} values must be finite Python floats, got {value!r}")
        try:
            packed = struct.pack(">f", value)
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{field} value cannot be represented as float32: {value!r}") from error
        if not math.isfinite(struct.unpack(">f", packed)[0]):
            raise ValueError(f"{field} value overflows float32: {value!r}")
        encoded.extend(packed)
    return bytes(encoded)


def _encode_extra(value: Any) -> bytes:
    if value is None:
        return b"N"
    if type(value) is bool:
        return b"B\x01" if value else b"B\x00"
    if type(value) is int:
        if not -(2**63) <= value <= (2**63 - 1):
            raise ValueError(f"extras integer is outside signed 64-bit range: {value}")
        return b"I" + struct.pack(">q", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"extras floats must be finite, got {value!r}")
        return b"F" + struct.pack(">d", value)
    if type(value) is str:
        return b"S" + _encode_text(value)
    if type(value) is list:
        return (
            b"L"
            + _encode_uint(len(value), field="extras list length")
            + b"".join(_encode_extra(item) for item in value)
        )
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("extras mappings must have string keys")
        keys = sorted(value, key=lambda key: key.encode("utf-8"))
        return (
            b"D"
            + _encode_uint(len(keys), field="extras mapping length")
            + b"".join(_encode_text(key) + _encode_extra(value[key]) for key in keys)
        )
    raise TypeError(f"extras contain unsupported value type {type(value).__name__}")


def compute_extras_digest(extras: Mapping[str, Any] | None) -> str:
    """Digest a normalized JSON-like extras envelope.

    Supported values are ``None``, exact ``bool``/``int``/``float``/``str``
    scalars, lists, and string-keyed dictionaries. This deliberately rejects
    numpy scalars, tuples, bytes, NaN, and infinity so different runtimes
    cannot silently choose different encodings.
    """
    normalized: dict[str, Any] | None
    if extras is None:
        normalized = None
    elif isinstance(extras, Mapping):
        normalized = dict(extras)
    else:
        raise TypeError(f"extras must be a mapping or None, got {type(extras).__name__}")
    payload = struct.pack(">B", EXTRAS_DIGEST_VERSION) + _encode_extra(normalized)
    return hashlib.sha256(_EXTRAS_DIGEST_DOMAIN + payload).hexdigest()


EMPTY_EXTRAS_DIGEST = compute_extras_digest(None)


def compute_staging_digest(
    *,
    schema_version: int,
    digest_version: int,
    extras_digest_version: int,
    rollout_id: str,
    model_call_id: str,
    parent_call_id: str | None,
    mode: str,
    prev_len: int,
    delta_len: int,
    cum_len: int,
    weight_version: int,
    token_ids_delta: Sequence[int],
    token_mask_delta: Sequence[float],
    generation_log_probs_delta: Sequence[float],
    extras_digest: str,
    chain_hash: str | None = None,
    cumulative_hash: str | None = None,
) -> str:
    """Compute the v2 digest for one staged call delta."""
    if type(schema_version) is not int or schema_version != STAGING_SCHEMA_VERSION:
        raise ValueError(f"unsupported staging schema version {schema_version}")
    if type(digest_version) is not int or digest_version != STAGING_DIGEST_VERSION:
        raise ValueError(f"unsupported staging digest version {digest_version}")
    if type(extras_digest_version) is not int or extras_digest_version != EXTRAS_DIGEST_VERSION:
        raise ValueError(f"unsupported extras digest version {extras_digest_version}")
    if not rollout_id or not model_call_id:
        raise ValueError("rollout_id and model_call_id must be non-empty")
    if parent_call_id == "":
        raise ValueError("parent_call_id must be non-empty when present")
    if mode not in ("text", "token_in"):
        raise ValueError(f"unsupported capture mode {mode!r}")
    if parent_call_id is None and (prev_len != 0 or mode != "text"):
        raise ValueError("a parentless call must be a text-mode root with prev_len == 0")
    if parent_call_id is not None and (prev_len == 0 or mode != "token_in"):
        raise ValueError("a child call must use token_in mode with prev_len > 0")
    if delta_len == 0:
        raise ValueError("a staged call delta must contain at least one token")
    if delta_len != len(token_ids_delta):
        raise ValueError(f"delta_len {delta_len} does not match {len(token_ids_delta)} token IDs")
    if not (len(token_ids_delta) == len(token_mask_delta) == len(generation_log_probs_delta)):
        raise ValueError("token IDs, masks, and log probabilities must have equal lengths")
    if any(mask not in (0.0, 1.0) for mask in token_mask_delta):
        raise ValueError("token_mask_delta must contain only 0.0 or 1.0")
    if any(mask == 0.0 and log_prob != 0.0 for mask, log_prob in zip(token_mask_delta, generation_log_probs_delta)):
        raise ValueError("prompt-carry log probabilities must be 0.0")
    if cum_len != prev_len + delta_len:
        raise ValueError(f"cum_len {cum_len} does not equal prev_len + delta_len ({prev_len + delta_len})")
    _validate_digest(extras_digest, field="extras_digest")

    payload = bytearray(struct.pack(">BBB", schema_version, digest_version, extras_digest_version))
    payload.extend(_encode_text(rollout_id))
    payload.extend(_encode_text(model_call_id))
    payload.extend(_encode_optional_text(parent_call_id))
    payload.extend(_encode_text(mode))
    payload.extend(_encode_uint(prev_len, field="prev_len"))
    payload.extend(_encode_uint(delta_len, field="delta_len"))
    payload.extend(_encode_uint(cum_len, field="cum_len"))
    payload.extend(_encode_uint(weight_version, field="weight_version"))
    payload.extend(_encode_bytes(encode_token_ids(token_ids_delta)))
    payload.extend(_encode_bytes(_encode_float32_values(token_mask_delta, field="token_mask_delta")))
    payload.extend(
        _encode_bytes(_encode_float32_values(generation_log_probs_delta, field="generation_log_probs_delta"))
    )
    payload.extend(bytes.fromhex(extras_digest))
    payload.extend(_encode_optional_digest(chain_hash, field="chain_hash"))
    payload.extend(_encode_optional_digest(cumulative_hash, field="cumulative_hash"))
    return hashlib.sha256(_CALL_DIGEST_DOMAIN + bytes(payload)).hexdigest()


def build_staging_delta(
    *,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    generated_log_probs: list[float],
    prev_len: int,
) -> tuple[list[int], list[float], list[float]]:
    """Slice a full prompt/generation pair into the next staged delta."""
    if prev_len < 0 or prev_len > len(prompt_token_ids):
        raise ValueError(f"prev_len={prev_len} is outside prompt length {len(prompt_token_ids)}")
    if len(generated_token_ids) != len(generated_log_probs):
        raise ValueError(
            "generated token and log-probability lengths differ: "
            f"{len(generated_token_ids)} != {len(generated_log_probs)}"
        )
    prompt_delta = prompt_token_ids[prev_len:]
    token_ids_delta = prompt_delta + generated_token_ids
    token_mask_delta = [0.0] * len(prompt_delta) + [1.0] * len(generated_token_ids)
    generation_log_probs_delta = [0.0] * len(prompt_delta) + generated_log_probs
    if not token_ids_delta:
        raise ValueError("staging delta must contain at least one token")
    return token_ids_delta, token_mask_delta, generation_log_probs_delta
