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

"""Encode staged records for integrity verification.

The staged-record digest covers the call metadata and token columns.
Optional extras are encoded separately.
Their digest is included in the staged-record digest.
This lets one consumer verify token data without loading extras.
A later consumer can verify extras against the same recorded digest.

All values use versioned, length-delimited encodings.
Masks and log probabilities use IEEE-754 float32 bit patterns.
Extras use a typed binary encoding instead of JSON serialization.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.records import StagedCallRecord


STAGING_SCHEMA_VERSION = 2
STAGING_DIGEST_VERSION = 2
EXTRAS_DIGEST_VERSION = 1

_CALL_DIGEST_DOMAIN = b"nemo-gym-staging-call-v2"
_EXTRAS_DIGEST_DOMAIN = b"nemo-gym-staging-extras-v1"
_TOKEN_DIGEST_DOMAIN = b"nemo-gym-staging-prefix-v2"
_CHAIN_DIGEST_DOMAIN = b"nemo-gym-staging-chain-v1"
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


def _encode_present_digest(value: str, *, field: str) -> bytes:
    # The leading presence byte is part of the frozen v2 layout; every staged
    # record carries both chain digests, so it is always ``0x01``.
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a digest string")
    _validate_digest(value, field=field)
    return b"\x01" + bytes.fromhex(value)


def encode_token_ids(token_ids: Sequence[int]) -> bytes:
    """Encode token IDs as a length followed by unsigned big-endian values."""
    length = _encode_uint(len(token_ids), field="token_ids length")
    for token_id in token_ids:
        if type(token_id) is not int or not 0 <= token_id <= (2**64 - 1):
            raise ValueError(f"token_id must be an unsigned 64-bit integer, got {token_id!r}")
    # Pack the column in C rather than allocating and extending once per token.
    # Explicit big-endian widths preserve the v2 wire bytes on every host.
    return length + struct.pack(f">{len(token_ids)}Q", *token_ids)


def hash_token_ids(token_ids: Sequence[int]) -> str:
    """Hash one exact cumulative token prefix."""
    return hashlib.sha256(_TOKEN_DIGEST_DOMAIN + encode_token_ids(token_ids)).hexdigest()


def compute_chain_hash(parent_chain_hash: str | None, token_ids_delta: Sequence[int]) -> str:
    """Chain one staged delta onto its parent's chain hash.

    ``chain_hash_N = H(chain_hash_{N-1} || token_ids_delta_N)`` lets every
    consumer verify parent-child token continuity without materializing the
    cumulative sequence. A root delta chains from ``None``. The marker byte
    keeps a root encoding from colliding with a child encoding.

    Args:
        parent_chain_hash: The parent call's chain hash, or ``None`` for a root.
        token_ids_delta: This call's exact staged token delta.

    Returns:
        The lowercase SHA-256 hex chain hash for this call.
    """
    payload = bytearray()
    if parent_chain_hash is None:
        payload += b"\x00"
    else:
        _validate_digest(parent_chain_hash, field="parent_chain_hash")
        payload += b"\x01" + bytes.fromhex(parent_chain_hash)
    payload += encode_token_ids(token_ids_delta)
    return hashlib.sha256(_CHAIN_DIGEST_DOMAIN + bytes(payload)).hexdigest()


def _encode_float32_values(values: Sequence[float], *, field: str) -> bytes:
    length = _encode_uint(len(values), field=f"{field} length")
    for value in values:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{field} values must be finite Python floats, got {value!r}")
    format_string = f">{len(values)}f"
    try:
        packed = struct.pack(format_string, *values)
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{field} value cannot be represented as float32") from error
    # Some runtimes round a finite float64 to float32 infinity instead of raising.
    if not all(map(math.isfinite, struct.unpack(format_string, packed))):
        raise ValueError(f"{field} value overflows float32")
    return length + packed


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
    chain_hash: str,
    cumulative_hash: str,
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
    _encode_uint(delta_len, field="delta_len")
    _encode_uint(cum_len, field="cum_len")
    _validate_digest(extras_digest, field="extras_digest")

    identity = _staging_identity(
        schema_version, digest_version, extras_digest_version, rollout_id, model_call_id, parent_call_id, mode
    )
    return _hash_staging_columns(
        identity=identity,
        prev_len=prev_len,
        weight_version=weight_version,
        token_bytes=encode_token_ids(token_ids_delta)[8:],
        mask_bytes=_encode_float32_values(token_mask_delta, field="token_mask_delta")[8:],
        logprob_bytes=_encode_float32_values(generation_log_probs_delta, field="generation_log_probs_delta")[8:],
        extras_digest=extras_digest,
        chain_hash=chain_hash,
        cumulative_hash=cumulative_hash,
    )


def _staging_identity(
    schema_version: int,
    digest_version: int,
    extras_digest_version: int,
    rollout_id: str,
    model_call_id: str,
    parent_call_id: str | None,
    mode: str,
) -> bytes:
    payload = bytearray(struct.pack(">BBB", schema_version, digest_version, extras_digest_version))
    payload.extend(_encode_text(rollout_id))
    payload.extend(_encode_text(model_call_id))
    payload.extend(_encode_optional_text(parent_call_id))
    payload.extend(_encode_text(mode))
    return bytes(payload)


def _hash_staging_columns(
    *,
    identity: bytes,
    prev_len: int,
    weight_version: int,
    token_bytes: bytes,
    mask_bytes: bytes,
    logprob_bytes: bytes,
    extras_digest: str,
    chain_hash: str,
    cumulative_hash: str,
) -> str:
    """Hash validated columns using the unchanged, length-delimited v2 layout."""
    delta_len = len(token_bytes) // 8
    if len(token_bytes) % 8 or len(mask_bytes) != delta_len * 4 or len(logprob_bytes) != delta_len * 4:
        raise ValueError("encoded staging columns have inconsistent lengths")
    digest = hashlib.sha256(_CALL_DIGEST_DOMAIN)
    digest.update(identity)
    digest.update(_encode_uint(prev_len, field="prev_len"))
    digest.update(_encode_uint(delta_len, field="delta_len"))
    digest.update(_encode_uint(prev_len + delta_len, field="cum_len"))
    digest.update(_encode_uint(weight_version, field="weight_version"))
    for column in (token_bytes, mask_bytes, logprob_bytes):
        digest.update(struct.pack(">QQ", len(column) + 8, delta_len))
        digest.update(column)
    digest.update(bytes.fromhex(extras_digest))
    digest.update(_encode_present_digest(chain_hash, field="chain_hash"))
    digest.update(_encode_present_digest(cumulative_hash, field="cumulative_hash"))
    return digest.hexdigest()


@dataclass(frozen=True)
class EncodedPrefixDigest:
    """Immutable per-call cache of validated v2 prefix columns.

    ``extend`` returns a candidate; adopt it only after the cut is staged and
    sealed. The cache is process-local, never serialized as checkpoint state.
    Hashing still covers the full prefix, but old columns are not re-encoded.
    """

    _identity: bytes
    _prev_len: int
    _weight_version: int
    _parent_chain_hash: str | None
    _parent_token_bytes: bytes
    _token_bytes: bytes
    _mask_bytes: bytes
    _logprob_bytes: bytes
    _extras_digest: str
    digest: str

    @classmethod
    def from_record(
        cls,
        record: StagedCallRecord,
        *,
        parent_chain_hash: str | None,
        parent_token_ids: Sequence[int],
    ) -> EncodedPrefixDigest:
        """Copy a validated record into immutable columns and verify its binding."""
        if len(parent_token_ids) != record.prev_len:
            raise ValueError("prefix cache parent length does not match record")
        cache = cls(
            _identity=_staging_identity(
                record.schema_version,
                record.digest_version,
                record.extras_digest_version,
                record.rollout_id,
                record.model_call_id,
                record.parent_call_id,
                record.mode,
            ),
            _prev_len=record.prev_len,
            _weight_version=record.weight_version,
            _parent_chain_hash=parent_chain_hash,
            _parent_token_bytes=encode_token_ids(parent_token_ids)[8:],
            _token_bytes=encode_token_ids(record.token_ids_delta)[8:],
            _mask_bytes=_encode_float32_values(record.token_mask_delta, field="token_mask_delta")[8:],
            _logprob_bytes=_encode_float32_values(
                record.generation_log_probs_delta, field="generation_log_probs_delta"
            )[8:],
            _extras_digest=record.extras_digest,
            digest=record.digest,
        )
        if cache._compute_digest() != record.digest:
            raise ValueError("prefix cache contents or parent do not match record digest")
        return cache

    def extend(
        self, *, generated_token_ids: Sequence[int], generated_logprobs: Sequence[float]
    ) -> EncodedPrefixDigest:
        """Validate new generated tokens and return an independently owned candidate."""
        if len(generated_token_ids) != len(generated_logprobs):
            raise ValueError("generated token IDs and log probabilities must have equal lengths")
        if not generated_token_ids:
            return self
        candidate = replace(
            self,
            _token_bytes=self._token_bytes + encode_token_ids(generated_token_ids)[8:],
            _mask_bytes=self._mask_bytes + struct.pack(">f", 1.0) * len(generated_token_ids),
            _logprob_bytes=self._logprob_bytes
            + _encode_float32_values(generated_logprobs, field="generation_log_probs_delta")[8:],
        )
        return replace(candidate, digest=candidate._compute_digest())

    def _compute_digest(self) -> str:
        delta_len = len(self._token_bytes) // 8
        chain = hashlib.sha256(_CHAIN_DIGEST_DOMAIN)
        if self._parent_chain_hash is None:
            chain.update(b"\x00")
        else:
            chain.update(_encode_present_digest(self._parent_chain_hash, field="parent_chain_hash"))
        chain.update(struct.pack(">Q", delta_len))
        chain.update(self._token_bytes)
        cumulative = hashlib.sha256(_TOKEN_DIGEST_DOMAIN)
        cumulative.update(struct.pack(">Q", self._prev_len + delta_len))
        cumulative.update(self._parent_token_bytes)
        cumulative.update(self._token_bytes)
        return _hash_staging_columns(
            identity=self._identity,
            prev_len=self._prev_len,
            weight_version=self._weight_version,
            token_bytes=self._token_bytes,
            mask_bytes=self._mask_bytes,
            logprob_bytes=self._logprob_bytes,
            extras_digest=self._extras_digest,
            chain_hash=chain.hexdigest(),
            cumulative_hash=cumulative.hexdigest(),
        )


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
