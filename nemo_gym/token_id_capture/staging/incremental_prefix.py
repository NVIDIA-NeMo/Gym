# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned active-prefix identities with append-only token column digests.

Staged chunks retain their v2 format. Only the cumulative active-prefix identity
uses v3: independently hashed token, mask and logprob columns plus call metadata,
lineage hashes and extras. Restorers recompute these hashes from verified chunks.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from nemo_gym.token_id_capture.staging.digest import (
    EXTRAS_DIGEST_VERSION,
    STAGING_SCHEMA_VERSION,
    PackedFloat32Values,
    PackedTokenIds,
    _encode_optional_text,
    _encode_present_digest,
    _encode_text,
    _encode_uint,
    _validate_digest,
)


if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot

GENERATION_PREFIX_DIGEST_VERSION = 3
_PREFIX_DOMAIN = b"nemo-gym-generation-prefix-v3"
_CHAIN_DOMAIN = b"nemo-gym-generation-chain-v3"
_CUMULATIVE_DOMAIN = b"nemo-gym-generation-tokens-v3"


class _HashState(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def copy(self) -> _HashState: ...
    def digest(self) -> bytes: ...


@dataclass(frozen=True)
class PrefixMetadata:
    schema_version: int
    rollout_id: str
    model_call_id: str
    parent_call_id: str | None
    mode: str
    prev_len: int
    weight_version: int


@dataclass(frozen=True)
class PrefixIdentity:
    digest: str
    chain_hash: str
    cumulative_hash: str


class IncrementalPrefixDigest:
    """Owned hash states advanced transactionally after a chunk becomes durable."""

    def __init__(self, *, prefix_token_ids: Sequence[int], parent_chain_hash: str | None) -> None:
        self._prefix_count = len(prefix_token_ids)
        self._prefix_hash = hashlib.sha256(PackedTokenIds(prefix_token_ids).encoded[8:]).digest()
        if parent_chain_hash is not None:
            _validate_digest(parent_chain_hash, field="parent_chain_hash")
        self._parent_chain_hash = parent_chain_hash
        self._tokens: _HashState = hashlib.sha256()
        self._mask: _HashState = hashlib.sha256()
        self._logprobs: _HashState = hashlib.sha256()
        self.count = 0
        self.generated_count = 0

    def copy(self) -> IncrementalPrefixDigest:
        """Fork a candidate; a failed TQ write must not advance committed state."""
        candidate = object.__new__(IncrementalPrefixDigest)
        candidate._prefix_count = self._prefix_count
        candidate._prefix_hash = self._prefix_hash
        candidate._parent_chain_hash = self._parent_chain_hash
        candidate._tokens = self._tokens.copy()
        candidate._mask = self._mask.copy()
        candidate._logprobs = self._logprobs.copy()
        candidate.count = self.count
        candidate.generated_count = self.generated_count
        return candidate

    def append(self, *, tokens: Sequence[int], mask: Sequence[float], logprobs: Sequence[float]) -> None:
        """Validate and hash only new column bytes; never retain engine lists."""
        if not len(tokens) == len(mask) == len(logprobs):
            raise ValueError("prefix token, mask and logprob lengths differ")
        packed_tokens = PackedTokenIds(tokens).encoded[8:]
        packed_mask = PackedFloat32Values(mask, field="token_mask_delta").encoded[8:]
        packed_logprobs = PackedFloat32Values(logprobs, field="generation_log_probs_delta").encoded[8:]
        prompt_count = mask.count(0.0)
        generated_count = mask.count(1.0)
        if prompt_count + generated_count != len(mask):
            raise ValueError("prefix masks must contain only 0.0 or 1.0")
        if prompt_count and any(
            value != 0.0 for value, selected in zip(logprobs, mask, strict=True) if selected == 0.0
        ):
            raise ValueError("prompt-carry log probabilities must be 0.0")
        # Publish only after every column passed validation.
        self._tokens.update(packed_tokens)
        self._mask.update(packed_mask)
        self._logprobs.update(packed_logprobs)
        self.count += len(tokens)
        self.generated_count += generated_count

    @property
    def chain_hash(self) -> str:
        parent = b"\x00" if self._parent_chain_hash is None else b"\x01" + bytes.fromhex(self._parent_chain_hash)
        return hashlib.sha256(
            _CHAIN_DOMAIN + parent + struct.pack(">Q", self.count) + self._tokens.digest()
        ).hexdigest()

    @property
    def cumulative_hash(self) -> str:
        return hashlib.sha256(
            _CUMULATIVE_DOMAIN
            + struct.pack(">QQ", self._prefix_count, self.count)
            + self._prefix_hash
            + self._tokens.digest()
        ).hexdigest()

    def digest(
        self,
        metadata: PrefixMetadata,
        *,
        extras_digest: str,
        chain_hash: str,
        cumulative_hash: str,
    ) -> str:
        """Bind column contents, lengths, identity, policy and parent lineage."""
        if metadata.schema_version != STAGING_SCHEMA_VERSION or self.count == 0:
            raise ValueError("unsupported or empty generation prefix")
        if not metadata.rollout_id or not metadata.model_call_id:
            raise ValueError("prefix identity requires rollout and model-call IDs")
        if metadata.parent_call_id is None:
            if metadata.mode != "text" or metadata.prev_len != 0:
                raise ValueError("root prefix must use text mode with prev_len=0")
        elif not metadata.parent_call_id or metadata.mode != "token_in" or metadata.prev_len == 0:
            raise ValueError("child prefix must use token_in mode with prev_len>0")
        _validate_digest(extras_digest, field="extras_digest")
        payload = bytearray(
            struct.pack(
                ">BBB",
                metadata.schema_version,
                GENERATION_PREFIX_DIGEST_VERSION,
                EXTRAS_DIGEST_VERSION,
            )
        )
        payload.extend(_encode_text(metadata.rollout_id))
        payload.extend(_encode_text(metadata.model_call_id))
        payload.extend(_encode_optional_text(metadata.parent_call_id))
        payload.extend(_encode_text(metadata.mode))
        for name, value in (
            ("prev_len", metadata.prev_len),
            ("delta_len", self.count),
            ("cum_len", metadata.prev_len + self.count),
            ("weight_version", metadata.weight_version),
        ):
            payload.extend(_encode_uint(value, field=name))
        payload.extend(self._tokens.digest())
        payload.extend(self._mask.digest())
        payload.extend(self._logprobs.digest())
        payload.extend(bytes.fromhex(extras_digest))
        payload.extend(_encode_present_digest(chain_hash, field="chain_hash"))
        payload.extend(_encode_present_digest(cumulative_hash, field="cumulative_hash"))
        return hashlib.sha256(_PREFIX_DOMAIN + payload).hexdigest()

    def identity(self, metadata: PrefixMetadata, *, extras_digest: str) -> PrefixIdentity:
        if metadata.prev_len != self._prefix_count:
            raise ValueError("generation prefix parent length changed")
        chain, cumulative = self.chain_hash, self.cumulative_hash
        return PrefixIdentity(
            self.digest(
                metadata,
                extras_digest=extras_digest,
                chain_hash=chain,
                cumulative_hash=cumulative,
            ),
            chain,
            cumulative,
        )


def snapshot_prefix_digest(snapshot: StagedCallBaseSnapshot) -> str:
    """Cold-path independent verification of a restored v3 snapshot's columns."""
    state = IncrementalPrefixDigest(prefix_token_ids=(), parent_chain_hash=None)
    state.append(
        tokens=snapshot.token_ids_delta,
        mask=snapshot.token_mask_delta,
        logprobs=snapshot.generation_log_probs_delta,
    )
    if snapshot.delta_len != state.count or snapshot.cum_len != snapshot.prev_len + state.count:
        raise ValueError("generation-prefix lengths do not match its columns")
    metadata = PrefixMetadata(
        snapshot.schema_version,
        snapshot.rollout_id,
        snapshot.model_call_id,
        snapshot.parent_call_id,
        snapshot.mode,
        snapshot.prev_len,
        snapshot.weight_version,
    )
    return state.digest(
        metadata,
        extras_digest=snapshot.extras_digest,
        chain_hash=snapshot.chain_hash,
        cumulative_hash=snapshot.cumulative_hash,
    )
