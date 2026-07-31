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

"""Content digest for one staged token delta.

``compute_staging_digest`` fingerprints a staged row -- token ids, mask values,
logprob bit patterns, and the identifying metadata -- at the worker, before the
row is handed to the framework's ``StagingSink``. The finalizer recomputes it over
the fetched row, so any storage-layer corruption or key substitution is
detected without the tokens ever making a second trip.

The encoding is frozen (see ``HASH_VERSION``): length-delimited big-endian
token ids, and masks/logprobs quantized to float32 BIT PATTERNS. Staging
columns are float32, so quantizing here makes the worker's digest (over
pre-tensorized python floats) and the finalizer's recomputation (over fetched
storage values) byte-identical, while -0.0 vs 0.0 and NaN payloads still
cannot alias.

This module is part of the dependency-free capture core: stdlib only.
"""

from __future__ import annotations

import hashlib
import struct


HASH_VERSION = 1
EMPTY_PREFIX_HASH = hashlib.sha256(b"nemo-rl-prefix-v1").hexdigest()


def _encode_bytes(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _encode_text(value: str) -> bytes:
    return _encode_bytes(value.encode("utf-8"))


def encode_token_ids(token_ids: list[int]) -> bytes:
    """Return a stable, length-delimited big-endian token encoding."""
    encoded = bytearray(struct.pack(">BQ", HASH_VERSION, len(token_ids)))
    for token_id in token_ids:
        if token_id < 0:
            raise ValueError(f"token IDs must be non-negative, got {token_id}")
        encoded.extend(struct.pack(">Q", token_id))
    return bytes(encoded)


def hash_token_ids(token_ids: list[int]) -> str:
    """Hash an exact token sequence using the frozen prefix encoding.

    Not used on the MVP hot path; reserved for the hardening layer's
    ``cum_hash``/``chain_hash`` fields so both sides share one encoding.
    """
    if not token_ids:
        return EMPTY_PREFIX_HASH
    return hashlib.sha256(b"nemo-rl-prefix" + encode_token_ids(token_ids)).hexdigest()


def compute_staging_digest(
    *,
    rollout_id: str,
    call_id: str,
    prev_len: int,
    token_ids_delta: list[int],
    token_mask_delta: list[float],
    logprobs_delta: list[float],
) -> str:
    """Digest one staged row: token ids, mask values, logprob bit patterns,
    and the identifying metadata. The finalizer recomputes it over the fetched
    row, so any storage-layer corruption or substitution is detected.
    """
    payload = bytearray(struct.pack(">B", HASH_VERSION))
    payload.extend(_encode_text(rollout_id))
    payload.extend(_encode_text(call_id))
    payload.extend(struct.pack(">Q", prev_len))
    payload.extend(_encode_bytes(encode_token_ids(token_ids_delta)))
    for mask_value in token_mask_delta:
        payload.extend(struct.pack(">f", mask_value))
    for logprob in logprobs_delta:
        payload.extend(struct.pack(">f", logprob))
    return hashlib.sha256(b"nemo-rl-staging-digest" + bytes(payload)).hexdigest()


def build_staging_delta(
    *,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    generated_logprobs: list[float],
    prev_len: int,
) -> tuple[list[int], list[float], list[float]]:
    """Slice one full request/response into the next per-call delta.

    The delta is ``rendered_prompt[prev_len:] + generated``, with mask 0.0 on
    the prompt carry and 1.0 on generated tokens; logprobs are 0.0 on the
    carry. ``rebuild.snapshots_to_entries`` is the exact inverse.
    """
    if prev_len < 0 or prev_len > len(prompt_token_ids):
        raise ValueError(f"prev_len={prev_len} is outside prompt length {len(prompt_token_ids)}")
    if len(generated_token_ids) != len(generated_logprobs):
        raise ValueError(
            "generated token and log-probability lengths differ: "
            f"{len(generated_token_ids)} != {len(generated_logprobs)}"
        )
    prompt_delta = prompt_token_ids[prev_len:]
    token_ids_delta = prompt_delta + generated_token_ids
    token_mask_delta = [0.0] * len(prompt_delta) + [1.0] * len(generated_token_ids)
    logprobs_delta = [0.0] * len(prompt_delta) + generated_logprobs
    if not token_ids_delta:
        raise ValueError("staging delta must contain at least one token")
    return token_ids_delta, token_mask_delta, logprobs_delta
