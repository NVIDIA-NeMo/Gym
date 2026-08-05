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

"""Define training-token records extracted from served responses.

A ``TokenEntry`` contains one model call's training data.
It stores the exact prompt token ids.
It stores generated token ids and their log probabilities.
Evaluation uses a separate ``ModelCallRecord``.
Evaluation records do not carry token arrays.
Both records share a ``model_call_id``.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# These fields carry token metadata on a served response.
# ``routed_experts`` is optional for MoE backends.
TOKEN_FIELDS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs", "routed_experts")

# Increment this version when a field or its meaning changes.
# Writers and readers may run in different processes or repositories.
# Records may outlive a deployment.
# Readers must reject unsupported newer records.
# ``extra="allow"`` otherwise hides unknown fields.
#
#   1  rollout and call identity, the token arrays, the output items and their carrier index
#   2  parent_call_id, cum_len and digest, added when calls began being linked to their parent
TOKEN_ENTRY_RECORD_SCHEMA_VERSION = 2

# Bumped if the digest encoding below ever changes, so a stale digest fails to
# verify instead of silently comparing equal.
DIGEST_VERSION = 1
_DIGEST_DOMAIN = b"nemo-gym-tokens"
_EMPTY_DIGEST = hashlib.sha256(_DIGEST_DOMAIN).hexdigest()


def encode_token_ids(token_ids: list[int]) -> bytes:
    """Stable, length-delimited, big-endian encoding of a token sequence.

    The encoding is fixed and length-delimited so an independent implementation
    can hash the same bytes and get the same digest.
    """
    encoded = bytearray(struct.pack(">BQ", DIGEST_VERSION, len(token_ids)))
    for token_id in token_ids:
        if token_id < 0:
            raise ValueError(f"token ids must be non-negative, got {token_id}")
        encoded.extend(struct.pack(">Q", token_id))
    return bytes(encoded)


def compute_digest(token_ids: list[int]) -> str:
    """Digest of an exact token sequence.

    Used to verify a claimed ``parent_call_id`` in O(1) instead of comparing
    whole arrays, and to detect a stale or interleaved store (a rerun that
    appended onto a previous attempt's records fails the check rather than
    merging silently).
    """
    if not token_ids:
        return _EMPTY_DIGEST
    return hashlib.sha256(_DIGEST_DOMAIN + encode_token_ids(token_ids)).hexdigest()


class TokenEntry(BaseModel):
    """Store one model call's content and token metadata.

    The rollout id identifies the training sample.
    The model call id joins evaluation context.
    ``output_items`` preserves assistant text and tool calls.
    Text-based penalties require that content.
    Token arrays are stored once at the top level.
    ``token_item_index`` identifies their original output item.
    A trajectory builder can restore chain-correct token fields there.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = TOKEN_ENTRY_RECORD_SCHEMA_VERSION
    rollout_id: str
    model_call_id: str
    model: str = ""
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    generation_log_probs: list[float]
    routed_experts: Any | None = None
    # Preserve response output items without token arrays.
    output_items: list[dict] = Field(default_factory=list)
    # This index identifies the item that carried token arrays.
    # ``None`` means no item carried them.
    token_item_index: int | None = None
    # This non-semantic timestamp helps diagnose retries and sibling branches.
    created_at: float = 0.0

    # --- Added with parent resolution (schema version 2).
    # Optional: null when the model server could not identify the parent, in which case the
    # builder matches token prefixes instead.
    #
    # When present these make the chain exact rather than matched. Two calls
    # can share a prompt and differ only in their generation (a harness retry,
    # since capture records a response the client may never have received);
    # prefix matching has to guess between them, whereas the next call's parent
    # link names the one the harness actually kept. They also turn the builder's
    # O(N^2) prefix scan into an O(1) lookup, and let a claimed parent be
    # verified rather than trusted.
    parent_call_id: str | None = None
    # len(prompt_token_ids) + len(generation_token_ids) for THIS call: the
    # length of the prefix a child of this call must start with.
    cum_len: int | None = None
    # compute_digest(prompt_token_ids + generation_token_ids).
    digest: str | None = None

    @model_validator(mode="after")
    def _refuse_a_newer_record(self) -> "TokenEntry":
        """Accept older records and reject newer records.

        Missing older fields use their defaults.
        Unknown newer fields may change token semantics.
        Rejecting them prevents silent training corruption.
        """
        if self.schema_version > TOKEN_ENTRY_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"token record is schema_version {self.schema_version}, but this reader understands "
                f"up to {TOKEN_ENTRY_RECORD_SCHEMA_VERSION}. Upgrade the reader, or point it at "
                "records written by a writer it matches."
            )
        if len(self.generation_token_ids) != len(self.generation_log_probs):
            raise ValueError(
                "generation_token_ids and generation_log_probs must have the same length "
                f"(got {len(self.generation_token_ids)} and {len(self.generation_log_probs)})"
            )
        if self.token_item_index is not None and not 0 <= self.token_item_index < len(self.output_items):
            raise ValueError(
                f"token_item_index {self.token_item_index} is outside output_items of length {len(self.output_items)}"
            )
        return self


def cumulative_tokens(entry: TokenEntry) -> list[int]:
    """The full sequence a child of this call must start with."""
    return list(entry.prompt_token_ids) + list(entry.generation_token_ids)


def stamp_lineage(entry: TokenEntry, parent_call_id: str | None) -> TokenEntry:
    """Fill ``cum_len`` and ``digest`` (always) and ``parent_call_id`` (when known).

    ``cum_len``/``digest`` describe this call and are always computable; the
    parent link is only known when the model server resolved one.
    """
    cumulative = cumulative_tokens(entry)
    entry.cum_len = len(cumulative)
    entry.digest = compute_digest(cumulative)
    entry.parent_call_id = parent_call_id
    return entry


def response_to_output_items(payload: dict) -> list[dict]:
    """Normalize a served response to a list of content-bearing Responses output items.

    Responses payloads already carry ``output``.
    Chat payloads carry ``choices[*].message``.
    Wrap each assistant message as a Responses ``message`` item.
    """
    output = payload.get("output")
    if isinstance(output, list) and output:
        return [item for item in output if isinstance(item, dict)]
    items: list[dict] = []
    for choice in payload.get("choices") or []:
        message = (choice or {}).get("message") or {}
        if not isinstance(message, dict):
            continue
        item = dict(message)
        item.setdefault("type", "message")
        item.setdefault("role", "assistant")
        items.append(item)
    return items


def strip_token_fields(items: list[dict]) -> tuple[list[dict], int | None]:
    """Drop the token arrays from output items, keeping the content.

    Return the stripped items and the index of their token-bearing item.
    Capture requires exactly one token-bearing item.
    The arrays are held once on the entry.
    Storing them again per item would roughly double the record size.
    """
    indices: list[int] = []
    stripped: list[dict] = []
    for position, item in enumerate(items):
        if item.get("generation_token_ids") is not None:
            indices.append(position)
        stripped.append({key: value for key, value in item.items() if key not in TOKEN_FIELDS})
    if len(indices) > 1:
        raise ValueError("multiple output items carry token metadata")
    return stripped, indices[0] if indices else None


def extract_token_fields(response_json: dict) -> dict | None:
    """Pull the token-id fields off a served response, or ``None`` if absent.

    Handle Responses output items and Chat Completions messages.
    Exactly one item may carry token metadata.
    Return ``None`` when no item carries token ids.
    """
    candidates: list[dict] = []
    required = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")
    for item in response_json.get("output") or []:
        if isinstance(item, dict) and any(item.get(field) is not None for field in required):
            candidates.append(item)
    for choice in response_json.get("choices") or []:
        message = (choice or {}).get("message") or {}
        if isinstance(message, dict) and any(message.get(field) is not None for field in required):
            candidates.append(message)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError("multiple response items carry token metadata")
    source = candidates[0]
    missing = [field for field in required if source.get(field) is None]
    if missing:
        raise ValueError(f"partial token metadata is missing: {', '.join(missing)}")
    return {field: source.get(field) for field in TOKEN_FIELDS}
