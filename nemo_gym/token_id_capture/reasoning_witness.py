# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disambiguate reasoning-only continuations using exact served output witnesses.

The caller supplies only otherwise ambiguous, committed lineage candidates.
The echoed output must immediately follow an unchanged request context, retain
all server-issued reasoning IDs and content, and contain no additional model
turn. Missing, altered or duplicate witnesses abstain. Persist the witness in
the same atomic custody row as the captured call; legacy rows remain unchanged.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningWitness:
    """Token-free identity of a served reasoning-only output and its context."""

    call_id: str
    context_len: int
    context_digest: str
    output_digest: str
    output_ids: tuple[str, ...]


def _normalize_item(item: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(item)
    if item.get("role") and item.get("type") in {None, "message"}:
        normalized["type"] = "message"
    if item.get("type") == "reasoning":
        # The SDK drops these two absent optional fields. Preserve every
        # non-null value, including summaries and encrypted reasoning.
        for key in ("content", "encrypted_content"):
            if normalized.get(key) is None:
                normalized.pop(key, None)
    return normalized


def _digest(items: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        [_normalize_item(item) for item in items], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _reasoning_digest(items: Sequence[Mapping[str, object]]) -> tuple[str, tuple[str, ...]] | None:
    if not items or any(item.get("type") != "reasoning" for item in items):
        return None
    raw_ids = tuple(item.get("id") for item in items)
    if any(not isinstance(value, str) or not value for value in raw_ids):
        return None
    ids = tuple(str(value) for value in raw_ids)
    if len(set(ids)) != len(ids):
        return None
    return _digest(items), ids


def make_witness(
    call_id: str, request: Sequence[Mapping[str, object]], output: Sequence[Mapping[str, object]]
) -> ReasoningWitness | None:
    """Derive a witness only from an exact served reasoning-only output."""
    identity = _reasoning_digest(output)
    if identity is None:
        return None
    digest, ids = identity
    return ReasoningWitness(call_id, len(request), _digest(request), digest, ids)


def select_witness(request: Sequence[Mapping[str, object]], candidates: Sequence[ReasoningWitness]) -> str | None:
    """Select one exact, contiguous output witness among ambiguous candidates.

    After that output, only newly appended user messages and tool results are
    allowed. No user/tool item may precede or interrupt the witnessed output.
    Candidate order never affects selection.
    """
    selected = []
    for witness in candidates:
        output_end = witness.context_len + len(witness.output_ids)
        if len(request) < output_end or _digest(request[: witness.context_len]) != witness.context_digest:
            continue
        output = request[witness.context_len : output_end]
        if _reasoning_digest(output) != (witness.output_digest, witness.output_ids):
            continue
        if any(
            not (
                item.get("role") == "user"
                and item.get("type") in {None, "message"}
                or item.get("type") == "function_call_output"
            )
            for item in request[output_end:]
        ):
            continue
        selected.append(witness.call_id)
    return selected[0] if len(selected) == 1 else None
