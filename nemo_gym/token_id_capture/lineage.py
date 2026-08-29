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

"""Resolve the recorded call that a request continues.

A rollout can contain several model calls.
Training consumes their exact tokens as one contiguous sequence.
Request-time lineage identifies the earlier call that each request continues.

``assistant_fingerprint`` is the lookup key.
It hashes model-authored turns and ignores user and tool content added between calls.
``conversation_digest`` verifies the unchanged request context.
A digest mismatch rejects the claimed lineage before any parent tokens are reused.

The shared ``LineageStore`` resolves entries already committed by ``TokenSink``.
``FileLineageStore`` tails the token JSONL through the token store's lock.
Each child receives its parent's cumulative tokens.
Downstream inference consumes those tokens to supply the exact prompt prefix.

Every supported record distinguishes a root, a resolved parent, and an unresolved boundary.
The builder uses token-prefix matching only when a verified parent is absent from the frozen snapshot.
It never uses prefix matching to cross an unresolved boundary.

A delivered chain contains exactly the tokens the policy emitted over the recorded context.
The hashes ignore reasoning and selected items that a harness may omit when it echoes model output.
These differences do not change the captured token sequence.
Ambiguous matches remain unresolved rather than risking tokens from the wrong call.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from nemo_gym.token_id_capture.protocols import LineageMatch, LineageResolution
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry, cumulative_tokens


# Increment when fingerprint canonicalization or hash layout changes.
# Resolvers ignore entries stamped with a different version.
FINGERPRINT_VERSION = 1

_FINGERPRINT_DOMAIN = b"nemo-gym-lineage"
_CONTEXT_DOMAIN = b"nemo-gym-lineage-context"

# Canonicalization version of ``assistant_fingerprint``. Stamped on ledger
# custody rows beside the recorded fingerprints; terminal attribution ignores
# fingerprints stamped with a different version. Increment when the hash
# layout or content normalization changes.
LINEAGE_FINGERPRINT_VERSION = 1

# Names of the token-free CallRecord custody columns a ledger row carries.
# ``staging_digest`` is the worker's staged-record digest; the row's ``digest``
# key remains the lineage digest over the cumulative token IDs.
_CUSTODY_FIELDS = (
    "parent_call_id",
    "staging_key",
    "weight_version",
    "prev_len",
    "delta_len",
    "cum_len",
    "staging_digest",
    "extras_digest",
    "mode",
    "logical_request_id",
    "admitted_at",
    "staging_chain",
    "chain_hash",
    "cumulative_hash",
    "response_id",
    "output_fingerprint",
    "continuation_fingerprint",
    "fingerprint_version",
)


def _custody_columns(
    parent_call_id: str | None,
    staging_key: str | None,
    weight_version: int | None,
    prev_len: int | None,
    delta_len: int | None,
    cum_len: int | None,
    staging_digest: str | None,
    extras_digest: str | None,
    mode: str | None,
    logical_request_id: str | None,
    admitted_at: float | None,
    staging_chain: list[str] | None = None,
    chain_hash: str | None = None,
    cumulative_hash: str | None = None,
    response_id: str | None = None,
    output_fingerprint: str | None = None,
    continuation_fingerprint: str | None = None,
    fingerprint_version: int = 0,
) -> dict:
    """Return the ledger custody columns, or an empty dict for a lineage-only row."""
    if staging_key is None:
        return {}
    return {
        "parent_call_id": parent_call_id,
        "staging_key": staging_key,
        "weight_version": weight_version,
        "prev_len": prev_len,
        "delta_len": delta_len,
        "cum_len": cum_len,
        "staging_digest": staging_digest,
        "extras_digest": extras_digest,
        "mode": mode,
        "logical_request_id": logical_request_id,
        "admitted_at": admitted_at,
        "staging_chain": list(staging_chain) if staging_chain else [],
        "chain_hash": chain_hash,
        "cumulative_hash": cumulative_hash,
        "response_id": response_id,
        "output_fingerprint": output_fingerprint or None,
        "continuation_fingerprint": continuation_fingerprint or None,
        "fingerprint_version": fingerprint_version,
    }


def _manifest_from_rows(rollout_id: str, rows: list[dict]) -> dict:
    """Build the token-free ``RolloutManifest`` payload from ledger rows.

    Committed custody rows become ``CallRecord`` payloads; failure rows become
    ``failures`` entries. Lineage-only rows (local capture) carry no custody
    columns and are not part of a capture manifest.
    """
    # Deferred: staging.records pulls in the digest module; lineage stays light.
    from nemo_gym.token_id_capture.staging.records import (
        CallRecord,
        ManifestFailure,
        RolloutManifest,
    )

    records = []
    failures = []
    for row in rows:
        if row.get("failure_reason") is not None:
            failures.append(
                ManifestFailure(
                    model_call_id=str(row["model_call_id"]),
                    reason=str(row["failure_reason"]),
                )
            )
        elif row.get("staging_key") is not None:
            if not row.get("response_id"):
                # A custody row without a served response id is a stamping
                # bug, not a tolerated legacy shape. Poison the rollout
                # (fail-closed) instead of crashing the manifest route.
                failures.append(
                    ManifestFailure(
                        model_call_id=str(row["model_call_id"]),
                        reason="ledger_row_missing_response_id",
                    )
                )
                continue
            records.append(
                CallRecord(
                    model_call_id=str(row["model_call_id"]),
                    parent_call_id=row.get("parent_call_id"),
                    prev_len=int(row["prev_len"]),
                    delta_len=int(row["delta_len"]),
                    cum_len=int(row["cum_len"]),
                    weight_version=int(row["weight_version"]),
                    digest=str(row["staging_digest"]),
                    extras_digest=str(row["extras_digest"]),
                    staging_key=str(row["staging_key"]),
                    mode=str(row["mode"]),
                    chain_hash=row.get("chain_hash"),
                    cumulative_hash=row.get("cumulative_hash"),
                    response_id=str(row["response_id"]),
                    logical_request_id=row.get("logical_request_id"),
                    admitted_at=row.get("admitted_at"),
                    output_fingerprint=row.get("output_fingerprint") or None,
                    continuation_fingerprint=row.get("continuation_fingerprint") or None,
                    fingerprint_version=int(row.get("fingerprint_version") or 0),
                )
            )
    manifest = RolloutManifest(rollout_id=rollout_id, records=records, failures=failures)
    return manifest.model_dump(mode="json")


def _update_field(hasher: Any, tag: bytes, value: str) -> None:
    """Hash one tagged, length-delimited UTF-8 field."""
    encoded = value.encode("utf-8")
    hasher.update(tag)
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def _canonical_json(value: Any) -> str:
    """Serialize JSON-compatible prompt content without losing structure."""
    try:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    except (TypeError, orjson.JSONEncodeError) as error:
        raise ValueError(f"unsupported prompt content: {type(value).__name__}") from error


def canonicalize_tool_arguments(value: Any) -> str:
    """Normalize a tool call's arguments for comparison only.

    Harnesses can reserialize tool-call arguments between turns.
    Comparison uses sorted-key JSON with normalized separators.
    The record retains the model's original string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return value.strip()
    else:
        parsed = value
    return _canonical_json(parsed)


def _content_of(content: Any) -> list[tuple[str, str]]:
    """Return typed content parts without discarding prompt-shaping blocks.

    Tool calls are normalized separately by ``_tools_of``.
    Tool results are normalized separately by ``_tool_results_of``.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [("text", content)] if content else []
    if not isinstance(content, list):
        raise ValueError(f"unsupported message content: {type(content).__name__}")
    parts: list[tuple[str, str]] = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append(("text", block))
            continue
        if not isinstance(block, dict):
            raise ValueError(f"unsupported content block: {type(block).__name__}")
        block_type = str(block.get("type") or "")
        if block_type in {"tool_use", "tool_result"}:
            continue
        if isinstance(block.get("text"), str) and block_type in {
            "",
            "text",
            "input_text",
            "output_text",
        }:
            if block["text"]:
                parts.append(("text", block["text"]))
            continue
        if not block_type:
            raise ValueError("content block has no supported type")
        parts.append((block_type, _canonical_json(block)))
    return parts


def _tools_of(message: dict) -> list[tuple[str, str, str]]:
    """Return tool calls as ``(id, name, canonical arguments)`` tuples.

    Chat stores calls in the message's ``tool_calls`` field.
    Anthropic stores calls in ``tool_use`` content blocks.
    Responses stores each call as a standalone ``function_call`` item.
    """
    tools: list[tuple[str, str, str]] = []
    # A Responses item is the tool call.
    if message.get("type") == "function_call":
        tools.append(
            (
                str(message.get("call_id") or message.get("id") or ""),
                str(message.get("name", "")),
                canonicalize_tool_arguments(message.get("arguments")),
            )
        )
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append(
                    (
                        str(block.get("id") or ""),
                        str(block.get("name", "")),
                        canonicalize_tool_arguments(block.get("input")),
                    )
                )
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        tools.append(
            (
                str((call or {}).get("id") or ""),
                str(function.get("name", "")),
                canonicalize_tool_arguments(function.get("arguments")),
            )
        )
    return tools


def _tool_results_of(message: dict) -> list[tuple[str, str]]:
    """Return tool result identities and payloads across dialects.

    Responses stores results in standalone ``function_call_output`` items.
    Anthropic stores results in ``tool_result`` content blocks.
    Chat stores results as plain message content.
    """
    parts: list[tuple[str, str]] = []
    if message.get("type") == "function_call_output":
        output = message.get("output")
        parts.append(
            (
                str(message.get("call_id") or message.get("id") or ""),
                output if isinstance(output, str) else _canonical_json(output),
            )
        )
    elif message.get("role") == "tool":
        content = message.get("content")
        parts.append(
            (
                str(message.get("tool_call_id") or ""),
                content if isinstance(content, str) else _canonical_json(content),
            )
        )
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                payload = inner if isinstance(inner, str) else _canonical_json(inner)
                parts.append((str(block.get("tool_use_id") or block.get("id") or ""), payload))
    return parts


def _is_assistant_authored(message: dict) -> bool:
    """Return whether the model produced this item.

    Chat and Anthropic use the ``assistant`` role.
    Responses tool calls are roleless ``function_call`` items.
    """
    if message.get("role") == "assistant":
        return True
    # Reasoning is deliberately excluded.
    # A harness need not echo standalone reasoning items.
    # Including reasoning would make fingerprints depend on the dialect and echo behavior.
    # Reasoning-only collisions resolve as ambiguous and fall back.
    return message.get("type") == "function_call"


def conversation_digest(messages: list[dict]) -> str:
    """Hash every turn of a conversation, model-authored or not.

    ``assistant_fingerprint`` ignores user and tool content.
    This digest covers that omitted context.
    A mismatch rejects the parent before its tokens are reused.
    """
    hasher = hashlib.sha256(_CONTEXT_DOMAIN)
    for message in messages or []:
        if not isinstance(message, dict):
            raise ValueError(f"request item is not an object: {type(message).__name__}")
        _update_field(hasher, b"\x00", str(message.get("role") or message.get("type") or ""))
        for content_type, payload in _content_of(message.get("content")):
            _update_field(hasher, b"\x01", content_type)
            _update_field(hasher, b"\x02", payload)
        for call_id, name, arguments in _tools_of(message):
            _update_field(hasher, b"\x03", call_id)
            _update_field(hasher, b"\x04", name)
            _update_field(hasher, b"\x05", arguments)
        # Include tool results.
        # Summarizing, redacting, or truncating a result changes the request context.
        for call_id, output in _tool_results_of(message):
            _update_field(hasher, b"\x06", call_id)
            _update_field(hasher, b"\x07", output)
    return hasher.hexdigest()


def assistant_fingerprint(messages: list[dict]) -> str:
    """Fingerprint the model-authored turns of a request, in order.

    The fingerprint identifies the call that produced the last model-authored turn.
    User and tool content is excluded from the lookup key.
    Dialect-specific tool-call shapes normalize to the same hash input.
    """
    hasher = hashlib.sha256(_FINGERPRINT_DOMAIN)
    count = 0
    for message in messages or []:
        if not isinstance(message, dict):
            raise ValueError(f"request item is not an object: {type(message).__name__}")
        if not _is_assistant_authored(message):
            continue
        count += 1
        for content_type, payload in _content_of(message.get("content")):
            _update_field(hasher, b"\x00", content_type)
            _update_field(hasher, b"\x01", payload)
        for call_id, name, arguments in _tools_of(message):
            _update_field(hasher, b"\x02", call_id)
            _update_field(hasher, b"\x03", name)
            _update_field(hasher, b"\x04", arguments)
    if count == 0:
        return ""
    return hasher.hexdigest()


@dataclass
class LineageNode:
    call_id: str
    # ``None`` means the index is metadata-only.
    # A resolved match loads tokens from ``entry_offset``.
    cum_tokens: list[int] | None
    cum_len: int
    digest: str
    entry_offset: int = -1
    # These fields describe the request context sent for this call.
    # They exclude the model's response.
    # The item count is stable while the harness stays in one dialect.
    # A mid-rollout dialect switch can misalign it; verification then fails closed.
    context_len: int = 0
    context_digest: str = ""
    parent_call_id: str | None = None
    prompt_is_delta: bool = False
    staging_key: str = ""
    staging_chain: list[str] = field(default_factory=list)
    chain_hash: str = ""


def stamp_continuation(entry: TokenEntry, request_items: list[dict]) -> TokenEntry:
    """Add compact lookup metadata before the token entry is committed."""
    entry.continuation_fingerprint = assistant_fingerprint(list(request_items) + list(entry.output_items))
    entry.continuation_context_len = len(request_items)
    entry.continuation_context_digest = conversation_digest(request_items)
    entry.fingerprint_version = FINGERPRINT_VERSION
    return entry


@dataclass
class RolloutLineage:
    """Keep an append-only per-rollout call index."""

    by_fingerprint: dict[str, list[str]] = field(default_factory=dict)
    by_call_id: dict[str, LineageNode] = field(default_factory=dict)
    # Cache the cumulative token count for memory bounds.
    total_tokens: int = 0

    def resolve_node(self, messages: list[dict]) -> tuple[ParentResolutionStatus, "LineageNode | None", str]:
        """Return the parent decision without touching token arrays.

        Matching needs only fingerprints, digests, and lengths.
        The caller materializes tokens for the single winner.
        """
        fingerprint = assistant_fingerprint(messages)
        if not fingerprint:
            return ParentResolutionStatus.ROOT, None, ""
        # dict.fromkeys: a call id indexed twice (e.g. by racing refreshes) is one candidate.
        call_ids = list(dict.fromkeys(self.by_fingerprint.get(fingerprint) or []))
        candidates = [
            node
            for call_id in call_ids
            if (node := self.by_call_id.get(call_id)) is not None and self._continues(node, messages)
        ]
        if len(candidates) > 1:
            # Calls with identical cumulative tokens are interchangeable.
            # Keep different token sequences unresolved.
            digests = {(node.digest, node.cum_len) for node in candidates}
            if len(digests) == 1 and candidates[0].digest:
                candidates = [min(candidates, key=lambda node: node.call_id)]
        if len(candidates) != 1:
            return ParentResolutionStatus.UNRESOLVED, None, "no_match" if not candidates else "ambiguous"
        return ParentResolutionStatus.RESOLVED, candidates[0], ""

    def resolve(self, messages: list[dict]) -> LineageResolution:
        """Return the immutable parent decision for this request.

        A request without model-authored history is a root.
        A request with unverified history is unresolved.
        Never guess among calls with identical output.
        """
        status, node, reason = self.resolve_node(messages)
        if status != ParentResolutionStatus.RESOLVED:
            return LineageResolution(status, reason=reason)
        if node.cum_tokens is None:
            raise ValueError("metadata-only lineage node requires caller-side materialization")
        return LineageResolution(
            ParentResolutionStatus.RESOLVED,
            match=LineageMatch(
                model_call_id=node.call_id,
                cumulative_token_ids=tuple(node.cum_tokens),
                digest=node.digest,
                staging_chain=tuple(node.staging_chain),
                prev_len=node.cum_len,
                chain_hash=node.chain_hash,
            ),
        )

    @staticmethod
    def _continues(node: LineageNode, messages: list[dict]) -> bool:
        """Return whether this request extends the node's recorded context.

        The leading ``context_len`` items must match the recorded request.
        A rewritten or summarized context fails verification.
        Verification excludes the model response because dialects can echo it as different item counts.
        """
        if not node.context_digest:
            # Fail closed when no context digest is available.
            return False
        if len(messages) < node.context_len:
            return False
        return conversation_digest(messages[: node.context_len]) == node.context_digest

    def add_entry(self, entry: TokenEntry, *, store_tokens: bool = True, entry_offset: int = -1) -> None:
        """Index lookup metadata carried by one committed token entry.

        ``store_tokens=False`` keeps token arrays in the durable log.
        """
        if not entry.continuation_fingerprint:
            return
        if entry.fingerprint_version is not None and entry.fingerprint_version != FINGERPRINT_VERSION:
            # A different algorithm produced this fingerprint; matching it would be luck.
            return
        if getattr(entry, "prompt_is_delta", False) and store_tokens:
            # A memory-only index cannot reconstruct a delta chain.
            raise ValueError("delta records require a durable-log-backed lineage store")
        node = LineageNode(
            call_id=entry.model_call_id,
            cum_tokens=cumulative_tokens(entry) if store_tokens else None,
            cum_len=entry.cum_len if entry.cum_len is not None else len(cumulative_tokens(entry)),
            digest=entry.digest or "",
            entry_offset=entry_offset,
            context_len=entry.continuation_context_len,
            context_digest=entry.continuation_context_digest,
            parent_call_id=entry.parent_call_id,
            prompt_is_delta=entry.prompt_is_delta,
        )
        previous = self.by_call_id.get(entry.model_call_id)
        if previous is not None:
            if previous != node:
                raise ValueError(f"conflicting lineage record for model call {entry.model_call_id}")
            return
        self.total_tokens += node.cum_len
        self.by_call_id[entry.model_call_id] = node
        self.by_fingerprint.setdefault(entry.continuation_fingerprint, []).append(entry.model_call_id)

    def record(
        self,
        call_id: str,
        messages: list[dict],
        cum_tokens: list[int],
        digest: str,
        context_len: int | None = None,
        *,
        staging_key: str = "",
        parent_staging_chain: list[str] | None = None,
        cum_len: int | None = None,
        chain_hash: str = "",
    ) -> None:
        """Index a completed call by its continuation fingerprint.

        ``context_len`` counts the request items before the model response.
        The default assumes one synthesized response item.
        ``cum_len`` must be passed explicitly for token-free custody rows,
        where ``cum_tokens`` is empty; a child's ``prev_len`` reads it.
        """
        node = LineageNode(
            call_id=call_id,
            cum_tokens=list(cum_tokens),
            cum_len=cum_len if cum_len is not None else len(cum_tokens),
            digest=digest,
            context_len=context_len if context_len is not None else max(len(messages or []) - 1, 0),
            context_digest=conversation_digest(
                (messages or [])[: context_len if context_len is not None else max(len(messages or []) - 1, 0)]
            ),
            staging_key=staging_key,
            staging_chain=list(parent_staging_chain) if parent_staging_chain else [],
            chain_hash=chain_hash,
        )
        previous = self.by_call_id.get(call_id)
        if previous is not None:
            if previous != node:
                raise ValueError(f"conflicting lineage record for model call {call_id}")
            return
        self.total_tokens += node.cum_len
        self.by_call_id[call_id] = node
        fingerprint = assistant_fingerprint(messages)
        if fingerprint:
            self.by_fingerprint.setdefault(fingerprint, []).append(call_id)


class LineageIndex:
    """Bound worker-local lineage by rollout and cumulative token counts.

    This index backs the single-worker fallback.
    Shared stores provide cross-worker visibility.
    Eviction removes the oldest rollout.
    An evicted parent leaves later continuations unresolved and the builder masks them.
    The only live rollout is never evicted.
    """

    def __init__(self, max_rollouts: int = 512, max_tokens: int = 8_000_000) -> None:
        self._max_rollouts = max_rollouts
        self._max_tokens = max_tokens
        self._rollouts: dict[str, RolloutLineage] = {}

    def for_rollout(self, rollout_id: str) -> RolloutLineage:
        lineage = self._rollouts.get(rollout_id)
        if lineage is None:
            lineage = RolloutLineage()
            self._rollouts[rollout_id] = lineage
        self._evict()
        return lineage

    def _evict(self) -> None:
        # Check after every access because existing rollouts can grow.
        while self._rollouts and (len(self._rollouts) > self._max_rollouts or self.total_tokens > self._max_tokens):
            oldest = next(iter(self._rollouts))
            # Never evict the only rollout.
            if len(self._rollouts) == 1:
                return
            self._rollouts.pop(oldest)

    @property
    def total_tokens(self) -> int:
        return sum(lineage.total_tokens for lineage in self._rollouts.values())

    def drop(self, rollout_id: str) -> None:
        """Release a rollout's lineage early.

        Gym's model server has no rollout-completion signal.
        An in-process framework can call this when it retires the records.
        """
        self._rollouts.pop(rollout_id, None)

    def clear(self) -> None:
        self._rollouts.clear()

    def __len__(self) -> int:
        return len(self._rollouts)


class InMemoryLineageStore:
    """Reference resolver for in-process framework backends and tests.

    Production wiring uses ``FileLineageStore`` when a token store exists.
    Its index is memory-only.
    Eviction or restart leaves affected continuations unresolved.
    That failure mode is safe but can mask otherwise usable rollouts.
    Multi-worker deployments require a shared ``LineageStore``.
    The resolution index evicts rollouts under memory bounds, so this store
    cannot serve as an external-staging capture ledger (completeness would
    break); it remains for unit tests and single-worker development. Its
    ledger rows are kept in a separate unbounded map so ledger unit tests see
    file-store semantics.
    """

    def __init__(self, max_rollouts: int = 512, max_tokens: int = 8_000_000) -> None:
        self.index = LineageIndex(max_rollouts=max_rollouts, max_tokens=max_tokens)
        self._ledgers: dict[str, list[dict]] = {}

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        return self.index.for_rollout(rollout_id).resolve(request_items)

    async def put(self, entry: TokenEntry) -> None:
        """Publish one committed entry to the worker-local index."""
        self.index.for_rollout(entry.rollout_id).add_entry(entry)

    def is_process_shared(self) -> bool:
        return False

    async def record(
        self,
        rollout_id: str,
        model_call_id: str,
        request_items: list[dict],
        response_items: list[dict],
        cumulative_token_ids: list[int],
        digest: str,
        *,
        parent_call_id: str | None = None,
        staging_key: str | None = None,
        weight_version: int | None = None,
        prev_len: int | None = None,
        delta_len: int | None = None,
        cum_len: int | None = None,
        staging_digest: str | None = None,
        extras_digest: str | None = None,
        mode: str | None = None,
        logical_request_id: str | None = None,
        admitted_at: float | None = None,
        staging_chain: list[str] | None = None,
        chain_hash: str | None = None,
        cumulative_hash: str | None = None,
        response_id: str | None = None,
        output_fingerprint: str | None = None,
        continuation_fingerprint: str | None = None,
        fingerprint_version: int = 0,
    ) -> None:
        # Custody rows are token-free (mirrors FileLineageStore): the chain
        # hash covers continuity, so the index keeps tokens only for
        # lineage-only local-capture rows that inject prompt prefixes.
        self.index.for_rollout(rollout_id).record(
            model_call_id,
            list(request_items) + list(response_items),
            [] if staging_key else cumulative_token_ids,
            digest,
            context_len=len(request_items),
            staging_key=staging_key or "",
            parent_staging_chain=staging_chain,
            cum_len=cum_len,
            chain_hash=chain_hash or "",
        )
        custody = _custody_columns(
            parent_call_id,
            staging_key,
            weight_version,
            prev_len,
            delta_len,
            cum_len,
            staging_digest,
            extras_digest,
            mode,
            logical_request_id,
            admitted_at,
            staging_chain,
            chain_hash,
            cumulative_hash,
            response_id,
            output_fingerprint,
            continuation_fingerprint,
            fingerprint_version,
        )
        if custody:
            rows = self._ledgers.setdefault(rollout_id, [])
            row = {"model_call_id": model_call_id, **custody}
            if not any(existing == row for existing in rows):
                rows.append(row)

    async def record_failure(self, rollout_id: str, model_call_id: str, reason: str) -> None:
        rows = self._ledgers.setdefault(rollout_id, [])
        row = {"model_call_id": model_call_id, "failure_reason": reason}
        if not any(existing == row for existing in rows):
            rows.append(row)

    async def manifest(self, rollout_id: str) -> dict:
        return _manifest_from_rows(rollout_id, list(self._ledgers.get(rollout_id) or []))

    async def has_rows(self, rollout_id: str) -> bool:
        if self._ledgers.get(rollout_id):
            return True
        return bool(self.index.for_rollout(rollout_id).by_call_id)

    async def close(self) -> None:
        self.index.clear()
        self._ledgers.clear()


class IncrementalLineageStore:
    """Base class for lineage resolvers over any committed-entry backend.

    An external backend implements two hooks.
    It inherits Gym's matcher, bounded index, locking, and token materialization.
    Hash-for-hash agreement is the wire contract.
    The backend remains the source of truth when cache rows are evicted.
    A resolved match loads only the winning call's token chain.

    Required hooks:
      ``_fetch_new_entries(rollout_id, cursor)`` -> ``(items, new_cursor)`` where
        ``items`` is ``[(TokenEntry, ref), ...]`` in commit order since ``cursor``
        (``None`` means from the beginning) and ``ref`` is any handle that
        ``_load_entry`` can use later (byte offset, KV key, ...). Raise
        ``CursorReset`` when the cursor no longer describes the backend (file
        rotated, namespace recreated); the base refetches from the beginning.
      ``_load_entry(rollout_id, ref)`` -> ``TokenEntry`` for one committed record.

    Optional hooks:
      ``_load_entries(rollout_id, refs)`` — batch-load one parent chain
        (default: call ``_load_entry`` for each reference).
      ``_read_locked(rollout_id)`` — context manager held around fetch+resolve
        for backends with a read-lock discipline (default: no lock).
      ``is_process_shared()`` — default ``True``; an external backend exists to
        be shared, and the multi-worker startup check trusts this answer.
    """

    class CursorReset(Exception):
        """The stored cursor no longer describes the backend; refetch from scratch."""

    def __init__(self, *, max_cached_rollouts: int = 65536, max_cached_tokens: int = 8_000_000) -> None:
        import threading

        if max_cached_rollouts < 1:
            raise ValueError("max_cached_rollouts must be positive")
        if max_cached_tokens < 1:
            raise ValueError("max_cached_tokens must be positive")
        # (cursor, refs, lineage): lineage stays at index 2 for diagnostics/tooling.
        self._cache: dict[str, tuple[Any, dict[str, Any], RolloutLineage]] = {}
        self._max_cached_rollouts = max_cached_rollouts
        # Keep only the latest materialized parent for each rollout.
        # The global token bound avoids recreating full-record memory growth.
        self._materialized: dict[str, tuple[str, tuple[int, ...]]] = {}
        self._materialized_tokens = 0
        self._max_cached_tokens = max_cached_tokens
        self._cache_guard = threading.Lock()
        # Fixed lock striping bounds synchronization metadata.
        # Hash collisions only serialize unrelated rollouts.
        self._rollout_locks = tuple(threading.Lock() for _ in range(256))

    # -- hooks ----------------------------------------------------------------
    def _fetch_new_entries(self, rollout_id: str, cursor: Any) -> tuple[list[tuple[TokenEntry, Any]], Any]:
        raise NotImplementedError

    def _load_entry(self, rollout_id: str, ref: Any) -> TokenEntry:
        raise NotImplementedError

    def _load_entries(self, rollout_id: str, refs: list[Any]) -> list[TokenEntry]:
        """Load several committed entries.

        Backends can override this hook to fetch a parent chain in one operation.
        """
        return [self._load_entry(rollout_id, ref) for ref in refs]

    def _read_locked(self, rollout_id: str):
        from contextlib import nullcontext

        return nullcontext()

    # -- shared machinery -----------------------------------------------------
    def _rollout_lock(self, rollout_id: str):
        return self._rollout_locks[hash(rollout_id) % len(self._rollout_locks)]

    def _cache_put(self, rollout_id: str, value: tuple[Any, dict[str, Any], RolloutLineage]) -> None:
        """Insert or touch a cache row with LRU semantics.

        Reinsert a touched row so dictionary order tracks recency.
        Eviction only requires a later backend refetch.
        """
        with self._cache_guard:
            self._cache.pop(rollout_id, None)
            self._cache[rollout_id] = value
            while len(self._cache) > self._max_cached_rollouts:
                oldest = next(iter(self._cache))
                if oldest == rollout_id:
                    break
                self._cache.pop(oldest)
                materialized = self._materialized.pop(oldest, None)
                if materialized is not None:
                    self._materialized_tokens -= len(materialized[1])

    def _cached_materialized(self, rollout_id: str) -> tuple[str, tuple[int, ...]] | None:
        with self._cache_guard:
            return self._materialized.get(rollout_id)

    def _remember_materialized(self, rollout_id: str, call_id: str, tokens: tuple[int, ...]) -> None:
        with self._cache_guard:
            previous = self._materialized.pop(rollout_id, None)
            if previous is not None:
                self._materialized_tokens -= len(previous[1])
            if len(tokens) > self._max_cached_tokens:
                return
            self._materialized[rollout_id] = (call_id, tokens)
            self._materialized_tokens += len(tokens)
            while self._materialized_tokens > self._max_cached_tokens:
                oldest = next(iter(self._materialized))
                evicted = self._materialized.pop(oldest)
                self._materialized_tokens -= len(evicted[1])

    def _refresh(self, rollout_id: str) -> tuple[dict[str, Any], RolloutLineage]:
        with self._cache_guard:
            cached = self._cache.get(rollout_id)
        cursor, refs, lineage = cached if cached is not None else (None, {}, RolloutLineage())
        try:
            items, cursor = self._fetch_new_entries(rollout_id, cursor)
        except IncrementalLineageStore.CursorReset:
            refs, lineage = {}, RolloutLineage()
            items, cursor = self._fetch_new_entries(rollout_id, None)
        for entry, ref in items:
            refs[entry.model_call_id] = ref
            # Metadata-only: tokens stay in the backend behind ``ref``.
            lineage.add_entry(entry, store_tokens=False, entry_offset=ref if isinstance(ref, int) else -1)
        self._cache_put(rollout_id, (cursor, refs, lineage))
        return refs, lineage

    def _materialize(
        self, rollout_id: str, node: LineageNode, refs: dict[str, Any], lineage: RolloutLineage
    ) -> tuple[int, ...]:
        """Load one RESOLVED parent's cumulative tokens from the backend.

        Read the chain in one batch and append each token segment once.
        Digest verification makes stale references fail closed.
        """
        from nemo_gym.token_id_capture.records import compute_digest

        # Metadata carries enough lineage to collect every backend reference before loading tokens.
        cached = self._cached_materialized(rollout_id)
        chain: list[LineageNode] = []
        seen: set[str] = set()
        current = node
        cached_tokens: tuple[int, ...] | None = None
        while True:
            if cached is not None and current.call_id == cached[0]:
                cached_tokens = cached[1]
                break
            if current.call_id in seen:
                raise ValueError(f"delta chain for {node.call_id} contains a cycle")
            if len(chain) >= 10_000:
                raise ValueError(f"delta chain for {node.call_id} exceeds sane depth")
            seen.add(current.call_id)
            chain.append(current)
            if not current.prompt_is_delta:
                break
            if not current.parent_call_id:
                raise ValueError(f"delta record {current.call_id} has no parent call id")
            parent = lineage.by_call_id.get(current.parent_call_id)
            if parent is None:
                raise ValueError(f"delta record {current.call_id} has no indexed parent")
            current = parent

        ordered_nodes = list(reversed(chain))
        missing_ref = next((item.call_id for item in ordered_nodes if item.call_id not in refs), None)
        if missing_ref is not None:
            raise ValueError(f"lineage node for {missing_ref} has no backend ref")
        entries = (
            self._load_entries(rollout_id, [refs[item.call_id] for item in ordered_nodes]) if ordered_nodes else []
        )

        tokens = list(cached_tokens or ())
        for expected, entry in zip(ordered_nodes, entries, strict=True):
            if entry.model_call_id != expected.call_id:
                raise ValueError(f"ref for {expected.call_id} points at {entry.model_call_id}")
            if entry.prompt_is_delta != expected.prompt_is_delta:
                raise ValueError(f"metadata for {expected.call_id} disagrees with its stored entry")
            tokens.extend(entry.prompt_token_ids)
            tokens.extend(entry.generation_token_ids)
        if node.digest and compute_digest(tokens) != node.digest:
            raise ValueError(f"materialized tokens for {node.call_id} fail their digest")
        materialized = tuple(tokens)
        self._remember_materialized(rollout_id, node.call_id, materialized)
        return materialized

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        return await asyncio.to_thread(self._resolve, rollout_id, request_items)

    def _resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        with self._rollout_lock(rollout_id), self._read_locked(rollout_id):
            refs, lineage = self._refresh(rollout_id)
            status, node, reason = lineage.resolve_node(request_items)
            if status != ParentResolutionStatus.RESOLVED:
                return LineageResolution(status, reason=reason)
            tokens = (
                node.cum_tokens if node.cum_tokens is not None else self._materialize(rollout_id, node, refs, lineage)
            )
            return LineageResolution(
                ParentResolutionStatus.RESOLVED,
                match=LineageMatch(
                    model_call_id=node.call_id,
                    cumulative_token_ids=tuple(tokens),
                    digest=node.digest,
                ),
            )

    def is_process_shared(self) -> bool:
        return True

    async def close(self) -> None:
        with self._cache_guard:
            self._cache.clear()
            self._materialized.clear()
            self._materialized_tokens = 0


class FileLineageStore(IncrementalLineageStore):
    """Resolve lineage from the token JSONL committed by ``TokenCaptureStore``.

    The reference ``IncrementalLineageStore`` backend: cursor = (inode, offset),
    ref = byte offset, reads under the store's shared flock so a committed
    ``put`` is immediately visible.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_cached_rollouts: int = 65536,
        max_cached_tokens: int = 8_000_000,
    ) -> None:
        from nemo_gym.token_id_capture.store import TokenCaptureStore

        super().__init__(max_cached_rollouts=max_cached_rollouts, max_cached_tokens=max_cached_tokens)
        self._store = TokenCaptureStore(root)
        # The capture-ledger rows live beside the token JSONL. Each worker
        # caches parsed rows through its last byte offset; locked appends
        # provide read-after-write visibility across workers.
        self._ledger_root = Path(root)
        self._ledger_root.mkdir(parents=True, exist_ok=True)
        self._ledger_cache: dict[str, tuple[int, int, list[dict]]] = {}

    def _read_locked(self, rollout_id: str):
        return self._store._locked(rollout_id, shared=True)

    def _fetch_new_entries(self, rollout_id: str, cursor: Any) -> tuple[list[tuple[TokenEntry, Any]], Any]:
        path = self._store.path_for(rollout_id)
        if not path.exists():
            if cursor is not None:
                raise IncrementalLineageStore.CursorReset
            return [], None
        file_stat = path.stat()
        inode, offset = cursor if cursor is not None else (file_stat.st_ino, 0)
        if inode != file_stat.st_ino or offset < 0 or offset > file_stat.st_size:
            raise IncrementalLineageStore.CursorReset
        items: list[tuple[TokenEntry, Any]] = []
        if offset < file_stat.st_size:
            with path.open("rb") as handle:
                handle.seek(offset)
                while True:
                    line_offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    payload = line.strip()
                    if not payload:
                        continue
                    items.append((TokenEntry.model_validate(orjson.loads(payload)), line_offset))
                offset = handle.tell()
        return items, (inode, offset)

    def _load_entry(self, rollout_id: str, ref: Any) -> TokenEntry:
        with self._store.path_for(rollout_id).open("rb") as handle:
            handle.seek(ref)
            return TokenEntry.model_validate(orjson.loads(handle.readline()))

    def _load_entries(self, rollout_id: str, refs: list[Any]) -> list[TokenEntry]:
        entries = []
        with self._store.path_for(rollout_id).open("rb") as handle:
            for ref in refs:
                handle.seek(ref)
                entries.append(TokenEntry.model_validate(orjson.loads(handle.readline())))
        return entries

    # --- CaptureLedger surface -------------------------------------------
    # External staging commits token-bearing ledger rows through ``record``
    # instead of TokenEntry JSONL, so resolution falls back to those rows
    # when the incremental entry index has no match.

    @staticmethod
    def _validate_rollout_id(rollout_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", rollout_id):
            raise ValueError(f"unsafe rollout_id {rollout_id!r}")

    def _ledger_path(self, rollout_id: str) -> Path:
        self._validate_rollout_id(rollout_id)
        return self._ledger_root / f"{rollout_id}.lineage.jsonl"

    def _ledger_lock_path(self, rollout_id: str) -> Path:
        self._validate_rollout_id(rollout_id)
        return self._ledger_root / f"{rollout_id}.lineage.lock"

    @contextmanager
    def _locked(self, rollout_id: str):
        lock_path = self._ledger_lock_path(rollout_id)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self, rollout_id: str) -> list[dict]:
        path = self._ledger_path(rollout_id)
        if not path.exists():
            self._ledger_cache.pop(rollout_id, None)
            return []
        file_stat = path.stat()
        inode, offset, cached = self._ledger_cache.get(rollout_id, (file_stat.st_ino, 0, []))
        if inode != file_stat.st_ino or offset < 0 or offset > file_stat.st_size:
            inode, offset, cached = file_stat.st_ino, 0, []
        if offset == file_stat.st_size:
            return cached

        records = list(cached)
        with path.open("rb") as handle:
            handle.seek(offset)
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                record = json.loads(payload)
                if not isinstance(record, dict):
                    raise ValueError(f"lineage record for {rollout_id} is not an object")
                records.append(record)
            offset = handle.tell()
        self._ledger_cache[rollout_id] = (inode, offset, records)
        return records

    def _append(self, rollout_id: str, record: dict, records: list[dict]) -> None:
        path = self._ledger_path(rollout_id)
        created = not path.exists()
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            offset = handle.tell()
            inode = os.fstat(handle.fileno()).st_ino
        if created:
            directory_fd = os.open(self._ledger_root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        records.append(record)
        self._ledger_cache[rollout_id] = (inode, offset, records)

    def _resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        resolution = super()._resolve(rollout_id, request_items)
        if resolution.status != ParentResolutionStatus.UNRESOLVED:
            return resolution
        match = self._resolve_row(rollout_id, request_items)
        if match is not None:
            return LineageResolution(ParentResolutionStatus.RESOLVED, match=match)
        return resolution

    def _resolve_row(self, rollout_id: str, request_items: list[dict]) -> LineageMatch | None:
        fingerprint = assistant_fingerprint(request_items)
        if not fingerprint:
            return None
        with self._locked(rollout_id):
            # Failure rows carry no fingerprint and can never resolve as parents.
            records = [record for record in self._read(rollout_id) if record.get("fingerprint") == fingerprint]
        if len(records) != 1:
            return None
        record = records[0]
        context_len = int(record["context_len"])
        if len(request_items) < context_len:
            return None
        if conversation_digest(request_items[:context_len]) != record["context_digest"]:
            return None
        return LineageMatch(
            model_call_id=str(record["model_call_id"]),
            # Token-free custody rows omit the column; legacy rows keep it.
            cumulative_token_ids=tuple(int(token) for token in record.get("cumulative_token_ids") or ()),
            digest=str(record["digest"]),
            staging_chain=tuple(record.get("staging_chain") or []),
            prev_len=int(record.get("cum_len") or 0),
            chain_hash=str(record.get("chain_hash") or ""),
        )

    async def record(
        self,
        rollout_id: str,
        model_call_id: str,
        request_items: list[dict],
        response_items: list[dict],
        cumulative_token_ids: list[int],
        digest: str,
        *,
        parent_call_id: str | None = None,
        staging_key: str | None = None,
        weight_version: int | None = None,
        prev_len: int | None = None,
        delta_len: int | None = None,
        cum_len: int | None = None,
        staging_digest: str | None = None,
        extras_digest: str | None = None,
        mode: str | None = None,
        logical_request_id: str | None = None,
        admitted_at: float | None = None,
        staging_chain: list[str] | None = None,
        chain_hash: str | None = None,
        cumulative_hash: str | None = None,
        response_id: str | None = None,
        output_fingerprint: str | None = None,
        continuation_fingerprint: str | None = None,
        fingerprint_version: int = 0,
    ) -> None:
        custody = _custody_columns(
            parent_call_id,
            staging_key,
            weight_version,
            prev_len,
            delta_len,
            cum_len,
            staging_digest,
            extras_digest,
            mode,
            logical_request_id,
            admitted_at,
            staging_chain,
            chain_hash,
            cumulative_hash,
            response_id,
            output_fingerprint,
            continuation_fingerprint,
            fingerprint_version,
        )
        await asyncio.to_thread(
            self._record,
            rollout_id,
            model_call_id,
            request_items,
            response_items,
            cumulative_token_ids,
            digest,
            custody,
        )

    def _record(
        self,
        rollout_id: str,
        model_call_id: str,
        request_items: list[dict],
        response_items: list[dict],
        cumulative_token_ids: list[int],
        digest: str,
        custody: dict | None = None,
    ) -> None:
        record = {
            "model_call_id": model_call_id,
            "fingerprint": assistant_fingerprint(list(request_items) + list(response_items)),
            "context_len": len(request_items),
            "context_digest": conversation_digest(request_items),
            "digest": digest,
            **(custody or {}),
        }
        if not custody:
            # Custody rows are token-free: the chained ``chain_hash`` covers
            # continuity, so only lineage-only (local capture) rows store the
            # cumulative sequence for prompt-prefix injection.
            record["cumulative_token_ids"] = list(cumulative_token_ids)
        with self._locked(rollout_id):
            records = self._read(rollout_id)
            matches = [existing for existing in records if existing["model_call_id"] == model_call_id]
            if matches:
                if matches[0] != record:
                    raise ValueError(f"conflicting lineage record for model call {model_call_id}")
                return
            self._append(rollout_id, record, records)

    async def record_failure(self, rollout_id: str, model_call_id: str, reason: str) -> None:
        await asyncio.to_thread(self._record_failure, rollout_id, model_call_id, reason)

    def _record_failure(self, rollout_id: str, model_call_id: str, reason: str) -> None:
        # No fingerprint: ``_resolve_row`` filters on it, so a failure row can
        # never be returned as a parent. Idempotent for the identical reason.
        record = {"model_call_id": model_call_id, "failure_reason": reason}
        with self._locked(rollout_id):
            records = self._read(rollout_id)
            if any(existing == record for existing in records):
                return
            self._append(rollout_id, record, records)

    async def manifest(self, rollout_id: str) -> dict:
        return await asyncio.to_thread(self._manifest, rollout_id)

    def _manifest(self, rollout_id: str) -> dict:
        with self._locked(rollout_id):
            rows = list(self._read(rollout_id))
        return _manifest_from_rows(rollout_id, rows)

    async def has_rows(self, rollout_id: str) -> bool:
        return await asyncio.to_thread(self._has_rows, rollout_id)

    def _has_rows(self, rollout_id: str) -> bool:
        with self._locked(rollout_id):
            return bool(self._read(rollout_id))
