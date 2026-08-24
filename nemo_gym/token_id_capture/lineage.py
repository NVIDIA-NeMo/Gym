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

The shared ``LineageStore`` provides read-after-write visibility across workers.
``FileLineageStore`` implements that contract with append-only JSONL files and a per-worker tail cache.
Each child receives its parent's cumulative tokens.
Downstream inference consumes those tokens to supply the exact prompt prefix.

The builder remains the fallback when request-time lineage is unavailable.
A missing ``parent_call_id`` target falls back to strict longest token-prefix matching.
A digest mismatch quarantines the call instead of falling back.
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

from nemo_gym.token_id_capture.protocols import LineageMatch


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
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(parsed)


def _text_of(content: Any) -> str:
    """Flatten a message's text content across dialects.

    Tool calls are normalized separately by ``_tools_of``.
    Dialects store tool calls in different locations.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(parts)


def _tools_of(message: dict) -> list[tuple[str, str]]:
    """Return tool calls as ``(name, canonical arguments)`` pairs.

    Chat stores calls in the message's ``tool_calls`` field.
    Anthropic stores calls in ``tool_use`` content blocks.
    Responses stores each call as a standalone ``function_call`` item.
    """
    tools: list[tuple[str, str]] = []
    # A Responses item is the tool call.
    if message.get("type") == "function_call":
        tools.append((str(message.get("name", "")), canonicalize_tool_arguments(message.get("arguments"))))
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append((str(block.get("name", "")), canonicalize_tool_arguments(block.get("input"))))
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        tools.append((str(function.get("name", "")), canonicalize_tool_arguments(function.get("arguments"))))
    return tools


def _tool_result_text(message: dict) -> str:
    """Return a tool result payload across dialects.

    Responses stores results in standalone ``function_call_output`` items.
    Anthropic stores results in ``tool_result`` content blocks.
    Chat stores results as plain message content.
    """
    parts: list[str] = []
    if message.get("type") == "function_call_output":
        output = message.get("output")
        parts.append(output if isinstance(output, str) else json.dumps(output, sort_keys=True, default=str))
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                parts.append(inner if isinstance(inner, str) else _text_of(inner))
    return "\n".join(part for part in parts if part)


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
            continue
        hasher.update(b"\x00")
        hasher.update(str(message.get("role") or message.get("type") or "").encode("utf-8"))
        hasher.update(b"\x01")
        hasher.update(_text_of(message.get("content")).encode("utf-8"))
        for name, arguments in _tools_of(message):
            hasher.update(b"\x02")
            hasher.update(name.encode("utf-8"))
            hasher.update(arguments.encode("utf-8"))
        # Include tool results.
        # Summarizing, redacting, or truncating a result changes the request context.
        hasher.update(b"\x03")
        hasher.update(_tool_result_text(message).encode("utf-8"))
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
        if not isinstance(message, dict) or not _is_assistant_authored(message):
            continue
        count += 1
        hasher.update(b"\x00")
        hasher.update(_text_of(message.get("content")).encode("utf-8"))
        for name, arguments in _tools_of(message):
            hasher.update(b"\x01")
            hasher.update(name.encode("utf-8"))
            hasher.update(arguments.encode("utf-8"))
    if count == 0:
        return ""
    return hasher.hexdigest()


@dataclass
class LineageNode:
    call_id: str
    cum_tokens: list[int]
    cum_len: int
    digest: str
    # These fields describe the request context sent for this call.
    # They exclude the model's response.
    # The request context has a stable item count across dialect round trips.
    context_len: int = 0
    context_digest: str = ""
    staging_key: str = ""
    staging_chain: list[str] = field(default_factory=list)
    chain_hash: str = ""


@dataclass
class RolloutLineage:
    """Keep an append-only per-rollout call index."""

    by_fingerprint: dict[str, list[str]] = field(default_factory=dict)
    by_call_id: dict[str, LineageNode] = field(default_factory=dict)
    # Cache the cumulative token count for memory bounds.
    total_tokens: int = 0

    def resolve(self, messages: list[dict]) -> LineageNode | None:
        """Return the unique call that this request continues.

        Return ``None`` for a new root.
        Return ``None`` for an ambiguous match.
        Never guess among calls with identical output.
        """
        fingerprint = assistant_fingerprint(messages)
        if not fingerprint:
            return None
        call_ids = self.by_fingerprint.get(fingerprint) or []
        if len(call_ids) != 1:
            return None
        node = self.by_call_id.get(call_ids[0])
        if node is None or not self._continues(node, messages):
            return None
        return node

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
    An evicted parent degrades to strict token-prefix matching.
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
    """Provide lineage within one worker.

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

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageMatch | None:
        node = self.index.for_rollout(rollout_id).resolve(request_items)
        if node is None:
            return None
        return LineageMatch(
            model_call_id=node.call_id,
            cumulative_token_ids=tuple(node.cum_tokens),
            digest=node.digest,
            staging_chain=tuple(node.staging_chain),
            prev_len=node.cum_len,
            chain_hash=node.chain_hash,
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


class FileLineageStore:
    """Share append-only JSONL lineage across local workers.

    Each worker caches records through its last byte offset.
    Reads parse only the newly appended tail.
    Locked appends provide read-after-write visibility across workers.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[int, int, list[dict]]] = {}

    @staticmethod
    def _validate_rollout_id(rollout_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", rollout_id):
            raise ValueError(f"unsafe rollout_id {rollout_id!r}")

    def _state_path(self, rollout_id: str) -> Path:
        self._validate_rollout_id(rollout_id)
        return self.root / f"{rollout_id}.lineage.jsonl"

    def _lock_path(self, rollout_id: str) -> Path:
        self._validate_rollout_id(rollout_id)
        return self.root / f"{rollout_id}.lineage.lock"

    @contextmanager
    def _locked(self, rollout_id: str):
        lock_path = self._lock_path(rollout_id)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self, rollout_id: str) -> list[dict]:
        path = self._state_path(rollout_id)
        if not path.exists():
            self._cache.pop(rollout_id, None)
            return []
        file_stat = path.stat()
        inode, offset, cached = self._cache.get(rollout_id, (file_stat.st_ino, 0, []))
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
        self._cache[rollout_id] = (inode, offset, records)
        return records

    def _append(self, rollout_id: str, record: dict, records: list[dict]) -> None:
        path = self._state_path(rollout_id)
        created = not path.exists()
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            offset = handle.tell()
            inode = os.fstat(handle.fileno()).st_ino
        if created:
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        records.append(record)
        self._cache[rollout_id] = (inode, offset, records)

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageMatch | None:
        return await asyncio.to_thread(self._resolve, rollout_id, request_items)

    def _resolve(self, rollout_id: str, request_items: list[dict]) -> LineageMatch | None:
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
        # No fingerprint: ``_resolve`` filters on it, so a failure row can
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

    async def close(self) -> None:
        self._cache.clear()
