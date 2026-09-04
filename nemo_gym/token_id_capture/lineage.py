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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from nemo_gym.token_id_capture.fingerprint import (
    FINGERPRINT_VERSION,
    assistant_fingerprint,
    conversation_digest,
)
from nemo_gym.token_id_capture.fingerprint import (
    canonicalize_tool_arguments as canonicalize_tool_arguments,
)
from nemo_gym.token_id_capture.protocols import LineageMatch, LineageResolution
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry, cumulative_tokens


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
    ) -> None:
        """Build an in-memory entry for direct index tests."""
        request_len = context_len if context_len is not None else max(len(messages) - 1, 0)
        entry = TokenEntry(
            rollout_id="_in_memory",
            model_call_id=call_id,
            prompt_token_ids=[],
            generation_token_ids=list(cum_tokens),
            generation_log_probs=[0.0] * len(cum_tokens),
            output_items=list(messages[request_len:]),
            cum_len=len(cum_tokens),
            digest=digest,
            continuation_fingerprint=assistant_fingerprint(messages),
            continuation_context_len=request_len,
            continuation_context_digest=conversation_digest(messages[:request_len]),
        )
        self.add_entry(entry)


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
    This class supports in-process framework adapters and tests.
    Its index is memory-only.
    Eviction or restart leaves affected continuations unresolved.
    That failure mode is safe but can mask otherwise usable rollouts.
    Production adapters should back the incremental resolver with durable records.
    """

    def __init__(self, max_rollouts: int = 512, max_tokens: int = 8_000_000) -> None:
        self.index = LineageIndex(max_rollouts=max_rollouts, max_tokens=max_tokens)

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        return self.index.for_rollout(rollout_id).resolve(request_items)

    async def put(self, entry: TokenEntry) -> None:
        """Publish one committed entry to the worker-local index."""
        self.index.for_rollout(entry.rollout_id).add_entry(entry)

    def is_process_shared(self) -> bool:
        return False

    async def close(self) -> None:
        self.index.clear()


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
