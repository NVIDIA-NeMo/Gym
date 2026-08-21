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

Every new record distinguishes a root, a resolved parent, and an unresolved boundary.
Only records that predate this metadata use token-prefix fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from nemo_gym.token_id_capture.protocols import LineageMatch, LineageResolution
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry, cumulative_tokens


_FINGERPRINT_DOMAIN = b"nemo-gym-lineage"
_CONTEXT_DOMAIN = b"nemo-gym-lineage-context"


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
    cum_tokens: list[int]
    cum_len: int
    digest: str
    # These fields describe the request context sent for this call.
    # They exclude the model's response.
    # The request context has a stable item count across dialect round trips.
    context_len: int = 0
    context_digest: str = ""


def stamp_continuation(entry: TokenEntry, request_items: list[dict]) -> TokenEntry:
    """Add compact lookup metadata before the token entry is committed."""
    entry.continuation_fingerprint = assistant_fingerprint(list(request_items) + list(entry.output_items))
    entry.continuation_context_len = len(request_items)
    entry.continuation_context_digest = conversation_digest(request_items)
    return entry


@dataclass
class RolloutLineage:
    """Keep an append-only per-rollout call index."""

    by_fingerprint: dict[str, list[str]] = field(default_factory=dict)
    by_call_id: dict[str, LineageNode] = field(default_factory=dict)
    # Cache the cumulative token count for memory bounds.
    total_tokens: int = 0

    def resolve(self, messages: list[dict]) -> LineageResolution:
        """Return the immutable parent decision for this request.

        A request without model-authored history is a root.
        A request with unverified history is unresolved.
        Never guess among calls with identical output.
        """
        fingerprint = assistant_fingerprint(messages)
        if not fingerprint:
            return LineageResolution(ParentResolutionStatus.ROOT)
        call_ids = self.by_fingerprint.get(fingerprint) or []
        candidates = [
            node
            for call_id in call_ids
            if (node := self.by_call_id.get(call_id)) is not None and self._continues(node, messages)
        ]
        if len(candidates) != 1:
            reason = "no_match" if not candidates else "ambiguous"
            return LineageResolution(ParentResolutionStatus.UNRESOLVED, reason=reason)
        node = candidates[0]
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

    def add_entry(self, entry: TokenEntry) -> None:
        """Index lookup metadata carried by one committed token entry."""
        if not entry.continuation_fingerprint:
            return
        node = LineageNode(
            call_id=entry.model_call_id,
            cum_tokens=cumulative_tokens(entry),
            cum_len=entry.cum_len if entry.cum_len is not None else len(cumulative_tokens(entry)),
            digest=entry.digest or "",
            context_len=entry.continuation_context_len,
            context_digest=entry.continuation_context_digest,
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
    """Resolve entries committed to one worker-local index.

    Call ``put`` at the same publication boundary as the token sink.
    Multi-worker deployments require a shared backend.
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


class FileLineageStore:
    """Resolve lineage from the token JSONL committed by ``TokenCaptureStore``."""

    def __init__(self, root: str | Path) -> None:
        from nemo_gym.token_id_capture.store import TokenCaptureStore

        self._store = TokenCaptureStore(root)
        self._cache: dict[str, tuple[int, int, RolloutLineage]] = {}

    def _refresh(self, rollout_id: str) -> RolloutLineage:
        path = self._store.path_for(rollout_id)
        if not path.exists():
            self._cache.pop(rollout_id, None)
            return RolloutLineage()
        file_stat = path.stat()
        inode, offset, lineage = self._cache.get(rollout_id, (file_stat.st_ino, 0, RolloutLineage()))
        if inode != file_stat.st_ino or offset < 0 or offset > file_stat.st_size:
            inode, offset, lineage = file_stat.st_ino, 0, RolloutLineage()
        if offset == file_stat.st_size:
            return lineage
        with path.open("rb") as handle:
            handle.seek(offset)
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                lineage.add_entry(TokenEntry.model_validate(orjson.loads(payload)))
            offset = handle.tell()
        self._cache[rollout_id] = (inode, offset, lineage)
        return lineage

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        return await asyncio.to_thread(self._resolve, rollout_id, request_items)

    def _resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        # The resolver shares the token store's lock and durable log.
        # A successful ``TokenSink.put`` is therefore immediately visible.
        with self._store._locked(rollout_id, shared=True):
            lineage = self._refresh(rollout_id)
            return lineage.resolve(request_items)

    def is_process_shared(self) -> bool:
        return True

    async def close(self) -> None:
        self._cache.clear()
