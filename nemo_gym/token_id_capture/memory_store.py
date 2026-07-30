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

"""The gate's in-flight token custody: per-rollout delta buffers in memory.

Holds each committed call's ``token_ids_delta`` (ids only — never logprobs)
keyed by ``(rollout_id, call_id)`` with a parent pointer, so serving a child's
exact prefix is a chain walk that concatenates deltas root -> parent. Memory
is O(total committed tokens) per rollout — forks share their common prefix
through the parent chain instead of duplicating cumulative sequences.

State self-clears at seal / fail / TTL (`drop`). This is the hot-path
companion to #2124's durable ``TokenCaptureStore`` (JSONL), which remains the
debug/persistence backend; the buffer deliberately exposes a delta-chain
interface instead of the JSONL store's ``TokenEntry`` append/read, because
prefix serving needs parent-linked deltas, not per-call full snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class TokenBufferError(Exception):
    """A buffer operation violated the delta-chain contract (caller bug)."""


@dataclass
class _CallDelta:
    call_id: str
    parent_call_id: Optional[str]
    token_ids_delta: list[int]
    cum_len: int


@dataclass
class _RolloutBuffer:
    calls: dict[str, _CallDelta] = field(default_factory=dict)
    total_tokens: int = 0


class MemoryRolloutTokenBuffer:
    """In-memory rollout -> call-delta forest, single-owner per gate."""

    def __init__(self) -> None:
        self._rollouts: dict[str, _RolloutBuffer] = {}

    def register(self, rollout_id: str) -> None:
        """Create-only, mirroring lineage registration."""
        if rollout_id in self._rollouts:
            raise TokenBufferError(f"rollout {rollout_id} already has a token buffer (create-only)")
        self._rollouts[rollout_id] = _RolloutBuffer()

    def __contains__(self, rollout_id: str) -> bool:
        return rollout_id in self._rollouts

    def has_call(self, rollout_id: str, call_id: str) -> bool:
        buffer = self._rollouts.get(rollout_id)
        return buffer is not None and call_id in buffer.calls

    def add_call(
        self,
        rollout_id: str,
        call_id: str,
        *,
        parent_call_id: Optional[str],
        token_ids_delta: list[int],
    ) -> None:
        """Record one committed call's delta. The parent must already be
        buffered (children are admitted only against committed parents), and
        the delta must be non-empty — empty deltas never commit."""
        buffer = self._buffer(rollout_id)
        if call_id in buffer.calls:
            raise TokenBufferError(f"rollout {rollout_id}: call {call_id} already buffered")
        if not token_ids_delta:
            raise TokenBufferError(f"rollout {rollout_id}: call {call_id} buffered an empty delta")
        if parent_call_id is None:
            prev_len = 0
        else:
            parent = buffer.calls.get(parent_call_id)
            if parent is None:
                raise TokenBufferError(f"rollout {rollout_id}: call {call_id} parent {parent_call_id} is not buffered")
            prev_len = parent.cum_len
        buffer.calls[call_id] = _CallDelta(
            call_id=call_id,
            parent_call_id=parent_call_id,
            token_ids_delta=list(token_ids_delta),
            cum_len=prev_len + len(token_ids_delta),
        )
        buffer.total_tokens += len(token_ids_delta)

    def cumulative_ids(self, rollout_id: str, call_id: str) -> list[int]:
        """The exact cumulative token sequence through ``call_id`` — the
        prefix served to a child admitted against it."""
        buffer = self._buffer(rollout_id)
        chain: list[_CallDelta] = []
        cursor: Optional[str] = call_id
        while cursor is not None:
            node = buffer.calls.get(cursor)
            if node is None:
                raise TokenBufferError(f"rollout {rollout_id}: call {cursor} is not buffered")
            chain.append(node)
            cursor = node.parent_call_id
        ids: list[int] = []
        for node in reversed(chain):
            ids.extend(node.token_ids_delta)
        return ids

    def total_tokens(self, rollout_id: str) -> int:
        return self._buffer(rollout_id).total_tokens

    def drop(self, rollout_id: str) -> None:
        """Release a rollout's buffer (seal / fail / TTL). Idempotent."""
        self._rollouts.pop(rollout_id, None)

    def __len__(self) -> int:
        return len(self._rollouts)

    def _buffer(self, rollout_id: str) -> _RolloutBuffer:
        buffer = self._rollouts.get(rollout_id)
        if buffer is None:
            raise TokenBufferError(f"rollout {rollout_id} has no token buffer")
        return buffer
