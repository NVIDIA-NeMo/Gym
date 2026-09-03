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

"""Define interfaces for captured training tokens.

Gym owns the record shape and capture protocols.
A training framework may implement the transport.
The sink may run in a Gym model server.
It may instead run in a framework inference worker.
Engine-side placement keeps token arrays off Gym's HTTP response.
Consumers read through ``TokenSource.freeze``.
They identify the frozen state with ``snapshot_id``.
This module avoids FastAPI, Ray, Torch, and aiohttp imports.

The serving path awaits ``TokenSink.put`` before returning the model response.
When ``put`` returns, the record must be durable and visible to every worker's lineage resolver.
The harness can then send a continuation to any worker without racing publication of the previous call.
A transport that returns before cross-client visibility can produce intermittent unresolved samples under load.

A sink may additionally implement ``begin_call(rollout_id, model_call_id)``.
It is an optional extension and deliberately not part of the ``TokenSink`` protocol.
``begin_call`` durably records a pre-dispatch intent.
An intent with no matching entry at freeze must mask the rollout.
That closes the window where the final call's entry is lost without a trace.
``begin_call`` runs before generation, so the caller may fail the model call at zero compute cost.

``nemo_gym.token_id_capture.conformance`` checks an external implementation against these contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry


@dataclass(frozen=True)
class TokenCaptureSnapshot:
    """An immutable view of one rollout's frozen capture records."""

    rollout_id: str
    entries: tuple[TokenEntry, ...]
    incomplete: bool
    snapshot_id: str
    version: int


@dataclass(frozen=True)
class LineageMatch:
    """Describe a uniquely verified parent from a shared lineage store."""

    model_call_id: str
    cumulative_token_ids: tuple[int, ...]
    digest: str


@dataclass(frozen=True)
class LineageResolution:
    """Return one immutable request-time parent decision."""

    status: ParentResolutionStatus
    match: LineageMatch | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status == ParentResolutionStatus.RESOLVED and self.match is None:
            raise ValueError("resolved lineage requires a match")
        if self.status != ParentResolutionStatus.RESOLVED and self.match is not None:
            raise ValueError(f"{self.status.value} lineage cannot carry a match")


@runtime_checkable
class LineageStore(Protocol):
    """Resolve request-time lineage from entries committed by a token sink.

    This is a read-only view over sink-committed records.
    After ``TokenSink.put`` returns, a later ``resolve`` on any worker must see the entry.
    Visibility is required per rollout key only, so the store may shard by rollout.
    Implementations must never guess among candidates.
    ``resolve`` must return ``UNRESOLVED`` when it cannot prove one parent.
    The offline builder independently re-verifies every claimed link by digest.
    External implementations should embed Gym's ``RolloutLineage`` matcher rather than reimplement the hashing.
    ``nemo_gym.token_id_capture.lineage`` is importable without the server stack.
    Callers of ``commit_entry`` outside ``capture_tokens`` must run ``stamp_continuation`` themselves.
    Unstamped entries are invisible to resolution.
    """

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        """Return whether the request is a root, resolved, or unresolved.

        ``request_items`` are the unmodified harness items.
        The implementation must verify the recorded request context.
        A conflicting set of committed payloads for one call id must count as zero candidates.
        UNRESOLVED is always a safe answer; a wrong RESOLVED is caught later by digest verification.
        """
        ...

    def is_process_shared(self) -> bool:
        """Return whether separate model-server workers share committed entries."""
        ...

    async def close(self) -> None:
        """Release resources. Idempotent.

        The store is read-only, so there is never pending work to flush.
        """
        ...


@runtime_checkable
class TokenSink(Protocol):
    """Receive captured records through Gym's file store or a framework transport."""

    async def put(self, entry: TokenEntry) -> None:
        """Durably store one record before returning.

        The entry carries its continuation lookup metadata.
        Durability and resolver visibility are the return condition, not an eventual goal.
        Any worker's paired lineage resolver must see the entry once this method returns.
        Repeating the same call id with the same payload is a no-op.
        "Same payload" means the identical serialized entry, byte-for-byte, timestamps included.
        A retry must resend the same bytes; rebuilding the entry produces a conflict, not a retry.
        Reusing a call id with a different payload must fail.
        A transport without compare-and-swap may delegate that conflict to the reader.
        A resolver must then treat conflicting committed payloads for one call id as zero candidates.
        Writing after freeze must fail or bump the frozen version; see ``TokenSource.freeze``.

        This method may raise.
        The caller marks the rollout incomplete.
        A capture error never fails the model call.
        """
        ...

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        """Durably record that a call of this rollout failed to capture.

        The rollout is now missing a turn.
        A consumer must mask the sample instead of training on a chain with a hole.
        The model call itself still succeeds.
        This marker is therefore the durable signal that capture failed.
        It must succeed after freeze.
        It must change the observable version; that is what invalidates a stale retirement.
        Make it more available than ``put``, for example through a local spill.
        ``put`` and ``mark_incomplete`` failing together is the silent-loss case.
        """
        ...

    async def close(self) -> None:
        """Release resources. Idempotent.

        There is never buffered unwritten data here.
        ``put`` guaranteed durability before it returned.
        A close that must flush records means ``put`` broke its contract.
        """
        ...


@runtime_checkable
class TokenSource(Protocol):
    """Where a trajectory builder freezes, reads, and retires records."""

    async def freeze(self, rollout_id: str) -> TokenCaptureSnapshot:
        """Freeze a rollout and return one atomic snapshot.

        Freezing is idempotent.
        Entry order carries no meaning.
        Entries are unique per ``model_call_id``.
        An at-least-once transport must dedupe identical copies before snapshotting.
        The fence is relaxed: "no successful write after freeze" is not required cluster-wide.
        A strict fence is unimplementable without compare-and-swap.
        A write racing freeze may therefore succeed durably.
        It must then bump the version.
        A conditional ``drop`` of the consumed snapshot then fails, and the evidence is retained.
        Attempt-scoped rollout ids are the sanctioned strategy for retirement without compare-and-swap.
        """
        ...

    async def drop(self, rollout_id: str, *, snapshot_id: str, version: int) -> bool:
        """Conditionally retire the exact frozen snapshot that was consumed.

        Return ``False`` if state changed after the snapshot.
        Transports without delete return ``True`` and own retention.
        """
        ...

    async def close(self) -> None:
        """Release resources idempotently."""
        ...


# Install these defaults once in the process that owns them.
# The owner may be a Gym model server or a framework inference worker.
# Request-scoped sinks take precedence.
_INSTALLED_SINK: TokenSink | None = None
_INSTALLED_SOURCE: TokenSource | None = None
_INSTALLED_LINEAGE_STORE: LineageStore | None = None


def install_token_sink(sink: TokenSink | None) -> None:
    """Set (or clear, with ``None``) the process-wide default sink."""
    if sink is not None:
        # A sink missing mark_incomplete can hide incomplete rollouts.
        # Validate programmatically installed sinks too.
        missing = [name for name in ("put", "mark_incomplete", "close") if not callable(getattr(sink, name, None))]
        if missing:
            raise TypeError(f"installed token sink lacks required methods: {', '.join(missing)}")
    global _INSTALLED_SINK
    _INSTALLED_SINK = sink


def installed_token_sink() -> TokenSink | None:
    return _INSTALLED_SINK


def install_token_source(source: TokenSource | None) -> None:
    """Set (or clear) the caller-owned source in this process.

    Gym does not close an installed source.
    """
    global _INSTALLED_SOURCE
    _INSTALLED_SOURCE = source


def installed_token_source() -> TokenSource | None:
    return _INSTALLED_SOURCE


def install_lineage_store(store: LineageStore | None) -> None:
    """Set (or clear) the process-wide request-time lineage store."""
    global _INSTALLED_LINEAGE_STORE
    _INSTALLED_LINEAGE_STORE = store


def installed_lineage_store() -> LineageStore | None:
    return _INSTALLED_LINEAGE_STORE
