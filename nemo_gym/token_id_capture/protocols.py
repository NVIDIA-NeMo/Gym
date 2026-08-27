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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from nemo_gym.token_id_capture.records import TokenEntry


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
    # Empty for token-free external custody rows; populated for internal
    # lineage rows (prefix injection) and legacy external rows.
    cumulative_token_ids: tuple[int, ...]
    digest: str
    staging_chain: tuple[str, ...] = ()
    prev_len: int = 0
    chain_hash: str = ""


@runtime_checkable
class LineageStore(Protocol):
    """Share request-time lineage across every worker serving a rollout.

    Implementations must provide cross-worker read-after-write consistency.
    After ``record`` returns, a later ``resolve`` must see the record.
    A missing or ambiguous identity returns ``None``.
    Implementations must never guess among candidates.
    """

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageMatch | None:
        """Return one verified parent, or ``None`` for a root or ambiguity.

        ``request_items`` are the unmodified harness items.
        The implementation must verify the recorded request context.
        """
        ...

    async def record(
        self,
        rollout_id: str,
        model_call_id: str,
        request_items: list[dict],
        response_items: list[dict],
        cumulative_token_ids: list[int],
        digest: str,
        *,
        staging_chain: list[str] | None = None,
    ) -> None:
        """Publish a completed call for later request-time resolution.

        Request and response items retain their wire representations.
        Repeating a model call ID with the same payload is a no-op.
        Reusing a model call ID with different data must fail.
        Return only after every serving worker can read the record.
        """
        ...

    async def close(self) -> None:
        """Flush pending work and release resources. Idempotent."""
        ...


@runtime_checkable
class CaptureLedger(LineageStore, Protocol):
    """A lineage store that doubles as the per-rollout capture ledger.

    External staging requires this surface: rows additionally carry either
    the token-free ``CallRecord`` custody columns (passed to ``record`` as
    keyword arguments: ``parent_call_id``, ``staging_key``, ``weight_version``,
    ``prev_len``/``delta_len``/``cum_len``, ``staging_digest``,
    ``extras_digest``, ``mode``, ``logical_request_id``, ``admitted_at``,
    ``staging_chain``, ``chain_hash``, ``cumulative_hash``, ``response_id``,
    ``output_fingerprint``, ``continuation_fingerprint``,
    ``fingerprint_version``), or
    a deferred MInf reference (``ledger_request_uid`` plus its lineage,
    content hashes, served ``response_id``, and length columns),
    poison rows are
    appended with ``record_failure``, and the framework reads the rollout back
    token-free through ``manifest``.
    """

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
        ledger_request_uid: str | None = None,
    ) -> None:
        """Publish lineage with staged-custody or deferred-ledger columns."""
        ...

    async def record_failure(self, rollout_id: str, model_call_id: str, reason: str) -> None:
        """Append a poison row for a call whose capture did not commit.

        Failure rows carry no fingerprint, so ``resolve`` never returns them
        as parents. Any failure row poisons the rollout's manifest.
        """
        ...

    async def manifest(self, rollout_id: str) -> dict:
        """Return the rollout's token-free ledger as plain wire data.

        The shape validates as ``staging.records.RolloutManifest``:
        committed rows under ``records``, deferred MInf rows under
        ``pending_records``, and poison rows under ``failures``.
        Cumulative token IDs never appear in the manifest.
        """
        ...

    async def has_rows(self, rollout_id: str) -> bool:
        """Return whether any ledger row (committed or failed) exists."""
        ...


@runtime_checkable
class TokenSink(Protocol):
    """Receive captured records through Gym's file store or a framework transport."""

    async def put(self, entry: TokenEntry) -> None:
        """Durably store one record.

        Repeating the same call id with the same payload is a no-op.
        Reusing a call id with a different payload must fail.
        Writing after the rollout is frozen must fail.

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
        """
        ...

    async def close(self) -> None:
        """Flush pending work and release resources idempotently."""
        ...


@runtime_checkable
class TokenSource(Protocol):
    """Where a trajectory builder freezes, reads, and retires records."""

    async def freeze(self, rollout_id: str) -> TokenCaptureSnapshot:
        """Freeze a rollout and return one atomic snapshot.

        Freezing is idempotent.
        No successful writes may occur after it returns.
        Entry order carries no meaning.
        """
        ...

    async def drop(self, rollout_id: str, *, snapshot_id: str, version: int) -> bool:
        """Conditionally retire the exact frozen snapshot that was consumed.

        Return ``False`` if state changed after the snapshot.
        Implementations that cannot delete return ``True``.
        Their owner remains responsible for retention.
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
