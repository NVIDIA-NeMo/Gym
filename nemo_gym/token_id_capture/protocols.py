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

    Implementations must provide cross-worker read-after-write consistency.
    After ``TokenSink.put`` returns, a later ``resolve`` must see the entry.
    Implementations must never guess among candidates.
    """

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageResolution:
        """Return whether the request is a root, resolved, or unresolved.

        ``request_items`` are the unmodified harness items.
        The implementation must verify the recorded request context.
        """
        ...

    def is_process_shared(self) -> bool:
        """Return whether separate model-server workers share committed entries."""
        ...

    async def close(self) -> None:
        """Flush pending work and release resources. Idempotent."""
        ...


@runtime_checkable
class TokenSink(Protocol):
    """Receive captured records through Gym's file store or a framework transport."""

    async def put(self, entry: TokenEntry) -> None:
        """Durably store one record.

        The entry carries its continuation lookup metadata.
        A paired lineage resolver must see it after this method returns.
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
