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

"""Interfaces for writing and reading captured training tokens.

Gym owns the record shape, these protocols, and the code that builds a record. A
training framework supplies the implementations and runs them wherever its
tokens are produced. Neither side imports the other's transport.

Placement of the write is therefore a deployment choice:

- Gym owns serving (today): install the sink in the model server, which already
  holds the assembled response, so there is no extra hop.
- A framework owns the inference worker: install the sink there, so bulk token
  arrays go to the framework's data plane instead of riding back through Gym's
  HTTP response.

The capture code is the same in both cases. This module must stay free of
fastapi, ray, torch and aiohttp imports so a framework's worker can import it
without pulling in Gym's server stack. A unit test enforces that.
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


@runtime_checkable
class TokenSink(Protocol):
    """Where captured records go.

    Gym's file store and framework-owned transports implement this protocol.
    """

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
        """Flush pending work and release resources. Idempotent."""
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
        """Release resources. Idempotent."""
        ...


# Installed once at process startup by whoever owns the process: Gym's model
# server, or a framework's inference worker. The capture path reads it when a
# request-scoped context does not carry an explicit sink.
_INSTALLED_SINK: TokenSink | None = None
_INSTALLED_SOURCE: TokenSource | None = None


def install_token_sink(sink: TokenSink | None) -> None:
    """Set (or clear, with ``None``) the process-wide default sink."""
    global _INSTALLED_SINK
    _INSTALLED_SINK = sink


def installed_token_sink() -> TokenSink | None:
    return _INSTALLED_SINK


def install_token_source(source: TokenSource | None) -> None:
    """Set (or clear, with ``None``) the process-wide default source."""
    global _INSTALLED_SOURCE
    _INSTALLED_SOURCE = source


def installed_token_source() -> TokenSource | None:
    return _INSTALLED_SOURCE
