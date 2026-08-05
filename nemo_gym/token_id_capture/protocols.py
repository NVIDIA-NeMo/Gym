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

from typing import Protocol, runtime_checkable

from nemo_gym.token_id_capture.records import TokenEntry


@runtime_checkable
class TokenSink(Protocol):
    """Where captured records go. Implemented by Gym's file store, or by a
    framework over its own transport."""

    async def put(self, entry: TokenEntry) -> None:
        """Append one record.

        The record must be durable before this returns: a later ``tokens_for``
        for the same rollout has to see it. Delete-on-consume and post-rollout reads are only correct
        because of this.

        May raise. The caller counts the failure and marks the rollout, and
        never fails the model call because of it.
        """
        ...

    def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        """Record that a call of this rollout failed to capture.

        The rollout is now missing a turn, and a consumer must mask the sample rather
        than train on a chain with a hole in it. The model call itself still succeeds,
        so this is the only signal that anything went wrong: a sink that drops it makes
        an incomplete rollout indistinguishable from a complete one.

        Synchronous, because the caller is a failure path that cannot await. A transport
        that needs to send should queue here and flush elsewhere.
        """
        ...


@runtime_checkable
class TokenSource(Protocol):
    """Where a trajectory builder reads records from, and retires them afterwards."""

    async def tokens_for(self, rollout_id: str) -> list[TokenEntry]:
        """All records for a rollout, in any order.

        Order carries no meaning: calls run concurrently and may be served by
        different workers. The builder recovers structure from the records
        themselves, using parent links or token-prefix relationships.
        """
        ...

    async def drop(self, rollout_id: str) -> None:
        """Retire a rollout's records once they have been consumed.

        A transport that cannot delete implements this as a no-op and leaves
        retirement to whoever owns the storage.
        """
        ...


# Installed once at process startup by whoever owns the process: Gym's model
# server, or a framework's inference worker. The capture path reads it when a
# request-scoped context does not carry an explicit sink.
_INSTALLED_SINK: TokenSink | None = None


def install_token_sink(sink: TokenSink | None) -> None:
    """Set (or clear, with ``None``) the process-wide default sink."""
    global _INSTALLED_SINK
    _INSTALLED_SINK = sink


def installed_token_sink() -> TokenSink | None:
    return _INSTALLED_SINK
