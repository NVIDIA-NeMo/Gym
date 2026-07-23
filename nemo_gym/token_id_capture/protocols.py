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

"""The write and read seams for captured training tokens.

Gym owns the record shape, these two protocols, and the code that builds a
record. A training framework supplies the implementation and runs it wherever
its tokens are produced. Neither side imports the other's transport.

This makes the *placement* of the write a deployment choice rather than a fork
in the design:

- Gym owns serving (today): install the sink in the model server, which already
  holds the assembled response. No extra hop.
- A framework owns the inference worker: install the sink there, so bulk token
  arrays go straight to the framework's data plane instead of riding back
  through Gym's HTTP response.

Same capture code both times. This module must stay dependency-free (no
fastapi, ray, torch, aiohttp) so a framework's worker can import it without
pulling in Gym's server stack; ``tests/unit_tests/test_token_id_capture.py``
enforces that.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from nemo_gym.token_id_capture.records import TokenEntry


@runtime_checkable
class TokenSink(Protocol):
    """Where captured records go. Implemented by Gym's file store, or by a
    framework over its own transport."""

    async def put(self, entry: TokenEntry) -> None:
        """Append one record.

        MUST be durable on return: a later ``tokens_for`` for the same rollout
        has to see it. Delete-on-consume and post-rollout reads are only correct
        because of this.

        May raise. The caller counts the failure and marks the rollout, and
        never fails the model call because of it.
        """
        ...


@runtime_checkable
class TokenSource(Protocol):
    """Where a trajectory builder reads records from."""

    async def tokens_for(self, rollout_id: str) -> list[TokenEntry]:
        """All records for a rollout, in any order.

        Order carries no meaning: calls run concurrently and may be served by
        different workers. The builder recovers structure from the records
        themselves (parent links, or token-prefix relationships).
        """
        ...

    async def drop(self, rollout_id: str) -> None:
        """Retire a rollout's records once they have been consumed."""
        ...


# Installed once at process startup by whoever owns the process: Gym's model
# server, or a framework's inference worker. The capture path reads it when a
# request-scoped context does not carry an explicit sink.
_INSTALLED_SINK: Optional[TokenSink] = None


def install_token_sink(sink: Optional[TokenSink]) -> None:
    """Set (or clear, with ``None``) the process-wide default sink."""
    global _INSTALLED_SINK
    _INSTALLED_SINK = sink


def installed_token_sink() -> Optional[TokenSink]:
    return _INSTALLED_SINK
