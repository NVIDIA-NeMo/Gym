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

"""Append-only, rollout-keyed store for training ``TokenEntry`` records.

One file per rollout (``<rollout_id>.tokens.jsonl``), separate from the
evaluation capture file (``<rollout_id>.capture.jsonl``) so token payloads never
bloat eval reads. Each write fsyncs and holds a per-file ``flock`` (which
excludes other threads and worker processes writing the *same* rollout file),
because a killed box must not lose a rollout's training tokens.

Concurrency is per file, not global: there is deliberately no process-wide lock.
Every model call appends to its own rollout's file, so a global lock would
serialize all of them behind one fsync -- on a shared/network filesystem that
collapses throughput to ~1/fsync-latency regardless of core count. The per-file
flock keeps concurrent writers to one rollout correct while letting writes to
different rollouts proceed in parallel.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import orjson

from nemo_gym.token_id_capture.records import TokenEntry


def validate_rollout_id(rollout_id: str) -> str:
    """Reject anything that could escape the store directory or index a bad file."""
    if not rollout_id or any(not (char.isascii() and (char.isalnum() or char in "._-")) for char in rollout_id):
        raise ValueError(f"Invalid rollout id: {rollout_id!r}")
    return rollout_id


class TokenCaptureStore:
    """Durable, rollout-keyed JSONL sink for ``TokenEntry`` records."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, rollout_id: str) -> Path:
        return self._root / f"{validate_rollout_id(rollout_id)}.tokens.jsonl"

    def append(self, entry: TokenEntry) -> None:
        """Append one entry and fsync. Blocking file IO -- callers on the event
        loop must offload it (e.g. ``asyncio.to_thread``)."""
        line = orjson.dumps(entry.model_dump(), option=orjson.OPT_APPEND_NEWLINE)
        path = self.path_for(entry.rollout_id)
        with path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read_entries(self, rollout_id: str) -> list[TokenEntry]:
        path = self.path_for(rollout_id)
        if not path.exists():
            return []
        entries: list[TokenEntry] = []
        with path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        entries.append(TokenEntry.model_validate(orjson.loads(stripped)))
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return entries
