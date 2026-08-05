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
serialize all of them behind one fsync. On a shared or network filesystem that
collapses throughput to ~1/fsync-latency regardless of core count. The per-file
flock keeps concurrent writers to one rollout correct while letting writes to
different rollouts proceed in parallel.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
from pathlib import Path
from typing import Any

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

    def incomplete_path_for(self, rollout_id: str) -> Path:
        """Sentinel marking that at least one call of this rollout failed to capture."""
        return self._root / f"{validate_rollout_id(rollout_id)}.tokens.incomplete"

    def mark_incomplete(self, rollout_id: str, reason: str = "") -> None:
        """Record that a call was lost.

        Capture is best effort per call, because a bad payload must never break the
        harness, but a rollout that captured 9 of 10 calls must not be
        indistinguishable from a complete one. The marker is a file rather than
        an in-process counter because the writer (model server) and the reader
        (rollout collection, or the trainer) are different processes.
        """
        try:
            with self.incomplete_path_for(rollout_id).open("a") as handle:
                handle.write(f"{reason}\n")
        except OSError:
            # Never let bookkeeping about a failed capture cause another failure.
            pass

    def is_incomplete(self, rollout_id: str) -> bool:
        return self.incomplete_path_for(rollout_id).exists()

    def append(self, entry: TokenEntry) -> None:
        """Append one entry and fsync. Blocking file IO, so callers on the event
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

    # --- TokenSink / TokenSource. The file store is Gym's default implementation of both;
    # a framework swaps in its own without touching the capture path.
    #
    # Both offload to the default thread pool. The pool is shared process-wide and small
    # (min(32, cpus + 4)), so it is worth stating what this costs: measured on one NVMe
    # workstation, a 1.4k-token entry takes ~0.2 ms to write and 2560 concurrent writes
    # finish in 0.55 s; at a 131k-token context a write takes ~10 ms and throughput stays
    # linear in total bytes rather than collapsing. Serialization, not the write, dominates.

    async def put(self, entry: TokenEntry) -> None:
        """``TokenSink``: durable on return. The blocking append is offloaded so
        it does not sit on the event loop, and awaited so a reader after the
        rollout never races a partial file."""
        await asyncio.to_thread(self.append, entry)

    async def tokens_for(self, rollout_id: str) -> list[TokenEntry]:
        """``TokenSource``."""
        return await asyncio.to_thread(self.read_entries, rollout_id)

    async def drop(self, rollout_id: str) -> None:
        """``TokenSource``: delete-on-consume."""
        await asyncio.to_thread(self.delete, rollout_id)

    def delete(self, rollout_id: str) -> None:
        """Remove a rollout's records and its incomplete marker.

        Records are large (hundreds of KB per rollout) and the append opens in
        "ab" mode, so leaving a consumed file behind both grows the directory
        without bound and lets a rerun that reuses the id append onto stale
        records.
        """
        self.path_for(rollout_id).unlink(missing_ok=True)
        self.incomplete_path_for(rollout_id).unlink(missing_ok=True)

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


def make_token_store(global_config_dict: Any) -> TokenCaptureStore | None:
    """Build the training-token store, or ``None`` when this process is not writing one.

    ``None`` when capture is off, when no directory resolves, or when a sink is configured: the
    records go to that transport instead and there is no file store to build.
    """
    from nemo_gym.token_id_capture.config import TokenIdCaptureConfig

    config = TokenIdCaptureConfig.model_validate(global_config_dict)
    if not config.enabled or config.token_id_capture.sink is not None:
        return None
    directory = config.resolved_dir()
    return TokenCaptureStore(directory) if directory is not None else None
