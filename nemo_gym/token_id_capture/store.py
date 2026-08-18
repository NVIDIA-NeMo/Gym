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
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson

from nemo_gym.token_id_capture.protocols import TokenCaptureSnapshot
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

    def state_path_for(self, rollout_id: str) -> Path:
        return self._root / f"{validate_rollout_id(rollout_id)}.tokens.state.json"

    def lock_path_for(self, rollout_id: str) -> Path:
        return self._root / f"{validate_rollout_id(rollout_id)}.tokens.lock"

    @contextmanager
    def _locked(self, rollout_id: str, *, shared: bool = False):
        with self.lock_path_for(rollout_id).open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_state(self, rollout_id: str) -> dict[str, Any]:
        path = self.state_path_for(rollout_id)
        if not path.exists():
            return {"sealed": False, "incomplete": False, "seal_id": "", "version": 0}
        state = orjson.loads(path.read_bytes())
        if not isinstance(state, dict):
            raise ValueError(f"Invalid token-capture state for rollout {rollout_id}")
        return state

    def _write_state(self, rollout_id: str, state: dict[str, Any]) -> None:
        payload = orjson.dumps(state, option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE)
        with tempfile.NamedTemporaryFile(dir=self._root, prefix=".tokens-state-", delete=False) as handle:
            temporary_path = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary_path, self.state_path_for(rollout_id))
            self._fsync_root()
        finally:
            temporary_path.unlink(missing_ok=True)

    def _fsync_root(self) -> None:
        descriptor = os.open(self._root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        with self._locked(rollout_id):
            state = self._read_state(rollout_id)
            state["incomplete"] = True
            state["version"] = int(state.get("version", 0)) + 1
            self._write_state(rollout_id, state)
            with self.incomplete_path_for(rollout_id).open("a", encoding="utf-8") as handle:
                handle.write(f"{model_call_id}\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._fsync_root()

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        """Durably record that a call was lost."""
        await asyncio.to_thread(self._mark_incomplete, rollout_id, model_call_id)

    def is_incomplete(self, rollout_id: str) -> bool:
        with self._locked(rollout_id, shared=True):
            return bool(self._read_state(rollout_id).get("incomplete", False))

    def append(self, entry: TokenEntry) -> None:
        """Idempotently append one entry and fsync."""
        canonical = orjson.dumps(entry.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
        line = canonical + b"\n"
        rollout_id = entry.rollout_id
        with self._locked(rollout_id):
            state = self._read_state(rollout_id)
            if state.get("sealed", False):
                raise RuntimeError(f"Token capture for rollout {rollout_id} is already sealed")
            for existing in self._read_entries_unlocked(rollout_id):
                if existing.model_call_id != entry.model_call_id:
                    continue
                existing_bytes = orjson.dumps(existing.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
                if existing_bytes == canonical:
                    return
                state["incomplete"] = True
                state["version"] = int(state.get("version", 0)) + 1
                self._write_state(rollout_id, state)
                raise ValueError(
                    f"Model call id {entry.model_call_id!r} was reused with a different payload "
                    f"for rollout {rollout_id!r}"
                )
            with self.path_for(rollout_id).open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            state["version"] = int(state.get("version", 0)) + 1
            self._write_state(rollout_id, state)

    # --- TokenSink / TokenSource. The file store is Gym's default implementation of both;
    # a framework swaps in its own without touching the capture path.
    #
    # Both offload to the default thread pool, which is shared process-wide and small
    # (min(32, cpus + 4)). Serializing the entry dominates the cost rather than the write
    # itself, so a long context is the case to watch if this ever shows up in a profile.

    async def put(self, entry: TokenEntry) -> None:
        """``TokenSink``: durable on return. The blocking append is offloaded so
        it does not sit on the event loop, and awaited so a reader after the
        rollout never races a partial file."""
        await asyncio.to_thread(self.append, entry)

    async def seal(self, rollout_id: str) -> TokenCaptureSnapshot:
        return await asyncio.to_thread(self._seal, rollout_id)

    def _seal(self, rollout_id: str) -> TokenCaptureSnapshot:
        with self._locked(rollout_id):
            state = self._read_state(rollout_id)
            if not state.get("sealed", False):
                state["sealed"] = True
                state["seal_id"] = uuid4().hex
                state["version"] = int(state.get("version", 0)) + 1
                self._write_state(rollout_id, state)
            entries = tuple(self._read_entries_unlocked(rollout_id))
            return TokenCaptureSnapshot(
                rollout_id=rollout_id,
                entries=entries,
                incomplete=bool(state.get("incomplete", False)),
                seal_id=str(state["seal_id"]),
                version=int(state["version"]),
            )

    async def tokens_for(self, rollout_id: str) -> list[TokenEntry]:
        """Compatibility read for diagnostics. Consumers should use ``seal``."""
        return await asyncio.to_thread(self.read_entries, rollout_id)

    async def drop(self, rollout_id: str, *, seal_id: str, version: int) -> bool:
        """Conditionally delete the sealed snapshot."""
        return await asyncio.to_thread(self._drop, rollout_id, seal_id, version)

    def _drop(self, rollout_id: str, seal_id: str, version: int) -> bool:
        with self._locked(rollout_id):
            state = self._read_state(rollout_id)
            if (
                not state.get("sealed", False)
                or state.get("seal_id") != seal_id
                or int(state.get("version", 0)) != version
            ):
                return False
            self.path_for(rollout_id).unlink(missing_ok=True)
            self.incomplete_path_for(rollout_id).unlink(missing_ok=True)
            self.state_path_for(rollout_id).unlink(missing_ok=True)
            self._fsync_root()
            return True

    async def close(self) -> None:
        """The file store owns no persistent handles."""

    def delete(self, rollout_id: str) -> None:
        """Unconditionally remove a rollout's records.

        This compatibility helper is for administrative cleanup. Normal
        consumers use conditional ``drop``.
        """
        with self._locked(rollout_id):
            self.path_for(rollout_id).unlink(missing_ok=True)
            self.incomplete_path_for(rollout_id).unlink(missing_ok=True)
            self.state_path_for(rollout_id).unlink(missing_ok=True)
            self._fsync_root()

    def read_entries(self, rollout_id: str) -> list[TokenEntry]:
        with self._locked(rollout_id, shared=True):
            return self._read_entries_unlocked(rollout_id)

    def _read_entries_unlocked(self, rollout_id: str) -> list[TokenEntry]:
        path = self.path_for(rollout_id)
        if not path.exists():
            return []
        entries: list[TokenEntry] = []
        with path.open("rb") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    entries.append(TokenEntry.model_validate(orjson.loads(stripped)))
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
