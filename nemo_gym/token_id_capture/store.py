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

"""Store training ``TokenEntry`` records by rollout.

Each rollout uses one ``<rollout_id>.tokens.jsonl`` file.
Evaluation records use a separate file.
Every entry line is ``fsync``ed before ``put`` returns — that is the durability
guarantee. The state index is written atomically but fsynced only on lifecycle
transitions (freeze, mark, drop, refusal); it is reconstructible from the JSONL tail
except for those transitions, which is why they are fsynced.
A per-rollout file lock serializes writers to the same rollout.
Different rollouts can write concurrently.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import logging
import os
import tempfile
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson

from nemo_gym.token_id_capture.protocols import (
    RolloutRemovalPayload,
    TokenCaptureFrozenError,
    TokenCaptureRetiredError,
    TokenCaptureSnapshot,
)
from nemo_gym.token_id_capture.records import RefusalRecord, TokenEntry


logger = logging.getLogger(__name__)


def validate_rollout_id(rollout_id: str) -> str:
    """Reject anything that could escape the store directory or index a bad file."""
    if not rollout_id or any(not (char.isascii() and (char.isalnum() or char in "._-")) for char in rollout_id):
        raise ValueError(f"Invalid rollout id: {rollout_id!r}")
    return rollout_id


def validate_rollout_ids(rollout_ids: Sequence[str]) -> list[str]:
    """Validate a whole batch before anything changes, and drop duplicates.

    A bare string is rejected: it is a sequence of one-character rollout IDs.
    """
    if isinstance(rollout_ids, str):
        raise TypeError(f"expected a sequence of rollout ids, got the string {rollout_ids!r}")
    return list(dict.fromkeys(validate_rollout_id(rollout_id) for rollout_id in rollout_ids))


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

    def intents_path_for(self, rollout_id: str) -> Path:
        """Return the durable per-call intent path."""
        return self._root / f"{validate_rollout_id(rollout_id)}.tokens.intents"

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
            return {
                "frozen": False,
                "retired": False,
                "incomplete": False,
                "snapshot_id": "",
                "version": 0,
                "entry_digests": {},
                "indexed_size": 0,
                "refusals": [],
            }
        state = orjson.loads(path.read_bytes())
        if not isinstance(state, dict):
            raise ValueError(f"Invalid token-capture state for rollout {rollout_id}")
        return state

    @staticmethod
    def _entry_digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _sync_entry_index(self, rollout_id: str, state: dict[str, Any]) -> bool:
        """Reconcile an entry index with any durable JSONL tail.

        The JSONL write is durable before its state update.
        A process can therefore stop with one unindexed entry.
        Normal writes use the state index without parsing prior token arrays.
        Recovery parses only the unindexed tail.
        """
        path = self.path_for(rollout_id)
        file_size = path.stat().st_size if path.exists() else 0
        stored_index = state.get("entry_digests")
        stored_size = state.get("indexed_size")
        legacy_state = not isinstance(stored_index, dict) or not isinstance(stored_size, int)
        entry_digests = dict(stored_index) if isinstance(stored_index, dict) else {}
        indexed_size = stored_size if isinstance(stored_size, int) else 0
        if indexed_size < 0 or indexed_size > file_size:
            raise ValueError(f"Invalid token-capture index offset for rollout {rollout_id}")
        if indexed_size == file_size and not legacy_state:
            return False

        recovered = 0
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(indexed_size)
                tail = handle.read()
            position = 0
            while position < len(tail):
                newline = tail.find(b"\n", position)
                end = len(tail) if newline == -1 else newline
                payload = tail[position:end].strip()
                if payload:
                    try:
                        parsed = orjson.loads(payload)
                    except orjson.JSONDecodeError:
                        remainder = tail[end + 1 :] if newline != -1 else b""
                        if remainder.strip():
                            # A malformed line before more content is corrupt.
                            raise
                        # A torn final line was never acknowledged.
                        # Drop it while the caller holds the exclusive lock.
                        os.truncate(path, indexed_size + position)
                        file_size = indexed_size + position
                        logger.warning("Dropped %d torn trailing bytes from %s", len(tail) - position, path)
                        break
                    entry = TokenEntry.model_validate(parsed)
                    digest = self._entry_digest(payload)
                    existing = entry_digests.get(entry.model_call_id)
                    if existing is not None and existing != digest:
                        state["incomplete"] = True
                        state["version"] = int(state.get("version", 0)) + 1
                        self._write_state(rollout_id, state)
                        raise ValueError(
                            f"Model call id {entry.model_call_id!r} has conflicting durable payloads "
                            f"for rollout {rollout_id!r}"
                        )
                    entry_digests[entry.model_call_id] = digest
                    recovered += 1
                position = end + 1

        state["entry_digests"] = entry_digests
        state["indexed_size"] = file_size
        if recovered and not legacy_state:
            state["version"] = int(state.get("version", 0)) + recovered
        return True

    def _write_state(
        self, rollout_id: str, state: dict[str, Any], *, durable: bool = True, sync_root: bool = True
    ) -> None:
        # durable=False skips both fsyncs; sync_root=False leaves the directory fsync to the caller.
        # The temporary-file replacement remains atomic.
        # Use it only for state reconstructed from the JSONL tail.
        # Lifecycle flags are not reconstructible.
        payload = orjson.dumps(state, option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE)
        with tempfile.NamedTemporaryFile(dir=self._root, prefix=".tokens-state-", delete=False) as handle:
            temporary_path = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                if durable:
                    os.fsync(handle.fileno())
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary_path, self.state_path_for(rollout_id))
            if durable and sync_root:
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
            if state.get("retired", False):
                raise TokenCaptureRetiredError(f"Token capture for rollout {rollout_id} is retired")
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
        digest = self._entry_digest(canonical)
        rollout_id = entry.rollout_id
        with self._locked(rollout_id):
            state = self._read_state(rollout_id)
            if state.get("retired", False):
                raise TokenCaptureRetiredError(f"Token capture for rollout {rollout_id} is retired")
            if state.get("frozen", False):
                raise TokenCaptureFrozenError(f"Token capture for rollout {rollout_id} is already frozen")
            index_changed = self._sync_entry_index(rollout_id, state)
            entry_digests = state["entry_digests"]
            existing_digest = entry_digests.get(entry.model_call_id)
            if existing_digest is not None:
                if existing_digest == digest:
                    if index_changed:
                        self._write_state(rollout_id, state)
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
                state["indexed_size"] = handle.tell()
            entry_digests[entry.model_call_id] = digest
            state["version"] = int(state.get("version", 0)) + 1
            # The entry line's fsync is the durability guarantee.
            # The index is reconstructible from the JSONL tail.
            self._write_state(rollout_id, state, durable=False)

    # The file store is Gym's default TokenSink and TokenSource.
    # A framework can replace it without changing the capture path.
    #
    # Both interfaces offload blocking work to the process-wide default thread pool.

    async def put(self, entry: TokenEntry) -> None:
        """Store an entry durably without blocking the event loop.

        Await the append so later consumers cannot race a partial file.
        """
        await asyncio.to_thread(self.append, entry)

    def _begin_call(self, rollout_id: str, model_call_id: str) -> None:
        """Durably record that a captured call is about to be dispatched.

        A lost entry leaves a dangling intent.
        ``freeze_now`` then masks the rollout.
        A failure here happens before generation.
        """
        with self._locked(rollout_id):
            state = self._read_state(rollout_id)
            if state.get("retired", False):
                raise TokenCaptureRetiredError(f"Token capture for rollout {rollout_id} is retired")
            if state.get("frozen", False):
                raise TokenCaptureFrozenError(f"Token capture for rollout {rollout_id} is already frozen")
            with self.intents_path_for(rollout_id).open("ab") as handle:
                handle.write(model_call_id.encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())

    async def begin_call(self, rollout_id: str, model_call_id: str) -> None:
        await asyncio.to_thread(self._begin_call, rollout_id, model_call_id)

    def _refuse_call(self, rollout_id: str, model_call_id: str, code: str | None) -> None:
        """Durably record that the engine refused a captured call before it generated anything.

        The refusal resolves the call's pre-dispatch intent: no entry can ever arrive for the
        call, so ``freeze_now`` stops reporting the intent as dangling. The record lives in the
        state file and is fsynced here because it cannot be rebuilt from the JSONL tail.
        Repeating a call id with the same code is a no-op; a different code is a conflict.
        """
        with self._locked(rollout_id):
            state = self._read_state(rollout_id)
            if state.get("retired", False):
                raise TokenCaptureRetiredError(f"Token capture for rollout {rollout_id} is retired")
            if state.get("frozen", False):
                raise TokenCaptureFrozenError(f"Token capture for rollout {rollout_id} is already frozen")
            refusals = [RefusalRecord.model_validate(record) for record in state.get("refusals") or []]
            for existing in refusals:
                if existing.model_call_id != model_call_id:
                    continue
                if existing.code == code:
                    return
                raise ValueError(
                    f"Model call id {model_call_id!r} was refused with a different code for rollout {rollout_id!r}"
                )
            refusals.append(RefusalRecord(model_call_id=model_call_id, code=code, created_at=time.time()))
            state["refusals"] = [record.model_dump(mode="json") for record in refusals]
            state["version"] = int(state.get("version", 0)) + 1
            self._write_state(rollout_id, state)

    async def refuse_call(self, rollout_id: str, *, model_call_id: str, code: str | None) -> None:
        """Record an engine refusal; see ``_refuse_call``."""
        await asyncio.to_thread(self._refuse_call, rollout_id, model_call_id, code)

    def _dangling_intents(self, rollout_id: str, entries: tuple[TokenEntry, ...], state: dict[str, Any]) -> list[str]:
        path = self.intents_path_for(rollout_id)
        if not path.exists():
            return []
        # An entry or a refusal resolves an intent; a refused call never generated, so it was
        # not lost.
        resolved = {entry.model_call_id for entry in entries}
        resolved.update(str(record["model_call_id"]) for record in state.get("refusals") or [])
        intents = [line.strip().decode("utf-8") for line in path.read_bytes().splitlines() if line.strip()]
        return [call_id for call_id in intents if call_id not in resolved]

    async def freeze(self, rollout_id: str) -> TokenCaptureSnapshot:
        return await asyncio.to_thread(self.freeze_now, rollout_id)

    def freeze_now(self, rollout_id: str) -> TokenCaptureSnapshot:
        """Synchronously freeze one rollout and return its stable snapshot."""
        with self._locked(rollout_id):
            state = self._read_state(rollout_id)
            if state.get("retired", False):
                raise TokenCaptureRetiredError(f"Token capture for rollout {rollout_id} is retired")
            index_changed = self._sync_entry_index(rollout_id, state)
            if not state.get("frozen", False):
                state["frozen"] = True
                state["snapshot_id"] = uuid4().hex
                state["version"] = int(state.get("version", 0)) + 1
                self._write_state(rollout_id, state)
            elif index_changed:
                self._write_state(rollout_id, state)
            entries = tuple(self._read_entries_unlocked(rollout_id))
            # A dispatched call with neither an entry nor a refusal was lost.
            # The rollout must be masked.
            incomplete = bool(state.get("incomplete", False)) or bool(
                self._dangling_intents(rollout_id, entries, state)
            )
            return TokenCaptureSnapshot(
                rollout_id=rollout_id,
                entries=entries,
                incomplete=incomplete,
                snapshot_id=str(state["snapshot_id"]),
                version=int(state["version"]),
                refusals=tuple(RefusalRecord.model_validate(record) for record in state.get("refusals") or []),
            )

    async def drop(self, rollout_id: str, *, snapshot_id: str, version: int) -> bool:
        """Delete snapshot payloads while retaining its fence."""
        return await asyncio.to_thread(self._drop, rollout_id, snapshot_id, version)

    def _drop(self, rollout_id: str, snapshot_id: str, version: int) -> bool:
        with self._locked(rollout_id):
            state = self._read_state(rollout_id)
            if (
                not state.get("frozen", False)
                or state.get("snapshot_id") != snapshot_id
                or int(state.get("version", 0)) != version
            ):
                return False
            # The same two phases as ``retire_now``: a durable fence first, then the records.
            self._write_fence_unlocked(rollout_id, state)
            self._fsync_root()
            self._remove_record_files(rollout_id)
            self._fsync_root()
            return True

    def _write_fence_unlocked(self, rollout_id: str, state: dict[str, Any]) -> None:
        # The fence stays until ``delete``: a late writer from this attempt must still observe it. It is written,
        # and synced by the caller, before any record is removed, so a rollout is never left without records and
        # without a fence, and a late writer never finds an index that points past missing records.
        state["indexed_size"] = 0
        state["entry_digests"] = {}
        state["refusals"] = []
        state["retired"] = True
        self._write_state(rollout_id, state, sync_root=False)

    def _remove_record_files(self, rollout_id: str) -> None:
        """Remove a rollout's records, intents, and incomplete marker, whichever exist."""
        for path in (
            self.path_for(rollout_id),
            self.incomplete_path_for(rollout_id),
            self.intents_path_for(rollout_id),
        ):
            path.unlink(missing_ok=True)

    async def retire(self, rollout_ids: Sequence[str]) -> RolloutRemovalPayload:
        """Remove rollouts' records and keep a fence, whatever their state.

        This is ``drop`` without the snapshot check, for rollouts the consumer is done with but won't
        drop as a consumed snapshot: masked or failed captures once their evidence is no longer needed,
        and abandoned attempts. Like ``drop``, it leaves the fence that makes later writes for the
        rollout fail, including writes from a duplicate execution of the same rollout. Retiring again
        is a no-op.
        """
        return await asyncio.to_thread(self.retire_now, rollout_ids)

    def retire_now(self, rollout_ids: Sequence[str]) -> RolloutRemovalPayload:
        """Synchronous ``retire``.

        Two phases, each ending in one directory sync: write every missing fence, then remove the records. The
        first sync runs for every non-empty batch, even one that writes no fence, because the fences it finds
        may have been written by an overlapping or crashed retire that has not synced yet: a removal must never
        become durable before its fence, and the call must not return before the fence is durable.
        """
        removed, absent, to_clear = [], [], []
        unsynced = False
        try:
            for rollout_id in validate_rollout_ids(rollout_ids):
                with self._locked(rollout_id):
                    had_records = self._has_records(rollout_id)
                    state = self._read_state(rollout_id)
                    if not state.get("retired", False):
                        self._write_fence_unlocked(rollout_id, state)
                        unsynced = True
                    (removed if had_records else absent).append(rollout_id)
                    if had_records:
                        to_clear.append(rollout_id)
            # Sync every non-empty batch, even one that wrote no fence: an overlapping or crashed retire may have
            # written a fence this batch found without syncing it, and this call must not report it durable early.
            if unsynced or to_clear or removed or absent:
                self._fsync_root()
                unsynced = False
            if to_clear:
                # Set before removing: a removal that fails part way has already removed some files.
                unsynced = True
                for rollout_id in to_clear:
                    with self._locked(rollout_id):
                        # The lock was released after phase one. If a delete cleared the fence since, the rollout
                        # ID may already be in use again, and its records belong to the new attempt.
                        if self._read_state(rollout_id).get("retired", False):
                            self._remove_record_files(rollout_id)
                self._fsync_root()
                unsynced = False
        finally:
            if unsynced:
                self._sync_after_error()
        return {"removed": removed, "absent": absent}

    def _sync_after_error(self) -> None:
        # Make what a failed batch already changed durable, since a retry may find nothing left to change.
        # A failing sync is logged, not raised, so that it does not hide the error that failed the batch.
        try:
            self._fsync_root()
        except OSError:
            logger.warning("Could not sync capture directory %s after a failed batch.", self._root, exc_info=True)

    async def delete(self, rollout_ids: Sequence[str]) -> RolloutRemovalPayload:
        """Remove rollouts' records and fences.

        Delete retired rollouts once nothing of those attempts can still write, for example at the end
        of the run, which removes their fences. Delete a rollout ID before reusing it, so the new
        execution starts empty. Deleting again is a no-op.
        """
        return await asyncio.to_thread(self.delete_now, rollout_ids)

    def delete_now(self, rollout_ids: Sequence[str]) -> RolloutRemovalPayload:
        """Synchronous ``delete``.

        Every non-empty batch syncs the directory before returning, even when it found nothing to remove: an
        overlapping delete may have removed the files without syncing yet, and a caller that reuses the rollout
        ID after this returns must not see the old fence or records come back after a crash.
        """
        removed, absent = [], []
        rollout_ids = validate_rollout_ids(rollout_ids)
        if not rollout_ids:
            return {"removed": removed, "absent": absent}
        try:
            for rollout_id in rollout_ids:
                with self._locked(rollout_id):
                    (removed if self._has_records(rollout_id) else absent).append(rollout_id)
                    self._remove_record_files(rollout_id)
                    self.state_path_for(rollout_id).unlink(missing_ok=True)
        except BaseException:
            # A batch that fails part way still syncs what it removed, since a retry would find nothing left.
            self._sync_after_error()
            raise
        self._fsync_root()
        return {"removed": removed, "absent": absent}

    def _has_records(self, rollout_id: str) -> bool:
        return any(
            path.exists()
            for path in (
                self.path_for(rollout_id),
                self.incomplete_path_for(rollout_id),
                self.intents_path_for(rollout_id),
            )
        )

    async def close(self) -> None:
        """The file store owns no persistent handles."""

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
    """Build the training-token file store.

    Return ``None`` when capture is disabled.
    Return ``None`` when no directory resolves.
    Return ``None`` when a custom sink owns the records.
    """
    from nemo_gym.token_id_capture.config import TokenIdCaptureConfig

    config = TokenIdCaptureConfig.model_validate(global_config_dict)
    if not config.enabled or config.token_id_capture.sink is not None:
        return None
    directory = config.resolved_dir()
    return TokenCaptureStore(directory) if directory is not None else None
