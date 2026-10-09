# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the token-capture conformance checks against Gym's backends.

External frameworks run the same checks against their sink, source, and lineage adapters.
For example, NeMo-RL can use them to verify adapters backed by a transfer queue.
These tests verify Gym's file store and an in-memory backend.
"""

import asyncio

import pytest

from nemo_gym.token_id_capture import (
    FileLineageStore,
    InMemoryLineageStore,
    TokenCaptureFrozenError,
    TokenCaptureRetiredError,
    TokenCaptureSnapshot,
    TokenCaptureStore,
    TokenEntry,
)
from nemo_gym.token_id_capture.conformance import ConformanceError, run_conformance


def test_file_store_passes_all_checks(tmp_path):
    passed = asyncio.run(
        run_conformance(
            lambda: TokenCaptureStore(tmp_path),
            lambda: TokenCaptureStore(tmp_path),
            lambda: FileLineageStore(tmp_path),
        )
    )
    assert "begin_call_custody" in passed
    assert "lineage_visibility" in passed
    assert {"unconditional_retirement", "delete_clears_retirement"} <= set(passed)
    assert len(passed) >= 10


class _MemoryBackend:
    """Store records in memory for protocol conformance tests."""

    def __init__(self):
        self.entries: dict[str, dict[str, TokenEntry]] = {}
        self.incomplete: set[str] = set()
        self.frozen: dict[str, tuple[str, int]] = {}
        self.versions: dict[str, int] = {}
        self.retired: set[str] = set()
        self.lineage = InMemoryLineageStore()

    def remove(self, rollout_ids) -> dict:
        removed, absent = [], []
        for rollout_id in rollout_ids:
            had_records = rollout_id in self.entries or rollout_id in self.incomplete
            for records in (self.entries, self.frozen, self.versions):
                records.pop(rollout_id, None)
            self.incomplete.discard(rollout_id)
            (removed if had_records else absent).append(rollout_id)
        return {"removed": removed, "absent": absent}


class _MemorySink:
    def __init__(self, backend):
        self.backend = backend

    async def put(self, entry: TokenEntry) -> None:
        backend = self.backend
        if entry.rollout_id in backend.retired:
            raise TokenCaptureRetiredError("retired")
        if entry.rollout_id in backend.frozen:
            backend.versions[entry.rollout_id] = backend.versions.get(entry.rollout_id, 0) + 1
            raise RuntimeError("frozen")
        rollout = backend.entries.setdefault(entry.rollout_id, {})
        previous = rollout.get(entry.model_call_id)
        if previous is not None:
            if previous != entry:
                backend.incomplete.add(entry.rollout_id)
                backend.versions[entry.rollout_id] = backend.versions.get(entry.rollout_id, 0) + 1
                raise ValueError("conflicting payload")
            return
        rollout[entry.model_call_id] = entry
        backend.versions[entry.rollout_id] = backend.versions.get(entry.rollout_id, 0) + 1
        await backend.lineage.put(entry)

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        if rollout_id in self.backend.retired:
            raise TokenCaptureRetiredError("retired")
        self.backend.incomplete.add(rollout_id)
        self.backend.versions[rollout_id] = self.backend.versions.get(rollout_id, 0) + 1

    async def close(self) -> None:
        pass


class _MemorySource:
    def __init__(self, backend):
        self.backend = backend

    async def freeze(self, rollout_id: str) -> TokenCaptureSnapshot:
        backend = self.backend
        if rollout_id not in backend.frozen:
            backend.versions[rollout_id] = backend.versions.get(rollout_id, 0) + 1
            backend.frozen[rollout_id] = (f"snap-{rollout_id}", backend.versions[rollout_id])
        snapshot_id, version = backend.frozen[rollout_id]
        return TokenCaptureSnapshot(
            rollout_id=rollout_id,
            entries=tuple(backend.entries.get(rollout_id, {}).values()),
            incomplete=rollout_id in backend.incomplete,
            snapshot_id=snapshot_id,
            version=backend.versions[rollout_id],
        )

    async def drop(self, rollout_id: str, *, snapshot_id: str, version: int) -> bool:
        backend = self.backend
        frozen = backend.frozen.get(rollout_id)
        if frozen is None or frozen[0] != snapshot_id or backend.versions.get(rollout_id) != version:
            return False
        backend.entries.pop(rollout_id, None)
        return True

    async def retire(self, rollout_ids) -> dict:
        self.backend.retired.update(rollout_ids)
        return self.backend.remove(rollout_ids)

    async def delete(self, rollout_ids) -> dict:
        self.backend.retired.difference_update(rollout_ids)
        return self.backend.remove(rollout_ids)

    async def close(self) -> None:
        pass


class _MemoryLineage:
    def __init__(self, backend):
        self.backend = backend

    async def resolve(self, rollout_id: str, request_items: list[dict]):
        return await self.backend.lineage.resolve(rollout_id, request_items)

    def is_process_shared(self) -> bool:
        return False

    async def close(self) -> None:
        pass


def test_memory_backend_passes_applicable_checks():
    backend = _MemoryBackend()
    passed = asyncio.run(
        run_conformance(
            lambda: _MemorySink(backend),
            lambda: _MemorySource(backend),
            lambda: _MemoryLineage(backend),
        )
    )
    # No begin_call on this sink: the custody check is skipped, everything else passes.
    assert "begin_call_custody" not in passed
    assert "lineage_visibility" in passed
    assert len(passed) >= 9


class _FeedLineage:
    """Implement backend hooks while reusing Gym's lineage matching."""

    def __new__(cls, backend):
        from nemo_gym.token_id_capture import IncrementalLineageStore

        class _Impl(IncrementalLineageStore):
            def __init__(self, backend):
                super().__init__()
                self.backend = backend

            def _fetch_new_entries(self, rollout_id, cursor):
                entries = list(self.backend.entries.get(rollout_id, {}).values())
                start = cursor or 0
                items = [(entry, entry.model_call_id) for entry in entries[start:]]
                return items, len(entries)

            def _load_entry(self, rollout_id, ref):
                return self.backend.entries[rollout_id][ref]

        return _Impl(backend)


def test_memory_backend_passes_via_the_incremental_base():
    """Verify that an incremental adapter can reuse Gym's protocol implementation."""
    backend = _MemoryBackend()
    passed = asyncio.run(
        run_conformance(
            lambda: _MemorySink(backend),
            lambda: _MemorySource(backend),
            lambda: _FeedLineage(backend),
        )
    )
    assert "lineage_visibility" in passed
    assert "fresh_client_lineage_visibility" in passed


def test_kit_rejects_a_broken_backend(tmp_path):
    class _Amnesiac(TokenCaptureStore):
        def append(self, entry):  # drops writes: put acks without durability
            return

    from nemo_gym.token_id_capture.conformance import ConformanceError

    with pytest.raises(ConformanceError):
        asyncio.run(
            run_conformance(
                lambda: _Amnesiac(tmp_path),
                lambda: TokenCaptureStore(tmp_path),
            )
        )


class _LateMarkSink(_MemorySink):
    """Lets a late ``mark_incomplete`` through after retire, which resurrects the rollout's capture state."""

    async def mark_incomplete(self, rollout_id: str, model_call_id: str = "") -> None:
        self.backend.incomplete.add(rollout_id)
        self.backend.versions[rollout_id] = self.backend.versions.get(rollout_id, 0) + 1


class _UntypedLateWriteSink(_MemorySink):
    """Rejects a late write with a plain error, which the capture sink would report as a capture failure."""

    async def put(self, entry: TokenEntry) -> None:
        if entry.rollout_id in self.backend.retired:
            raise RuntimeError("retired")
        await super().put(entry)


@pytest.mark.parametrize(
    ("sink_type", "detail"),
    [
        (_LateMarkSink, "a late mark_incomplete after retirement was accepted"),
        (_UntypedLateWriteSink, "a late put after retirement raised RuntimeError, not TokenCaptureRetiredError"),
    ],
)
def test_conformance_rejects_a_backend_that_does_not_fence_late_writes(sink_type, detail):
    backend = _MemoryBackend()
    with pytest.raises(ConformanceError) as raised:
        asyncio.run(run_conformance(lambda: sink_type(backend), lambda: _MemorySource(backend)))
    assert raised.value.check_name == "unconditional_retirement"
    assert raised.value.detail == detail


class _FrozenNotRetiredSink(_MemorySink):
    """Rejects a late write as frozen, not retired, contrary to ``TokenSource.retire``."""

    async def put(self, entry: TokenEntry) -> None:
        if entry.rollout_id in self.backend.retired:
            raise TokenCaptureFrozenError("frozen")
        await super().put(entry)


class _BrokenFreezeAfterRetireSource(_MemorySource):
    """A source whose freeze fails for an unrelated reason once a rollout is retired."""

    async def freeze(self, rollout_id: str) -> TokenCaptureSnapshot:
        if rollout_id in self.backend.retired:
            raise ConnectionError("backend unreachable")
        return await super().freeze(rollout_id)


def test_conformance_requires_the_retired_error_for_late_writes():
    backend = _MemoryBackend()
    with pytest.raises(ConformanceError) as raised:
        asyncio.run(run_conformance(lambda: _FrozenNotRetiredSink(backend), lambda: _MemorySource(backend)))
    assert raised.value.check_name == "unconditional_retirement"
    assert (
        raised.value.detail
        == "a late put after retirement raised TokenCaptureFrozenError, not TokenCaptureRetiredError"
    )


def test_conformance_does_not_treat_a_broken_freeze_as_an_empty_rollout():
    backend = _MemoryBackend()
    with pytest.raises(ConformanceError, match="unconditional_retirement"):
        asyncio.run(run_conformance(lambda: _MemorySink(backend), lambda: _BrokenFreezeAfterRetireSource(backend)))
