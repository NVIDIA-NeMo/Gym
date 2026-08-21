# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the conformance kit against Gym's own backends.

The kit is what an external framework (e.g. NeMo-RL over TransferQueue) runs
against its sink/source/lineage adapters; Gym's file store and an in-memory
backend must both pass every applicable check.
"""

import asyncio

import pytest

from nemo_gym.token_id_capture import (
    FileLineageStore,
    InMemoryLineageStore,
    TokenCaptureSnapshot,
    TokenCaptureStore,
    TokenEntry,
)
from nemo_gym.token_id_capture.conformance import run_conformance


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
    assert len(passed) >= 10


class _MemoryBackend:
    """The minimal external-transport shape from the integration tests."""

    def __init__(self):
        self.entries: dict[str, dict[str, TokenEntry]] = {}
        self.incomplete: set[str] = set()
        self.frozen: dict[str, tuple[str, int]] = {}
        self.versions: dict[str, int] = {}
        self.lineage = InMemoryLineageStore()


class _MemorySink:
    def __init__(self, backend):
        self.backend = backend

    async def put(self, entry: TokenEntry) -> None:
        backend = self.backend
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
    """The intended external-adapter shape: subclass the base, implement two hooks."""

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
    """An adapter built on IncrementalLineageStore passes the kit with ~15 lines of
    backend-specific code — the pattern a TransferQueue adapter follows."""
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
