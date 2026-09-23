# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``FileManifestReader``: cache-free, shared-locked snapshots of a rollout ledger.

The reader must return exactly what the HTTP manifest route
(``FileLineageStore.manifest``) returns, hold no per-rollout state, respect
the writer's per-rollout lock, and bound every wait by the caller's deadline.
"""

from __future__ import annotations

import asyncio
import fcntl
import threading
import time

import pytest

from nemo_gym.token_id_capture.lineage import (
    WRITER_IDENTITY_DIRNAME,
    FileLineageStore,
    FileManifestReader,
    LedgerRootMismatch,
    ManifestReadCancelled,
    ManifestReadStats,
    ManifestReadTimeout,
    read_writer_identity_markers,
    verify_ledger_root_visibility,
)
from nemo_gym.token_id_capture.staging.digest import EMPTY_EXTRAS_DIGEST, compute_chain_hash, hash_token_ids
from nemo_gym.token_id_capture.staging.records import CallRecord, CaptureLedgerCommit, RolloutManifest


def _commit(rollout_id: str, index: int, parent: str | None = None, prev_len: int = 0) -> CaptureLedgerCommit:
    tokens = list(range(prev_len, prev_len + 16))
    record = CallRecord(
        model_call_id=f"c{index}",
        parent_call_id=parent,
        staging_key=f"{rollout_id}/c{index}",
        weight_version=1,
        prev_len=prev_len,
        delta_len=16,
        cum_len=prev_len + 16,
        digest="a" * 64,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        mode="text" if parent is None else "token_in",
        chain_hash=compute_chain_hash(None, tokens),
        cumulative_hash=hash_token_ids(tokens),
        response_id=f"resp-{index}",
        admitted_at=1.0,
        fingerprint_version=1,
    )
    return CaptureLedgerCommit(
        rollout_id=rollout_id,
        record=record,
        staging_chain=(f"{rollout_id}/c{index}",),
        request_items=[{"role": "user", "content": f"q{index}"}],
        response_items=[{"role": "assistant", "content": f"a{index}"}],
    )


@pytest.fixture
def root(tmp_path):
    return tmp_path / "lineage"


@pytest.fixture
def store(root):
    return FileLineageStore(root)


@pytest.mark.asyncio
async def test_reader_matches_http_route_for_multi_turn_and_failure_rows(store, root):
    await store.record(_commit("r1", 1))
    await store.record(_commit("r1", 2, parent="c1", prev_len=16))
    await store.record(_commit("r1", 2, parent="c1", prev_len=16))  # duplicate commit: idempotent
    await store.record_failure("r1", "c3", "worker_capture_failed")
    reader = FileManifestReader(root)
    stats = ManifestReadStats()
    direct = reader.read_manifest("r1", deadline=time.monotonic() + 5, stats=stats)
    assert direct == await store.manifest("r1")
    manifest = RolloutManifest.model_validate(direct)
    assert [record.model_call_id for record in manifest.records] == ["c1", "c2"]
    assert [failure.model_call_id for failure in manifest.failures] == ["c3"]
    assert stats.rows == 3 and stats.bytes_read > 0


@pytest.mark.asyncio
async def test_reader_missing_ledger_is_the_same_empty_manifest(store, root):
    reader = FileManifestReader(root)
    assert reader.read_manifest("absent") == await store.manifest("absent")
    # The reader never mints a lock file for a rollout that was never written.
    assert not (root / "absent2.tokens.lock").exists()
    reader.read_manifest("absent2")
    assert not (root / "absent2.tokens.lock").exists()


def test_reader_rejects_invalid_rollout_ids_and_malformed_rows(store, root):
    reader = FileManifestReader(root)
    with pytest.raises(ValueError):
        reader.read_manifest("../escape")
    (root / "bad.lineage.jsonl").write_bytes(b'{"model_call_id": "c1"}\n[1, 2]\n')
    with pytest.raises(ValueError, match="not an object"):
        reader.read_manifest("bad")
    (root / "bad2.lineage.jsonl").write_bytes(b"{nope\n")
    with pytest.raises(ValueError, match="malformed JSON"):
        reader.read_manifest("bad2")


def test_reader_requires_an_existing_root(tmp_path):
    with pytest.raises(LedgerRootMismatch):
        FileManifestReader(tmp_path / "nope")


@pytest.mark.asyncio
async def test_reader_times_out_and_cancels_while_writer_lock_is_held(store, root):
    await store.record(_commit("r1", 1))
    reader = FileManifestReader(root)
    with open(root / "r1.tokens.lock", "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        started = time.monotonic()
        with pytest.raises(ManifestReadTimeout, match="lock acquisition"):
            reader.read_manifest("r1", deadline=time.monotonic() + 0.2)
        assert time.monotonic() - started < 1.0
        cancel = threading.Event()
        threading.Timer(0.05, cancel.set).start()
        with pytest.raises(ManifestReadCancelled):
            reader.read_manifest("r1", cancel_event=cancel)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    # Lock released: the same read now succeeds and the reader kept no state.
    assert reader.read_manifest("r1")["records"][0]["model_call_id"] == "c1"


@pytest.mark.asyncio
async def test_reader_shared_lock_does_not_block_other_readers(store, root):
    await store.record(_commit("r1", 1))
    reader = FileManifestReader(root)
    with open(root / "r1.tokens.lock", "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        assert reader.read_manifest("r1", deadline=time.monotonic() + 1)["rollout_id"] == "r1"
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@pytest.mark.asyncio
async def test_reader_snapshots_stay_consistent_under_concurrent_appends(store, root):
    reader = FileManifestReader(root)

    async def writer():
        for index in range(1, 120):
            await store.record(_commit("r2", index))

    async def readers():
        seen = set()
        for _ in range(200):
            manifest = await asyncio.to_thread(reader.read_manifest, "r2")
            # Every snapshot parses as a whole manifest with monotone ids.
            ids = [int(record["model_call_id"][1:]) for record in manifest["records"]]
            assert ids == list(range(1, len(ids) + 1))
            seen.add(len(ids))
            await asyncio.sleep(0)
        return seen

    _, seen = await asyncio.gather(writer(), readers())
    assert len(seen) > 1


def test_writer_identity_marker_and_visibility_check(store, root):
    markers = read_writer_identity_markers(root)
    assert len(markers) == 1
    assert markers[0]["hostname"]
    assert (root / WRITER_IDENTITY_DIRNAME).is_dir()
    diagnostics = verify_ledger_root_visibility(root)
    assert diagnostics["writers"] == 1
    assert diagnostics["matching_writers"] == 1
    assert diagnostics["reader"]["inode"] == markers[0]["inode"]


def test_visibility_check_fails_without_root_or_marker_or_on_mismatch(tmp_path, root, store):
    with pytest.raises(LedgerRootMismatch, match="does not exist"):
        verify_ledger_root_visibility(tmp_path / "missing", wait_s=0.0)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(LedgerRootMismatch, match="no writer identity marker"):
        verify_ledger_root_visibility(empty, wait_s=0.0)
    assert verify_ledger_root_visibility(empty, require_writer_marker=False)["writers"] == 0
    # A marker written from a different directory (different inode) is a mismatch.
    other = tmp_path / "other"
    FileLineageStore(other)
    marker_dir = root / WRITER_IDENTITY_DIRNAME
    for path in marker_dir.glob("*.json"):
        path.unlink()
    for path in (other / WRITER_IDENTITY_DIRNAME).glob("*.json"):
        (marker_dir / path.name).write_bytes(path.read_bytes())
    with pytest.raises(LedgerRootMismatch, match="different directories"):
        verify_ledger_root_visibility(root, wait_s=0.0)


def test_manifest_from_rows_stays_the_single_schema_owner(store, root):
    """The reader reuses ``_manifest_from_rows``; a poison row shape is handled identically."""
    (root / "r9.lineage.jsonl").write_bytes(
        b'{"model_call_id": "c1", "staging_key": "r9/c1", "prev_len": 0, "delta_len": 1, '
        b'"cum_len": 1, "weight_version": 1, "staging_digest": "a", "extras_digest": "b", '
        b'"mode": "text", "chain_hash": "", "cumulative_hash": ""}\n'
    )
    reader = FileManifestReader(root)
    direct = reader.read_manifest("r9")
    assert direct["records"] == []
    assert direct["failures"][0]["model_call_id"] == "c1"
    assert direct == asyncio.run(store.manifest("r9"))


@pytest.mark.parametrize("fields", [{}, {"prev_len": None}])
def test_reader_normalizes_invalid_field_errors(store, root, fields):
    import orjson

    row = {
        "model_call_id": "c1",
        "staging_key": "key",
        "response_id": "resp",
        "chain_hash": "chain",
        "cumulative_hash": "cumulative",
        **fields,
    }
    (root / "bad.lineage.jsonl").write_bytes(orjson.dumps(row) + b"\n")
    reader = FileManifestReader(root)
    with pytest.raises(ValueError, match="invalid fields"):
        reader.read_manifest("bad")
    assert reader.read_manifest("absent")["records"] == []


def test_reader_rechecks_deadline_after_manifest_conversion(store, root, monkeypatch):
    import nemo_gym.token_id_capture.lineage as module

    real_convert = module._manifest_from_rows

    def convert(*args):
        time.sleep(0.03)
        return real_convert(*args)

    monkeypatch.setattr(module, "_manifest_from_rows", convert)
    with pytest.raises(ManifestReadTimeout, match="manifest conversion"):
        FileManifestReader(root).read_manifest("absent", deadline=time.monotonic() + 0.01)


def test_reader_sees_commits_from_a_separate_writer_process(root):
    import json
    import os
    import subprocess
    import sys

    writer = """
import asyncio
import json
import sys
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.staging.records import CaptureLedgerCommit
async def main():
    store = FileLineageStore(sys.argv[1])
    commit = CaptureLedgerCommit.model_validate_json(sys.stdin.read())
    await store.record(commit)
    print(json.dumps(await store.manifest(commit.rollout_id)))
asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-c", writer, str(root)],
        input=_commit("separate-process", 1).model_dump_json(),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert all(marker["pid"] != os.getpid() for marker in read_writer_identity_markers(root))
    assert verify_ledger_root_visibility(root)["matching_writers"] == 1
    assert FileManifestReader(root).read_manifest("separate-process") == json.loads(completed.stdout)
