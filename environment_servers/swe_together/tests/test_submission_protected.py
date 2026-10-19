# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment_servers.swe_together import submission_worker as worker
from environment_servers.swe_together.submission import record_command


@pytest.fixture
def protected_submission(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    protected = root / "nested" / "protected"
    protected.mkdir(parents=True)
    (protected / "data").write_bytes(b"frozen\x00\xff")
    (protected / "link").symlink_to("data")
    owned = root / "owned"
    owned.write_text("candidate")
    chunks = tmp_path / "chunks"
    worker.freeze([str(root)], chunks)
    before = {path: path.lstat() for path in (protected, *protected.iterdir())}
    owned.write_text("fresh image")
    owned.chmod(0o640)
    (root / "deleted").write_text("baseline only")
    # These tests isolate equivalence checks; a real nonroot Linux fixture covers
    # discovery and preservation of a root-owned directory in the live image.
    monkeypatch.setattr(worker, "protected_directories", lambda path: [protected])
    return root, protected, owned, chunks, before


def test_verified_subtree_retains_metadata_and_owned_changes_are_restored(protected_submission):
    root, protected, owned, chunks, before = protected_submission
    result = worker.restore(chunks)
    assert result["preserved_baseline"] == [{"root": str(protected), "members": 3}]
    for path, attributes in before.items():
        current = path.lstat()
        for field in ("st_ino", "st_uid", "st_gid", "st_mode", "st_mtime_ns"):
            assert getattr(current, field) == getattr(attributes, field)
    assert (protected / "data").read_bytes() == b"frozen\x00\xff"
    assert os.readlink(protected / "link") == "data"
    assert owned.read_text() == "candidate"
    assert not (root / "deleted").exists()


def change_archive_owner(chunks: Path, path: Path, field: str) -> None:
    manifest = json.loads((chunks / "manifest.json").read_text())
    compressed = (chunks / "state.tar.gz").read_bytes()
    output = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:gz") as source:
        with tarfile.open(fileobj=output, mode="w:gz") as target:
            for item in source.getmembers():
                if item.name == str(path).lstrip("/"):
                    setattr(item, field, getattr(item, field) + 1)
                target.addfile(item, source.extractfile(item) if item.isfile() else None)
    raw = output.getvalue()
    checksum = hashlib.sha256(raw).hexdigest()
    (chunks / "state.tar.gz").write_bytes(raw)
    manifest.update(
        archive_sha256=checksum,
        bytes=len(raw),
    )
    (chunks / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("difference", ["bytes", "missing", "extra", "type", "mode", "uid", "gid", "mtime", "link"])
def test_protected_mismatch_fails_before_owned_tree_changes(protected_submission, difference):
    root, protected, owned, chunks, before = protected_submission
    data = protected / "data"
    link = protected / "link"
    if difference == "bytes":
        data.write_bytes(b"changed bytes")
    elif difference == "missing":
        data.unlink()
    elif difference == "extra":
        (protected / "extra").write_text("extra")
    elif difference == "type":
        data.unlink()
        data.mkdir()
    elif difference == "mode":
        data.chmod(0o400)
    elif difference in {"uid", "gid"}:
        change_archive_owner(chunks, data, difference)
    elif difference == "mtime":
        os.utime(data, (before[data].st_atime, before[data].st_mtime + 10))
    elif difference == "link":
        link.unlink()
        link.symlink_to("different-target")
    for path, attributes in before.items():
        if path.exists() or path.is_symlink():
            if not (difference == "mtime" and path == data):
                os.utime(path, ns=(attributes.st_atime_ns, attributes.st_mtime_ns), follow_symlinks=False)
    owned_before = owned.stat()
    with pytest.raises(ValueError, match="Protected baseline differs") as failure:
        worker.restore(chunks)
    expected_reason = {
        "bytes": "file bytes",
        "missing": "missing or extra member",
        "extra": "missing or extra member",
        "link": "symlink target",
    }.get(difference, difference)
    assert expected_reason in str(failure.value)
    assert owned.read_text() == "fresh image"
    assert owned.stat().st_mode == owned_before.st_mode
    assert owned.stat().st_ino == owned_before.st_ino
    assert (root / "deleted").read_text() == "baseline only"


@pytest.mark.skipif(os.geteuid() == 0, reason="Requires real unprivileged access checks")
def test_protection_uses_effective_access_and_keeps_owned_readonly_behavior(tmp_path, monkeypatch):
    root = tmp_path / "directory"
    root.mkdir()
    root.chmod(0o555)
    try:
        # Owned read-only directories can already be made writable by the worker.
        assert worker.protected_directories(root) == []
        current_uid = os.geteuid()
        monkeypatch.setattr(worker.os, "geteuid", lambda: current_uid + 1)
        assert worker.protected_directories(root) == [root]
        root.chmod(0o755)
        # Ownership alone must not mark a directory protected when it is writable.
        assert worker.protected_directories(root) == []
    finally:
        root.chmod(0o755)


@pytest.mark.parametrize("location", ["a-outside", "z-outside", "nested/protected/internal"])
def test_hardlinks_involving_preserved_subtree_fail_before_mutation(protected_submission, location):
    root, protected, owned, chunks, _ = protected_submission
    os.link(protected / "data", root / location)
    worker.freeze([str(root)], chunks)
    with pytest.raises(ValueError, match="Protected baseline hardlink"):
        worker.restore(chunks)
    assert owned.read_text() == "fresh image"
    assert (root / "deleted").exists()


def test_baseline_hardlink_cannot_replace_distinct_archived_files(protected_submission):
    root, protected, owned, chunks, _ = protected_submission
    data = protected / "data"
    other = protected / "other"
    other.write_bytes(data.read_bytes())
    attributes = data.stat()
    other.chmod(attributes.st_mode & 0o7777)
    os.utime(other, ns=(attributes.st_atime_ns, attributes.st_mtime_ns))
    worker.freeze([str(root)], chunks)
    directory_mtime = protected.stat().st_mtime_ns
    other.unlink()
    os.link(data, other)
    os.utime(protected, ns=(protected.stat().st_atime_ns, directory_mtime))
    owned_before = owned.stat()
    with pytest.raises(ValueError, match="Protected baseline hardlink"):
        worker.restore(chunks)
    assert owned.read_text() == "fresh image"
    assert owned.stat().st_mode == owned_before.st_mode
    assert owned.stat().st_ino == owned_before.st_ino
    assert (root / "deleted").exists()


def test_command_failure_retains_both_provider_streams(tmp_path):
    result = SimpleNamespace(
        return_code=1,
        error_type=None,
        stdout="Traceback: PermissionError /workspace/protected/HEAD",
        stderr="exit status 1",
    )
    with pytest.raises(RuntimeError) as failure:
        record_command(result, action="restore", artifact_dir=tmp_path)
    assert result.stdout in str(failure.value)
    assert result.stderr in str(failure.value)
    assert json.loads((tmp_path / "restore-command.json").read_text()) == vars(result)
