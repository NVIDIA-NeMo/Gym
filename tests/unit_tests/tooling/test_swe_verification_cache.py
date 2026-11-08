# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check hash-proof reuse and invalidation with actual filesystem changes."""

import hashlib
import os
from pathlib import Path

import pytest
from scripts.swe_verification_cache import VerificationCache


def test_reuses_unchanged_file_without_reading_content(tmp_path, monkeypatch):
    path = tmp_path / "model.bin"
    path.write_bytes(b"model bytes")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = VerificationCache(tmp_path / "cache")
    cache.verify(path, algorithm="sha256", expected=expected)
    original = Path.open

    def open_file(self, *args, **kwargs):
        if self == path:
            raise AssertionError("unchanged model was read again")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    cache.verify(path, algorithm="sha256", expected=expected)
    assert [x["evidence"] for x in cache.events] == ["content_hashed", "reused_unchanged_file_identity"]


@pytest.mark.parametrize("change", ["bytes", "replace", "restored_mtime", "expected", "receipt"])
def test_changed_identity_or_expectation_invalidates_receipt(tmp_path, change):
    path = tmp_path / "image.sqsh"
    path.write_bytes(b"image A")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = VerificationCache(tmp_path / "cache")
    cache.verify(path, algorithm="sha256", expected=expected)
    before = path.stat()
    if change == "replace":
        other = tmp_path / "other"
        other.write_bytes(b"image B")
        other.replace(path)
    elif change in {"bytes", "restored_mtime"}:
        path.write_bytes(b"image B")
        if change == "restored_mtime":
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "expected":
        expected = "0" * 64
    else:
        next(cache.directory.glob("*.json")).write_text("corrupt receipt")
        cache.verify(path, algorithm="sha256", expected=expected)
        assert cache.events[-1]["evidence"] == "content_hashed"
        return
    with pytest.raises(ValueError, match="Content differs"):
        cache.verify(path, algorithm="sha256", expected=expected)
    assert len(cache.events) == 1


def test_symlink_target_change_invalidates_receipt(tmp_path):
    target = tmp_path / "blob"
    target.write_bytes(b"expected")
    path = tmp_path / "snapshot"
    path.symlink_to(target)
    cache = VerificationCache(tmp_path / "cache")
    digest = hashlib.sha1(b"blob 8\0expected").hexdigest()
    cache.verify(path, algorithm="git_blob_sha1", expected=digest)
    other = tmp_path / "other"
    other.write_bytes(b"different")
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(ValueError, match="Content differs"):
        cache.verify(path, algorithm="git_blob_sha1", expected=digest)


def test_concurrent_edit_during_verification_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "model"
    path.write_bytes(b"same content")
    cache = VerificationCache(tmp_path / "cache")
    identity = cache.identity
    calls = 0

    def check_identity(file):
        nonlocal calls
        calls += 1
        if calls == 2:
            file.write_bytes(b"replacement")
        return identity(file)

    monkeypatch.setattr(cache, "identity", check_identity)
    with pytest.raises(ValueError, match="changed during verification"):
        cache.verify(path, algorithm="sha256", expected=hashlib.sha256(b"same content").hexdigest())
    assert not cache.directory.exists()
