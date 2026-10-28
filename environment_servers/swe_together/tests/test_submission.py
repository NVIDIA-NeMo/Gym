# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from environment_servers.swe_together.submission_worker import freeze, restore, task_roots


def test_full_task_state_roundtrip(tmp_path):
    root = tmp_path / "task"
    nested = root / "nested-repo"
    nested.mkdir(parents=True)
    (root / ".gitignore").write_text("ignored.bin\n")
    (root / "ignored.bin").write_bytes(b"\x00\xffbinary")
    (nested / "untracked").write_text("nested change")
    (root / "executable").write_text("#!/bin/sh\n")
    (root / "executable").chmod(0o751)
    (root / "link").symlink_to("ignored.bin")
    chunks = tmp_path / "submission"
    manifest = freeze([str(root), str(nested)], chunks)
    assert manifest["roots"] == [str(root)]
    shutil.rmtree(root)
    root.mkdir()
    (root / "deleted").write_text("fresh image file deleted by candidate")
    inode = root.stat().st_ino
    result = restore(chunks)
    assert root.stat().st_ino == inode
    assert result["archive_sha256"] == manifest["archive_sha256"]
    assert (root / "ignored.bin").read_bytes() == b"\x00\xffbinary"
    assert (nested / "untracked").read_text() == "nested change"
    assert (root / "executable").stat().st_mode & 0o777 == 0o751
    assert os.readlink(root / "link") == "ignored.bin"
    assert not (root / "deleted").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="Requires an unprivileged image user")
def test_restore_as_image_user_without_write_access_to_root_parent(tmp_path):
    parent = tmp_path / "image"
    root = parent / "workspace"
    root.mkdir(parents=True)
    (root / "nested").mkdir()
    (root / "nested" / "ignored.bin").write_bytes(b"\x00\xff")
    (root / "executable").write_text("#!/bin/sh\n")
    (root / "executable").chmod(0o751)
    (root / "link").symlink_to("nested/ignored.bin")
    root.chmod(0o750)
    chunks = tmp_path / "chunks"
    freeze([str(root)], chunks)
    inode = root.stat().st_ino
    parent.chmod(0o555)
    try:
        # Reproduce the old failure: contents are writable, the root's parent is not.
        with pytest.raises(PermissionError):
            shutil.rmtree(root)
        (root / "deleted").write_text("fresh image file")
        root.chmod(0o700)
        worker = Path(__file__).resolve().parents[1] / "submission_worker.py"
        process = subprocess.run([sys.executable, str(worker), "restore", str(chunks)], capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        assert root.stat().st_ino == inode
        assert root.stat().st_uid == os.geteuid()
        assert root.stat().st_mode & 0o777 == 0o750
        assert (root / "nested" / "ignored.bin").read_bytes() == b"\x00\xff"
        assert (root / "nested" / "ignored.bin").stat().st_uid == os.geteuid()
        assert (root / "executable").stat().st_mode & 0o777 == 0o751
        assert os.readlink(root / "link") == "nested/ignored.bin"
        assert not (root / "deleted").exists()
    finally:
        parent.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="Requires an unprivileged image user")
def test_unreadable_submission_reports_path_and_image_identity(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    unreadable = root / "unreadable"
    unreadable.write_text("cannot freeze")
    unreadable.chmod(0)
    worker = Path(__file__).resolve().parents[1] / "submission_worker.py"
    try:
        process = subprocess.run(
            [sys.executable, str(worker), "freeze", str(tmp_path / "chunks"), f'["{root}"]'],
            capture_output=True,
            text=True,
        )
        assert process.returncode != 0
        assert f"Submission freeze denied for uid={os.geteuid()} gid={os.getegid()}" in process.stderr
        assert str(unreadable) in process.stderr
    finally:
        unreadable.chmod(0o644)


@pytest.mark.skipif(os.geteuid() == 0, reason="Requires an unprivileged image user")
def test_restore_readonly_directories_after_their_contents(tmp_path):
    root = tmp_path / "workspace"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "file").write_bytes(b"\x00\xffcandidate")
    (nested / "file").chmod(0o444)
    nested.chmod(0o555)
    root.chmod(0o555)
    chunks = tmp_path / "chunks"
    freeze([str(root)], chunks)
    try:
        root.chmod(0o755)
        nested.chmod(0o755)
        shutil.rmtree(nested)
        # The former per-member extraction makes the root read-only too soon.
        with tarfile.open(chunks / "state.tar.gz", "r:gz") as archive:
            with pytest.raises(PermissionError):
                for member in archive:
                    archive.extract(member, path="/", filter="fully_trusted")
        root.chmod(0o755)
        nested.mkdir()
        (nested / "deleted").write_text("baseline removed by candidate")
        nested.chmod(0o555)
        root.chmod(0o555)
        worker = Path(__file__).resolve().parents[1] / "submission_worker.py"
        process = subprocess.run([sys.executable, str(worker), "restore", str(chunks)], capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        assert (nested / "file").read_bytes() == b"\x00\xffcandidate"
        assert (nested / "file").stat().st_mode & 0o777 == 0o444
        assert root.stat().st_mode & 0o777 == 0o555
        assert nested.stat().st_mode & 0o777 == 0o555
        assert not (nested / "deleted").exists()
    finally:
        root.chmod(0o755)
        if nested.exists():
            nested.chmod(0o755)


def test_deleted_task_root_is_not_restored_as_empty_directory(tmp_path):
    root = tmp_path / "workspace"
    chunks = tmp_path / "chunks"
    freeze([str(root)], chunks)
    root.mkdir()
    (root / "deleted").write_text("fresh image file")
    restore(chunks)
    assert not root.exists()


@pytest.mark.parametrize(
    "path", ["/", "/tmp", "/opt", "/workspace/../root", "/opt/gym-runtime", "/tmp/nemo-gym-opencode-sessions/x"]
)
def test_snapshot_cannot_include_system_or_harness_roots(path):
    with pytest.raises(ValueError):
        task_roots([path])


def test_bad_chunk_cannot_replace_fresh_image(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    (root / "data").write_text("frozen")
    chunks = tmp_path / "chunks"
    freeze([str(root)], chunks)
    (root / "data").write_text("fresh")
    (chunks / "state.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore(chunks)
    assert (root / "data").read_text() == "fresh"
