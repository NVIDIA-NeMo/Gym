# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Staging cleanup must never chmod or delete a symlink target."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers import utils


pytestmark = pytest.mark.sandbox


@pytest.fixture
def require_permission_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise repair races even when the ordinary tree would be removable."""
    original = utils.shutil.rmtree
    first = True

    def rmtree(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise PermissionError("permission repair required")
        return original(*args, **kwargs)

    rmtree.avoids_symlink_attacks = original.avoids_symlink_attacks
    monkeypatch.setattr(utils.shutil, "rmtree", rmtree)


def test_cleanup_removes_read_only_tree_and_leaves_symlink_targets(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    locked = staging / "runtime" / "nested"
    locked.mkdir(parents=True)
    (locked / "artifact").write_text("sandbox output")
    (locked / "artifact").chmod(0o444)
    locked.chmod(0o555)
    locked.parent.chmod(0o555)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("host data")
    (staging / "link").symlink_to(outside, target_is_directory=True)
    (staging / "broken-link").symlink_to(tmp_path / "missing")
    outside.chmod(0o555)
    try:
        utils.remove_writable_tree(staging)
        assert not staging.exists()
        assert (outside / "keep").read_text() == "host data"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o555
    finally:
        outside.chmod(0o755)


@pytest.mark.parametrize("replace_ancestor", [False, True])
def test_cleanup_metadata_race_cannot_redirect_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_ancestor: bool, require_permission_repair: None
) -> None:
    staging = tmp_path / "staging"
    parent = staging / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (child / "output").write_text("sandbox output")
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "child" if replace_ancestor else outside
    target.mkdir(exist_ok=True)
    (target / "keep").write_text("host data")
    target.chmod(0o555)
    child_identity = child.stat()
    original_lstat, original_fstat = Path.lstat, os.fstat
    replaced = False

    def replace_after_inspection() -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        victim = parent if replace_ancestor else child
        victim.rename(staging / "parked")
        victim.symlink_to(outside, target_is_directory=True)

    # Inject the same race after metadata inspection in the old path-based
    # implementation or in the descriptor-based implementation.
    def racing_lstat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = original_lstat(path, *args, **kwargs)
        if path == child:
            replace_after_inspection()
        return result

    def racing_fstat(fd: int) -> os.stat_result:
        result = original_fstat(fd)
        if (result.st_dev, result.st_ino) == (child_identity.st_dev, child_identity.st_ino):
            replace_after_inspection()
        return result

    monkeypatch.setattr(Path, "lstat", racing_lstat)
    monkeypatch.setattr(os, "fstat", racing_fstat)
    try:
        if replace_ancestor:
            utils.remove_writable_tree(staging)
            assert not staging.exists()
        else:
            # This move changes the pinned child's parent; fail closed rather
            # than continuing a walk through an unexpected ancestor.
            with pytest.raises(OSError, match="moved during permission repair"):
                utils.remove_writable_tree(staging)
        assert replaced, "The replacement must happen during cleanup"
        assert (target / "keep").read_text() == "host data"
        assert stat.S_IMODE(target.stat().st_mode) == 0o555
    finally:
        target.chmod(0o755)


def test_cleanup_rejects_directory_replaced_by_symlink_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_permission_repair: None
) -> None:
    staging = tmp_path / "staging"
    child = staging / "child"
    child.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("host data")
    outside.chmod(0o555)
    original_open = os.open
    replaced = False

    def racing_open(path: str | bytes | Path, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if path == "child" and not replaced:
            replaced = True
            child.rename(staging / "parked")
            child.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    try:
        utils.remove_writable_tree(staging)
        assert replaced
        assert not staging.exists()
        assert (outside / "keep").read_text() == "host data"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o555
    finally:
        outside.chmod(0o755)


def test_cleanup_rejects_root_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("host data")
    staging = tmp_path / "staging"
    staging.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        utils.remove_writable_tree(staging)
    assert staging.is_symlink()
    assert (outside / "keep").read_text() == "host data"


def test_cleanup_rejects_unsafe_rmtree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(utils.shutil.rmtree, "avoids_symlink_attacks", False)
    with pytest.raises(OSError, match="symlink-safe"):
        utils.remove_writable_tree(staging)
    assert staging.is_dir()


def test_cleanup_root_replacement_after_open_cannot_redirect_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_permission_repair: None
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "output").write_text("sandbox output")
    parked = tmp_path / "parked"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("host data")
    outside.chmod(0o555)
    staging.chmod(0o555)
    original_fchmod = os.fchmod
    replaced = False

    def racing_fchmod(fd: int, mode: int) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            staging.rename(parked)
            staging.symlink_to(outside, target_is_directory=True)
        original_fchmod(fd, mode)

    monkeypatch.setattr(os, "fchmod", racing_fchmod)
    try:
        with pytest.raises(OSError):
            utils.remove_writable_tree(staging)
        assert replaced and staging.is_symlink()
        assert (parked / "output").read_text() == "sandbox output"
        assert (outside / "keep").read_text() == "host data"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o555
    finally:
        outside.chmod(0o755)


def test_cleanup_does_not_chmod_foreign_uid_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_permission_repair: None
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "output").write_text("result")
    original = os.fstat

    def foreign_owner(fd: int) -> os.stat_result:
        fields = list(original(fd))
        fields[4] = os.geteuid() + 1
        # Simulate a foreign-owned directory writable via group/other bits,
        # whose owner bits would trigger chmod if the ownership guard vanished.
        fields[0] = stat.S_IFDIR | 0o477
        return os.stat_result(fields)

    def forbidden_chmod(fd: int, mode: int) -> None:
        raise PermissionError("cannot chmod a foreign UID")

    monkeypatch.setattr(os, "fstat", foreign_owner)
    monkeypatch.setattr(os, "fchmod", forbidden_chmod)
    utils.remove_writable_tree(staging)
    assert not staging.exists()


def test_cleanup_writable_tree_needs_no_permission_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = tmp_path / "staging"
    (staging / "child").mkdir(parents=True)
    (staging / "child/output").write_text("keep no files")

    def unexpected_chmod(*args):
        raise AssertionError("A writable tree should not need a permission repair pass")

    original_scandir = os.scandir
    scanned = []

    def count_scans(fd):
        scanned.append(os.fstat(fd).st_ino)
        return original_scandir(fd)

    monkeypatch.setattr(os, "fchmod", unexpected_chmod)
    monkeypatch.setattr(os, "scandir", count_scans)
    utils.remove_writable_tree(staging)
    assert not staging.exists()
    assert len(scanned) == len(set(scanned)) == 2, "Writable directories should only be scanned once"


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("depth, fd_limit", [(1200, 4096), (160, 256)])
def test_cleanup_deep_tree_with_bounded_descriptors(
    tmp_path: Path, depth: int, fd_limit: int, read_only: bool
) -> None:
    # Limits affect only this child process. Directory-relative creation also
    # avoids the platform's maximum absolute path length.
    script = r"""
import os, resource, shutil, sys
from pathlib import Path
from nemo_gym.sandbox.providers.utils import remove_writable_tree
root = Path(sys.argv[1]) / 'staging'
depth, fd_limit, read_only = map(int, sys.argv[2:])
root.mkdir()
old_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
cleanup_limit = 4096 if old_limit[1] == resource.RLIM_INFINITY else min(4096, old_limit[1])
fd_limit = min(fd_limit, cleanup_limit)
if depth == 1200:
    # Preserve a recursion regression even on CI hosts with a lower hard fd cap.
    sys.setrecursionlimit(100)
    depth = min(depth, fd_limit - 64)
fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
try:
    for _ in range(depth):
        os.mkdir('d', dir_fd=fd)
        child = os.open('d', os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
        if read_only:
            os.fchmod(fd, 0o555)
        os.close(fd)
        fd = child
finally:
    os.close(fd)
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (fd_limit, old_limit[1]))
    remove_writable_tree(root)
    assert not root.exists(), 'staging directory leaked'
finally:
    # Clean up even when exercising a deliberately broken implementation.
    resource.setrlimit(resource.RLIMIT_NOFILE, (cleanup_limit, old_limit[1]))
    if root.exists():
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            while True:
                os.fchmod(fd, 0o755)
                try:
                    child = os.open('d', os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
                except FileNotFoundError:
                    break
                os.close(fd)
                fd = child
        finally:
            os.close(fd)
        shutil.rmtree(root)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(depth), str(fd_limit), str(int(read_only))],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cleanup_rejects_changed_parent_during_ascent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_permission_repair: None
) -> None:
    staging = tmp_path / "staging"
    child = staging / "child"
    child.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("host data")
    identity = child.stat()
    original_scandir = os.scandir
    moved = False

    def move_during_scan(fd):
        nonlocal moved
        info = os.fstat(fd)
        if not moved and (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            moved = True
            child.rename(outside / "moved")
            outside.chmod(0o555)
        return original_scandir(fd)

    monkeypatch.setattr(os, "scandir", move_during_scan)
    try:
        with pytest.raises(OSError, match="moved during permission repair"):
            utils.remove_writable_tree(staging)
        assert moved
        assert (outside / "keep").read_text() == "host data"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o555
    finally:
        outside.chmod(0o755)
