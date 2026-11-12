# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Staging cleanup must never chmod or delete a symlink target."""

import os
import stat
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers import utils


pytestmark = pytest.mark.sandbox


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_ancestor: bool
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
        utils.remove_writable_tree(staging)
        assert replaced, "The replacement must happen during cleanup"
        assert not staging.exists()
        assert (target / "keep").read_text() == "host data"
        assert stat.S_IMODE(target.stat().st_mode) == 0o555
    finally:
        target.chmod(0o755)


def test_cleanup_rejects_directory_replaced_by_symlink_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_cleanup_does_not_chmod_foreign_uid_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
