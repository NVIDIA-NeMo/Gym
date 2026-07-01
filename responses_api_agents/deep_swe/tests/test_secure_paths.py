# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.deep_swe import secure_paths


def test_private_root_and_children_are_owner_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "private"
    monkeypatch.setattr(secure_paths, "DEFAULT_PRIVATE_ROOT", root)
    child = secure_paths.ensure_private_directory(root / "runtime")
    assert child == root / "runtime"
    assert (root.stat().st_mode & 0o777) == 0o700
    assert (child.stat().st_mode & 0o777) == 0o700


def test_private_directory_rejects_symlink_and_loose_mode(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unavailable"):
        secure_paths._validate_private_directory(tmp_path / "missing")

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="not a real directory"):
        secure_paths.ensure_private_directory(link)

    loose = tmp_path / "loose"
    loose.mkdir(mode=0o700)
    os.chmod(loose, 0o755)
    with pytest.raises(RuntimeError, match="mode 0700"):
        secure_paths.ensure_private_directory(loose)

    owned = tmp_path / "owned"
    owned.mkdir(mode=0o700)
    actual_uid = os.geteuid()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(secure_paths.os, "geteuid", lambda: actual_uid + 1)
        with pytest.raises(RuntimeError, match="not owned"):
            secure_paths._validate_private_directory(owned)

    writable_parent = tmp_path / "writable-parent"
    writable_parent.mkdir(mode=0o700)
    os.chmod(writable_parent, 0o777)
    with pytest.raises(RuntimeError, match="attacker-writable ancestor"):
        secure_paths.ensure_private_directory(writable_parent / "child")

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="missing or symlinked ancestor"):
        secure_paths.ensure_private_directory(linked_parent / "child")


def test_ancestor_ownership_and_sticky_writable_policy() -> None:
    trusted_sticky = SimpleNamespace(st_mode=stat.S_IFDIR | stat.S_ISVTX | 0o777, st_uid=0)
    secure_paths._validate_trusted_ancestor(trusted_sticky, Path("/tmp"))

    untrusted_owner = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=os.geteuid() + 1)
    with pytest.raises(RuntimeError, match="untrusted ownership"):
        secure_paths._validate_trusted_ancestor(untrusted_owner, Path("/untrusted"))

    untrusted_sticky = SimpleNamespace(
        st_mode=stat.S_IFDIR | stat.S_ISVTX | 0o777,
        st_uid=os.geteuid() + 1,
    )
    with pytest.raises(RuntimeError, match="untrusted ownership"):
        secure_paths._validate_trusted_ancestor(untrusted_sticky, Path("/untrusted-sticky"))


def test_system_temporary_directory_ancestor_chain_is_trusted() -> None:
    # On macOS this traverses the root-owned /private/var hierarchy used by
    # tempfile.gettempdir(); on Linux it covers the root-owned sticky /tmp.
    secure_paths._validate_ancestor_chain(Path(tempfile.gettempdir()).resolve(strict=True))


def test_creation_rejects_component_replaced_before_no_follow_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    target = parent / "private"
    attacker = tmp_path / "attacker"
    attacker.mkdir(mode=0o700)
    real_mkdir = secure_paths.os.mkdir

    def replacing_mkdir(path: str, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        real_mkdir(path, mode, dir_fd=dir_fd)
        if path == target.name and dir_fd is not None:
            target.rmdir()
            target.symlink_to(attacker, target_is_directory=True)

    monkeypatch.setattr(secure_paths.os, "mkdir", replacing_mkdir)
    with pytest.raises(RuntimeError, match="not a real directory"):
        secure_paths.ensure_private_directory(target)
