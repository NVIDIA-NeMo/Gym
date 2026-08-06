# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared primitives for safe repository artifact writes."""

from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _checked_parent(path: Path, *, create: bool) -> Path:
    parent = path.parent
    if parent.is_symlink():
        raise OSError(f"refusing to write through symbolic-link directory '{parent}'")
    if not parent.exists():
        if not create:
            raise FileNotFoundError(parent)
        parent.mkdir(parents=True)
    if parent.is_symlink() or not parent.is_dir():
        raise OSError(f"write destination parent '{parent}' is not a regular directory")
    return parent


def find_repository_root(path: str | Path) -> Path | None:
    """Return the nearest Git worktree root above a file or directory."""

    candidate = Path(os.path.abspath(path))
    if not candidate.is_dir():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    return None


@contextmanager
def exclusive_directory_lock(path: str | Path, *, create: bool = False) -> Iterator[None]:
    """Serialize cooperating repository writers without creating a lock artifact."""

    directory = Path(path)
    if directory.is_symlink():
        raise OSError(f"refusing to lock symbolic-link directory '{directory}'")
    if not directory.exists():
        if not create:
            raise FileNotFoundError(directory)
        directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise OSError(f"lock target '{directory}' is not a regular directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError:
        os.close(descriptor)
        raise
    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def atomic_write_text(
    path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
    create_parent: bool = False,
    mode: int | None = None,
) -> None:
    """Replace one regular text file atomically without following a destination symlink."""

    destination = Path(path)
    parent = _checked_parent(destination, create=create_parent)
    if destination.is_symlink():
        raise OSError(f"refusing to replace symbolic-link destination '{destination}'")
    target_mode = (
        mode if mode is not None else (destination.lstat().st_mode & 0o777 if destination.exists() else 0o644)
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as stream:
            stream.write(content)
        os.chmod(temporary, target_mode)
        if parent.is_symlink() or destination.is_symlink():
            raise OSError(f"refusing to replace symbolic-link destination '{destination}'")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(
    path: str | Path,
    content: bytes,
    *,
    create_parent: bool = False,
    mode: int | None = None,
) -> None:
    """Replace one regular binary file atomically without following symlinks."""

    destination = Path(path)
    parent = _checked_parent(destination, create=create_parent)
    if destination.is_symlink():
        raise OSError(f"refusing to replace symbolic-link destination '{destination}'")
    target_mode = (
        mode if mode is not None else (destination.lstat().st_mode & 0o777 if destination.exists() else 0o644)
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.chmod(temporary, target_mode)
        if parent.is_symlink() or destination.is_symlink():
            raise OSError(f"refusing to replace symbolic-link destination '{destination}'")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_text(
    path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
    create_parent: bool = False,
    mode: int = 0o644,
) -> None:
    """Create one text file atomically and fail if the destination already exists."""

    destination = Path(path)
    parent = _checked_parent(destination, create=create_parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as stream:
            stream.write(content)
        os.chmod(temporary, mode)
        if parent.is_symlink() or destination.is_symlink():
            raise OSError(f"refusing to create symbolic-link destination '{destination}'")
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "atomic_create_text",
    "atomic_write_bytes",
    "atomic_write_text",
    "exclusive_directory_lock",
    "find_repository_root",
]
