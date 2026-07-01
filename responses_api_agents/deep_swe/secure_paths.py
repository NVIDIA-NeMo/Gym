# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private filesystem roots for executable caches and retained trajectories."""

import os
import stat
import tempfile
from pathlib import Path


# macOS exposes its system temporary directory through ``/var``, which is a
# root-owned compatibility symlink to ``/private/var``. Resolve that trusted,
# OS-selected base once so the strict no-follow checks below still reject
# symlinks in every user-configurable path component.
DEFAULT_PRIVATE_ROOT = Path(tempfile.gettempdir()).resolve(strict=True) / f"nemo-gym-deep-swe-{os.geteuid()}"


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def _validate_private_metadata(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"DeepSWE private path is not a real directory: {path}")
    if metadata.st_uid != os.geteuid():
        raise RuntimeError(f"DeepSWE private directory is not owned by the current user: {path}")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RuntimeError(f"DeepSWE private directory must have mode 0700: {path}")


def _validate_trusted_ancestor(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"DeepSWE path has a missing or symlinked ancestor: {path}")
    if metadata.st_uid not in {0, os.geteuid()}:
        raise RuntimeError(f"DeepSWE path has an ancestor with untrusted ownership: {path}")
    writable_by_others = stat.S_IMODE(metadata.st_mode) & 0o022
    if writable_by_others and not metadata.st_mode & stat.S_ISVTX:
        raise RuntimeError(f"DeepSWE path has an attacker-writable ancestor: {path}")


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_directory_chain(path: Path, *, private_from: Path | None, create_private: bool) -> None:
    """Walk and optionally create a directory chain using retained parent fds."""
    path = _absolute_without_symlink_resolution(path)
    if private_from is not None:
        private_from = _absolute_without_symlink_resolution(private_from)
        if path != private_from and not path.is_relative_to(private_from):
            raise ValueError(f"Private directory {path} is outside its creation root {private_from}")

    descriptor = os.open("/", _directory_open_flags())
    current = Path("/")
    try:
        for index, component in enumerate(path.parts[1:]):
            candidate = current / component
            is_final = index == len(path.parts[1:]) - 1
            is_private = private_from is not None and (
                candidate == private_from or candidate.is_relative_to(private_from)
            )
            may_create = create_private and is_private
            try:
                next_descriptor = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            except FileNotFoundError as error:
                if not may_create:
                    if is_final and is_private:
                        raise RuntimeError(f"DeepSWE private directory is unavailable: {path}") from error
                    raise RuntimeError(f"DeepSWE path has a missing or symlinked ancestor: {path}") from error
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    # Another creator won the race. The no-follow open and
                    # metadata checks below still decide whether it is safe.
                    pass
                except OSError as mkdir_error:
                    raise RuntimeError(f"DeepSWE private directory is unavailable: {candidate}") from mkdir_error
                try:
                    next_descriptor = os.open(component, _directory_open_flags(), dir_fd=descriptor)
                except OSError as open_error:
                    raise RuntimeError(f"DeepSWE private path is not a real directory: {candidate}") from open_error
            except OSError as error:
                if is_final and is_private:
                    raise RuntimeError(f"DeepSWE private path is not a real directory: {candidate}") from error
                raise RuntimeError(f"DeepSWE path has a missing or symlinked ancestor: {path}") from error

            os.close(descriptor)
            descriptor = next_descriptor
            current = candidate
            metadata = os.fstat(descriptor)
            if is_private:
                _validate_private_metadata(metadata, candidate)
            else:
                _validate_trusted_ancestor(metadata, candidate)

        metadata = os.fstat(descriptor)
        if path == Path("/"):
            if private_from is not None:
                _validate_private_metadata(metadata, path)
        else:
            try:
                path_metadata = path.lstat()
            except OSError as error:
                raise RuntimeError(f"DeepSWE directory changed while it was being validated: {path}") from error
            if (path_metadata.st_dev, path_metadata.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise RuntimeError(f"DeepSWE directory changed while it was being validated: {path}")
    finally:
        os.close(descriptor)


def _validate_private_directory(path: Path) -> None:
    path = _absolute_without_symlink_resolution(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"DeepSWE private directory is unavailable: {path}") from error
    _validate_private_metadata(metadata, path)


def _validate_ancestor_chain(path: Path) -> None:
    """Reject symlinked, untrusted, or non-sticky writable ancestors."""
    path = _absolute_without_symlink_resolution(path)
    _open_directory_chain(path, private_from=None, create_private=False)


def ensure_private_directory(path: Path) -> Path:
    """Create an owner-only directory without accepting a symlink or loose mode."""
    path = _absolute_without_symlink_resolution(path)
    private_root = _absolute_without_symlink_resolution(DEFAULT_PRIVATE_ROOT)
    if path == private_root or path.is_relative_to(private_root):
        _open_directory_chain(path, private_from=private_root, create_private=True)
        return path

    _open_directory_chain(path, private_from=path, create_private=True)
    return path


def default_private_path(*parts: str) -> Path:
    return DEFAULT_PRIVATE_ROOT.joinpath(*parts)


__all__ = ["DEFAULT_PRIVATE_ROOT", "default_private_path", "ensure_private_directory"]
