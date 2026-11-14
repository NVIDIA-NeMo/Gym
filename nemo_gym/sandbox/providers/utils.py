# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers shared by the built-in sandbox providers.

These were previously duplicated verbatim across the enroot, apptainer, and
opensandbox providers. They live here so there is a single implementation.
"""

import errno
import os
import posixpath
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def coerce_config(value: Any, config_cls: type[Any]) -> Any:
    """Accept either a config dataclass instance or a plain mapping (Hydra YAML)."""
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


def path_under_mount(mount_point: str, path: str) -> str | None:
    """If `path` is inside the mount, return its path relative to the mount; else None."""
    if not path.startswith("/"):
        return None
    mp = posixpath.normpath(mount_point.rstrip("/") or "/")
    normalized = posixpath.normpath(path)
    if normalized == mp:
        return ""
    try:
        if posixpath.commonpath([mp, normalized]) != mp:
            return None
    except ValueError:
        return None
    if mp == "/":
        return normalized.lstrip("/")
    return normalized[len(mp) + 1 :]


def remove_writable_tree(path: Path) -> None:
    """Remove read-only staging directories without following agent-controlled links.

    Pin each directory before changing its permissions: a concurrent rename or
    symlink replacement must not redirect a host chmod outside the staging tree.
    Removal uses shutil's descriptor-based traversal under the same parent.
    """
    if not shutil.rmtree.avoids_symlink_attacks:
        raise OSError("Sandbox staging cleanup requires symlink-safe shutil.rmtree")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

    def make_writable(root_fd: int) -> None:
        # Keep entry names and parent identities, not open ancestor descriptors.
        # On ascent, verify '..' still names the parent we visited before using
        # it. A concurrent move must not send permission repair outside the tree.
        stack: list[tuple[tuple[int, int], list[str]]] = []
        directory_fd = os.dup(root_fd)
        try:
            while True:
                info = os.fstat(directory_fd)
                mode = stat.S_IMODE(info.st_mode)
                if info.st_uid == os.geteuid():
                    writable = mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
                    if mode != writable:
                        os.fchmod(directory_fd, writable)
                # Other-UID directories may already be writable via group/other
                # bits or ACLs. Do not try to change their permissions.
                with os.scandir(directory_fd) as entries:
                    children = [entry.name for entry in entries if entry.is_dir(follow_symlinks=False)]
                stack.append(((info.st_dev, info.st_ino), children))
                while stack:
                    if stack[-1][1]:
                        name = stack[-1][1].pop()
                        try:
                            child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                        except OSError as error:
                            if error.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
                                continue
                            raise
                        os.close(directory_fd)
                        directory_fd = child_fd
                        break
                    stack.pop()
                    if not stack:
                        return
                    parent = os.open("..", directory_flags, dir_fd=directory_fd)
                    try:
                        info = os.fstat(parent)
                        if (info.st_dev, info.st_ino) != stack[-1][0]:
                            raise OSError("Sandbox staging directory moved during permission repair")
                    except BaseException:
                        os.close(parent)
                        raise
                    os.close(directory_fd)
                    directory_fd = parent
        finally:
            os.close(directory_fd)

    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            shutil.rmtree(path.name, dir_fd=parent_fd)
            return
        except PermissionError:
            # Most workspaces need no chmod pass. Repair only after deletion
            # encounters a read-only directory, retaining descriptor-safe access.
            pass
        root_fd = os.open(path.name, directory_flags, dir_fd=parent_fd)
        try:
            make_writable(root_fd)
        finally:
            os.close(root_fd)
        shutil.rmtree(path.name, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
