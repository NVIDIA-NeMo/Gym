# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Remove and copy folder trees on storage that other tools and users also write to."""

import logging
import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path


logger = logging.getLogger(__name__)


def remove_tree(path: str | Path) -> bool:
    """Remove ``path`` and everything in it; return whether it is gone. Never raises.

    A folder this process cannot write into, such as a read-only module cache a tool left behind, is made writable
    and removed too. An entry that still cannot be removed, such as a file another user owns, is skipped, so the
    rest of the tree is removed; what is left is logged once.
    """
    path = os.fspath(path)
    retried: set[str] = set()
    left: list[str] = []

    def on_error(func: Callable[..., object], failed: str, error: BaseException) -> None:
        if isinstance(error, FileNotFoundError):
            return
        if not isinstance(error, PermissionError) or failed in retried:
            left.append(failed)
            return
        retried.add(failed)
        if failed != path:
            _make_writable(os.path.dirname(failed))
        if not os.path.islink(failed):
            _make_writable(failed)
        try:
            os.unlink(failed)
        except IsADirectoryError:
            shutil.rmtree(failed, onexc=on_error)
        except FileNotFoundError:
            pass
        except OSError:
            left.append(failed)

    try:
        shutil.rmtree(path, onexc=on_error)
    except FileNotFoundError:
        return True
    except Exception:
        # Such as a RecursionError from a very deep tree; the caller's own work must not fail with it.
        logger.warning("Could not remove %s", path, exc_info=True)
        return not os.path.lexists(path)
    if left:
        logger.warning("Could not remove %d entries under %s, such as %s", len(left), path, left[0])
    return not os.path.lexists(path)


def copy_tree(src: Path, dest: Path) -> Path:
    """Copy the files under ``src`` into the new folder ``dest``.

    Folders are created afresh, so they take the user's usual permissions rather than the source's, and files are
    copied by content: no permission, time or extended-attribute copies, each a metadata operation on shared
    storage. Symlinks are copied as symlinks, and sockets, pipes and other special files are skipped. An entry that
    cannot be read is skipped, so the rest is kept; a failure to write fails the copy, which removes what it copied.
    """
    unread: list[str] = []
    created = False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.mkdir()
        created = True
        for root, dirs, files in os.walk(src, onerror=lambda error: unread.append(str(error.filename))):
            target = dest / os.path.relpath(root, src)
            for entry in [*dirs, *files]:
                source = os.path.join(root, entry)
                try:
                    mode = os.lstat(source).st_mode
                    link = os.readlink(source) if stat.S_ISLNK(mode) else None
                    reader = open(source, "rb") if stat.S_ISREG(mode) else None
                except OSError:
                    unread.append(source)
                    continue
                if link is not None:
                    os.symlink(link, target / entry)
                elif reader is not None:
                    # One create on the destination, with no stat of it, unlike shutil.copyfile.
                    with reader, open(target / entry, "xb") as writer:
                        shutil.copyfileobj(reader, writer)
                elif stat.S_ISDIR(mode):
                    (target / entry).mkdir()
                else:
                    logger.debug("Not copying %s: it is not a regular file, a folder or a symlink.", source)
    except BaseException:
        # Logged here, since a caller that was cancelled while waiting no longer receives the error.
        logger.warning("Could not copy %s to %s", src, dest, exc_info=True)
        # Only a folder this copy created: dest.mkdir() fails on one that was already there.
        if created:
            remove_tree(dest)
        raise
    if unread:
        logger.warning("Copied %s without %d entries it could not read, such as %s", src, len(unread), unread[0])
    return dest


def _make_writable(path: str) -> None:
    try:
        os.chmod(path, stat.S_IMODE(os.lstat(path).st_mode) | stat.S_IRWXU)
    except OSError:
        pass
