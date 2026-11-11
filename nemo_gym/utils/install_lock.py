# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Share one local tool installation among the worker processes of a server.

With `num_workers > 1`, each uvicorn worker installs a missing tool on its own first call,
and the workers' first calls can arrive together.
Without coordination they would download, unpack and `npm install` into the same directory at once.
An installer takes `install_lock` around its work,
so one worker installs while the others wait and then reuse that install.

An install records where it put the tool, and which version it installed, only after it finished,
with `record_install`, so an interrupted install is never mistaken for a complete one.
A record is reused only for the version it names, so a changed pin reinstalls instead of running the old tool.

A waiter waits as long as the holder installs, with no timeout.
A hung install already hung the worker that ran it, and a timeout would let a waiter install
into the directory the holder may still be writing.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import IO, AsyncIterator, Iterator, Optional


try:
    import fcntl
except ImportError:  # pragma: no cover
    # Windows has no fcntl.
    # Gym runs several workers only on POSIX hosts, so there the lock is not needed.
    fcntl = None

LOG = logging.getLogger(__name__)


def _warn_unlocked(lock_path: Path, error: OSError) -> None:
    # A filesystem without file locks (NFS without lockd, Lustre mounted noflock) refuses flock,
    # and a read-only package directory refuses the lock file itself.
    # The install still runs, unlocked, as it did before workers shared it.
    LOG.warning("Cannot lock %s (%s); installing without the lock", lock_path, error)


def _open_lock_file(lock_path: Path) -> Optional[IO[str]]:
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        return open(lock_path, "a")
    except OSError as error:
        _warn_unlocked(lock_path, error)
        return None


@contextmanager
def install_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive lock on `lock_path` across processes until the block exits.

    The lock file must live outside any directory the install may remove,
    because a lock on a removed file no longer excludes a process that opens the path again.
    When the lock file cannot be created or locked, it logs a warning and runs the block without exclusion.
    """
    handle = _open_lock_file(lock_path)
    try:
        if handle is not None and fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError as error:
                _warn_unlocked(lock_path, error)
        yield
    finally:
        if handle is not None:
            handle.close()


@asynccontextmanager
async def async_install_lock(lock_path: Path, *, poll_seconds: float = 0.1) -> AsyncIterator[None]:
    """`install_lock` for a coroutine, which waits without blocking the event loop.

    It polls instead of waiting in a thread,
    so a cancelled waiter leaves no thread behind that could take the lock later.
    When the lock file cannot be created or locked, it logs a warning and runs the block without exclusion.
    """
    handle = _open_lock_file(lock_path)
    try:
        while handle is not None and fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(poll_seconds)
            except OSError as error:
                _warn_unlocked(lock_path, error)
                break
        yield
    finally:
        if handle is not None:
            handle.close()


def record_install(marker: Path, bin_dir: Path, *, version: Optional[str]) -> None:
    """Record that an install of `version` finished and put its executables in `bin_dir`.

    `version` is the version the caller asked for, None for an unpinned install.
    The marker is replaced atomically, so a reader sees either no record or a complete one.
    Where the record cannot be written, such as a read-only package directory, it logs a warning:
    the install itself succeeded, and a later process without the record installs again.
    """
    temporary = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps({"bin_dir": str(bin_dir), "version": version}))
        temporary.replace(marker)
    except OSError as error:
        LOG.warning("Cannot record the install at %s (%s); a later process will install again", marker, error)
        with suppress(OSError):
            temporary.unlink()


def adopt_recorded_install(marker: Path, executable: str, *, version: Optional[str]) -> bool:
    """Put a recorded install's directory first on `PATH` when it holds `executable` at `version`.

    Another worker may have installed the tool into a directory that is not on this process's `PATH`.
    Returns False when there is no record, the record is for another version (or predates versioned records),
    or the recorded directory no longer holds the executable. The caller then installs `version` again.
    """
    try:
        record = json.loads(marker.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if not isinstance(record, dict) or record.get("version") != version:
        return False
    bin_dir = record.get("bin_dir")
    if not isinstance(bin_dir, str) or not bin_dir or not (Path(bin_dir) / executable).is_file():
        return False
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return True
