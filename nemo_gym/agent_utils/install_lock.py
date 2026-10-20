# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Share one local tool installation among the worker processes of a server.

With `num_workers > 1`, each uvicorn worker installs a missing tool on its own first call,
and the workers' first calls can arrive together.
Without coordination they would download, unpack and `npm install` into the same directory at once.
An installer takes `install_lock` around its work,
so one worker installs while the others wait and then reuse that install.

An install records where it put the tool only after it finished, with `record_install`,
so an interrupted install is never mistaken for a complete one.
"""

import asyncio
import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator


try:
    import fcntl
except ImportError:  # pragma: no cover
    # Windows has no fcntl.
    # Gym runs several workers only on POSIX hosts, so there the lock is not needed.
    fcntl = None


@contextmanager
def install_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive lock on `lock_path` across processes until the block exits.

    The lock file must live outside any directory the install may remove,
    because a lock on a removed file no longer excludes a process that opens the path again.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


@asynccontextmanager
async def async_install_lock(lock_path: Path, *, poll_seconds: float = 0.1) -> AsyncIterator[None]:
    """`install_lock` for a coroutine, which waits without blocking the event loop.

    It polls instead of waiting in a thread,
    so a cancelled waiter leaves no thread behind that could take the lock later.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as handle:
        while fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(poll_seconds)
        yield


def record_install(marker: Path, bin_dir: Path) -> None:
    """Record that an install finished and put its executables in `bin_dir`.

    The marker is replaced atomically, so a reader sees either no record or a complete one.
    """
    temporary = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(str(bin_dir))
    temporary.replace(marker)


def adopt_recorded_install(marker: Path, executable: str) -> bool:
    """Put a recorded install's directory first on `PATH` when it still holds `executable`.

    Another worker may have installed the tool into a directory that is not on this process's `PATH`.
    Returns False when there is no record or the recorded directory no longer holds the executable.
    """
    try:
        bin_dir = marker.read_text().strip()
    except FileNotFoundError:
        return False
    if not bin_dir or not (Path(bin_dir) / executable).is_file():
        return False
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return True
