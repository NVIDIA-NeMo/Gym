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
"""Where servers and agents put the files they write while they run.

A server starts with its own source folder as the working directory.
A relative path, or one built from ``Path.cwd()``, therefore writes into the Gym checkout.
On a cluster that checkout is on a shared filesystem, and every such write loads its metadata servers.

Runtime files go to one of three places:

- ``temp_dir``: temporary folders, deleted when their work ends. By default ``<system temp>/nemo_gym/<user id>/``.
  :func:`request_temp_dir` and :func:`async_request_temp_dir` last one request, :func:`session_temp_dir` one session,
  and :func:`process_temp_dir` the whole server process.
- ``results_dir``: everything kept after the job. A server keeps a finished temporary folder with
  :func:`save_dir`, which copies it once under ``<results_dir>/<server name>/`` or the server's ``output_dir``.
  State a later run reads back goes in the server's ``output_dir``, written as whole files.
- ``cache_dir``: files reused across runs.

Each server decides what to keep, with its own setting; nothing here turns kept files on or off.
Model-call capture keeps its own ``model_call_capture_dir``, because several components read it.
"""

import asyncio
import atexit
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from nemo_gym.config.global_config import (
    RESULTS_DIR_KEY_NAME,
    TEMP_DIR_KEY_NAME,
    maybe_get_global_config_dict,
)
from nemo_gym.core.episode_context import current_rollout_id
from nemo_gym.core.filesystems import SHARED_FILESYSTEM_TYPES, warn_if_unsuitable
from nemo_gym.core.processes import process_may_be_running


logger = logging.getLogger(__name__)

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
# Keeps the leaf folder name well under the 255-byte limit after the timestamp and random suffix.
_MAX_LABEL_LENGTH = 100
# The rollout folder of kept files for a request that carries no rollout id.
NO_ROLLOUT_SEGMENT = "no-rollout"
# The folder under the system temporary directory that holds each user's default temp_dir.
DEFAULT_TEMP_ROOT_NAME = "nemo_gym"
# How many save_dir copies one process runs at once; the rest wait in their worker threads.
SAVE_FOLDER_CONCURRENCY = 4
# Each process keeps its temporary folders in one folder of its own under a parent, named for the process:
# nemo-gym-<host boot id>-<pid namespace>-<pid>. Folders inside it carry no process data.
# PID namespace numbers repeat across hosts (every host's initial namespace has the same one), so the boot id is
# what tells hosts apart when two of them share a directory.
_PROCESS_DIR = re.compile(r"^nemo-gym-(?P<host>[0-9a-f]{12})-(?P<namespace>\d+)-(?P<pid>\d+)$")
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
_UNKNOWN_HOST = "unknown"
# Parents this process has already checked and cleared of folders left by dead processes.
_checked_temp_parents: set[Path] = set()
# This process's default temp_dir, keyed by process id so a forked child resolves its own.
_default_temp_dirs: dict[int, Path] = {}
# This process's own folder under each parent, keyed by process id so a forked child makes its own.
_process_dirs: dict[tuple[int, Path], Path] = {}
# Folders that last the whole process, keyed by process id, parent and name.
_process_temp_dirs: dict[tuple[int, Path, str], Path] = {}
# Relative output folders this process has already warned about, so each is reported once.
_warned_relative_output_dirs: set[Path] = set()
# Guards the first use of a parent: its check and cleanup, and creating this process's folder in it.
# Requests start their temporary folders in worker threads, so a burst of first requests arrives together.
_first_use_lock = threading.Lock()
_process_temp_dir_lock = threading.Lock()
_save_slots = threading.BoundedSemaphore(SAVE_FOLDER_CONCURRENCY)


def _reset_locks_after_fork() -> None:
    """Give a forked child its own locks: a thread of the parent may have held the inherited ones at the fork."""
    global _first_use_lock, _process_temp_dir_lock, _save_slots
    _first_use_lock = threading.Lock()
    _process_temp_dir_lock = threading.Lock()
    _save_slots = threading.BoundedSemaphore(SAVE_FOLDER_CONCURRENCY)


os.register_at_fork(after_in_child=_reset_locks_after_fork)


def _host_id() -> str:
    """This host's current boot, as 12 hex digits, or ``unknown`` where Linux does not expose it."""
    try:
        return _BOOT_ID.read_text().strip().replace("-", "")[:12] or _UNKNOWN_HOST
    except OSError:
        return _UNKNOWN_HOST


def _pid_namespace() -> int | None:
    """This process's PID namespace, or ``None`` where Linux does not expose one."""
    try:
        return os.stat("/proc/self/ns/pid").st_ino
    except OSError:
        return None


def _process_dir_name() -> str:
    return f"nemo-gym-{_host_id()}-{_pid_namespace() or 0}-{os.getpid()}"


def remove_stale_temp_dirs(parent: Path, *, private_to_host: bool = False) -> list[Path]:
    """Remove this user's process folders under ``parent`` whose process is gone, with everything inside.

    A killed process cannot remove its own folder.
    Each process folder's name records its host's boot, its PID namespace and its process.
    A folder goes when it belongs to this user, this boot of this host and this PID namespace,
    and its process is no longer running.
    When ``parent`` is private to this host, a folder from an earlier boot goes too: no process of that boot runs.
    On storage many hosts share, a different boot may be another host, so those folders stay.
    """
    host = _host_id()
    namespace = _pid_namespace()
    if host == _UNKNOWN_HOST or namespace is None:
        return []
    removed = []
    uid = os.getuid()
    try:
        entries = list(os.scandir(parent))
    except OSError:
        return []
    for entry in entries:
        match = _PROCESS_DIR.fullmatch(entry.name)
        if match is None:
            continue
        earlier_boot = match["host"] != host
        if earlier_boot and not private_to_host:
            continue
        if not earlier_boot and int(match["namespace"]) != namespace:
            continue
        try:
            if not entry.is_dir(follow_symlinks=False) or entry.stat(follow_symlinks=False).st_uid != uid:
                continue
        except OSError:
            continue
        if not earlier_boot and process_may_be_running(int(match["pid"])):
            continue
        shutil.rmtree(entry.path, ignore_errors=True)
        removed.append(Path(entry.path))
    return removed


def _remove_own_process_dirs(pid: int) -> None:
    """Remove the folders this process made, at its exit; a forked child that inherited the handler does nothing."""
    if os.getpid() != pid:
        return
    for (owner, _), folder in list(_process_dirs.items()):
        if owner == pid:
            shutil.rmtree(folder, ignore_errors=True)


def _process_dir(parent: Path, *, recreate: bool = False) -> Path:
    """This process's own folder under ``parent``, created once and removed when the process exits.

    ``recreate`` makes it again after something outside the process removed it, keeping any files that remain.
    """
    pid = os.getpid()
    folder = _process_dirs.get((pid, parent))
    if folder is not None and not recreate:
        return folder
    with _first_use_lock:
        folder = _process_dirs.get((pid, parent))
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        folder = parent / _process_dir_name()
        # Before this process records it, a folder with this name can only be one a dead process of the same id left.
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        if not any(owner == pid for owner, _ in _process_dirs):
            atexit.register(_remove_own_process_dirs, pid)
        _process_dirs[(pid, parent)] = folder
    return folder


def _shared_root_problem(root: Path) -> str | None:
    """Why ``root``, the folder every user's default temp_dir sits in, cannot be used safely; ``None`` if it can.

    It is created like ``/tmp`` itself: writable by every user, with the sticky bit, so one user cannot rename or
    remove another user's folder.
    """
    try:
        root.mkdir(mode=0o1777)
        # The mode passed to mkdir is reduced by the umask; set it in full.
        os.chmod(root, 0o1777)
    except FileExistsError:
        pass
    except OSError as error:
        return f"it could not be created: {error.strerror}"
    try:
        info = os.lstat(root)
    except OSError as error:
        return f"it could not be checked: {error.strerror}"
    if stat.S_ISLNK(info.st_mode):
        return "it is a symlink"
    if not stat.S_ISDIR(info.st_mode):
        return "it is not a folder"
    if info.st_uid not in (0, os.getuid()):
        return f"it belongs to user {info.st_uid}"
    if stat.S_IMODE(info.st_mode) & 0o1777 != 0o1777:
        return f"its mode is {stat.S_IMODE(info.st_mode):o}, not 1777"
    return None


def _make_private_dir(folder: Path) -> None:
    """Create ``folder`` for this user only, or check that an existing one is this user's own real folder."""
    try:
        folder.mkdir(mode=0o700)
        return
    except FileExistsError:
        pass
    except OSError as error:
        raise type(error)(
            error.errno, f"Could not create the default temp_dir {folder}: {error.strerror}. Set temp_dir."
        ) from error
    info = os.lstat(folder)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise PermissionError(
            f"The default temp_dir {folder} is a symlink, not a folder, or another user's folder. "
            f"Set temp_dir to a folder of your own."
        )


def _default_temp_dir() -> Path:
    """This user's default temp_dir: ``<system temp>/nemo_gym/<user id>/``, private to the user.

    When ``<system temp>/nemo_gym/`` cannot be used safely, ``<system temp>/nemo_gym-<user id>/`` instead.
    """
    pid = os.getpid()
    folder = _default_temp_dirs.get(pid)
    if folder is not None:
        return folder
    system_temp = Path(tempfile.gettempdir())
    uid = os.getuid()
    root = system_temp / DEFAULT_TEMP_ROOT_NAME
    problem = _shared_root_problem(root)
    if problem is None:
        folder = root / str(uid)
    else:
        folder = system_temp / f"{DEFAULT_TEMP_ROOT_NAME}-{uid}"
        logger.warning("Not using %s for temporary folders because %s; using %s instead.", root, problem, folder)
    _make_private_dir(folder)
    _default_temp_dirs[pid] = folder
    return folder


def _temp_parent(temp_dir: str | Path | None) -> Path:
    """The folder temporary folders go in: ``temp_dir``, else the global ``temp_dir``, else the user's default."""
    if temp_dir is not None:
        parent, source = Path(temp_dir), "the temp_dir argument"
    elif configured := (maybe_get_global_config_dict() or {}).get(TEMP_DIR_KEY_NAME):
        parent, source = Path(configured), "the temp_dir setting"
    else:
        parent, source = None, "the default temp_dir"
    if parent is not None and parent in _checked_temp_parents:
        return parent
    with _first_use_lock:
        if parent is None:
            # Raises an error that names the default temp_dir when it cannot be used.
            parent = _default_temp_dir()
        if parent in _checked_temp_parents:
            return parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise type(error)(
                error.errno, f"Could not create temporary folders under {parent}, chosen by {source}: {error.strerror}"
            ) from error
        # Checked here, on the node that writes, because the mount table differs per node.
        shared = warn_if_unsuitable(TEMP_DIR_KEY_NAME, parent) in SHARED_FILESYSTEM_TYPES
        # Removing leftovers is housekeeping: it must never fail the request that triggered it.
        try:
            remove_stale_temp_dirs(parent, private_to_host=not shared)
        except Exception:
            logger.warning("Could not remove leftover temporary folders under %s", parent, exc_info=True)
        _checked_temp_parents.add(parent)
    return parent


def _check_name(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise ValueError(f"name must be lowercase letters, digits, '-' and '_', got {name!r}")


def _create_temp_dir(name: str, temp_dir: str | Path | None) -> Path:
    _check_name(name)
    parent = _temp_parent(temp_dir)
    try:
        return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=_process_dir(parent)))
    except FileNotFoundError:
        # Something outside this process removed its folder, such as a cleaner of old temporary files.
        return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=_process_dir(parent, recreate=True)))


@contextmanager
def request_temp_dir(name: str = "tmp", *, temp_dir: str | Path | None = None) -> Iterator[Path]:
    """Create a temporary folder for one request and always remove it when the block exits.

    ``name`` says what the folder holds; it starts the folder's name, so a reader of ``temp_dir`` can tell folders apart.

    The folder is under ``temp_dir``: this argument, else the global ``temp_dir`` setting, else
    ``<system temp>/nemo_gym/<user id>/``. Pass ``temp_dir`` from a server's own setting, or into a Ray task that
    does not see Gym's configuration.
    The folder is private to this process: pass its content to another process, never its path.
    To keep what is in it, call :func:`save_dir` before the block exits.

    This blocks on filesystem calls; use :func:`async_request_temp_dir` in a coroutine.
    """
    path = _create_temp_dir(name, temp_dir)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@asynccontextmanager
async def async_request_temp_dir(name: str = "tmp", *, temp_dir: str | Path | None = None) -> AsyncIterator[Path]:
    """:func:`request_temp_dir` for a coroutine: creating and removing the folder run in a worker thread.

    The removal is shielded from cancellation.
    Under an anyio cancel scope, which cancels again at every await, the block can exit before the folder is gone;
    the removal then finishes in the background.
    """
    creating = asyncio.ensure_future(asyncio.to_thread(_create_temp_dir, name, temp_dir))
    try:
        path = await asyncio.shield(creating)
    except asyncio.CancelledError:
        # The thread may still create the folder after this task is cancelled; remove it once it does.
        creating.add_done_callback(_remove_created_dir)
        raise
    try:
        yield path
    finally:
        # Shielded, so a second cancellation while the folder is being removed does not leave it behind.
        await asyncio.shield(asyncio.to_thread(shutil.rmtree, path, True))


def _remove_created_dir(creating: asyncio.Future[Path]) -> None:
    if not creating.cancelled() and creating.exception() is None:
        shutil.rmtree(creating.result(), ignore_errors=True)


def session_temp_dir(name: str = "tmp", *, temp_dir: str | Path | None = None) -> Path:
    """Create a temporary folder for one session, under ``temp_dir`` as :func:`request_temp_dir` places it.

    Keep it on the session and remove it with :func:`remove_temp_dir` after the session's in-flight requests
    finish, under the session's lock. Remove what remains when the server shuts down; the process exit removes
    anything left after that.
    """
    return _create_temp_dir(name, temp_dir)


def remove_temp_dir(path: Path) -> bool:
    """Remove a folder from :func:`session_temp_dir`; return whether it is gone.

    A failure is logged, never raised, so a caller can try again later. A folder that is already gone counts as removed.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not remove temporary folder %s; remove it again to retry.", path, exc_info=True)
        return False
    return True


def process_temp_dir(name: str, *, temp_dir: str | Path | None = None) -> Path:
    """Return the temporary folder called ``name`` that lasts as long as this process, creating it on first use.

    Every call with the same ``name`` and ``temp_dir`` gets the same folder, for state shared across sessions,
    such as a prepared workspace template; writers in it coordinate themselves.
    It is removed when the process exits. A forked child starts with no run folders of its own.
    """
    parent = _temp_parent(temp_dir)
    key = (os.getpid(), parent, name)
    folder = _process_temp_dirs.get(key)
    if folder is not None:
        return folder
    with _process_temp_dir_lock:
        folder = _process_temp_dirs.get(key)
        if folder is None:
            folder = _create_temp_dir(name, parent)
            _process_temp_dirs[key] = folder
    return folder


def _kept_dir(server_name: str, name: str, label: str | None, output_dir: str | Path | None) -> Path | None:
    """A new folder path for kept files: ``<output>/<name>/<rollout>/<UTC time>-[<label>-]<random>``."""
    if output_dir is not None:
        output = Path(output_dir)
        if not output.is_absolute():
            if output not in _warned_relative_output_dirs:
                _warned_relative_output_dirs.add(output)
                logger.warning(
                    "%s's output_dir %r is not an absolute path, so its files are not kept; "
                    "a relative path would resolve against the server's own source folder.",
                    server_name,
                    str(output),
                )
            return None
    else:
        if server_name in ("", ".", ".."):
            raise ValueError(f"server_name must name a server, got {server_name!r}")
        results_dir = (maybe_get_global_config_dict() or {}).get(RESULTS_DIR_KEY_NAME)
        if not results_dir:
            # Only a process started without Gym's configuration gets here, and it has no results folder.
            logger.warning("No results_dir is set in this process, so %s's files are not kept.", server_name)
            return None
        output = Path(results_dir) / quote(server_name, safe="")
    leaf = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if label:
        leaf += "-" + quote(label, safe="")[:_MAX_LABEL_LENGTH]
    leaf += "-" + uuid4().hex[:12]
    rollout = quote(current_rollout_id() or NO_ROLLOUT_SEGMENT, safe="")
    return output / name / rollout / leaf


def _copy_dir(src: Path, dest: Path) -> Path:
    """Copy ``src`` to the new folder ``dest``, at most a few copies at once in this process.

    Files are copied by contents only: copying their permissions and times would add a chmod and a utime per file,
    each a metadata operation on the shared storage ``dest`` is usually on. A failed copy leaves nothing behind.
    """
    with _save_slots:
        try:
            shutil.copytree(src, dest, symlinks=True, copy_function=shutil.copyfile)
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise
    return dest


async def save_dir(
    src: str | Path,
    *,
    server_name: str,
    name: str,
    label: str | None = None,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Copy a finished folder into the run's kept files once; return the folder it copied to, or ``None``.

    ``name`` says what the files are, such as ``verification``, and groups them in the results.
    The copy goes to a new folder ``<output>/<name>/<rollout>/<UTC time>-[<label>-]<random>``, where ``<output>``
    is the server's ``output_dir`` or ``<results_dir>/<server_name>``.
    The rollout comes from the request, and is ``no-rollout`` outside one; put the attempt in ``label`` when a
    request is retried.
    It copies file contents only, in a worker thread, with at most a few copies running at once in this process.

    It never fails the request: a failed copy is logged and returns ``None``.
    It also returns ``None`` without copying when ``output_dir`` is relative, when no ``results_dir`` is set, and
    while the calling task is being cancelled, so shutdown stays fast.
    Call it in a ``finally`` so a failed request's files are kept too.
    """
    _check_name(name)
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        logger.info("Not keeping %s for %s: the request is being cancelled.", src, server_name)
        return None
    dest = _kept_dir(server_name, name, label, output_dir)
    if dest is None:
        return None
    try:
        return await asyncio.to_thread(_copy_dir, Path(src), dest)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Could not keep %s at %s", src, dest, exc_info=True)
        return None
