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

- ``temp_dir``: temporary folders, deleted when their work ends. By default ``<system temp>/nemo_gym-<user id>/``.
  :func:`request_temp_dir` and :func:`async_request_temp_dir` last one request, :func:`session_temp_dir` one
  session, and :func:`process_temp_dir` the whole server process.
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
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote
from uuid import uuid4

from nemo_gym.config.global_config import (
    RESULTS_DIR_KEY_NAME,
    TEMP_DIR_KEY_NAME,
    maybe_get_global_config_dict,
)
from nemo_gym.core.episode_context import current_rollout_id
from nemo_gym.core.filesystems import warn_if_unsuitable
from nemo_gym.core.processes import process_may_be_running


logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
# Keeps the leaf folder name well under the 255-byte limit after the timestamp and random suffix.
_MAX_LABEL_LENGTH = 100
# The rollout folder of kept files for a request that carries no rollout id.
_NO_ROLLOUT = "no-rollout"
# How many save_dir copies one process runs at once. They get their own threads, so queued copies never hold the
# event loop's default executor, which DNS lookups and other file work need.
_SAVE_WORKERS = 4
# Each process keeps its temporary folders in one folder of its own under a parent, named for the process:
# nemo-gym-<host boot id>-<pid namespace>-<pid>. Folders inside it carry no process data.
# PID namespace numbers repeat across hosts (every host's initial namespace has the same one), so the boot id is
# what tells hosts apart when two of them share a directory.
_PROCESS_DIR = re.compile(r"^nemo-gym-(?P<host>[0-9a-f]{12})-(?P<namespace>\d+)-(?P<pid>\d+)$")
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
_UNKNOWN_HOST = "unknown"

# Per-process state. The after-fork hook resets all of it, so a forked child starts with none of its parent's.
_default_temp_dir: Path | None = None
# Parents this process has already checked and cleared of folders left by dead processes.
_checked_temp_parents: set[Path] = set()
# This process's own folder under each parent.
_process_dirs: dict[Path, Path] = {}
_exit_handler_registered = False
# Guards the first use of a parent: its check and cleanup, and creating this process's folder in it.
# Requests start their temporary folders in worker threads, so a burst of first requests arrives together.
_first_use_lock = threading.Lock()
_save_executor = ThreadPoolExecutor(max_workers=_SAVE_WORKERS, thread_name_prefix="nemo-gym-save-dir")
# Warnings this process has already logged, so each is reported once.
_warned: set[str] = set()


def _reset_after_fork() -> None:
    """Give a forked child state of its own: its parent's folders, locks and copy threads are not its own."""
    global _default_temp_dir, _checked_temp_parents, _process_dirs, _exit_handler_registered
    global _first_use_lock, _save_executor
    _default_temp_dir = None
    _checked_temp_parents = set()
    _process_dirs = {}
    _exit_handler_registered = False
    _first_use_lock = threading.Lock()
    _save_executor = ThreadPoolExecutor(max_workers=_SAVE_WORKERS, thread_name_prefix="nemo-gym-save-dir")


os.register_at_fork(after_in_child=_reset_after_fork)


def _warn_once(key: str, message: str, *args: object) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _make_writable(path: str) -> None:
    try:
        os.chmod(path, stat.S_IMODE(os.lstat(path).st_mode) | stat.S_IRWXU)
    except OSError:
        pass


def _remove_tree(path: str | Path) -> bool:
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


def _remove_stale_process_dirs(parent: Path) -> list[Path]:
    """Remove this user's process folders under ``parent`` whose process is gone, with everything inside.

    A killed process cannot remove its own folder.
    Each process folder's name records its host's boot, its PID namespace and its process.
    A folder goes only when it belongs to this user, this boot of this host and this PID namespace,
    and its process is no longer running.
    A folder of another host, another boot or another container's namespace stays, which holds on a filesystem many
    hosts share too: the launcher's per-job temp_dir and the system's temporary-file cleaner bound those.
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
        if match is None or match["host"] != host or int(match["namespace"]) != namespace:
            continue
        try:
            if not entry.is_dir(follow_symlinks=False) or entry.stat(follow_symlinks=False).st_uid != uid:
                continue
        except OSError:
            continue
        if process_may_be_running(int(match["pid"])):
            continue
        if _remove_tree(entry.path):
            removed.append(Path(entry.path))
    return removed


def _remove_own_process_dirs(pid: int) -> None:
    """Remove the folders this process made, at its exit; a forked child that inherited the handler does nothing."""
    if os.getpid() != pid:
        return
    for folder in list(_process_dirs.values()):
        _remove_tree(folder)


def _process_dir(parent: Path) -> Path:
    """This process's own folder under ``parent``, created once and removed when the process exits."""
    global _exit_handler_registered
    folder = _process_dirs.get(parent)
    if folder is not None:
        return folder
    with _first_use_lock:
        folder = _process_dirs.get(parent)
        if folder is not None:
            return folder
        folder = parent / _process_dir_name()
        # Before this process records it, a folder with this name can only be one a dead process of the same id left.
        _remove_tree(folder)
        folder.mkdir()
        if not _exit_handler_registered:
            atexit.register(_remove_own_process_dirs, os.getpid())
            _exit_handler_registered = True
        _process_dirs[parent] = folder
    return folder


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
    # lstat does not follow a symlink, so a symlink is not a folder here.
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise PermissionError(
            f"The default temp_dir {folder} is not a folder of this user's own. Set temp_dir to a folder of your own."
        )


def _resolve_default_temp_dir() -> Path:
    """This user's default temp_dir, ``<system temp>/nemo_gym-<user id>/``, private to the user."""
    global _default_temp_dir
    if _default_temp_dir is None:
        folder = Path(tempfile.gettempdir()) / f"nemo_gym-{os.getuid()}"
        _make_private_dir(folder)
        _default_temp_dir = folder
    return _default_temp_dir


def _temp_parent(temp_dir: str | Path | None) -> Path:
    """The folder temporary folders go in: ``temp_dir``, else the global ``temp_dir``, else the user's default."""
    if temp_dir is not None:
        parent, source = Path(temp_dir).expanduser(), "the temp_dir argument"
        if not parent.is_absolute():
            # A server would resolve it against its own source folder.
            raise ValueError(f"temp_dir must be an absolute path, got {str(temp_dir)!r}")
    elif configured := (maybe_get_global_config_dict() or {}).get(TEMP_DIR_KEY_NAME):
        parent, source = Path(configured), "the temp_dir setting"
    else:
        parent, source = _default_temp_dir, "the default temp_dir"
    if parent is not None and parent in _checked_temp_parents:
        return parent
    with _first_use_lock:
        if parent is None:
            # Raises an error that names the default temp_dir when it cannot be used.
            parent = _resolve_default_temp_dir()
        if parent in _checked_temp_parents:
            return parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise type(error)(
                error.errno, f"Could not create temporary folders under {parent}, chosen by {source}: {error.strerror}"
            ) from error
        # Checked here, on the node that writes, because the mount table differs per node.
        warn_if_unsuitable(TEMP_DIR_KEY_NAME, parent)
        # Removing leftovers is housekeeping: it must never fail the request that triggered it.
        try:
            _remove_stale_process_dirs(parent)
        except Exception:
            logger.warning("Could not remove leftover temporary folders under %s", parent, exc_info=True)
        _checked_temp_parents.add(parent)
    return parent


def _forget(parent: Path) -> None:
    """Drop what this process knows about ``parent`` after something outside removed it, so it is checked again.

    Only what is really gone is dropped: another thread that hit the same removal may already have made the folders
    again, and its new process folder must not count as a dead process's leftover.
    """
    global _default_temp_dir
    with _first_use_lock:
        if not parent.is_dir():
            _checked_temp_parents.discard(parent)
            if parent == _default_temp_dir:
                _default_temp_dir = None
        folder = _process_dirs.get(parent)
        if folder is not None and not folder.is_dir():
            del _process_dirs[parent]


def _in_process_dir(temp_dir: str | Path | None, make: Callable[[Path], _T]) -> _T:
    """Call ``make`` with this process's folder under ``temp_dir``, checking the folders again if one was removed."""
    parent = _temp_parent(temp_dir)
    try:
        return make(_process_dir(parent))
    except FileNotFoundError:
        # Something outside this process removed its folder, such as a cleaner of old temporary files.
        _forget(parent)
        return make(_process_dir(_temp_parent(temp_dir)))


def _check_name(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise ValueError(f"name must be lowercase letters, digits, '-' and '_', got {name!r}")


def _create_temp_dir(name: str, temp_dir: str | Path | None) -> Path:
    _check_name(name)
    return _in_process_dir(temp_dir, lambda folder: Path(tempfile.mkdtemp(prefix=f"{name}-", dir=folder)))


@contextmanager
def request_temp_dir(name: str = "tmp", *, temp_dir: str | Path | None = None) -> Iterator[Path]:
    """Create a temporary folder for one request and always remove it when the block exits.

    ``name`` says what the folder holds; it starts the folder's name, so a reader of ``temp_dir`` can tell folders
    apart.

    The folder is under ``temp_dir``: this argument, which must be absolute, else the global ``temp_dir`` setting,
    else ``<system temp>/nemo_gym-<user id>/``. Pass ``temp_dir`` from a server's own setting, or into a Ray task
    that does not see Gym's configuration.
    Processes and containers this process starts on the same node may use the folder; never send its path to another
    Gym server, a Ray task or anything that may run on another node.
    To keep what is in it, call :func:`save_dir` before the block exits.

    This blocks on filesystem calls; use :func:`async_request_temp_dir` in a coroutine.
    """
    path = _create_temp_dir(name, temp_dir)
    try:
        yield path
    finally:
        _remove_tree(path)


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
        await asyncio.shield(asyncio.to_thread(_remove_tree, path))


def _remove_created_dir(creating: asyncio.Future[Path]) -> None:
    if not creating.cancelled() and creating.exception() is None:
        _remove_tree(creating.result())


def session_temp_dir(name: str = "tmp", *, temp_dir: str | Path | None = None) -> Path:
    """Create a temporary folder for one session, under ``temp_dir`` as :func:`request_temp_dir` places it.

    Keep it on the session and remove it with :func:`remove_temp_dir` after the session's in-flight requests
    finish, under the session's lock. Remove what remains when the server shuts down; the process exit removes
    anything left after that.
    """
    return _create_temp_dir(name, temp_dir)


def remove_temp_dir(path: Path) -> bool:
    """Remove a folder from :func:`session_temp_dir`; return whether it is gone.

    It refuses, with a warning, any path that is not a temporary folder this process created, so a wrong or empty
    path never deletes something else. A failure is logged, never raised, so a caller can try again later. A folder
    that is already gone counts as removed.
    """
    # Normalized first: pathlib keeps "..", so "<process folder>/.." would otherwise pass as a folder in it.
    path = Path(os.path.normpath(path))
    if not path.is_absolute() or path.parent not in _process_dirs.values() or path.name.startswith("process."):
        _warn_once("refuse", "Not removing %s: it is not a temporary folder this process created.", path)
        return False
    return _remove_tree(path)


def process_temp_dir(name: str, *, temp_dir: str | Path | None = None) -> Path:
    """Return the temporary folder called ``name`` that lasts as long as this process, creating it when missing.

    Every call with the same ``name`` and ``temp_dir`` gets the same folder, for state shared across sessions,
    such as a prepared workspace template; writers in it coordinate themselves. A server with several workers that
    shares prepared state creates it in its main process before the workers start, and passes them the path.
    The folder is removed when the process exits. A forked child gets its own.
    """
    _check_name(name)

    def make(folder: Path) -> Path:
        # A name has no dot, so this never collides with a request or session folder.
        path = folder / f"process.{name}"
        path.mkdir(exist_ok=True)
        return path

    return _in_process_dir(temp_dir, make)


def _kept_dir(server_name: str, name: str, label: str | None, output_dir: str | Path | None) -> Path | None:
    """A new folder path for kept files: ``<output>/<name>/<rollout>/<UTC time>-[<label>-]<random>``."""
    if output_dir is not None:
        output = Path(output_dir)
        if not output.is_absolute():
            _warn_once(
                f"relative:{output}",
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
            _warn_once("no-results-dir", "No results_dir is set in this process, so kept files are not saved.")
            return None
        output = Path(results_dir) / quote(server_name, safe="")
    leaf = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if label:
        leaf += "-" + quote(label, safe="")[:_MAX_LABEL_LENGTH]
    leaf += "-" + uuid4().hex[:12]
    rollout = quote(current_rollout_id() or _NO_ROLLOUT, safe="")
    return output / name / rollout / leaf


def _copy_dir(src: Path, dest: Path) -> Path:
    """Copy the files under ``src`` into the new folder ``dest``.

    Folders are created afresh, so they take the user's usual permissions rather than the temporary folder's
    private ones, and files are copied by content: no permission, time or extended-attribute copies, each a metadata
    operation on the shared storage ``dest`` is usually on. Symlinks are copied as symlinks, and sockets, pipes and
    other special files are skipped. An entry that cannot be read is skipped, so the rest is kept; a failure to
    write fails the copy, which removes what it copied.
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
                    logger.debug("Not keeping %s: it is not a regular file, a folder or a symlink.", source)
    except BaseException:
        # Logged here, since a cancelled caller no longer receives the error.
        logger.warning("Could not keep %s at %s", src, dest, exc_info=True)
        # Only a folder this copy created: dest.mkdir() fails on one that was already there.
        if created:
            _remove_tree(dest)
        raise
    if unread:
        logger.warning("Kept %s without %d entries it could not read, such as %s", src, len(unread), unread[0])
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
    is the server's ``output_dir`` or ``<results_dir>/<server_name>``. The rollout comes from the request, and is
    ``no-rollout`` outside one.
    Copies run on a few threads of their own, so a burst of them never holds up the server's other work.
    A copy has no time limit. A caller that must not wait longer, such as a request with a time budget, wraps the
    call in ``asyncio.wait_for``: it returns on time, and the copy finishes on its thread.

    It never fails the request: a failed copy is logged, its partial folder removed, and it returns ``None``.
    An entry it cannot read is skipped and logged, and the rest is kept.
    It also returns ``None`` without copying when ``output_dir`` is relative, when no ``results_dir`` is set, and
    while the calling task is being cancelled, so shutdown stays fast: it skips copies that have not started.
    Call it in a ``finally`` so a failed request's files are kept too, and outside any ``asyncio.timeout`` block:
    inside one that has expired, the task counts as being cancelled and nothing is kept.
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
        return await asyncio.get_running_loop().run_in_executor(_save_executor, _copy_dir, Path(src), dest)
    except asyncio.CancelledError:
        raise
    except Exception:
        # _copy_dir has logged it.
        return None
