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
Each kind of runtime file has one global setting, named by what it holds, and one helper here:

- A request's or session's temporary folder: :func:`temp_folder` or :func:`async_temp_folder`, under ``temp_dir``.
  The folder is removed when the block exits.
  Each process keeps its request folders inside one folder of its own, removed when the process exits;
  the next process on the same host removes the folder of a process that was killed.
- Debug files, such as verification logs, screenshots, trajectories and failure records: :func:`debug_folder`.
  They are written only when ``debug_dir`` or the server's own ``debug_dir`` names a folder.
- Model-call capture: ``model_call_capture_dir``.
- Caches: ``cache_dir``. Run output that a later run reads back: the server's ``output_dir``, under ``results_dir``.

A kind of file gets a top-level setting of its own only when its lifetime or cleanup owner differs from every
existing setting, and where it lives is a deployment decision an entry point makes for every process that uses it.
Files that one server writes go through these helpers instead.
"""

import asyncio
import atexit
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import quote
from uuid import uuid4

from pydantic import AfterValidator

from nemo_gym.config.global_config import (
    DEBUG_DIR_KEY_NAME,
    TEMP_DIR_KEY_NAME,
    maybe_get_global_config_dict,
)
from nemo_gym.core.episode_context import current_rollout_id
from nemo_gym.core.filesystems import warn_if_unsuitable
from nemo_gym.core.processes import process_may_be_running


logger = logging.getLogger(__name__)

_KIND = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
# Keeps the leaf folder name well under the 255-byte limit after the timestamp and random suffix.
_MAX_LABEL_LENGTH = 100
# The debug folder segment for a request that carries no rollout id.
NO_ROLLOUT_SEGMENT = "no-rollout"
# The debug folder segment for a request that runs as no named agent instance.
UNNAMED_AGENT_INSTANCE_SEGMENT = "unnamed"
# Each process keeps its temporary folders in one folder of its own under a parent, named for the process:
# nemo-gym-<host boot id>-<pid namespace>-<pid>. Request folders inside it carry no process data.
# PID namespace numbers repeat across hosts (every host's initial namespace has the same one), so the boot id is
# what tells hosts apart when two of them share a directory.
_PROCESS_FOLDER = re.compile(r"^nemo-gym-(?P<host>[0-9a-f]{12})-(?P<namespace>\d+)-(?P<pid>\d+)$")
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
_UNKNOWN_HOST = "unknown"
# Parents this process has already checked and cleared of folders left by dead processes.
_checked_temp_parents: set[Path] = set()
# This process's own folder under each parent, keyed by process id so a forked child makes its own.
_process_folders: dict[tuple[int, Path], Path] = {}
# Guards the first use of a parent: its check and cleanup, and creating this process's folder in it.
# Requests start their temporary folders in worker threads, so a burst of first requests arrives together.
_first_use_lock = threading.Lock()
_checked_debug_roots: set[Path] = set()


def _require_absolute(path: Path | None) -> Path | None:
    """Refuse a relative per-server folder: a server would resolve it against its own source folder."""
    if path is not None and not path.is_absolute():
        raise ValueError(
            f"A debug folder must be an absolute path, got {str(path)!r}: a server resolves a "
            "relative path against its own source folder."
        )
    return path


# A server's own ``debug_dir`` setting: ``None`` defers to the global one, and a relative path fails when the
# server's configuration loads, rather than on every request that writes debug files.
DebugDir = Annotated[Path | None, AfterValidator(_require_absolute)]


def debug_folder(
    *, server_name: str, kind: str, label: str | None = None, override: str | Path | None = None
) -> Path | None:
    """Return a new folder for one unit of work's debug files, or ``None`` when debug files are off.

    The root is ``override`` when the server's own ``debug_dir`` names a folder.
    Otherwise it is ``<debug_dir>/<server_name>`` when the global ``debug_dir`` names one.
    Otherwise there is none.
    Below the root the folder is ``<kind>/<capture key>/<agent instance>/<UTC time>-[<label>-]<random>``.
    The capture key and agent instance come from the request context.
    So every server separates rollouts and agent instances the same way.
    Outside a rollout those segments are ``no-rollout`` and ``unnamed``.
    The random part keeps two calls from sharing a folder.
    That covers two calls in one rollout, and two runs that reuse task and rollout indices.

    Call this once per unit of work that owns the files.
    That is once per request, or once when a session opens, keeping the folder on the session.
    Nothing is created here; the folder exists only if the caller writes into it.

    Debug folders are grouped by rollout only while the request carries a rollout id.
    Today that holds when model-call capture is on; with it off, folders fall under ``no-rollout``.

    Only a process started with Gym's configuration sees the global ``debug_dir``, and only a request sees its rollout.
    So call this in the server's request and pass the folder it returns to a Ray task or helper process.
    """
    if not _KIND.fullmatch(kind):
        raise ValueError(f"kind must be lowercase letters, digits, '-' and '_', got {kind!r}")
    if override is not None:
        root = _require_absolute(Path(override))
    else:
        global_root = (maybe_get_global_config_dict() or {}).get(DEBUG_DIR_KEY_NAME)
        if not global_root:
            return None
        # The name is used only here, so a request with debug files off never fails over it.
        if server_name in ("", ".", ".."):
            raise ValueError(f"server_name must name a server, got {server_name!r}")
        root = Path(global_root) / quote(server_name, safe="")

    leaf = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if label:
        leaf += "-" + quote(label, safe="")[:_MAX_LABEL_LENGTH]
    leaf += "-" + uuid4().hex[:12]
    if root not in _checked_debug_roots:
        _checked_debug_roots.add(root)
        warn_if_unsuitable(DEBUG_DIR_KEY_NAME, root)
    rollout = quote(current_rollout_id() or NO_ROLLOUT_SEGMENT, safe="")
    agent_instance = quote(_current_agent_instance() or UNNAMED_AGENT_INSTANCE_SEGMENT, safe="")
    return root / kind / rollout / agent_instance / leaf


def _current_agent_instance() -> str | None:
    """The agent instance the current request runs as, or ``None`` when it is unnamed.

    Requests carry no agent instance name in context yet.
    Once named agent sessions set one, this function reads it, and callers do not change.
    """
    return None


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


def _process_folder_name() -> str:
    return f"nemo-gym-{_host_id()}-{_pid_namespace() or 0}-{os.getpid()}"


def remove_stale_temp_folders(parent: Path) -> list[Path]:
    """Remove this user's process folders under ``parent`` whose process is gone, with every request folder inside.

    A killed process cannot remove its own folder.
    Each process folder's name records its host's boot, its PID namespace and its process.
    A folder goes only when it belongs to this user, this boot of this host and this PID namespace,
    and its process is no longer running.
    A process on another host, or in another container's namespace, keeps its folder,
    which holds on a filesystem many hosts share too.
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
        match = _PROCESS_FOLDER.fullmatch(entry.name)
        if match is None or match["host"] != host or int(match["namespace"]) != namespace:
            continue
        try:
            if not entry.is_dir(follow_symlinks=False) or entry.stat(follow_symlinks=False).st_uid != uid:
                continue
        except OSError:
            continue
        pid = int(match["pid"])
        if process_may_be_running(pid):
            continue
        shutil.rmtree(entry.path, ignore_errors=True)
        removed.append(Path(entry.path))
    return removed


def _remove_own_process_folders(pid: int) -> None:
    """Remove the folders this process made, at its exit; a forked child that inherited the handler does nothing."""
    if os.getpid() != pid:
        return
    for (owner, _), folder in list(_process_folders.items()):
        if owner == pid:
            shutil.rmtree(folder, ignore_errors=True)


def _process_folder(parent: Path, *, recreate: bool = False) -> Path:
    """This process's own folder under ``parent``, created once and removed when the process exits.

    ``recreate`` makes it again after something outside the process removed it, keeping any files that remain.
    """
    pid = os.getpid()
    folder = _process_folders.get((pid, parent))
    if folder is not None and not recreate:
        return folder
    with _first_use_lock:
        folder = _process_folders.get((pid, parent))
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        folder = parent / _process_folder_name()
        # Before this process records it, a folder with this name can only be one a dead process of the same id left.
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        if not any(owner == pid for owner, _ in _process_folders):
            atexit.register(_remove_own_process_folders, pid)
        _process_folders[(pid, parent)] = folder
    return folder


def _temp_parent(parent: Path | None) -> Path:
    if parent is None:
        configured = (maybe_get_global_config_dict() or {}).get(TEMP_DIR_KEY_NAME)
        # Without a setting, the node's own temporary directory, as this process sees it.
        parent = Path(configured) if configured else Path(tempfile.gettempdir())
    if parent in _checked_temp_parents:
        return parent
    with _first_use_lock:
        if parent not in _checked_temp_parents:
            parent.mkdir(parents=True, exist_ok=True)
            # Checked here, on the node that writes, because the mount table differs per node.
            warn_if_unsuitable(TEMP_DIR_KEY_NAME, parent)
            # Removing leftovers is housekeeping: it must never fail the request that triggered it.
            try:
                remove_stale_temp_folders(parent)
            except Exception:
                logger.warning("Could not remove leftover temporary folders under %s", parent, exc_info=True)
            _checked_temp_parents.add(parent)
    return parent


def _create_temp_folder(kind: str, parent: Path | None) -> Path:
    if not _KIND.fullmatch(kind):
        raise ValueError(f"kind must be lowercase letters, digits, '-' and '_', got {kind!r}")
    parent = _temp_parent(parent)
    try:
        return Path(tempfile.mkdtemp(prefix=f"{kind}-", dir=_process_folder(parent)))
    except FileNotFoundError:
        # Something outside this process removed its folder, such as a cleaner of old temporary files.
        return Path(tempfile.mkdtemp(prefix=f"{kind}-", dir=_process_folder(parent, recreate=True)))


def _finish_temp_folder(path: Path, keep_as: Path | None) -> None:
    if keep_as is not None:
        # Debug files must never fail the request they describe.
        try:
            keep_as.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, keep_as)
        except Exception:
            logger.warning("Could not keep temporary folder %s as debug files at %s", path, keep_as, exc_info=True)
    shutil.rmtree(path, ignore_errors=True)


@contextmanager
def temp_folder(*, kind: str, keep_as: Path | None = None, parent: Path | None = None) -> Iterator[Path]:
    """Create a temporary folder for one request or session and remove it when the block exits.

    The folder is under ``temp_dir``, which defaults to the node's temporary directory.
    It is private to this request: pass its content to another process in the request, never its path.

    ``keep_as`` is a folder from :func:`debug_folder`.
    When it is set, the folder is moved there on exit instead of being deleted.
    A server then writes to one place whether debug files are on or off.
    The whole folder moves, across filesystems by copying each file, so use it for a folder of logs, never a workspace.
    Give each temporary folder its own ``keep_as`` that does not exist yet, such as a fresh :func:`debug_folder`:
    a folder that already exists receives the temporary folder inside it.

    ``parent`` replaces ``temp_dir`` for code that runs without Gym's configuration, such as a Ray task:
    the server that starts it passes the folder it reads from its own configuration.

    This blocks on filesystem calls; use :func:`async_temp_folder` in a coroutine.
    """
    path = _create_temp_folder(kind, parent)
    try:
        yield path
    finally:
        _finish_temp_folder(path, keep_as)


@asynccontextmanager
async def async_temp_folder(
    *, kind: str, keep_as: Path | None = None, parent: Path | None = None
) -> AsyncIterator[Path]:
    """:func:`temp_folder` for a coroutine: creating and removing the folder run in a worker thread.

    The removal is shielded from cancellation.
    Under an anyio cancel scope, which cancels again at every await, the block can exit before the folder is gone;
    the removal then finishes in the background.
    """
    creating = asyncio.ensure_future(asyncio.to_thread(_create_temp_folder, kind, parent))
    try:
        path = await asyncio.shield(creating)
    except asyncio.CancelledError:
        # The thread may still create the folder after this task is cancelled; remove it once it does.
        creating.add_done_callback(_remove_created_folder)
        raise
    try:
        yield path
    finally:
        # Shielded, so a second cancellation while the folder is being removed does not leave it behind.
        await asyncio.shield(asyncio.to_thread(_finish_temp_folder, path, keep_as))


def _remove_created_folder(creating: asyncio.Future[Path]) -> None:
    if not creating.cancelled() and creating.exception() is None:
        shutil.rmtree(creating.result(), ignore_errors=True)
