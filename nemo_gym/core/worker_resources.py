# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""What a live worker needs to find the host resources a dead peer left: its workspaces and child process groups.

Every worker process has one routing ID.
With several workers, session routing sets it (see ``install_session_routing``).
A process that never installs routing gets one generated on first use.
The ID groups everything the worker creates on the host:

- **Workspaces** live under ``<root>/<routing ID>/``.
  The root is ``NEMO_GYM_WORKSPACE_ROOT``, or a per-user directory in the system temporary directory.
  The worker directory is private to its user (mode 0700), and a worker that exits removes it once it is empty.
- **Child process groups** are recorded under ``<root>/<routing ID>/.process-groups/``.
  Each group has one file, holding the group ID and its leader's start time in clock ticks since boot.
  A child keeps its own process group, so a timeout still kills only that command's processes.
  The record is what ties the group to the worker.
  A reader checks the record against ``/proc`` before it signals the group, so a reused process ID is never killed.

A record stays while its group runs, including processes a command left running in the background.
Whoever owns the group, such as a local sandbox at its close, ends it with ``ProcessGroupRecord.end``.
Ending the groups of a worker that died is the reap's job.
Process group records need Linux's ``/proc``; elsewhere children run unrecorded.
"""

import atexit
import logging
import os
import secrets
import signal
import stat
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path


LOGGER = logging.getLogger(__name__)

WORKSPACE_ROOT_ENV_VAR = "NEMO_GYM_WORKSPACE_ROOT"
PROCESS_GROUPS_DIR_NAME = ".process-groups"

# (process ID, routing ID, whether the ID was handed out): a forked child is a different worker and gets its own ID.
_routing_id: tuple[int, str, bool] | None = None
# The process that registered removal of its worker directory at exit.
_cleanup_registered_pid: int | None = None


def worker_routing_id() -> str:
    """This worker's routing ID: the one session routing set, else one generated once for this process."""
    global _routing_id
    pid = os.getpid()
    if _routing_id is None or _routing_id[0] != pid:
        _routing_id = (pid, uuid.uuid4().hex, True)
    elif not _routing_id[2]:
        _routing_id = (pid, _routing_id[1], True)
    return _routing_id[1]


def set_worker_routing_id(routing_id: str) -> None:
    """Make ``routing_id`` this process's routing ID.

    Session routing calls it before the worker creates anything.
    Anything already created under an ID that was handed out stays filed under that ID, which no peer knows.
    Replacing such an ID is therefore logged.
    """
    global _routing_id
    if not routing_id or os.sep in routing_id or routing_id in (os.curdir, os.pardir):
        raise ValueError(f"A routing ID must be a non-empty path component, got {routing_id!r}")
    pid = os.getpid()
    if _routing_id is not None and _routing_id[0] == pid and _routing_id[2] and _routing_id[1] != routing_id:
        LOGGER.warning(
            "Routing ID %s replaces %s, which this worker already used; what it created under %s cannot be reaped",
            routing_id,
            _routing_id[1],
            _routing_id[1],
        )
    _routing_id = (pid, routing_id, False)


def workspace_root() -> Path:
    """The directory that holds one directory per worker.

    It is ``NEMO_GYM_WORKSPACE_ROOT``, else a per-user directory in the system temporary directory.
    The default is per user, so two users on one node never share one directory or fight over its mode.
    The root must be on a node-local filesystem.
    Process group records name processes of this node, and each local command writes and removes one in the event loop.
    """
    configured = os.environ.get(WORKSPACE_ROOT_ENV_VAR, "").strip()
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / f"nemo-gym-workspaces-{os.getuid()}"


def worker_workspace_dir() -> Path:
    """``<root>/<routing ID>/`` for this worker, created private to this user when missing.

    Raises:
        PermissionError: The default root or the worker directory is a symbolic link, belongs to another user,
            or is writable by others.
            Someone else could then read or replace this worker's files.
    """
    root = workspace_root()
    if os.environ.get(WORKSPACE_ROOT_ENV_VAR, "").strip():
        # An operator's root may be shared by several users, for example with mode 1777.
        # Only the worker directory is ours.
        root.mkdir(parents=True, exist_ok=True)
    else:
        _private_dir(root)
    path = _private_dir(root / worker_routing_id())
    _register_cleanup(path)
    return path


def worker_mkdtemp(*, prefix: str | None = None, suffix: str | None = None) -> str:
    """``tempfile.mkdtemp`` under this worker's workspace directory, so a peer can remove it if the worker dies."""
    return tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=worker_workspace_dir())


@dataclass(frozen=True)
class ProcessGroupRecord:
    """A recorded child process group: its ID, which is its leader's process ID, and the leader's start time.

    ``start_ticks`` is field 22 of ``/proc/<pid>/stat``: clock ticks since boot.
    A change of the wall clock does not move it.
    """

    pgid: int
    start_ticks: int
    # The record's file, removed by ``forget``; ``None`` for a record that was never written.
    path: Path | None = field(default=None, compare=False)

    def is_running(self) -> bool:
        """Whether the recorded group still has a process.

        A leader that still runs must have the recorded start time; one that differs is a reused process ID.
        Once the leader has exited, its members keep the group.
        Linux does not hand out a process ID still in use as a group ID, so a process in group ``pgid`` is one of ours.
        """
        leader_ticks = _start_ticks(self.pgid)
        if leader_ticks is not None and leader_ticks != self.start_ticks:
            return False
        return _group_has_members(self.pgid)

    def end(self) -> None:
        """Kill the group if it still runs, then remove its record."""
        if self.is_running():
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.forget()

    def forget(self) -> None:
        """Remove the record, leaving the group as it is; logs a record it cannot remove."""
        if self.path is None:
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError as error:
            LOGGER.warning("Could not remove the record of child process group %d: %r", self.pgid, error)


def record_process_group(pid: int) -> ProcessGroupRecord | None:
    """Record the process group that child ``pid`` leads under this worker.

    Call it right after starting the child with ``start_new_session=True``, before waiting for it.
    The caller owns the record: once the group has no process left it calls ``forget``, and to stop the group it calls
    ``end``.
    Each record has its own file, so a reused process ID never removes another command's record.
    It logs and returns ``None`` when the child is gone, ``/proc`` is unavailable or the record cannot be written.
    The child then runs unrecorded.
    """
    ticks = _start_ticks(pid)
    if ticks is None:
        LOGGER.warning("Could not record child process group %d of this worker: its start time is unavailable", pid)
        return None
    try:
        records = _private_dir(worker_workspace_dir() / PROCESS_GROUPS_DIR_NAME)
        path = records / f"{pid}-{secrets.token_hex(4)}"
        path.write_text(f"{pid} {ticks}", encoding="ascii")
    except OSError as error:
        LOGGER.warning("Could not record child process group %d of this worker: %r", pid, error)
        return None
    return ProcessGroupRecord(pgid=pid, start_ticks=ticks, path=path)


def recorded_process_groups(routing_id: str) -> list[ProcessGroupRecord]:
    """The process groups recorded under ``routing_id``; a record that cannot be read is skipped."""
    records: list[ProcessGroupRecord] = []
    try:
        entries = sorted((workspace_root() / routing_id / PROCESS_GROUPS_DIR_NAME).iterdir())
    except FileNotFoundError:
        return records
    for entry in entries:
        try:
            pgid, ticks = entry.read_text("ascii").split()
            records.append(ProcessGroupRecord(pgid=int(pgid), start_ticks=int(ticks), path=entry))
        except (OSError, ValueError):
            LOGGER.warning("Skipping unreadable process group record %s", entry)
    return records


def _private_dir(path: Path) -> Path:
    """``path`` as a directory only this user can write, created with mode 0700 when missing."""
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise PermissionError(
            f"{path} must be a directory owned by this user and writable by nobody else; "
            f"remove it or set {WORKSPACE_ROOT_ENV_VAR} to a private directory"
        )
    return path


def _register_cleanup(path: Path) -> None:
    global _cleanup_registered_pid
    if _cleanup_registered_pid == os.getpid():
        return
    _cleanup_registered_pid = os.getpid()
    atexit.register(_remove_if_empty, path)


def _remove_if_empty(path: Path) -> None:
    """At a clean exit, remove the worker directory unless something in it is still owned by its creator."""
    for directory in (path / PROCESS_GROUPS_DIR_NAME, path):
        try:
            directory.rmdir()
        except OSError:
            pass


def _start_ticks(pid: int) -> int | None:
    """Field 22 of ``/proc/<pid>/stat``, the start time in clock ticks since boot; ``None`` when unavailable."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as stat_file:
            fields = stat_file.read().rpartition(")")[2].split()
        # The fields after the command name start at field 3; field 22 is the start time.
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _group_has_members(pgid: int) -> bool:
    """Whether any process is in group ``pgid``, asked with one signal-0 system call."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A process in the group exists but belongs to another user.
        return True
    return True
