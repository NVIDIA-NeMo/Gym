# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One directory per server process for its AF_UNIX sockets, removed when the process exits.

A server with several uvicorn workers coordinates them over sockets: its checkpoint coordinator's,
and each worker's private session-routing socket.
They all live in ``/tmp/ng-<pid>-<random>/``,
under /tmp rather than the configured temporary directory because AF_UNIX paths are limited to about 100 bytes
and cluster temporary directories can be long.

A clean exit removes the directory.
A killed process cannot, so each new directory first removes those left behind:
a directory goes only when its owner process no longer exists and none of its sockets accepts a connection.
The connection check keeps a live server's directory when the process check is wrong,
as it is for a server in another PID namespace that shares /tmp, so a directory is never removed by age.
"""

import atexit
import os
import re
import shutil
import socket
import stat
import uuid
from typing import Optional


RUNTIME_ROOT = "/tmp"
_NAME = re.compile(r"ng-(?P<pid>\d+)-[0-9a-f]{8}")
_CONNECT_TIMEOUT_SECONDS = 0.5

_runtime_dir: Optional[tuple[int, str]] = None


def server_runtime_dir() -> str:
    """This process's runtime directory, created on first use."""
    global _runtime_dir
    pid = os.getpid()
    if _runtime_dir is not None and _runtime_dir[0] == pid:
        return _runtime_dir[1]
    remove_stale_runtime_dirs()
    while True:
        path = os.path.join(RUNTIME_ROOT, f"ng-{pid}-{uuid.uuid4().hex[:8]}")
        try:
            os.mkdir(path, 0o700)
            break
        except FileExistsError:
            continue
    atexit.register(_remove, pid, path)
    _runtime_dir = (pid, path)
    return path


def remove_stale_runtime_dirs(root: Optional[str] = None) -> list[str]:
    """Remove this user's runtime directories whose process is gone and whose sockets are all closed."""
    removed = []
    uid = os.getuid()
    with os.scandir(root or RUNTIME_ROOT) as entries:
        for entry in entries:
            match = _NAME.fullmatch(entry.name)
            if match is None:
                continue
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid:
                continue
            if _process_exists(int(match["pid"])) or _any_socket_accepts(entry.path):
                continue
            shutil.rmtree(entry.path, ignore_errors=True)
            removed.append(entry.path)
    return removed


def _remove(pid: int, path: str) -> None:
    # A forked child inherits the exit handler; only the process that created the directory removes it.
    if os.getpid() == pid:
        shutil.rmtree(path, ignore_errors=True)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _any_socket_accepts(directory: str) -> bool:
    try:
        with os.scandir(directory) as entries:
            paths = [entry.path for entry in entries if stat.S_ISSOCK(entry.stat(follow_symlinks=False).st_mode)]
    except OSError:
        # Unreadable, or changing under us: keep it.
        return True
    for path in paths:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(_CONNECT_TIMEOUT_SECONDS)
            try:
                probe.connect(path)
            except (ConnectionRefusedError, FileNotFoundError):
                continue
            except OSError:
                # A timeout or any other failure may be a live server too busy to accept: keep it.
                return True
            return True
    return False
