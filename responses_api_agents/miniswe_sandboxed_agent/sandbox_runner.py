# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervise the unmodified mini-SWE CLI and its detached tool descendants."""

import ctypes
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


def descendants(parent: int) -> set[int]:
    """Include adopted orphans and children that started new sessions/groups."""
    parents = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                parents[int(entry.name)] = int(fields[1])
            except (FileNotFoundError, ProcessLookupError):
                pass
    found = set()
    frontier = {parent}
    while frontier:
        frontier = {pid for pid, ppid in parents.items() if ppid in frontier} - found
        found.update(frontier)
    return found


def reap() -> bool:
    """Return true only when the kernel confirms no owned children remain."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        if pid == 0:
            return False


def supervise(directory: Path, command: str | None = None) -> None:
    """One Linux subreaper per activation, owning even detached tool descendants."""
    if platform.system() != "Linux" or ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("mini-SWE requires Linux PR_SET_CHILD_SUBREAPER for verified cleanup")
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    worker = None
    client = directory / "mcp"
    if (client / "servers.json").exists() and not (directory / "stop").exists():
        subprocess.Popen([str(client / "bin/python"), str(client / "client.py"), "serve"])
        deadline = time.monotonic() + 60
        while not (client / "server.sock").exists():
            if time.monotonic() > deadline:
                stop_requested = True
                break
            time.sleep(0.1)
    if not stop_requested and not (directory / "stop").exists():
        worker = (
            subprocess.Popen(["bash", "-c", command])
            if command
            else subprocess.Popen([sys.executable, __file__, str(directory), "--worker"])
        )
    while worker is not None and worker.poll() is None and not stop_requested and not (directory / "stop").exists():
        time.sleep(0.1)
    started = time.monotonic()
    terminated = set()
    remaining = descendants(os.getpid())
    no_children = reap()
    while (remaining or not no_children) and time.monotonic() - started < 8:
        sig = signal.SIGTERM if time.monotonic() - started < 1 else signal.SIGKILL
        for pid in remaining:
            try:
                os.kill(pid, sig)
                terminated.add(pid)
            except ProcessLookupError:
                pass
        time.sleep(0.1)
        if worker is not None:
            worker.poll()
        no_children = reap()
        remaining = descendants(os.getpid())
    write_json(
        directory / "cleanup.json",
        {
            "status": "stopped" if not remaining and no_children else "failed",
            "remaining_pids": sorted(remaining),
            "terminated_pids": sorted(terminated),
            "supervisor_pid": os.getpid(),
            "worker_pid": worker.pid if worker is not None else None,
            "duration_ms": (time.monotonic() - started) * 1000,
        },
    )


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    write_json(
        directory / "runtime.json",
        {"hostname": platform.node(), "pid": os.getpid(), "user": os.getuid(), "python": sys.executable},
    )
    supervise(directory, sys.argv[2])
