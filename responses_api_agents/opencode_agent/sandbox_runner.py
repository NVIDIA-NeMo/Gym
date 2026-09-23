# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run OpenCode inside a Linux task sandbox; confirm descendant cleanup before verification."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep
from typing import TypedDict


class RunnerInput(TypedDict):
    """Adapter-owned settings for one supervised harness invocation."""

    directory: str
    command: list[str]
    cwd: str
    env: dict[str, str]
    prompt: str
    timeout: float
    cleanup_timeout: float


class RunnerResult(TypedDict):
    """Completion and descendant-cleanup evidence consumed by session close."""

    return_code: int
    timed_out: bool
    cleanup_confirmed: bool
    error: str | None
    hostname: str
    pid: int


def enable_subreaper() -> None:
    """Adopt detached tool processes so they cannot outlive a successful close."""
    if sys.platform != "linux":
        raise RuntimeError("Native OpenCode sessions require a Linux sandbox")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot establish OpenCode child-subreaper boundary")


def drain_children(timeout: float) -> None:
    """Kill and reap descendants, including double-forked terminal commands."""
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    deadline = monotonic() + timeout
    while True:
        for child in children.read_text().split():
            try:
                os.kill(int(child), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return
        if monotonic() >= deadline:
            raise TimeoutError("OpenCode descendants remain alive; verification must not proceed")
        sleep(0.01)


def run(params: RunnerInput) -> RunnerResult:
    """Execute one invocation and preserve stdout, SQLite state, and cleanup evidence."""
    directory = Path(params["directory"])
    process = None
    error = None
    timed_out = False
    cleanup_confirmed = False
    stopping = False

    def interrupt(*_: object) -> None:
        nonlocal stopping
        # Do not interrupt Popen between process creation and handle assignment.
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    for key in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        if key in params["env"]:
            Path(params["env"][key]).mkdir(parents=True, exist_ok=True)
    (directory / "prompt.txt").write_text(params["prompt"])
    with (
        (directory / "stderr.log").open("wb") as stderr,
        (directory / "stdout.jsonl").open("wb") as stdout,
        (directory / "prompt.txt").open("rb") as stdin,
    ):
        try:
            enable_subreaper()
            process = subprocess.Popen(
                params["command"],
                cwd=params["cwd"],
                env=dict(os.environ) | params["env"],
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            deadline = monotonic() + params["timeout"]
            while process.poll() is None:
                if stopping or monotonic() >= deadline:
                    timed_out = True
                    break
                sleep(0.05)
        except Exception as exc:
            error = str(exc)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=params["cleanup_timeout"])
                # Popen may create a child and then raise before returning its handle.
                drain_children(params["cleanup_timeout"])
                cleanup_confirmed = True
            except Exception as exc:
                error = f"cleanup: {exc}"
    # Snapshot WAL-backed state only after all writers and tools have exited.
    try:
        snapshot(directory)
    except Exception as exc:
        error = error or f"OpenCode transcript capture failed: {exc}"
        (directory / "export.json").write_text("{}")
    return {
        "return_code": process.returncode if process else 1,
        "timed_out": timed_out,
        "cleanup_confirmed": cleanup_confirmed,
        "error": error,
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
    }


def snapshot(directory: Path) -> None:
    """Export root messages and retain the full session tree for observations."""
    database = directory / "data/opencode/opencode.db"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source:
        with sqlite3.connect(directory / "observations.db") as destination:
            source.backup(destination)
        source.row_factory = sqlite3.Row
        sessions = source.execute(
            "select id from session where parent_id is null order by time_created, id"
        ).fetchall()
        if len(sessions) != 1:
            raise RuntimeError(f"Expected one OpenCode root session, found {len(sessions)}")
        rows = source.execute(
            "select id, data from message where session_id=? order by time_created, id", (sessions[0]["id"],)
        ).fetchall()
        messages = []
        for row in rows:
            info = json.loads(row["data"])
            parts = [
                json.loads(part[0])
                for part in source.execute(
                    "select data from part where message_id=? order by time_created, id", (row["id"],)
                )
            ]
            messages.append({"info": info, "parts": parts})
        usage_messages = [json.loads(row[0]) for row in source.execute("select data from message")]
        (directory / "export.json").write_text(json.dumps({"messages": messages, "usage_messages": usage_messages}))


def main() -> None:
    params = json.loads(Path(sys.argv[1]).read_text())
    result = run(params)
    output = Path(params["directory"]) / "result.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result))
    temporary.replace(output)


if __name__ == "__main__":
    main()
