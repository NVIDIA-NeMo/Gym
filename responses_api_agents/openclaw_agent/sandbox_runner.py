# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supervise one OpenClaw invocation and reap all Linux tool descendants."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep
from typing import TypedDict


class RunnerRequest(TypedDict):
    """Invocation data uploaded by the adapter."""

    directory: str
    prompt: str
    command: list[str]
    cwd: str
    env: dict[str, str]
    timeout: float
    cleanup_timeout: float


class RunnerResult(TypedDict):
    """Receipt proving whether verification can safely inspect the task."""

    return_code: int
    timed_out: bool
    cleanup_confirmed: bool
    error: str | None
    hostname: str
    pid: int


def enable_subreaper() -> None:
    """Adopt detached children that escape the original process group."""
    if sys.platform != "linux":
        raise RuntimeError("Native OpenClaw sessions require Linux")
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot establish OpenClaw child-subreaper boundary")


def drain_children(timeout: float) -> None:
    """Kill and reap descendants before acknowledging cleanup."""
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
            raise TimeoutError("OpenClaw descendants remain alive; verification must not proceed")
        sleep(0.01)


def run(params: RunnerRequest) -> RunnerResult:
    """Write stdout incrementally; the transcript remains available after failures."""
    directory = Path(params["directory"])
    process = None
    error = None
    timed_out = False
    cleanup_confirmed = False
    stopping = False

    def interrupt(_signum: int, _frame: object) -> None:
        nonlocal stopping
        # Signals must not interrupt Popen between child creation and handle assignment.
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    (directory / "prompt.txt").write_text(params["prompt"])
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        try:
            enable_subreaper()
            process = subprocess.Popen(
                params["command"],
                cwd=params["cwd"],
                env=dict(os.environ) | params["env"],
                stdin=subprocess.DEVNULL,
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
                drain_children(params["cleanup_timeout"])
                cleanup_confirmed = True
            except Exception as exc:
                error = f"cleanup: {exc}"
    return {
        "return_code": process.returncode if process is not None and process.returncode is not None else 1,
        "timed_out": timed_out,
        "cleanup_confirmed": cleanup_confirmed,
        "error": error,
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
    }


def main() -> None:
    """Atomically publish the cleanup receipt after all descendants have exited."""
    params = json.loads(Path(sys.argv[1]).read_text())
    result = run(params)
    output = Path(params["directory"]) / "result.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result))
    temporary.replace(output)


if __name__ == "__main__":
    main()
