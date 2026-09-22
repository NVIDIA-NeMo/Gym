# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Codex inside a Linux task sandbox; confirm descendant cleanup before verification."""

import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from time import monotonic, sleep, time
from typing import Optional, TypedDict


class RunnerInput(TypedDict):
    """One adapter-authored activation, uploaded outside the task repository."""

    directory: str
    command: list[str]
    cwd: str
    env: dict[str, str]
    prompt: str
    timeout: float
    cleanup_timeout: float


class RunnerResult(TypedDict):
    """Atomic evidence that the adapter checks before permitting verification."""

    return_code: int
    timed_out: bool
    cleanup_confirmed: bool
    error: Optional[str]
    hostname: str
    pid: int


def enable_subreaper() -> None:
    """Adopt detached tool processes so they cannot outlive a successful close."""
    if sys.platform != "linux":
        raise RuntimeError("Native Codex sessions require a Linux sandbox")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot establish Codex child-subreaper boundary")


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
            raise TimeoutError("Codex descendants remain alive; verification must not proceed")
        sleep(0.01)


def run(params: RunnerInput) -> RunnerResult:
    """Execute one invocation and write timestamped JSON events without buffering stdout in memory."""
    directory = Path(params["directory"])
    process = None
    reader = None
    error = None
    timed_out = False
    cleanup_confirmed = False
    capture_errors = []
    stopping = False

    def interrupt(*_):
        nonlocal stopping
        # Do not interrupt Popen between process creation and handle assignment.
        stopping = True

    def capture(stream) -> None:
        try:
            with (directory / "events.jsonl").open("w") as events:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except (ValueError, RecursionError):
                        continue
                    if isinstance(event, dict):
                        events.write(json.dumps([time(), event]) + "\n")
                        events.flush()
        except Exception as exc:
            capture_errors.append(str(exc))

    signal.signal(signal.SIGTERM, interrupt)
    (directory / "prompt.txt").write_text(params["prompt"])
    (directory / "events.jsonl").touch()
    with (directory / "stderr.log").open("wb") as stderr, (directory / "prompt.txt").open("rb") as stdin:
        try:
            enable_subreaper()
            process = subprocess.Popen(
                params["command"],
                cwd=params["cwd"],
                env={key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ} | params["env"],
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
            )
            reader = threading.Thread(target=capture, args=(process.stdout,), daemon=True)
            reader.start()
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
                if reader is not None:
                    reader.join(timeout=params["cleanup_timeout"])
                    if reader.is_alive():
                        raise TimeoutError("Codex event capture did not finish")
                cleanup_confirmed = True
            except Exception as exc:
                error = f"cleanup: {exc}"
    if capture_errors:
        error = f"Codex event capture failed: {capture_errors[0]}"
    return {
        "return_code": process.returncode if process and process.returncode is not None else 1,
        "timed_out": timed_out,
        "cleanup_confirmed": cleanup_confirmed,
        "error": error,
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
    }


def main() -> None:
    params = json.loads(Path(sys.argv[1]).read_text())
    result = run(params)
    output = Path(params["directory"]) / "result.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result))
    temporary.replace(output)


if __name__ == "__main__":
    main()
