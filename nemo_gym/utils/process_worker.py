# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one command in a Linux child subreaper and confirm cleanup before reporting its outcome.

This file uses only the standard library so a launcher can embed it in a job whose host has no Gym install.
The dedicated worker owns every child it starts, including descendants that create another session. An optional
cleanup command releases resources outside the local process tree after local children have stopped.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _children() -> list[int]:
    """Only signal direct children: they cannot reuse a PID until this single-threaded worker reaps them.

    Killing a parent adopts its descendants here, including children launched from another thread or session.
    """
    pid = os.getpid()
    return [int(child) for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]


def _signal_children(signum: int) -> None:
    for pid in _children():
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def _reap() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _drain(grace_seconds: float) -> None:
    deadline = time.monotonic() + grace_seconds
    while True:
        _reap()
        if not _children():
            return
        if time.monotonic() >= deadline + grace_seconds:
            raise TimeoutError("Local descendant termination was not confirmed")
        _signal_children(signal.SIGKILL if time.monotonic() >= deadline else signal.SIGTERM)
        time.sleep(0.05)


def run(command: list[str], *, receipt: Path, grace_seconds: float, cleanup_command: str | None = None) -> int:
    """Execute once, drain descendants and external resources, then atomically finish a fresh receipt.

    Call only in a dedicated process: becoming a subreaper and reaping children are process-wide operations.
    No retry or recovery environment is set. SIGKILL of the worker itself remains the enclosing scheduler's cleanup duty.
    """
    if sys.platform != "linux":
        raise RuntimeError("Driver supervision requires Linux")
    if not math.isfinite(grace_seconds) or grace_seconds <= 0:
        raise ValueError("grace_seconds must be finite and positive")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot become a child subreaper")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    value: dict[str, object] = {"state": "running", "started_at": started_at, "cleanup_confirmed": False}
    with receipt.open("x") as stream:
        json.dump(value, stream)
        stream.write("\n")
    stopped: int | None = None
    stop_at: float | None = None

    def stop(signum: int, _frame: object) -> None:
        nonlocal stopped, stop_at
        if stopped is None:
            stopped, stop_at = signum, time.monotonic()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    code = 127
    error: str | None = None
    try:
        try:
            if stopped is None:
                child = subprocess.Popen(command, start_new_session=True)
                while child.poll() is None:
                    if stop_at is not None:
                        if time.monotonic() - stop_at >= 2 * grace_seconds:
                            raise TimeoutError("Command termination was not confirmed")
                        _signal_children(signal.SIGKILL if time.monotonic() - stop_at >= grace_seconds else stopped)
                    time.sleep(0.05)
                code = child.wait()
        except OSError as exc:
            error = str(exc)
        finally:
            # An exited shell can leave children in other sessions. Drain them before any external cleanup.
            cleanup_errors: list[str] = []
            try:
                _drain(grace_seconds)
            except OSError as exc:
                cleanup_errors.append(str(exc))
            if cleanup_command is not None:
                try:
                    subprocess.run(["bash", "-c", cleanup_command], check=True, timeout=4 * grace_seconds)
                except (OSError, subprocess.SubprocessError) as exc:
                    cleanup_errors.append(str(exc))
            # A timed-out cleanup command can itself leave local children.
            try:
                _drain(grace_seconds)
            except OSError as exc:
                cleanup_errors.append(str(exc))
            value["cleanup_confirmed"] = not cleanup_errors
            if cleanup_errors:
                error = "Cleanup failed: " + "; ".join(cleanup_errors)
        result = 70 if cleanup_errors else 128 + stopped if stopped is not None else code
        if result < 0:
            result = 128 - result
        value.update(
            state=(
                "cleanup_failed"
                if not value["cleanup_confirmed"]
                else "cancelled"
                if stopped
                else "complete"
                if result == 0
                else "failed"
            ),
            finished_at=time.time(),
            exit_code=result,
            command_exit_code=code,
            signal=stopped,
        )
        if error is not None:
            value["error"] = error
        temporary = receipt.with_suffix(receipt.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(receipt)
        return result
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--grace-seconds", type=float, default=10)
    parser.add_argument("--cleanup-command")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    return run(command, receipt=args.receipt, grace_seconds=args.grace_seconds, cleanup_command=args.cleanup_command)


if __name__ == "__main__":
    raise SystemExit(main())
