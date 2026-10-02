# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Linux process supervisor to upload beside a sandboxed harness.

Uses only the standard library; Gym need not be installed in the task sandbox.
The receipt confirms descendant cleanup independently of the worker's result.
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
from typing import TypedDict


DEFAULT_CLEANUP_TIMEOUT = 10.0


class CleanupReceipt(TypedDict):
    """Outcome written only after supervision and bounded cleanup finish."""

    cleanup_confirmed: bool
    return_code: int | None
    error: str | None
    timed_out: bool


def exec_timeout(*, timeout: float, cleanup_timeout: float = DEFAULT_CLEANUP_TIMEOUT) -> float:
    """Leave room for TERM grace, worker reaping, descendant draining, and receipt I/O."""
    return timeout + 3 * cleanup_timeout + 30


def _drain_children(timeout: float) -> None:
    """Reap tool descendants, including processes that detached with setsid()."""
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    deadline = time.monotonic() + timeout
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
        if time.monotonic() >= deadline:
            raise TimeoutError("Sandbox tool processes remain alive")
        time.sleep(0.01)


def _supervise(
    command: list[str],
    *,
    timeout: float,
    cleanup_timeout: float = DEFAULT_CLEANUP_TIMEOUT,
    stop_path: Path | None = None,
) -> CleanupReceipt:
    """Enforce a worker deadline, then acknowledge cleanup after all descendants exit."""
    process = None
    subreaping = False
    stopping = False
    receipt: CleanupReceipt = {"cleanup_confirmed": False, "return_code": None, "error": None, "timed_out": False}

    def interrupt(*_: object) -> None:
        nonlocal stopping
        # A signal between Popen and handle assignment must not lose the child.
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    try:
        # TERM can be ignored during interpreter startup. A durable stop marker fences
        # that window; later signals are handled without interrupting Popen.
        if stopping or (stop_path is not None and stop_path.exists()):
            return receipt  # The finally block confirms that no worker was launched.
        if sys.platform != "linux":
            raise RuntimeError("Sandbox process supervision requires Linux")
        # Leave the provider group when possible. An exec launcher may already make us
        # its group leader, so setsid cannot protect us from that group's SIGKILL.
        # Callers must confirm cleanup before cancelling provider exec, in either case.
        if os.getpgrp() != os.getpid():
            os.setsid()
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "Cannot supervise sandbox tool processes")
        subreaping = True
        process = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + timeout
        while process.poll() is None and not stopping and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            receipt["timed_out"] = not stopping
            # Let the worker checkpoint before the bounded hard cleanup.
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=cleanup_timeout)
            except subprocess.TimeoutExpired:
                pass
    except Exception as exc:
        receipt["error"] = str(exc)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                receipt["return_code"] = process.wait(timeout=cleanup_timeout)
            if subreaping:
                # Popen can fail after creating a child, before returning its handle.
                _drain_children(cleanup_timeout)
            receipt["cleanup_confirmed"] = True
        except Exception as exc:
            receipt["error"] = f"cleanup: {exc}"
    return receipt


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return seconds


def main() -> int:
    """Run COMMAND and atomically write its cleanup receipt outside the task workdir."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=_positive_seconds, required=True)
    parser.add_argument("--cleanup-timeout", type=_positive_seconds, default=DEFAULT_CLEANUP_TIMEOUT)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a worker command is required after --")
    receipt = _supervise(command, timeout=args.timeout, cleanup_timeout=args.cleanup_timeout, stop_path=args.stop_file)
    temporary = args.receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt))
    temporary.replace(args.receipt)
    return 0 if receipt["cleanup_confirmed"] and receipt["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
