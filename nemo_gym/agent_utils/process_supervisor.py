# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Linux process supervisor to upload beside a sandboxed harness.

Uses only the standard library; Gym need not be installed in the task sandbox.
The receipt confirms termination independently of the harness result. On forced
stop, newly adopted children are also drained, including tools in other sessions.
Detached services adopted before stopping remain for verification. Natural exit
only drains the harness group: orphaned services and unfinished detached tools
cannot be distinguished without cooperation from the harness.
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
CLEANUP_PHASE_COUNT = 3


class CleanupReceipt(TypedDict):
    """Outcome after bounded harness termination.

    ``cleanup_confirmed`` means no live harness-group members remain. Forced
    stops also drain newly adopted children. Previously adopted detached services
    remain until the owner closes the sandbox; this is not a sandbox-empty check.
    """

    cleanup_confirmed: bool
    return_code: int | None
    error: str | None
    timed_out: bool


def exec_timeout(*, timeout: float, cleanup_timeout: float = DEFAULT_CLEANUP_TIMEOUT) -> float:
    """Reserve TERM grace, direct-process reaping, and process-group reaping.

    The extra 30 seconds cover startup, polling, and receipt I/O beyond those
    bounded cleanup phases. Keep their count shared with the controller budget.
    """
    return timeout + CLEANUP_PHASE_COUNT * cleanup_timeout + 30


def _child_pids() -> set[int]:
    """Read this single-threaded subreaper's current children."""
    return {int(pid) for pid in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()}


def _group_has_live_processes(pgid: int) -> bool:
    # killpg(..., 0) also succeeds for zombies parented outside this subreaper.
    # They cannot execute or write, and their owner alone can reap them. Any
    # unreadable/invalid proc state propagates instead of confirming termination.
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
            return True
    return False


def _reap_process_group(pgid: int, timeout: float, *, preserved_children: set[int] | None = None) -> None:
    """Drain the group and, on forced stop, newly adopted tools within one budget."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        group_children_pending = True
        try:
            while os.waitpid(-pgid, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            group_children_pending = False
        pending_children = set()
        if preserved_children is not None:
            # Killing one generation can orphan more children in other groups.
            # Re-read each round rather than taking only one post-stop snapshot.
            for pid in _child_pids() - preserved_children:
                try:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, os.WNOHANG)
                except ProcessLookupError:
                    pass
            pending_children = _child_pids() - preserved_children
        try:
            os.killpg(pgid, 0)
            group_alive = _group_has_live_processes(pgid)
        except ProcessLookupError:
            group_alive = False
        if not group_alive and not group_children_pending and not pending_children:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Harness process group or newly adopted tools have not terminated")
        time.sleep(0.01)


def _supervise(
    command: list[str],
    *,
    timeout: float,
    cleanup_timeout: float = DEFAULT_CLEANUP_TIMEOUT,
    stop_path: Path | None = None,
) -> CleanupReceipt:
    """Stop the harness and forced-stop tools; preserve earlier detached services."""
    process = None
    launch_attempted = False
    stopping = False
    forced_stop = False
    preserved_children = None
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
            return receipt  # The finally block confirms that no harness process was launched.
        if sys.platform != "linux":
            raise RuntimeError("Sandbox process supervision requires Linux")
        # Leave the provider group when possible. An exec launcher may already make us
        # its group leader, so setsid cannot protect us from that group's SIGKILL.
        # Callers must confirm cleanup before cancelling provider exec, in either case.
        if os.getpgrp() != os.getpid():
            os.setsid()
        # Reap killed group members ourselves instead of depending on the
        # container's PID 1. Otherwise zombies keep killpg(..., 0) succeeding.
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "Cannot supervise harness process group")
        launch_attempted = True
        process = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + timeout
        while process.poll() is None and not stopping and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            forced_stop = True
            receipt["timed_out"] = not stopping
            # Completed tool calls can leave services already adopted by us.
            # Preserve those, but drain children orphaned by this forced stop.
            preserved_children = _child_pids() - {process.pid}
            # Let the harness checkpoint before the bounded hard cleanup.
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=cleanup_timeout)
            except subprocess.TimeoutExpired:
                pass
    except Exception as exc:
        receipt["error"] = f"Harness launch failed: {exc}" if process is None and launch_attempted else str(exc)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if process is not None:
                # Any services still in the harness group stop here too. Only
                # previously adopted children outside that group are preserved
                # on forced stop; audit service-launching task recipes.
                # Natural exit has no pre-stop boundary for detached processes.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                receipt["return_code"] = process.wait(timeout=cleanup_timeout)
                _reap_process_group(process.pid, cleanup_timeout, preserved_children=preserved_children)
                if forced_stop and preserved_children is None:
                    raise RuntimeError("Cannot identify pre-stop adopted children")
                receipt["cleanup_confirmed"] = True
            else:
                # Popen can fail after spawning but before returning a handle.
                receipt["cleanup_confirmed"] = not launch_attempted
        except Exception as exc:
            receipt["error"] = f"Harness termination was not confirmed: {exc}"
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
        parser.error("a harness command is required after --")
    receipt = _supervise(
        command,
        timeout=args.timeout,
        cleanup_timeout=args.cleanup_timeout,
        stop_path=args.stop_file,
    )
    temporary = args.receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt))
    temporary.replace(args.receipt)
    return 0 if receipt["cleanup_confirmed"] and receipt["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
