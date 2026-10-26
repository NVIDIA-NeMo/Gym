# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture Pi JSON events; the supervisor bounds execution; the sandbox owner stops task services."""

import json
import os
import selectors
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from time import time


OOM_SCORE_ADJ = "/proc/self/oom_score_adj"
# Streaming events repeat the whole partial message (or tool output) on every delta, so their
# volume grows quadratically with a turn's length. The adapter reads only completed messages,
# tool start/end and lifecycle events; the streaming ones are dropped from every copy.
DROPPED_EVENT_TYPES = frozenset({"message_update", "tool_execution_update"})


def prefer_for_oom_kill(path: str = OOM_SCORE_ADJ) -> str:
    """Make this process and its children the memory-cgroup OOM killer's preferred victims.

    Raise oom_score_adj to 1000, or to 0 when the sandbox starts it below zero (a Guaranteed
    pod's -997), so the sandbox never becomes the node's first victim in a global OOM. Raising
    needs no privilege; a failure leaves the score unchanged. Returns what happened.
    """
    try:
        with open(path) as handle:
            before = int(handle.read())
        after = 0 if before < 0 else 1000
        with open(path, "w") as handle:
            handle.write(str(after))
    except (OSError, ValueError) as error:
        return f"unchanged: {error}"
    return f"{before} -> {after}"


def run(params: dict) -> int:
    """Run Pi with isolated input/environment and flush timestamped events as they arrive.

    Optional ``transcript`` and ``stderr`` paths (e.g. under a harness log directory) receive Pi's
    JSON lines, without the streaming events, and its stderr; otherwise stderr stays in the
    session directory.
    """
    directory = Path(params["directory"])
    (directory / "runtime.json").write_text(json.dumps({"hostname": os.uname().nodename, "pid": os.getpid()}))
    (directory / "prompt.txt").write_text(params["prompt"])
    stderr_path = Path(params.get("stderr") or directory / "stderr.log")
    if params.get("raise_oom_score_adj"):
        # Pi and its tools inherit the score; execd and the supervisor keep the sandbox's own.
        with stderr_path.open("a") as log:
            print(f"oom_score_adj: {prefer_for_oom_kill()}", file=log, flush=True)
    stopping = False

    def interrupt(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as stack:
        stderr = stack.enter_context(stderr_path.open("ab"))
        stdin = stack.enter_context((directory / "prompt.txt").open("rb"))
        events = stack.enter_context((directory / "events.jsonl").open("w"))
        transcript = stack.enter_context(open(params["transcript"], "wb")) if params.get("transcript") else None
        process = subprocess.Popen(
            params["command"],
            cwd=params["cwd"],
            env={**os.environ, **params["env"]},
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        try:
            pending = bytearray()

            def capture(line: bytes) -> None:
                try:
                    event = json.loads(line.decode(errors="replace"))
                except (ValueError, RecursionError):
                    event = None
                if isinstance(event, dict) and event.get("type") in DROPPED_EVENT_TYPES:
                    return
                if transcript is not None:
                    transcript.write(line.rstrip(b"\n") + b"\n")
                    transcript.flush()
                if isinstance(event, dict):
                    events.write(json.dumps([time(), event]) + "\n")
                    events.flush()

            # A detached tool can inherit stdout. Drain available data after Pi exits,
            # without waiting for that tool to close its pipe. Resources may retain
            # task services through verification.
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    if stopping and process.poll() is None:
                        process.terminate()
                        stopping = False
                    ready = selector.select(timeout=0.05)
                    if ready:
                        chunk = os.read(process.stdout.fileno(), 64 * 1024)
                        if not chunk:
                            break
                        pending.extend(chunk)
                        while b"\n" in pending:
                            line, _, pending = pending.partition(b"\n")
                            capture(line)
                    elif process.poll() is not None:
                        break
            if pending:
                capture(pending)
            return process.wait()
        finally:
            # An I/O/capture error must not leave the CLI changing the task.
            # Only stop our direct CLI; task services belong to the sandbox owner.
            if process.poll() is None:
                process.kill()
            process.wait()


def main() -> None:
    params = json.loads(Path(sys.argv[1]).read_text())
    raise SystemExit(run(params))


if __name__ == "__main__":
    main()
