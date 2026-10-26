# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture Copilot's JSONL events; the supervisor bounds execution; the sandbox owner stops task services."""

import json
import os
import selectors
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from time import time
from typing import Optional


# Streaming duplicates of later complete events. Dropping them bounds what collection downloads.
_STREAMING_EVENTS = frozenset(
    {
        "assistant.message_delta",
        "assistant.reasoning_delta",
        "assistant.tool_call_delta",
        "session.background_tasks_changed",
        "tool.execution_partial_result",
        "tool.shell_output",
    }
)


def run(params: dict) -> int:
    """Run Copilot with stdin closed and flush timestamped events as they arrive."""
    directory = Path(params["directory"])
    (directory / "runtime.json").write_text(json.dumps({"hostname": os.uname().nodename, "pid": os.getpid()}))
    stopping = False

    def interrupt(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as stack:
        stderr = stack.enter_context((directory / "stderr.log").open("wb"))
        events = stack.enter_context((directory / "events.jsonl").open("w"))
        process = subprocess.Popen(
            params["command"],
            cwd=params["cwd"],
            env={**os.environ, **params["env"]},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        try:
            pending = bytearray()

            def capture(line: bytes) -> None:
                try:
                    event = json.loads(line.decode(errors="replace"))
                except (ValueError, RecursionError):
                    return
                if isinstance(event, dict) and event.get("type") not in _STREAMING_EVENTS:
                    events.write(json.dumps([time(), event]) + "\n")
                    events.flush()

            # A detached tool can inherit stdout. Drain available data after Copilot exits,
            # without waiting for that tool to close its pipe.
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
            # An I/O or capture error must not leave the CLI changing the task.
            # Only stop the direct CLI; task services belong to the sandbox owner.
            if process.poll() is None:
                process.kill()
            process.wait()


def read_compaction_state(directory: Path) -> Optional[bool]:
    """Read persisted CLI events, including background compaction absent from stdout and spans."""
    observed = False
    try:
        for path in (directory / "copilot-home/session-state").glob("*/events.jsonl"):
            with path.open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                        return None
                    observed = True
                    if event["type"] in ("session.compaction_start", "session.compaction_complete"):
                        return True
    except (OSError, ValueError, RecursionError):
        return None
    return False if observed else None


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--compaction-state":
        print(json.dumps(read_compaction_state(Path(sys.argv[2]))))
        return
    params = json.loads(Path(sys.argv[1]).read_text())
    raise SystemExit(run(params))


if __name__ == "__main__":
    main()
