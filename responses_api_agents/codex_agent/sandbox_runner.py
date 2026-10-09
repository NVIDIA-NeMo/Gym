# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture Codex JSON events; the shared process supervisor owns deadlines and cleanup."""

import json
import os
import selectors
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from time import time


def run(params: dict) -> int:
    """Run Codex with isolated input/environment and flush timestamped events as they arrive."""
    directory = Path(params["directory"])
    (directory / "runtime.json").write_text(json.dumps({"hostname": os.uname().nodename, "pid": os.getpid()}))
    (directory / "prompt.txt").write_text(params["prompt"])
    stopping = False

    def interrupt(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as stack:
        stderr = stack.enter_context((directory / "stderr.log").open("wb"))
        stdin = stack.enter_context((directory / "prompt.txt").open("rb"))
        events = stack.enter_context((directory / "events.jsonl").open("w"))
        process = subprocess.Popen(
            params["command"],
            cwd=params["cwd"],
            env={**{key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}, **params["env"]},
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        pending = bytearray()

        def capture(line: bytes) -> None:
            try:
                event = json.loads(line.decode(errors="replace"))
            except (ValueError, RecursionError):
                return
            if isinstance(event, dict):
                events.write(json.dumps([time(), event]) + "\n")
                events.flush()

        # A detached tool can inherit stdout. Drain available data after Codex exits,
        # without waiting for that tool to close its pipe. Resources may retain
        # task services through verification after a successful harness exit.
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


def main() -> None:
    params = json.loads(Path(sys.argv[1]).read_text())
    raise SystemExit(run(params))


if __name__ == "__main__":
    main()
