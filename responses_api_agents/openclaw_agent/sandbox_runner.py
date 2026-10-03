# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture OpenClaw output; the shared process supervisor owns deadlines and cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TypedDict


class RunnerInput(TypedDict):
    """Harness input, independent of supervisor cleanup and runtime metadata."""

    directory: str
    prompt: str
    command: list[str]
    cwd: str
    env: dict[str, str]


def run(params: RunnerInput) -> int:
    """Run the harness with isolated files and let the supervisor reap descendants."""
    directory = Path(params["directory"])
    (directory / "runtime.json").write_text(json.dumps({"hostname": os.uname().nodename, "pid": os.getpid()}))
    (directory / "prompt.txt").write_text(params["prompt"])
    stopping = False

    def interrupt(*_: object) -> None:
        nonlocal stopping
        # Defer TERM until Popen returns so a signal cannot lose the child handle.
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as stack:
        stdout = stack.enter_context((directory / "stdout.log").open("wb"))
        stderr = stack.enter_context((directory / "stderr.log").open("wb"))
        stdin = subprocess.DEVNULL
        process = subprocess.Popen(
            params["command"],
            cwd=params["cwd"],
            env={**os.environ, **params["env"]},
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
        while True:
            if stopping:
                process.terminate()
                stopping = False
            try:
                return process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass


def main() -> None:
    params = json.loads(Path(sys.argv[1]).read_text())
    raise SystemExit(run(params))


if __name__ == "__main__":
    main()
