# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture Claude stdout independently of stderr and supervisor diagnostics."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def run(params: dict) -> int:
    """Keep raw stream-json and partial output; the shared supervisor owns process cleanup."""
    directory = Path(params["directory"])
    (directory / "runtime.json").write_text(json.dumps({"hostname": os.uname().nodename, "pid": os.getpid()}))
    stopping = False

    def interrupt(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    with (directory / "events.jsonl").open("wb") as events, (directory / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(
            params["command"],
            cwd=params["cwd"],
            env={**os.environ, **params["env"]},
            stdin=subprocess.DEVNULL,
            stdout=events,
            stderr=stderr,
        )
        try:
            # Regular files avoid waiting for detached task services to close inherited pipes.
            while True:
                if stopping and process.poll() is None:
                    process.terminate()
                    stopping = False
                try:
                    return process.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


if __name__ == "__main__":
    raise SystemExit(run(json.loads(Path(sys.argv[1]).read_text())))
