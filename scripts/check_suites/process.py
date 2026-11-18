# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a worker subprocess with a timeout and clean up everything it started."""

import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import psutil


def run_process(
    command: list[str], *, cwd: Path, env: Mapping[str, str], log_path: Path, timeout: float
) -> dict[str, object]:
    """Keep logs and reap the worker and its descendants, including new process groups."""
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        tracked = {}
        timed_out = False
        try:
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                # CLIs may start their own sessions, so killing only our process group
                # would leave their children alive after a worker timeout.
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        tracked[(child.pid, child.create_time())] = child
                except psutil.NoSuchProcess:
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    proc.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                proc.terminate()
            for child in reversed(list(tracked.values())):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(list(tracked.values()), timeout=2)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return {"returncode": proc.returncode, "timed_out": timed_out}
