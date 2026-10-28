# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a persistent IPython kernel inside a GDP.pdf python-tool sandbox.

The resources server places this file in ``<workdir>/.ipython_kernel/`` and runs it there with the sandbox's
``python3``, which needs ``ipykernel`` and ``jupyter_client`` (nothing from Gym):

- ``start`` launches a detached kernel whose working directory is ``<workdir>``, and waits until it answers.
- ``run`` executes ``cell.py`` and prints the cell's text output (streams, the last expression's value, tracebacks),
  keeping at most ``--max-chars`` of it. A cell still running after ``--timeout`` seconds is interrupted. A kernel that
  has exited, or that ignores the interrupt, is restarted, and the output says so.
- ``stop`` kills the kernel and anything it started.
"""

import argparse
import contextlib
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from queue import Empty

from jupyter_client import BlockingKernelClient
from jupyter_client.connect import write_connection_file


KERNEL_DIR = Path(__file__).resolve().parent
CONNECTION_FILE = KERNEL_DIR / "kernel.json"
PID_FILE = KERNEL_DIR / "kernel.pid"
CELL_FILE = KERNEL_DIR / "cell.py"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
START_TIMEOUT_S = 60
# A kernel that takes longer than this to answer is stuck in an earlier cell (or gone).
IDLE_TIMEOUT_S = 10
INTERRUPT_GRACE_S = 10
RESTARTED = "[The Python kernel was restarted; variables from earlier cells are gone.]\n"


def connect(timeout: float) -> BlockingKernelClient:
    client = BlockingKernelClient(connection_file=str(CONNECTION_FILE))
    client.load_connection_file()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=timeout)
    except BaseException:
        client.stop_channels()
        raise
    return client


def start() -> BlockingKernelClient:
    # Writing the connection file first means the kernel never has to, so no reader sees it half-written.
    write_connection_file(str(CONNECTION_FILE), key=uuid.uuid4().hex.encode())
    with open(KERNEL_DIR / "kernel.log", "ab") as log:
        # Detached, so the sandbox exec that runs `start` returns while the kernel keeps running.
        kernel = subprocess.Popen(
            [sys.executable, "-m", "ipykernel_launcher", "-f", str(CONNECTION_FILE), "--HistoryManager.enabled=False"],
            cwd=KERNEL_DIR.parent,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    PID_FILE.write_text(str(kernel.pid))
    return connect(START_TIMEOUT_S)


def stop() -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(int(PID_FILE.read_text()), signal.SIGKILL)


def run(timeout: float, max_chars: int) -> None:
    output, size, dropped, notes = [], 0, 0, []
    try:
        client = connect(IDLE_TIMEOUT_S)
    except (OSError, RuntimeError):
        stop()
        client = start()
        notes.append(RESTARTED)
    try:
        msg_id = client.execute(CELL_FILE.read_text(), allow_stdin=False)
        deadline = time.monotonic() + timeout
        interrupted = False
        while True:
            message = None
            remaining = deadline - time.monotonic()
            if remaining > 0:
                with contextlib.suppress(Empty):
                    message = client.get_iopub_msg(timeout=remaining)
            if message is None:
                if not interrupted:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(int(PID_FILE.read_text()), signal.SIGINT)
                    interrupted = True
                    deadline = time.monotonic() + INTERRUPT_GRACE_S
                    continue
                stop()
                start().stop_channels()
                notes.append(RESTARTED)
                break
            if message["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "status" and content["execution_state"] == "idle":
                break
            text = ""
            if kind == "stream":
                text = content["text"]
            elif kind in ("execute_result", "display_data") and "text/plain" in content["data"]:
                text = content["data"]["text/plain"] + "\n"
            elif kind == "error":
                text = ANSI_ESCAPE.sub("", "\n".join(content["traceback"])) + "\n"
            kept = text[: max(0, max_chars - size)]
            output.append(kept)
            size += len(kept)
            dropped += len(text) - len(kept)
        if interrupted:
            notes.insert(0, f"[Execution exceeded {timeout:g} seconds and was interrupted.]\n")
    finally:
        client.stop_channels()
    if dropped:
        output.append(f"\n[Output truncated to {max_chars} characters.]\n")
    print("".join(output + notes), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "run", "stop"])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-chars", type=int, default=20_000)
    args = parser.parse_args()
    if args.command == "start":
        start().stop_channels()
    elif args.command == "run":
        run(args.timeout, args.max_chars)
    else:
        stop()
