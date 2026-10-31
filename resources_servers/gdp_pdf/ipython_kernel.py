# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a persistent IPython kernel inside a GDP.pdf python-tool sandbox.

The resources server places this file in ``<workdir>/.ipython_kernel/`` and runs it there with the sandbox's
``python3``, which needs ``ipykernel`` and ``jupyter_client`` (nothing from Gym):

- ``start`` launches a detached kernel whose working directory is ``<workdir>``, and waits until it answers.
- ``run`` executes ``cell.py`` and prints JSON content containing text and displayed PNG/JPEG images, keeping at most
  ``--max-chars`` of text without dropping images. A cell still running after ``--timeout`` seconds is interrupted. A kernel that
  has exited, or that ignores the interrupt, is restarted, and the output says so.
- ``stop`` kills the kernel and anything it started.
"""

import argparse
import base64
import contextlib
import io
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from queue import Empty

from jupyter_client import BlockingKernelClient
from jupyter_client.connect import write_connection_file
from PIL import Image


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
    # A busy shared CPU can miss the default one-second heartbeat while the
    # kernel is still alive. Use the same deadline as the readiness request.
    client.hb_channel.time_to_dead = timeout
    client.start_channels()
    try:
        client.wait_for_ready(timeout=timeout)
    except BaseException:
        client.stop_channels()
        raise
    return client


def start(*, address_space_mib: int | None = None) -> BlockingKernelClient:
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
    try:
        if address_space_mib is not None:
            # Apply before accepting cells. prlimit avoids preexec_fn in the threaded
            # client, and the hard limit is inherited by subprocesses and cannot be raised.
            if not hasattr(resource, "prlimit"):
                raise RuntimeError("Kernel address-space limits require Linux resource.prlimit")
            existing = resource.prlimit(kernel.pid, resource.RLIMIT_AS)
            limit = min([address_space_mib * 2**20, *(v for v in existing if v != resource.RLIM_INFINITY)])
            resource.prlimit(kernel.pid, resource.RLIMIT_AS, (limit, limit))
        return connect(START_TIMEOUT_S)
    except (OSError, RuntimeError) as error:
        # Setup destroys a failed sandbox. Preserve the kernel's own diagnostic before that happens.
        stop()
        kernel.wait()
        with open(KERNEL_DIR / "kernel.log", "rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 4000))
            diagnostic = log.read().decode(errors="replace")
        raise RuntimeError(f"{error}\nIPython kernel log:\n{diagnostic}") from error


def stop() -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(int(PID_FILE.read_text()), signal.SIGKILL)


def asked_to_exit(client: BlockingKernelClient, msg_id: str) -> bool:
    """Whether the cell called exit(), after which the kernel shuts itself down."""
    with contextlib.suppress(Empty):
        while True:
            reply = client.get_shell_msg(timeout=IDLE_TIMEOUT_S)
            if reply["parent_header"].get("msg_id") == msg_id:
                return any(item.get("source") == "ask_exit" for item in reply["content"].get("payload", []))
    return False


def image_output(data: str, mime: str, max_size: int) -> list[dict[str, str]]:
    """Bound displayed images before they reach the model, retaining aspect ratio."""
    try:
        with Image.open(io.BytesIO(base64.b64decode(data))) as image:
            if max(image.size) <= max_size:
                image.load()
                return [{"type": "input_image", "image_url": f"data:{mime};base64,{data}", "detail": "auto"}]
            original_size = image.size
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return [
                {
                    "type": "input_text",
                    "text": f"[Displayed image resized from {original_size} to {image.size}; "
                    "crop a region to inspect finer detail.]\n",
                },
                {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}", "detail": "auto"},
            ]
    except (ValueError, OSError, Image.DecompressionBombError):
        # A malformed or unreasonably large display is a tool error the model
        # can correct, not a failed episode or an inference-engine crash.
        return [
            {
                "type": "input_text",
                "text": "[Cannot display image: invalid or too large to decode safely. "
                "Render a crop or use a lower resolution.]\n",
            }
        ]


def run(timeout: float, max_chars: int, max_image_size: int, *, address_space_mib: int | None = None) -> None:
    output, size, dropped, notes = [], 0, 0, []
    # An unexpected loss between calls invalidates the episode. Only intentional exit() or a
    # timed-out cell below resets the kernel with explicit model-visible feedback.
    client = connect(IDLE_TIMEOUT_S)
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
                start(address_space_mib=address_space_mib).stop_channels()
                notes.append(RESTARTED)
                break
            if message["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "status" and content["execution_state"] == "idle":
                if asked_to_exit(client, msg_id):
                    # Start the next kernel now, so the next cell never reaches the one shutting down.
                    stop()
                    start(address_space_mib=address_space_mib).stop_channels()
                    notes.append(RESTARTED)
                break
            text = ""
            if kind == "stream":
                text = content["text"]
            elif kind in ("execute_result", "display_data"):
                data = content["data"]
                image_type = next((mime for mime in ("image/png", "image/jpeg") if mime in data), None)
                if image_type and data[image_type]:
                    output.extend(image_output(data[image_type], image_type, max_image_size))
                elif image_type:
                    text = "[Cannot display image: the image data is empty.]\n"
                elif "text/plain" in data:
                    text = data["text/plain"] + "\n"
            elif kind == "error":
                text = ANSI_ESCAPE.sub("", "\n".join(content["traceback"])) + "\n"
            kept = text[: max(0, max_chars - size)]
            if kept:
                output.append({"type": "input_text", "text": kept})
            size += len(kept)
            dropped += len(text) - len(kept)
        if interrupted:
            notes.insert(0, f"[Execution exceeded {timeout:g} seconds and was interrupted.]\n")
    finally:
        client.stop_channels()
    if dropped:
        notes.append(f"\n[Text output truncated to {max_chars} characters.]\n")
    if notes:
        output.append({"type": "input_text", "text": "".join(notes)})
    print(json.dumps({"content": output}), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "run", "stop"])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--max-image-size", type=int, default=4096)
    parser.add_argument("--address-space-mib", type=int)
    args = parser.parse_args()
    if args.address_space_mib is not None and args.address_space_mib < 1:
        parser.error("--address-space-mib must be positive")
    if args.command == "start":
        start(address_space_mib=args.address_space_mib).stop_channels()
    elif args.command == "run":
        run(args.timeout, args.max_chars, args.max_image_size, address_space_mib=args.address_space_mib)
    else:
        stop()
