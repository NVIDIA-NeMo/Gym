# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start this task's media listener, with bounded proof that its own child is ready."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4


HOST = "127.0.0.1"
PORT = 17654


def main() -> None:
    instance = uuid4().hex
    script = Path(__file__).with_name("mcp_http.py")
    log_path = script.with_name("mcp-http.log")
    with log_path.open("ab") as log:
        child = subprocess.Popen(
            [sys.executable, str(script), "--instance", instance],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Claw media child exited before readiness")
            try:
                with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=1) as response:
                    ready = json.load(response)
                if ready == {"service": "claweval-media", "instance": instance}:
                    script.with_name("mcp-http-pid.json").write_text(
                        json.dumps(
                            {
                                "pid": child.pid,
                                "instance": instance,
                                "owner_uid": os.getuid(),
                            }
                        )
                        + "\n"
                    )
                    return
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        raise TimeoutError("Own Claw media listener was not ready within 30 seconds")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise


if __name__ == "__main__":
    main()
