# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import http.client
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.uvicorn_reuse_port import reuse_port_supported, run_with_reuse_port


pytestmark = pytest.mark.skipif(not reuse_port_supported(), reason="SO_REUSEPORT load balancing is Linux-only")

PID_APP = """
import os

async def app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": str(os.getpid()).encode()})
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_pid(port: int) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", "/")
        return connection.getresponse().read().decode()
    finally:
        connection.close()


def test_each_worker_takes_an_even_share_of_new_connections(tmp_path: Path) -> None:
    (tmp_path / "pid_app.py").write_text(PID_APP)
    port = _free_port()
    workers = 4
    server = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from nemo_gym.uvicorn_reuse_port import run_with_reuse_port; "
            f"run_with_reuse_port(app='pid_app:app', host='127.0.0.1', port={port}, workers={workers}, "
            "lifespan='off', log_level='warning')",
        ],
        cwd=str(PARENT_DIR),
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(PARENT_DIR)])},
    )
    try:
        seen = set()
        deadline = time.monotonic() + 60
        # Ready once every worker has answered at least once.
        while len(seen) < workers and time.monotonic() < deadline:
            try:
                seen.add(_get_pid(port))
            except OSError:
                time.sleep(0.2)
        assert len(seen) == workers, f"only {len(seen)} of {workers} workers answered"

        pids = Counter(_get_pid(port) for _ in range(400))

        assert len(pids) == workers
        assert max(pids.values()) / (400 / workers) < 1.6, pids
    finally:
        server.terminate()
        server.wait(timeout=30)


def test_a_single_worker_server_is_refused() -> None:
    with pytest.raises(ValueError, match="multi-worker"):
        run_with_reuse_port(app="pid_app:app", host="127.0.0.1", port=_free_port(), workers=1)
