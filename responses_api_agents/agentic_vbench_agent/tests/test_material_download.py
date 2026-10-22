# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shlex
import shutil
import socket
import struct
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from responses_api_agents.agentic_vbench_agent.remote_environment import _adapt_dockerfile


def test_download_adaptation_keeps_task_commands_and_installer_pipes() -> None:
    original = (
        b"FROM python:3.12-slim\n"
        b"RUN apt-get update\n"
        b"ARG MATERIALS_URL=https://example.com/original.zip\n"
        b"RUN curl --fail --silent --show-error --location --retry 5 --retry-delay 3 "
        b'"$MATERIALS_URL" -o /tmp/m.zip && unzip -q /tmp/m.zip -d /baked/materials/\n'
        b"RUN curl -o- https://example.com/install.sh | bash\n"
        b"COPY test.py /tests/test.py\n"
    )
    adapted = _adapt_dockerfile(original)
    # Only rootless APT/transport options differ; URLs, output, extraction,
    # verifier copy and piped installers remain byte-for-byte identical.
    restored = adapted.replace(b" -o APT::Sandbox::User=root", b"").replace(b" --http1.1 --retry-all-errors", b"")
    assert restored == original
    assert b"curl --http1.1 --retry-all-errors --fail" in adapted
    assert b"curl -o- https://example.com/install.sh | bash" in adapted


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl is required for a real transfer regression")
def test_material_get_retries_reset_and_replaces_partial_file(tmp_path: Path) -> None:
    payload = b"complete benchmark material\n" * 4096
    requests: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests[self.path] = requests.get(self.path, 0) + 1
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if requests[self.path] == 1:
                self.wfile.write(payload[:1024])
                self.wfile.flush()
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                self.close_connection = True
                return
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command = b"curl --fail --silent --show-error --location --retry 5 --retry-delay 3"
        for path, adapted in [("/original", False), ("/adapted", True)]:
            args = shlex.split((_adapt_dockerfile(command) if adapted else command).decode())
            target = tmp_path / path.removeprefix("/")
            result = subprocess.run(
                [*args, "--noproxy", "*", f"http://127.0.0.1:{server.server_port}{path}", "-o", str(target)],
                capture_output=True,
                timeout=25,
            )
            if adapted:
                assert result.returncode == 0, result.stderr.decode()
                assert requests[path] == 2
                assert target.read_bytes() == payload
            else:
                assert result.returncode != 0
                assert requests[path] == 1
                assert target.read_bytes() != payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
