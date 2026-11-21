# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise development bootstrap downloads with interrupted HTTP responses."""

import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest


ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(shutil.which("curl") is None, reason="curl is required")


@contextmanager
def _installer_server(*, download_fails: bool, installer_status: int) -> Iterator[tuple[str, list[str]]]:
    """Serve an interrupted response followed by an installer or HTTP error."""
    requests = []
    installer = f'printf "installed\\n" >> "$INSTALLER_MARKER"\nexit {installer_status}\n'.encode()

    class Handler(BaseHTTPRequestHandler):
        """Send controlled responses for curl's actual retry implementation."""

        def log_message(self, *args: object) -> None:
            """Suppress HTTP access logging during the test."""

        def do_GET(self) -> None:
            """Interrupt the first transfer after sending a partial shell script."""
            requests.append(self.path)
            if len(requests) == 1:
                body = installer if download_fails else b"case x in\nx)\n"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body) + 1000))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                self.close_connection = True
            elif download_fails:
                self.send_error(404)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(installer)))
                self.end_headers()
                self.wfile.write(installer)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/install.sh", requests
        finally:
            server.shutdown()
            worker.join(timeout=5)


@pytest.mark.parametrize(
    ("download_fails", "installer_status", "expected_status"),
    [(False, 0, 0), (True, 0, 22), (False, 73, 73)],
    ids=["retry-success", "download-failure", "installer-failure"],
)
def test_setup_dev_download_retries_and_cleanup(
    tmp_path: Path, download_fails: bool, installer_status: int, expected_status: int
) -> None:
    """Require complete downloads, preserve failures, and remove installer files."""
    repo = tmp_path / "repo"
    ci_dir = repo / "scripts" / "ci"
    ci_dir.mkdir(parents=True)
    shutil.copy2(ROOT / ".python-version", repo / ".python-version")
    venv = repo / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch(mode=0o755)
    (venv / "bin" / "activate").touch()
    installer_tmp = tmp_path / "installer-tmp"
    installer_tmp.mkdir()
    marker = tmp_path / "installed"
    env = dict(
        os.environ,
        INSTALLER_MARKER=str(marker),
        TMPDIR=str(installer_tmp),
        GYM_CI_DEV_VENV_DIR=str(venv),
        UV_CACHE_DIR=str(tmp_path / "uv-cache"),
        NO_PROXY="127.0.0.1",
        no_proxy="127.0.0.1",
    )
    command = """
uv() {
    case "$1" in
        --version)
            if [[ -f "$INSTALLER_MARKER" ]]; then echo 'uv 0.13.4'; else echo 'uv 0.0.0'; fi ;;
        cache) echo "$UV_CACHE_DIR" ;;
        lock|sync) return 0 ;;
        *) return 99 ;;
    esac
}
source "$1"
"""

    with _installer_server(download_fails=download_fails, installer_status=installer_status) as (url, requests):
        source = (ROOT / "scripts" / "ci" / "setup_dev.sh").read_text()
        source = source.replace("https://astral.sh/uv/0.13.4/install.sh", url).replace("--retry 5", "--retry 1")
        setup_dev = ci_dir / "setup_dev.sh"
        setup_dev.write_text(source)
        result = subprocess.run(
            ["bash", "-c", command, "setup-test", str(setup_dev)],
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
        )

    assert result.returncode == expected_status, result.stderr
    assert requests == ["/install.sh", "/install.sh"]
    assert marker.exists() is not download_fails
    if marker.exists():
        assert marker.read_text() == "installed\n"
    assert list(installer_tmp.iterdir()) == []
