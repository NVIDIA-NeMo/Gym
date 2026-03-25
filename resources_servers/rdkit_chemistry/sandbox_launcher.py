# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Start and supervise a nemo_skills sandbox subprocess.

Launched from ``RDKitChemistryResourcesServer.setup_webserver()`` so the
sandbox lifetime is tied to the resources server — no separate job to manage
and no risk of the sandbox going down while GPUs are still running.

A background watchdog thread monitors the process and auto-restarts on crash.

nemo_skills uses per-request UUIDs to keep sandbox sessions independent, so a
single sandbox instance handles concurrent requests without state collision.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import httpx


logger = logging.getLogger(__name__)

_HEALTH_POLL = 2.0
_HEALTH_TIMEOUT = 120.0
_PROXY_HEALTH_TIMEOUT = 30.0
_WATCHDOG_INTERVAL = 10.0
_DEFAULT_PROXY_REQUEST_TIMEOUT = 120.0
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

_lock = threading.Lock()
_sandbox_proc: subprocess.Popen | None = None
_sandbox_python: str | None = None
_sandbox_port: int = 6000
_proxy_port: int | None = None
_proxy_server: "_SandboxProxyServer | None" = None
_proxy_thread: threading.Thread | None = None


class _SandboxProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 512

    def __init__(
        self,
        server_address: tuple[str, int],
        upstream_port: int,
        max_concurrency: int,
        request_timeout_s: float,
        connect_retries: int,
        retry_backoff_s: float,
    ) -> None:
        super().__init__(server_address, _SandboxProxyHandler)
        self.upstream_base_url = f"http://127.0.0.1:{upstream_port}"
        self.semaphore = threading.Semaphore(max_concurrency)
        self.request_timeout_s = request_timeout_s
        self.connect_retries = connect_retries
        self.retry_backoff_s = retry_backoff_s
        self.client = httpx.Client(timeout=httpx.Timeout(request_timeout_s, connect=5.0))


class _SandboxProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_GET(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy_request()

    def log_message(self, fmt: str, *args) -> None:
        logger.debug("Sandbox proxy: " + fmt, *args)

    def _proxy_request(self) -> None:
        server = self.server
        assert isinstance(server, _SandboxProxyServer)

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else b""
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() not in {"content-length", "host"}
        }

        acquired = server.semaphore.acquire(timeout=server.request_timeout_s)
        if not acquired:
            self.send_error(503, "Sandbox proxy is saturated")
            return

        try:
            upstream_response = None
            last_error: Optional[Exception] = None
            for attempt in range(server.connect_retries + 1):
                try:
                    upstream_response = server.client.request(
                        self.command,
                        f"{server.upstream_base_url}{self.path}",
                        headers=headers,
                        content=body or None,
                    )
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                    last_error = e
                    if attempt >= server.connect_retries:
                        break
                    time.sleep(server.retry_backoff_s * (2**attempt))

            if upstream_response is None:
                logger.warning("Sandbox proxy upstream request failed: %s", last_error)
                self.send_error(502, f"Sandbox proxy upstream request failed: {last_error}")
                return

            response_content = upstream_response.content
            self.send_response(upstream_response.status_code)
            for key, value in upstream_response.headers.items():
                lower_key = key.lower()
                if lower_key in _HOP_BY_HOP_HEADERS or lower_key in {"content-length", "date", "server"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_content)))
            self.end_headers()
            if self.command != "HEAD" and response_content:
                self.wfile.write(response_content)
        finally:
            server.semaphore.release()


def start_sandbox(
    venv_path: str,
    port: int = 6000,
    proxy_port: int | None = None,
    proxy_max_concurrency: int = 128,
    proxy_request_timeout_s: float = _DEFAULT_PROXY_REQUEST_TIMEOUT,
    proxy_connect_retries: int = 3,
    proxy_retry_backoff_s: float = 0.25,
    extra_packages: list[str] | None = None,
    discovery_path: str | None = None,
) -> None:
    """Start a nemo_skills sandbox server as a managed subprocess.

    Safe to call multiple times — only the first call has effect (the sandbox
    is a process-wide singleton).

    Args:
        venv_path: Path to the ns_tools virtualenv that has ``nemo_skills``.
        port: Port for the sandbox (default 6000, matching ns_tools defaults).
        extra_packages: Pip packages to ensure are installed (e.g. rdkit).
        discovery_path: Optional path on shared FS to write a JSON file with
            the sandbox address (for other jobs to discover).
    """
    global _sandbox_proc, _sandbox_python, _sandbox_port

    with _lock:
        python = os.path.join(venv_path, "bin", "python")
        pip = os.path.join(venv_path, "bin", "pip")

        # ng_run creates all server venvs in parallel.  The ns_tools venv
        # may not be ready yet when rdkit_chemistry starts — wait for it.
        _wait_for_venv(python)

        _sandbox_python = python
        _sandbox_port = port

        if _sandbox_proc is None or _sandbox_proc.poll() is not None:
            _ensure_packages(python, pip, extra_packages or [])
            _sandbox_proc = _spawn(python, port)

    _wait_for_health(port)

    advertised_port = port
    if proxy_port is not None and proxy_port != port:
        _start_proxy(
            proxy_port=proxy_port,
            upstream_port=port,
            max_concurrency=proxy_max_concurrency,
            request_timeout_s=proxy_request_timeout_s,
            connect_retries=proxy_connect_retries,
            retry_backoff_s=proxy_retry_backoff_s,
        )
        _wait_for_health(proxy_port, timeout_s=_PROXY_HEALTH_TIMEOUT, check_sandbox_proc=False)
        advertised_port = proxy_port

    if discovery_path:
        _write_discovery(discovery_path, advertised_port)

    watchdog = threading.Thread(target=_watchdog, args=(python, port), daemon=True, name="sandbox-watchdog")
    watchdog.start()

    atexit.register(_stop_proxy)
    atexit.register(_stop_sandbox)
    if advertised_port != port:
        logger.info(
            "Sandbox ready on 127.0.0.1:%d via throttling proxy 127.0.0.1:%d (pid=%d)",
            port,
            advertised_port,
            _sandbox_proc.pid,
        )
    else:
        logger.info("Sandbox ready on 127.0.0.1:%d (pid=%d)", port, _sandbox_proc.pid)


_VENV_TIMEOUT = 600.0  # ng_run venv creation can take several minutes


def _wait_for_venv(python: str) -> None:
    """Block until the venv's python binary exists and nemo_skills is importable.

    ng_run creates all server venvs concurrently, so the ns_tools venv (which
    has nemo_skills) may still be installing when rdkit_chemistry starts.
    """
    deadline = time.monotonic() + _VENV_TIMEOUT
    phase = "binary"

    if not os.path.isfile(python):
        logger.info("Waiting for sandbox venv python at %s ...", python)
        while time.monotonic() < deadline:
            if os.path.isfile(python):
                break
            time.sleep(5.0)
        else:
            raise FileNotFoundError(
                f"Sandbox venv python not found at {python} after {_VENV_TIMEOUT}s. "
                "Ensure ns_tools is part of the ng_run config."
            )

    phase = "nemo_skills"
    logger.info("Waiting for nemo_skills to be importable in %s ...", python)
    while time.monotonic() < deadline:
        try:
            subprocess.run(
                [python, "-c", "import nemo_skills"],
                check=True,
                capture_output=True,
            )
            logger.info("Sandbox venv ready (nemo_skills importable)")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            time.sleep(5.0)

    raise TimeoutError(f"nemo_skills not importable in {python} after {_VENV_TIMEOUT}s ({phase} phase)")


def _ensure_packages(python: str, pip: str, packages: list[str]) -> None:
    for pkg in packages:
        try:
            subprocess.run(
                [python, "-c", f"import {pkg}"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            logger.info("Installing %s into sandbox venv...", pkg)
            subprocess.run(
                [pip, "install", "--quiet", pkg],
                check=True,
                capture_output=True,
            )


def _spawn(python: str, port: int) -> subprocess.Popen:
    log_path = f"/tmp/sandbox_{port}.log"
    log_file = open(log_path, "a")  # noqa: SIM115
    proc = subprocess.Popen(
        [python, "-m", "nemo_skills.code_execution.local_sandbox.local_sandbox_server"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    logger.info("Sandbox spawned (pid=%d, port=%d, log=%s)", proc.pid, port, log_path)
    return proc


def _start_proxy(
    proxy_port: int,
    upstream_port: int,
    max_concurrency: int,
    request_timeout_s: float,
    connect_retries: int,
    retry_backoff_s: float,
) -> None:
    global _proxy_port, _proxy_server, _proxy_thread

    old_server = None
    old_thread = None
    with _lock:
        if _proxy_thread is not None and _proxy_thread.is_alive() and _proxy_port == proxy_port:
            return

        if _proxy_server is not None:
            old_server = _proxy_server
            old_thread = _proxy_thread
            _proxy_server = None
            _proxy_thread = None
            _proxy_port = None

    if old_server is not None:
        old_server.shutdown()
        old_server.server_close()
        old_server.client.close()
        if old_thread is not None:
            old_thread.join(timeout=5)

    with _lock:
        _proxy_server = _SandboxProxyServer(
            ("127.0.0.1", proxy_port),
            upstream_port=upstream_port,
            max_concurrency=max_concurrency,
            request_timeout_s=request_timeout_s,
            connect_retries=connect_retries,
            retry_backoff_s=retry_backoff_s,
        )
        _proxy_port = proxy_port
        _proxy_thread = threading.Thread(target=_proxy_server.serve_forever, daemon=True, name="sandbox-proxy")
        _proxy_thread.start()
        logger.info(
            "Sandbox proxy listening on 127.0.0.1:%d -> 127.0.0.1:%d (max_concurrency=%d)",
            proxy_port,
            upstream_port,
            max_concurrency,
        )


def _wait_for_health(port: int, timeout_s: float = _HEALTH_TIMEOUT, check_sandbox_proc: bool = True) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check_sandbox_proc:
            with _lock:
                proc = _sandbox_proc
            if proc and proc.poll() is not None:
                log_tail = _tail_log(_sandbox_port)
                raise RuntimeError(
                    f"Sandbox died during startup (exit={proc.returncode})\n--- sandbox log tail ---\n{log_tail}"
                )
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    return
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        time.sleep(_HEALTH_POLL)

    raise TimeoutError(f"Sandbox not healthy after {timeout_s}s on port {port}")


def _tail_log(port: int, n: int = 30) -> str:
    log_path = f"/tmp/sandbox_{port}.log"
    if not os.path.exists(log_path):
        return "(no log file)"
    try:
        with open(log_path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as e:
        return f"(could not read log: {e})"


def _watchdog(python: str, port: int) -> None:
    global _sandbox_proc
    while True:
        time.sleep(_WATCHDOG_INTERVAL)
        with _lock:
            proc = _sandbox_proc
        if proc is None:
            return
        if proc.poll() is not None:
            logger.warning("Sandbox died (exit=%s) — restarting...", proc.returncode)
            with _lock:
                _sandbox_proc = _spawn(python, port)
            try:
                _wait_for_health(port)
                logger.info("Sandbox recovered (pid=%d)", _sandbox_proc.pid)
            except (RuntimeError, TimeoutError):
                logger.error("Sandbox failed to recover after restart")


def _stop_proxy() -> None:
    global _proxy_port, _proxy_server, _proxy_thread
    old_server = None
    old_thread = None
    with _lock:
        if _proxy_server is not None:
            old_server = _proxy_server
            _proxy_server = None
        if _proxy_thread is not None:
            old_thread = _proxy_thread
            _proxy_thread = None
        old_port = _proxy_port
        _proxy_port = None

    if old_server is not None:
        old_server.shutdown()
        old_server.server_close()
        old_server.client.close()
    if old_thread is not None:
        old_thread.join(timeout=5)
    if old_port is not None:
        logger.info("Sandbox proxy stopped")


def _stop_sandbox() -> None:
    global _sandbox_proc
    with _lock:
        if _sandbox_proc is not None:
            _sandbox_proc.terminate()
            try:
                _sandbox_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _sandbox_proc.kill()
            _sandbox_proc = None
            logger.info("Sandbox stopped")


def _write_discovery(path: str, port: int) -> None:
    host = socket.gethostname()
    discovery = {
        "sandbox_host": host,
        "sandbox_port": port,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(discovery, f, indent=2)
    os.replace(tmp, path)
    logger.info("Wrote sandbox discovery to %s", path)
