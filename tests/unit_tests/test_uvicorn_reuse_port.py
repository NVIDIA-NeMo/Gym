# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import http.client
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

import nemo_gym.uvicorn_reuse_port as uvicorn_reuse_port
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


def _start_server(tmp_path: Path, port: int, workers: int) -> subprocess.Popen:
    (tmp_path / "pid_app.py").write_text(PID_APP)
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from nemo_gym.uvicorn_reuse_port import run_with_reuse_port; "
            f"run_with_reuse_port(app='pid_app:app', host='127.0.0.1', port={port}, workers={workers}, "
            "lifespan='off', log_level='warning')",
        ],
        cwd=str(PARENT_DIR),
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(PARENT_DIR)])},
        start_new_session=True,
    )


def _stop_server(server: subprocess.Popen) -> None:
    # Kill the whole session, so workers never outlive a main process that failed to stop them.
    server.terminate()
    try:
        server.wait(timeout=30)
    finally:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _pids_answering(port: int, expected: int, timeout: float = 60) -> set[str]:
    seen: set[str] = set()
    deadline = time.monotonic() + timeout
    while len(seen) < expected and time.monotonic() < deadline:
        try:
            seen.add(_get_pid(port))
        except OSError:
            time.sleep(0.2)
    return seen


def _sockets_on_port(port: int) -> dict[int, str]:
    """Socket inode -> TCP state (0A is LISTEN) of the sockets the kernel lists for local port `port`."""
    states = {}
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if int(fields[1].split(":")[1], 16) == port:
            states[int(fields[9])] = fields[3]
    return states


def _socket_inodes(pid: int) -> set[int]:
    inodes = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(int(target[len("socket:[") : -1]))
    return inodes


def test_each_worker_takes_an_even_share_of_new_connections(tmp_path: Path) -> None:
    port = _free_port()
    workers = 4
    server = _start_server(tmp_path, port, workers)
    try:
        assert len(_pids_answering(port, workers)) == workers

        pids = Counter(_get_pid(port) for _ in range(400))

        assert len(pids) == workers
        assert max(pids.values()) / (400 / workers) < 1.6, pids
    finally:
        _stop_server(server)


def test_only_workers_listen_and_the_main_process_socket_takes_no_connections(tmp_path: Path) -> None:
    port = _free_port()
    workers = 3
    server = _start_server(tmp_path, port, workers)
    try:
        worker_pids = [int(pid) for pid in _pids_answering(port, workers)]
        assert len(worker_pids) == workers

        listening = {inode for inode, state in _sockets_on_port(port).items() if state == "0A"}

        assert len(listening) == workers
        assert not listening & _socket_inodes(server.pid), "the main process's socket is listening"
        # Each worker listens on a socket of its own, not on a copy of a shared one.
        for pid in worker_pids:
            assert len(listening & _socket_inodes(pid)) == 1, pid
    finally:
        _stop_server(server)


def test_a_killed_worker_is_replaced_by_one_listening_on_its_own_socket(tmp_path: Path) -> None:
    port = _free_port()
    workers = 3
    server = _start_server(tmp_path, port, workers)
    try:
        before = _pids_answering(port, workers)
        assert len(before) == workers

        killed = before.pop()
        os.kill(int(killed), signal.SIGKILL)

        # Uvicorn's main process polls every 0.5 s; the replacement binds its own socket on start-up.
        deadline = time.monotonic() + 60
        after: set[str] = set()
        while time.monotonic() < deadline and not (len(after) == workers and killed not in after):
            after = _pids_answering(port, workers, timeout=5) - {killed}

        assert killed not in after
        assert len(after) == workers, after
        assert before < after, "the surviving workers should keep serving"
        assert sorted(_sockets_on_port(port).values()).count("0A") == workers
    finally:
        _stop_server(server)


def test_a_second_server_on_a_port_in_use_exits_instead_of_sharing_it(tmp_path: Path) -> None:
    port = _free_port()
    first = _start_server(tmp_path, port, 2)
    try:
        assert len(_pids_answering(port, 2)) == 2

        second = _start_server(tmp_path, port, 2)
        try:
            assert second.wait(timeout=30) != 0
        finally:
            _stop_server(second)
        # The second server never joined the port, so every request still reaches the first server's workers.
        assert len({_get_pid(port) for _ in range(40)}) == 2
    finally:
        _stop_server(first)


def test_a_second_bind_with_reuse_port_shares_the_address() -> None:
    port = _free_port()
    first = uvicorn_reuse_port._bind_reuse_port_socket(socket.AF_INET, ("127.0.0.1", port))
    second = uvicorn_reuse_port._bind_reuse_port_socket(socket.AF_INET, ("127.0.0.1", port))
    first.close()
    second.close()


def test_bind_socket_exits_when_a_plain_listener_holds_the_port() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        config = uvicorn_reuse_port._ReusePortUvicornConfig(
            app="pid_app:app", host="127.0.0.1", port=listener.getsockname()[1], workers=2
        )

        with pytest.raises(SystemExit) as exit_info:
            config.bind_socket()

    assert exit_info.value.code == uvicorn_reuse_port.STARTUP_FAILURE


def test_bind_socket_binds_without_listening() -> None:
    port = _free_port()
    config = uvicorn_reuse_port._ReusePortUvicornConfig(app="pid_app:app", host="127.0.0.1", port=port, workers=2)

    sock = config.bind_socket()
    try:
        assert sock.getsockname()[1] == port
        assert sock.get_inheritable()
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 0
    finally:
        sock.close()


def test_a_single_worker_server_is_refused() -> None:
    with pytest.raises(ValueError, match="multi-worker"):
        run_with_reuse_port(app="pid_app:app", host="127.0.0.1", port=_free_port(), workers=1)


def test_an_app_object_is_refused() -> None:
    with pytest.raises(ValueError, match="import string"):
        run_with_reuse_port(app=object(), host="127.0.0.1", port=_free_port(), workers=4)


def test_non_linux_hosts_are_refused_before_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uvicorn_reuse_port.sys, "platform", "darwin")
    assert not reuse_port_supported()

    def bind(*args: object) -> None:
        pytest.fail("bound a socket before checking the platform")

    monkeypatch.setattr(uvicorn_reuse_port, "_bind_reuse_port_socket", bind)
    with pytest.raises(RuntimeError, match="needs Linux"):
        run_with_reuse_port(app="pid_app:app", host="127.0.0.1", port=_free_port(), workers=4)
