# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from pytest import MonkeyPatch, raises

from resources_servers.newton_bench.newton_bench_utils.sandbox import SessionHandle


def test_worker_releases_unrelated_socket_and_preserves_python_state() -> None:
    server_socket, peer_socket = socket.socketpair()
    handle = None
    try:
        # Create the connection before the worker, as happens to the HTTP connection running the first tool.
        handle = SessionHandle(max_execution_time=60)
        assert handle.exec("x = 41\nx")[2] == "41"
        assert handle._proc.is_alive()

        server_socket.close()
        peer_socket.settimeout(1)
        # A forked worker keeps the peer open even after the server closes its own socket.
        assert peer_socket.recv(1) == b""
        assert handle._proc.is_alive()
        assert handle.exec("x += 1\nx")[2] == "42"
    finally:
        server_socket.close()
        peer_socket.close()
        if handle is not None:
            handle.close()


def test_failed_worker_termination_remains_retryable() -> None:
    handle = SessionHandle.__new__(SessionHandle)
    handle._close_lock = threading.Lock()
    handle._conn = MagicMock()
    handle._proc = MagicMock()
    handle._proc.is_alive.return_value = True
    handle._proc.kill.side_effect = OSError("provider denied kill")
    handle.is_closed = False
    with raises(RuntimeError, match="still alive"):
        handle.close()
    assert not handle.is_closed
    handle._proc.close.assert_not_called()
    handle._proc.is_alive.return_value = False
    handle.close()
    assert handle.is_closed
    handle._proc.close.assert_called_once()


def test_worker_close_serializes_concurrent_callers(monkeypatch: MonkeyPatch) -> None:
    handle = SessionHandle(max_execution_time=60)
    original_send = handle._conn.send
    first_sending, second_started, second_sending, finish = (threading.Event() for _ in range(4))
    sends = []
    send_lock = threading.Lock()

    def paused_send(message) -> None:
        with send_lock:
            sends.append(message)
            first = len(sends) == 1
        if first:
            first_sending.set()
            if not finish.wait(5):
                raise TimeoutError("Test did not allow the first close to finish")
        else:
            second_sending.set()
        original_send(message)

    def second_close() -> None:
        second_started.set()
        handle.close()

    try:
        assert handle.exec("41 + 1")[2] == "42"
        monkeypatch.setattr(handle._conn, "send", paused_send)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(handle.close)
            try:
                assert first_sending.wait(5)
                second = pool.submit(second_close)
                assert second_started.wait(5)
                # Both callers entered close; only the owner can touch the pipe/process until it finishes.
                overlapped = second_sending.wait(0.2)
            finally:
                finish.set()
            first.result(timeout=10)
            second.result(timeout=10)
        assert not overlapped
        assert sends == [{"cmd": "close"}]
        assert handle.is_closed
        assert handle._proc._closed
    finally:
        finish.set()
        monkeypatch.setattr(handle._conn, "send", original_send)
        handle.close()


def test_concurrent_execs_each_get_their_own_reply() -> None:
    handle = SessionHandle(max_execution_time=60)
    try:
        assert handle.exec("1")[2] == "1"  # start the worker before the race
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i in range(8):
                slow = pool.submit(handle.exec, f"import time\ntime.sleep(0.2)\n'slow-{i}'")
                time.sleep(0.05)  # the slow cell is running when the fast one starts
                fast = pool.submit(handle.exec, f"'fast-{i}'")
                assert slow.result()[2] == f"slow-{i}"
                assert fast.result()[2] == f"fast-{i}"
    finally:
        handle.close()
