# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import socket
import threading
import time

import pytest
import requests

from nemo_gym.utils.http_deadline import get_within_deadline


def _peer(pieces: list[tuple[bytes, float]]) -> tuple[str, threading.Event]:
    """A loopback server that answers its first request by sending each piece after its delay, then closes."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    finished = threading.Event()

    def serve() -> None:
        connection, _ = listener.accept()
        with connection, listener:
            connection.recv(65536)
            try:
                for piece, delay in pieces:
                    time.sleep(delay)
                    connection.sendall(piece)
                finished.set()
            except OSError:
                pass  # The client gave up.

    threading.Thread(target=serve, daemon=True).start()
    return f"http://127.0.0.1:{listener.getsockname()[1]}", finished


def _head(length: int) -> bytes:
    return b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % length


def test_returns_the_status_and_body():
    url, _ = _peer([(_head(2), 0), (b"ok", 0)])

    assert get_within_deadline(url, timeout_s=10) == (200, b"ok")


def test_does_not_read_the_body_when_asked_not_to():
    url, _ = _peer([(_head(2), 0), (b"ok", 5)])  # the body would arrive long after the limit

    started = time.monotonic()
    assert get_within_deadline(url, timeout_s=2, read_body=False) == (200, b"")
    assert time.monotonic() - started < 1.5


def test_a_body_that_dribbles_past_the_limit_ends_at_the_limit():
    body = b"x" * 40
    url, finished = _peer([(_head(len(body)), 0)] + [(body[i : i + 1], 0.05) for i in range(len(body))])

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        get_within_deadline(url, timeout_s=0.2)

    assert time.monotonic() - started < 1.5 and not finished.is_set()


def test_headers_that_dribble_past_the_limit_end_at_the_limit():
    head = _head(2)
    url, finished = _peer([(head[i : i + 1], 0.05) for i in range(len(head))] + [(b"ok", 0)])

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        get_within_deadline(url, timeout_s=0.2)

    assert time.monotonic() - started < 1.5 and not finished.is_set()


def test_a_refused_connection_raises_the_requests_error_unchanged():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    with pytest.raises(requests.exceptions.ConnectionError):
        get_within_deadline(f"http://127.0.0.1:{port}", timeout_s=5)


def _dribbling_peer(head_first: bool) -> tuple[str, threading.Event, list[threading.Thread]]:
    """A loopback server that, for every connection, sends one byte every 10 ms until the client goes away.

    The event is set once every accepted connection has been seen closed by the client."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    handlers: list[threading.Thread] = []
    all_closed = threading.Event()
    open_connections = [0]
    lock = threading.Lock()

    def handle(connection: socket.socket) -> None:
        with connection:
            connection.settimeout(5)
            try:
                connection.recv(65536)
                pieces = iter(b"HTTP/1.1 200 OK\r\nX-Pad: " + b"a" * 100000 if head_first else b"")
                for byte in pieces:
                    connection.sendall(bytes([byte]))
                    time.sleep(0.01)
            except OSError:
                pass  # The client closed or shut down its end.
        with lock:
            open_connections[0] -= 1
            if open_connections[0] == 0:
                all_closed.set()

    def accept() -> None:
        while True:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            with lock:
                open_connections[0] += 1
                all_closed.clear()
            thread = threading.Thread(target=handle, args=(connection,), daemon=True)
            handlers.append(thread)
            thread.start()

    threading.Thread(target=accept, daemon=True).start()
    return f"http://127.0.0.1:{listener.getsockname()[1]}", all_closed, handlers


def _request_threads() -> int:
    return sum(thread.name == "get-within-deadline" for thread in threading.enumerate())


def test_repeated_expired_requests_release_their_threads_and_connections_while_the_peer_keeps_sending() -> None:
    url, all_closed, _ = _dribbling_peer(head_first=True)
    baseline = _request_threads()

    for _ in range(5):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            get_within_deadline(url, timeout_s=0.1, read_body=False)
        assert time.monotonic() - started < 1.0

    # The peer is still sending, so nothing but the shutdown of the sockets can end these requests.
    deadline = time.monotonic() + 3
    while _request_threads() > baseline and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _request_threads() == baseline
    assert all_closed.wait(3)
