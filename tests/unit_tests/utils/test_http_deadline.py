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
