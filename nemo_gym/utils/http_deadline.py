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
"""An HTTP GET that ends at a total time limit."""

import socket
import threading
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool


class _Sockets:
    """The sockets of one request. `shutdown` ends any read blocked on them, and a socket opened afterwards."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self._shut_down = False

    def add(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets.append(sock)
            shut_down = self._shut_down
        if shut_down:
            self._shutdown_one(sock)

    def shutdown(self) -> None:
        with self._lock:
            self._shut_down = True
            sockets = list(self._sockets)
        for sock in sockets:
            self._shutdown_one(sock)

    @staticmethod
    def _shutdown_one(sock: socket.socket) -> None:
        # shutdown, not close: it does not wait for the thread that is reading, and that read then ends.
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # Already closed or never connected.


class _TrackingAdapter(HTTPAdapter):
    """An adapter whose connections report their sockets, so that a request can be stopped from another thread."""

    def __init__(self, sockets: _Sockets) -> None:
        self._sockets = sockets
        super().__init__()

    def _tracked_pool_classes(self) -> dict[str, type]:
        sockets = self._sockets

        class TrackedHTTPConnection(HTTPConnection):
            def _new_conn(self) -> socket.socket:
                sock = super()._new_conn()
                sockets.add(sock)
                return sock

        class TrackedHTTPSConnection(HTTPSConnection):
            def _new_conn(self) -> socket.socket:
                sock = super()._new_conn()
                sockets.add(sock)
                return sock

        class TrackedHTTPConnectionPool(HTTPConnectionPool):
            ConnectionCls = TrackedHTTPConnection

        class TrackedHTTPSConnectionPool(HTTPSConnectionPool):
            ConnectionCls = TrackedHTTPSConnection

        return {"http": TrackedHTTPConnectionPool, "https": TrackedHTTPSConnectionPool}

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = self._tracked_pool_classes()

    def proxy_manager_for(self, *args: Any, **kwargs: Any) -> Any:
        manager = super().proxy_manager_for(*args, **kwargs)
        manager.pool_classes_by_scheme = self._tracked_pool_classes()
        return manager


def get_within_deadline(
    url: str, *, timeout_s: float, read_timeout_s: float | None = None, read_body: bool = True
) -> tuple[int, bytes]:
    """GET ``url`` and return its status code and body, raising ``TimeoutError`` when ``timeout_s`` seconds have passed.

    A requests timeout bounds the wait for each read, not the whole call: a peer that keeps sending a byte inside it,
    in the headers or in the body, keeps the call going. The request therefore runs on a daemon thread that the caller
    stops waiting for at the limit. At the limit the request's sockets are shut down, which ends a read that is still
    waiting, in the headers or in the body, so the thread and its connection are released promptly. With
    ``read_body=False`` the body is not read and ``b""`` is returned: for a readiness check, the answer is the headers.

    Exceptions of the request, such as ``requests.ConnectionError``, are raised unchanged.
    """
    outcome: dict[str, Any] = {}
    expired = threading.Event()
    sockets = _Sockets()

    def get() -> None:
        try:
            with requests.Session() as session:
                session.mount("http://", _TrackingAdapter(sockets))
                session.mount("https://", _TrackingAdapter(sockets))
                r = session.get(url, timeout=timeout_s if read_timeout_s is None else read_timeout_s, stream=True)
                with r:
                    body = b""
                    if read_body:
                        chunks = []
                        for chunk in r.iter_content(chunk_size=8192):
                            if expired.is_set():
                                return
                            chunks.append(chunk)
                        body = b"".join(chunks)
                    outcome["result"] = (r.status_code, body)
        except BaseException as error:  # Handed to the caller, which raises it.
            outcome["error"] = error

    worker = threading.Thread(target=get, name="get-within-deadline", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        expired.set()
        sockets.shutdown()
        raise TimeoutError(f"GET {url} did not finish within {timeout_s} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]
