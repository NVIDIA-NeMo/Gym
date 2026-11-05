# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import socket

from pytest import MonkeyPatch

from resources_servers.newton_bench.newton_bench_utils.sandbox import SessionHandle


def test_worker_releases_unrelated_socket_and_preserves_python_state(monkeypatch: MonkeyPatch) -> None:
    # Exercise Linux's default constructors even when the test host defaults to spawn.
    fork_context = multiprocessing.get_context("fork")
    monkeypatch.setattr(multiprocessing, "Pipe", fork_context.Pipe)
    monkeypatch.setattr(multiprocessing, "Process", fork_context.Process)
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
