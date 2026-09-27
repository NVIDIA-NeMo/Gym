# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket

from responses_api_agents.miniswe_sandboxed_agent.native_runner import KeepAliveSocket


def test_tcp_keepalive_survives_connection():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with KeepAliveSocket() as client:
            client.connect(listener.getsockname())
            with listener.accept()[0] as peer:
                peer.sendall(b"response")
                assert client.recv(8) == b"response"
                assert client.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
                if hasattr(socket, "TCP_KEEPIDLE"):
                    assert client.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 60
                    assert client.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 30
                    assert client.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == 5


def test_udp_does_not_get_tcp_options():
    with KeepAliveSocket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        assert not client.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
