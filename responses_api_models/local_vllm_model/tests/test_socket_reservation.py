# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import socket
import sys
from argparse import Namespace
from types import ModuleType

import pytest

from responses_api_models.local_vllm_model.socket_reservation import _exclusive_socket, reserve_server_socket


def install_api(monkeypatch, modern):
    api = ModuleType("vllm.entrypoints.openai.api_server")
    calls = []

    def original_factory(address):
        raise AssertionError("Legacy reuse-port factory must be replaced during setup")

    api.create_server_socket = original_factory
    if modern:

        def setup_server(args, *, reuse_port):
            calls.append(reuse_port)
            assert reuse_port is False
            return f"http://{args.host}:{args.port}", _exclusive_socket((args.host, args.port))
    else:

        def setup_server(args):
            calls.append("legacy")
            return f"http://{args.host}:{args.port}", api.create_server_socket((args.host, args.port))

    api.setup_server = setup_server
    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.openai"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.entrypoints.openai"].api_server = api
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.api_server", api)
    return api, original_factory, calls


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("modern", [False, True])
def test_bound_socket_preserves_policy_and_version_contract(monkeypatch, modern):
    api, original, calls = install_api(monkeypatch, modern)
    port = unused_port()
    address, reserved = reserve_server_socket(
        Namespace(host="127.0.0.1", uds=None), port_range=(port, port), disallowed_ports=()
    )
    try:
        assert reserved.getsockname()[1] == port
        assert address == f"http://127.0.0.1:{port}"
        assert api.create_server_socket is original
        with socket.socket() as competing:
            with pytest.raises(OSError) as error:
                competing.bind(("127.0.0.1", port))
            assert error.value.errno == errno.EADDRINUSE
        assert calls == ([False] if modern else ["legacy"])
    finally:
        reserved.close()


@pytest.mark.parametrize("modern", [False, True])
def test_excluded_and_occupied_ports_are_not_used(monkeypatch, modern):
    api, original, calls = install_api(monkeypatch, modern)
    port = unused_port()
    args = Namespace(host="127.0.0.1", uds=None)
    with pytest.raises(RuntimeError, match="allowed range"):
        reserve_server_socket(args, port_range=(port, port), disallowed_ports=(port,))
    assert calls == []
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", port))
        occupied.listen()
        with pytest.raises(RuntimeError, match="allowed range"):
            reserve_server_socket(args, port_range=(port, port), disallowed_ports=())
    assert api.create_server_socket is original


@pytest.mark.parametrize("modern", [False, True])
def test_setup_error_restores_legacy_factory(monkeypatch, modern):
    api, original, _ = install_api(monkeypatch, modern)
    if modern:

        def fail(args, *, reuse_port):
            raise ValueError("bad model args")
    else:

        def fail(args):
            raise ValueError("bad model args")

    api.setup_server = fail
    port = unused_port()
    with pytest.raises(ValueError, match="bad model args"):
        reserve_server_socket(Namespace(host="127.0.0.1", uds=None), port_range=(port, port), disallowed_ports=())
    assert api.create_server_socket is original
