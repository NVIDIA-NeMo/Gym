# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reserve the vLLM socket on its host without leaving Gym's port policy."""

import errno
import inspect
import random
import socket
from argparse import Namespace


def _exclusive_socket(address: tuple[str, int]) -> socket.socket:
    """Legacy vLLM socket factory without its unconditional SO_REUSEPORT."""
    family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(address)
    except BaseException:
        sock.close()
        raise
    return sock


def reserve_server_socket(
    server_args: Namespace,
    *,
    port_range: tuple[int, int],
    disallowed_ports: tuple[int, ...],
    max_attempts: int = 50,
) -> tuple[str, socket.socket]:
    """Validate vLLM args and retain a bound socket from Gym's allowed range.

    Runs before the actor starts its HTTP thread. vLLM 0.11 lacks the
    ``reuse_port`` option, so its socket factory is replaced only during setup
    and restored on every exit path. Later versions accept the option directly.
    """
    from vllm.entrypoints.openai import api_server

    low, high = port_range
    if not 1 <= low <= high <= 65535 or max_attempts < 1:
        raise ValueError("A valid non-empty TCP port range and positive attempt count are required")
    if getattr(server_args, "uds", None):
        raise ValueError("LocalVLLMModel publishes TCP URLs and does not support a Unix-domain socket")
    forbidden = set(disallowed_ports)
    candidates = [port for port in range(low, high + 1) if port not in forbidden]
    random.shuffle(candidates)
    modern = "reuse_port" in inspect.signature(api_server.setup_server).parameters
    last_error = None
    for port in candidates[:max_attempts]:
        server_args.port = port
        try:
            if modern:
                return api_server.setup_server(server_args, reuse_port=False)
            original_factory = api_server.create_server_socket
            api_server.create_server_socket = _exclusive_socket
            try:
                return api_server.setup_server(server_args)
            finally:
                api_server.create_server_socket = original_factory
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            last_error = error
    raise RuntimeError("No available vLLM TCP port in Gym's configured allowed range") from last_error
