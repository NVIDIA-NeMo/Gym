# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run native mini-SWE with keepalive on long, non-streaming model calls."""

import runpy
import socket
from functools import partial

import httpx
import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def model_transport() -> httpx.HTTPTransport:
    """Keep idle model sockets tracked while the server generates a response."""
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle_option = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    if idle_option is not None:
        options.append((socket.IPPROTO_TCP, idle_option, 60))
    if hasattr(socket, "TCP_KEEPINTVL"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30))
    if hasattr(socket, "TCP_KEEPCNT"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5))
    return httpx.HTTPTransport(socket_options=options)


def main() -> None:
    # An application request timeout does not keep intermediary connection
    # tracking alive. LiteLLM's Responses adapter accepts an HTTPHandler;
    # the native model's explicit per-request timeout still applies.
    previous = litellm.responses
    with httpx.Client(transport=model_transport()) as client:
        litellm.responses = partial(previous, client=HTTPHandler(client=client))
        try:
            runpy.run_module("minisweagent.run.mini", run_name="__main__")
        finally:
            litellm.responses = previous


if __name__ == "__main__":
    main()
