# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep idle TCP flows alive while native mini-SWE awaits non-streaming inference."""

import runpy
import socket


class KeepAliveSocket(socket.socket):
    """Apply transport liveness settings without changing request or agent budgets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.family not in (socket.AF_INET, socket.AF_INET6) or self.type != socket.SOCK_STREAM:
            return
        self.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 30), ("TCP_KEEPCNT", 5)):
            if option := getattr(socket, name, None):
                self.setsockopt(socket.IPPROTO_TCP, option, value)


if __name__ == "__main__":
    # Install before HTTP clients are imported. SSL wraps the same TCP socket,
    # so the options also survive TLS negotiation. Child task commands do not
    # inherit this Python-process-local change.
    socket.socket = KeepAliveSocket
    runpy.run_module("minisweagent.run.mini", run_name="__main__")
