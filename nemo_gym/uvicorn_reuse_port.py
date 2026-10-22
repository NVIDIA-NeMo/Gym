# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a multi-worker Uvicorn server whose workers each listen on their own SO_REUSEPORT socket.

Uvicorn's multi-worker mode hands every worker the same listening socket, and whichever worker
wakes first accepts the next connection. Under load that spreads connections unevenly, and since
clients keep connections alive, a few workers end up serving most of the traffic while the others
idle. Here each worker binds its own socket to the server's address with SO_REUSEPORT, so the
Linux kernel spreads new connections evenly across the workers' sockets.

The supervisor keeps Uvicorn's usual role (spawning, health-checking and restarting workers). Its
own socket reserves the address and is never listened on, so it takes no connections.
"""

import logging
import socket
import sys
from typing import Any, List, Optional

import uvicorn
from uvicorn.main import STARTUP_FAILURE
from uvicorn.supervisors import Multiprocess


LOG = logging.getLogger("uvicorn.error")


def reuse_port_supported() -> bool:
    """SO_REUSEPORT load-balances connections across sockets on Linux only."""
    return sys.platform == "linux" and hasattr(socket, "SO_REUSEPORT")


def _bind_reuse_port_socket(family: socket.AddressFamily, address: tuple) -> socket.socket:
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(address)
    except OSError:
        sock.close()
        raise
    sock.set_inheritable(True)
    return sock


class ReusePortConfig(uvicorn.Config):
    """A Uvicorn config whose bound socket can share its address with the workers' sockets."""

    def bind_socket(self) -> socket.socket:
        family = socket.AF_INET6 if self.host and ":" in self.host else socket.AF_INET
        try:
            sock = _bind_reuse_port_socket(family, (self.host, self.port))
        except OSError as exc:
            LOG.error(exc)
            sys.exit(STARTUP_FAILURE)
        LOG.info("Uvicorn running on %s:%d with one SO_REUSEPORT socket per worker", self.host, self.port)
        return sock


class ReusePortServer(uvicorn.Server):
    """A Uvicorn server that listens on its own SO_REUSEPORT socket instead of the inherited one."""

    async def startup(self, sockets: Optional[List[socket.socket]] = None) -> None:
        if sockets:
            own_sockets = []
            for inherited in sockets:
                own_sockets.append(_bind_reuse_port_socket(inherited.family, inherited.getsockname()))
                # The supervisor keeps its copy, which holds the address; this process does not need one.
                inherited.close()
            sockets = own_sockets
        await super().startup(sockets=sockets)


def run_with_reuse_port(**uvicorn_kwargs: Any) -> None:
    """``uvicorn.run`` for a multi-worker server, with one SO_REUSEPORT listening socket per worker.

    Takes the same keyword arguments as ``uvicorn.run``; ``app`` must be an import string and
    ``workers`` above 1.
    """
    if not reuse_port_supported():
        raise RuntimeError("uvicorn_reuse_port needs Linux, where SO_REUSEPORT spreads connections across sockets")
    config = ReusePortConfig(**uvicorn_kwargs)
    if config.workers <= 1 or not isinstance(config.app, str):
        raise ValueError("run_with_reuse_port serves a multi-worker server: pass the app as an import string")
    server = ReusePortServer(config=config)
    sock = config.bind_socket()
    try:
        Multiprocess(config, target=server.run, sockets=[sock]).run()
    except KeyboardInterrupt:
        pass
