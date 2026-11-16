# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a multi-worker Uvicorn server whose workers each listen on their own SO_REUSEPORT socket.

Uvicorn's multi-worker mode hands every worker the same listening socket, and whichever worker
wakes first accepts the next connection. Under load that spreads connections unevenly, and since
clients keep connections alive, a few workers end up serving most of the traffic while the others
idle. Here each worker binds its own socket to the server's address with SO_REUSEPORT, so the
Linux kernel spreads new connections evenly across the workers' sockets.

The main process keeps Uvicorn's usual role (spawning, health-checking and restarting workers). Its
own socket reserves the address, also while workers are still starting, and is never listened on, so
it takes no connections.
"""

import inspect
import logging
import socket
import sys
from typing import Any, List, Optional

import uvicorn
from uvicorn.main import STARTUP_FAILURE
from uvicorn.supervisors import Multiprocess
from uvicorn.supervisors import multiprocess as uvicorn_multiprocess


LOG = logging.getLogger("uvicorn.error")


def reuse_port_supported() -> bool:
    """SO_REUSEPORT load-balances connections across sockets on Linux only."""
    return sys.platform == "linux" and hasattr(socket, "SO_REUSEPORT")


def _bind_reuse_port_socket(
    family: socket.AddressFamily, address: tuple, *, reuse_address: bool = True
) -> socket.socket:
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if reuse_address:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(address)
    except OSError:
        sock.close()
        raise
    # Not inheritable (PEP 446): a process a worker execs must not keep the worker's socket in the port's
    # SO_REUSEPORT group after the worker dies, where the kernel would keep sending it connections nobody accepts.
    return sock


class _ReusePortUvicornConfig(uvicorn.Config):
    """A Uvicorn config whose bound socket can share its address with the workers' sockets."""

    def bind_socket(self) -> socket.socket:
        family = socket.AF_INET6 if self.host and ":" in self.host else socket.AF_INET
        try:
            # SO_REUSEPORT would let this server join another process's listening sockets on the address and split
            # its connections, so first make sure nothing listens there. SO_REUSEADDR keeps TIME_WAIT from blocking.
            # Workers bind in _ReusePortWorker.startup, not here, so restarts are unaffected.
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((self.host, self.port))
            # Without SO_REUSEADDR this socket holds the address against every bind that does not join the group,
            # including while no worker listens yet, so a worker's listen() cannot lose the port to another server.
            sock = _bind_reuse_port_socket(family, (self.host, self.port), reuse_address=False)
            # Like uvicorn.Config.bind_socket, so the workers can be handed this socket.
            sock.set_inheritable(True)
        except OSError as exc:
            LOG.error(exc)
            sys.exit(STARTUP_FAILURE)
        address = f"[{self.host}]" if family == socket.AF_INET6 else self.host
        LOG.info(
            "Uvicorn running on http://%s:%d with one SO_REUSEPORT socket per worker", address, sock.getsockname()[1]
        )
        return sock


class _ReusePortWorker(uvicorn.Server):
    """A Uvicorn server that listens on its own SO_REUSEPORT socket instead of the inherited one."""

    async def startup(self, sockets: Optional[List[socket.socket]] = None) -> None:
        if sockets:
            own_sockets = []
            for inherited in sockets:
                own_sockets.append(_bind_reuse_port_socket(inherited.family, inherited.getsockname()))
                # The main process keeps its copy, which holds the address; this process does not need one.
                inherited.close()
            sockets = own_sockets
        await super().startup(sockets=sockets)


class _ReusePortProcess(uvicorn_multiprocess.Process):
    """A Uvicorn worker process whose server is a _ReusePortWorker, for Uvicorn versions that build each
    worker's server inside its Process."""

    @property
    def server(self) -> uvicorn.Server:
        if self._server is None:
            self._server = _ReusePortWorker(config=self.config)
        return self._server


def run_with_reuse_port(**uvicorn_kwargs: Any) -> None:
    """``uvicorn.run`` for a multi-worker server, with one SO_REUSEPORT listening socket per worker.

    Takes the same keyword arguments as ``uvicorn.run``; ``app`` must be an import string and
    ``workers`` above 1.
    """
    if not reuse_port_supported():
        raise RuntimeError("uvicorn_reuse_port needs Linux, where SO_REUSEPORT spreads connections across sockets")
    config = _ReusePortUvicornConfig(**uvicorn_kwargs)
    if config.workers <= 1:
        raise ValueError(f"run_with_reuse_port serves a multi-worker server; got workers={config.workers}")
    if not isinstance(config.app, str):
        raise ValueError("run_with_reuse_port starts workers by import: pass the app as an import string")
    sock = config.bind_socket()
    if "target" in inspect.signature(Multiprocess.__init__).parameters:
        # Older Uvicorn runs the target it is given in each worker.
        supervisor = Multiprocess(config, target=_ReusePortWorker(config=config).run, sockets=[sock])
    else:
        # Newer Uvicorn builds each worker's server inside its Process, at every place it starts a worker
        # (startup, restarts, scale-up). This main process runs only this server, so swap the class.
        uvicorn_multiprocess.Process = _ReusePortProcess
        supervisor = Multiprocess(config, sockets=[sock])
    try:
        supervisor.run()
    except KeyboardInterrupt:
        pass
