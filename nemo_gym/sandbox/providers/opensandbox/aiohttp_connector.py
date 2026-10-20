# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""aiohttp TCP connector whose cancelled connects cannot poison a file descriptor under uvloop.

Root cause of ``RuntimeError: File descriptor N is used by transport <TCPTransport
closed=False reading=True ...>`` in the agent servers:

``aiohttp.TCPConnector._wrap_create_connection`` lets ``aiohappyeyeballs`` build and
connect a plain Python ``socket.socket`` and then hands it to
``loop.create_connection(..., sock=sock)``. uvloop's ``sock=`` path does
``tr._open(sock.fileno())`` (libuv now owns the descriptor), awaits the
``connection_made`` waiter, and only *after* that calls ``tr._attach_fileobj(sock)``,
which is what makes the eventual libuv close ``detach()`` the Python socket. If the
await is cancelled first (``asyncio.wait_for`` around an SDK call, aiohttp's own
``sock_connect`` timeout, a cancelled rollout), the ``except BaseException`` branch
calls ``tr._close()``: libuv closes the descriptor immediately, but the Python socket
object is simply dropped still holding the number. When it is garbage-collected (or
closed by aiohappyeyeballs' cleanup) it calls ``close(fd)`` on whatever connection the
kernel has since assigned that number to. That victim transport stays registered in
the loop, alive and "reading", on a descriptor it no longer owns; the kernel keeps
handing the (lowest free) number to new sockets, and every ``sock_connect`` that lands
on it trips the loop's ``_ensure_fd_no_transport`` guard. One cancellation therefore
fails a large share of all later connects in that worker for the rest of its life.

uvloop 0.22.1 and current master both order the calls this way; the stdlib selector
loop is unaffected because its transport closes the very same socket object.

The fix: record the sockets aiohappyeyeballs creates for one connect attempt (through
the public ``socket_factory`` hook) and, when the attempt ends in cancellation or a
timeout under uvloop, ``detach()`` any that is still open. By then libuv has closed the
descriptor (uvloop never raises from ``await waiter`` before ``_open`` succeeded), so
detaching only drops the stale duplicate reference; nothing leaks.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import socket
from typing import Any, Callable

import aiohttp


LOGGER = logging.getLogger(__name__)

# Sockets created for the connect attempt currently running in this task tree.
# aiohappyeyeballs runs each address attempt in its own task; task creation copies
# the context, so the attempts share the list the awaiting connect installed.
_PENDING_SOCKETS: contextvars.ContextVar[list[socket.socket] | None] = contextvars.ContextVar(
    "nemo_gym_pending_connect_sockets", default=None
)

_SOCKET_FACTORY_SUPPORTED = "socket_factory" in inspect.signature(aiohttp.TCPConnector.__init__).parameters

SocketFactory = Callable[[Any], socket.socket]


def loop_detaches_sockets_late(loop: asyncio.AbstractEventLoop | None = None) -> bool:
    """True when ``loop.create_connection(sock=...)`` can close the descriptor
    through libuv while leaving the Python socket attached (uvloop)."""
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
    return type(loop).__module__.split(".", 1)[0] == "uvloop"


def _default_socket_factory(addr_info: Any) -> socket.socket:
    family, type_, proto, _, _ = addr_info
    return socket.socket(family=family, type=type_, proto=proto)


def detach_orphaned_sockets(sockets: list[socket.socket]) -> list[socket.socket]:
    """Detach every still-open socket; return the ones detached."""
    detached = []
    for sock in sockets:
        try:
            if sock.fileno() < 0:
                continue
            sock.detach()
        except OSError:
            continue
        detached.append(sock)
    return detached


class FdSafeTCPConnector(aiohttp.TCPConnector):
    """``aiohttp.TCPConnector`` that detaches the Python socket of a connect
    attempt cancelled under uvloop, so its later close cannot hit a reused
    descriptor. See the module docstring for the failure it prevents."""

    def __init__(self, *args: Any, socket_factory: SocketFactory | None = None, **kwargs: Any) -> None:
        self._user_socket_factory = socket_factory
        if _SOCKET_FACTORY_SUPPORTED:
            kwargs["socket_factory"] = self._make_socket
        elif socket_factory is not None:
            raise TypeError("this aiohttp does not support TCPConnector(socket_factory=...)")
        else:
            LOGGER.warning(
                "aiohttp %s has no TCPConnector socket_factory hook; cancelled connects under uvloop "
                "can still poison file descriptors",
                aiohttp.__version__,
            )
        super().__init__(*args, **kwargs)

    def _make_socket(self, addr_info: Any) -> socket.socket:
        factory = self._user_socket_factory or _default_socket_factory
        sock = factory(addr_info)
        pending = _PENDING_SOCKETS.get()
        if pending is not None:
            pending.append(sock)
        return sock

    async def _wrap_create_connection(self, *args: Any, **kwargs: Any) -> Any:
        created: list[socket.socket] = []
        token = _PENDING_SOCKETS.set(created)
        try:
            return await super()._wrap_create_connection(*args, **kwargs)
        except (asyncio.CancelledError, TimeoutError):
            # Only these leave create_connection(sock=) between tr._open and
            # tr._attach_fileobj. Any other error (e.g. OSError from _open) means
            # libuv never took the descriptor and the socket must close it itself.
            if loop_detaches_sockets_late(self._loop):
                detached = detach_orphaned_sockets(created)
                if detached:
                    LOGGER.debug(
                        "Detached %d socket(s) of a cancelled connect; libuv already closed the descriptor(s)",
                        len(detached),
                    )
            raise
        finally:
            _PENDING_SOCKETS.reset(token)
