# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License.
"""
CRLF heartbeat for the global aiohttp client.

Some network intermediaries drop TCP connections that carry no payload for a fixed
period and do not count TCP keepalive probes (for example AWS Global Accelerator, whose
idle timeout is fixed at 340 seconds). A non-streaming model call that takes longer than
that to produce its first response byte fails with ``ServerDisconnectedError`` when the
intermediary closes the connection.

HTTP/2 clients solve this with PING frames. aiohttp speaks HTTP/1.1 only, so this module
uses the HTTP/1.1 equivalent: while a request is in flight, write a bare CRLF on the
connection every ``heartbeat`` seconds. The bytes travel as encrypted TLS records, so the
intermediary sees payload and resets its idle timer. HTTP/1.1 servers must ignore empty
lines received before a request-line (RFC 9112 section 2.2), so the origin discards them
once the pending response has been sent and the connection stays reusable. The same idea
is standardised for SIP as the "CRLF keep-alive technique" (RFC 5626 section 4.4.1).

Enabled with ``global_aiohttp_crlf_heartbeat_seconds`` in the global config (default 0,
off). Only connections with a request in flight receive heartbeats.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from aiohttp import TCPConnector


CRLF = b"\r\n"
_LOG = logging.getLogger(__name__)


class HeartbeatTCPConnector(TCPConnector):
    """aiohttp.TCPConnector that writes CRLF on every in-flight connection periodically.

    Args:
        heartbeat: seconds between heartbeats. Keep it well below the intermediary's idle
            timeout (for example 60 s for a 340 s intermediary timeout). 0 disables the
            heartbeat; the connector is then identical to TCPConnector.
        on_heartbeat: optional callback receiving the number of connections written to,
            for metrics.
        **kwargs: forwarded to TCPConnector.
    """

    def __init__(
        self,
        *args: Any,
        heartbeat: float = 0.0,
        on_heartbeat: Optional[Callable[[int], None]] = None,
        **kwargs: Any,
    ) -> None:
        if heartbeat < 0:
            raise ValueError("heartbeat must be >= 0")
        super().__init__(*args, **kwargs)
        self._hb_every = float(heartbeat)
        self._hb_cb = on_heartbeat
        self._hb_task: Optional[asyncio.Task[None]] = None
        self.heartbeats_sent = 0

    async def _create_connection(self, req: Any, traces: Any, timeout: Any) -> Any:  # type: ignore[override]
        proto = await super()._create_connection(req, traces, timeout)
        self._ensure_task()
        return proto

    def _ensure_task(self) -> None:
        if self._hb_every <= 0 or self._closed:
            return
        if self._hb_task is None or self._hb_task.done():
            self._hb_task = asyncio.get_running_loop().create_task(
                self._heartbeat_loop(), name="aiohttp-crlf-heartbeat"
            )

    async def _heartbeat_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self._hb_every)
                n = self._beat_once()
                if n and self._hb_cb is not None:
                    try:
                        self._hb_cb(n)
                    except Exception:
                        _LOG.exception("on_heartbeat callback raised")
        except asyncio.CancelledError:
            pass
        except Exception:
            _LOG.exception("CRLF heartbeat loop crashed; it restarts on the next new connection")

    def _beat_once(self) -> int:
        """Write CRLF on each acquired (request in flight) connection; return how many."""
        n = 0
        for proto in list(self._acquired):
            transport = getattr(proto, "transport", None)
            if transport is None or transport.is_closing():
                continue
            try:
                transport.write(CRLF)
                n += 1
            except Exception:
                _LOG.debug("CRLF heartbeat write failed", exc_info=True)
        self.heartbeats_sent += n
        return n

    async def close(self) -> Any:  # type: ignore[override]
        if self._hb_task is not None:
            self._hb_task.cancel()
            self._hb_task = None
        return await super().close()
