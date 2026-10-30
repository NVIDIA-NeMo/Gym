# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model relay uploaded into a task sandbox: a harness's model calls go through it to Gym's model server.

A sandbox restored from a memory snapshot still runs the harness, whose model client was configured for the model
server of the job that took the snapshot. Pointing the client at this relay instead lets the restoring job re-point
the calls: it writes the new model server into the endpoint file, and every call the relay has not yet answered goes
to the new server, byte for byte as the harness sent it. The harness sees a slow call, not an error.

- Endpoint file (``--endpoint``): ``{"gen": N, "upstream": URL}``. Read on every request and every 200 ms. A request
  for ``/v1/<rest>`` goes to ``<upstream>/<rest>``; Gym's model base URLs end in ``/v1``.
- Hold: while the supervisor's pause request in ``--session-dir`` names a later sequence number than its continue
  request (``pause.request`` and ``continue.request``, see supervisor_client.py), new requests wait unsent.
- Re-send: a request is kept until its response starts, and the response's headers reach the harness only together
  with its first body byte. When the endpoint's ``gen`` changes, every request sent under an earlier one that has
  not started its response is sent again to the new upstream; a response already started is cut with a reset.
- An upstream that cannot be reached, or resets a request before its response starts, is retried with backoff
  while the endpoint stays the same, for up to ``--hold-timeout-s``; then the harness gets a 502. Error replies of
  the model server reach the harness as they are.
- Status file (``--status``): ``{"pid", "start", "port", "gen", "held", "inflight", "resent"}``, about every second;
  ``start`` is the process start time from /proc, so a reader can tell a restarted relay from the one it knew.

Runs on Python 3.6 or later with the standard library only: task images ship old interpreters.

usage: model_relay.py --port PORT --endpoint ENDPOINT_JSON --session-dir DIR --status STATUS_JSON
                      [--hold-timeout-s SECONDS]
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import ssl
import sys
import time
from urllib.parse import urlsplit


LOG = logging.getLogger("model_relay")
POLL_S = 0.2
STATUS_INTERVAL_S = 1.0
PAUSE_REQUEST_FILE = "pause.request"
CONTINUE_REQUEST_FILE = "continue.request"
# Retry delays for an unreachable upstream; the last one repeats.
BACKOFF_S = (0.2, 0.5, 1.0, 2.0, 5.0)
READ_SIZE = 65536
# Headers that describe one connection, not the request: the relay frames each hop itself.
HOP_BY_HOP = frozenset(
    (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "expect",
    )
)


class RelayError(Exception):
    """A request the relay cannot read or forward."""


def read_seq(path):
    """The sequence number in a supervisor request file; 0 when it is missing or not a number."""
    try:
        with open(path) as handle:
            value = handle.readline().strip()
    except OSError:
        return 0
    return int(value) if value.isdigit() and len(value) <= 15 else 0


def read_endpoint(path):
    """``(gen, upstream)`` from the endpoint file, or None while it is missing or incomplete."""
    try:
        with open(path) as handle:
            data = json.load(handle)
        return int(data["gen"]), str(data["upstream"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def process_start_time():
    """This process's start time in clock ticks since boot (/proc/self/stat field 22), or None without /proc."""
    try:
        with open("/proc/self/stat") as handle:
            stat = handle.read()
    except OSError:
        return None
    fields = stat[stat.rindex(")") + 2 :].split()
    return int(fields[19])


def upstream_target(upstream, path):
    """Host, port, TLS and request target for ``path`` (``/v1/<rest>?query``) on ``upstream``."""
    parts = urlsplit(upstream)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise RelayError("unsupported upstream %r" % upstream)
    secure = parts.scheme == "https"
    port = parts.port or (443 if secure else 80)
    suffix = path[3:] if path == "/v1" or path.startswith(("/v1/", "/v1?")) else path
    host = parts.hostname if ":" not in parts.hostname else "[%s]" % parts.hostname
    authority = host if parts.port is None else "%s:%d" % (host, port)
    return parts.hostname, port, secure, authority, parts.path.rstrip("/") + suffix


class Request(object):
    def __init__(self, method, target, version, headers, body):
        self.method = method
        self.target = target
        self.version = version
        self.headers = headers
        self.body = body

    def header(self, name):
        for key, value in self.headers:
            if key.lower() == name:
                return value
        return None

    def keep_alive(self):
        connection = (self.header("connection") or "").lower()
        if self.version == "HTTP/1.0":
            return "keep-alive" in connection
        return "close" not in connection

    def encode(self, authority, target):
        """The request as sent upstream: the harness's method, headers and body, framed for one connection."""
        lines = ["%s %s HTTP/1.1" % (self.method, target), "Host: %s" % authority]
        lines.extend("%s: %s" % (key, value) for key, value in self.headers if key.lower() not in HOP_BY_HOP)
        if self.body or self.method in ("POST", "PUT", "PATCH"):
            lines.append("Content-Length: %d" % len(self.body))
        lines.append("Connection: close")
        return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + self.body


async def read_head(reader):
    """Start line and headers of an HTTP message; None at a clean end of stream."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError as error:
        if not error.partial.strip():
            return None
        raise RelayError("truncated message head")
    lines = head.decode("latin-1").split("\r\n")
    headers = []
    for line in lines[1:]:
        if line:
            name, _, value = line.partition(":")
            headers.append((name.strip(), value.strip()))
    return lines[0], headers


async def read_chunk(reader):
    """One chunk of a chunked body; b"" after the last one and its trailers."""
    line = await reader.readline()
    try:
        size = int(line.split(b";", 1)[0].strip(), 16)
    except ValueError:
        raise RelayError("bad chunk size %r" % line[:40])
    if size == 0:
        while (await reader.readline()) not in (b"\r\n", b"\n", b""):
            pass
        return b""
    data = await reader.readexactly(size)
    await reader.readexactly(2)
    return data


async def read_request(reader):
    head = await read_head(reader)
    if head is None:
        return None
    start, headers = head
    parts = start.split(" ")
    if len(parts) != 3:
        raise RelayError("bad request line %r" % start[:80])
    request = Request(parts[0], parts[1], parts[2], headers, b"")
    if (request.header("transfer-encoding") or "").lower().endswith("chunked"):
        chunks = []
        while True:
            chunk = await read_chunk(reader)
            if not chunk:
                break
            chunks.append(chunk)
        request.body = b"".join(chunks)
    elif request.header("content-length"):
        request.body = await reader.readexactly(int(request.header("content-length")))
    return request


class UpstreamResponse(object):
    """A response whose head and first body bytes arrived; the rest is read with ``next_chunk``."""

    def __init__(self, reader, writer, status_line, headers, framing, remaining):
        self.reader = reader
        self.writer = writer
        self.status_line = status_line
        self.headers = headers
        self.framing = framing  # "none", "length", "chunked" or "close"
        self.remaining = remaining
        self.first = b""

    async def next_chunk(self):
        if self.framing == "none":
            return b""
        if self.framing == "length":
            if self.remaining == 0:
                return b""
            data = await self.reader.read(min(READ_SIZE, self.remaining))
            if not data:
                raise asyncio.IncompleteReadError(b"", self.remaining)
            self.remaining -= len(data)
            return data
        if self.framing == "chunked":
            return await read_chunk(self.reader)
        return await self.reader.read(READ_SIZE)

    def close(self):
        self.writer.close()


class Relay(object):
    def __init__(self, endpoint_path, session_dir, status_path, hold_timeout_s):
        self.endpoint_path = endpoint_path
        self.pause_path = os.path.join(session_dir, PAUSE_REQUEST_FILE)
        self.continue_path = os.path.join(session_dir, CONTINUE_REQUEST_FILE)
        self.status_path = status_path
        self.hold_timeout_s = hold_timeout_s
        self.start = process_start_time()
        self.port = 0
        self.gen = None
        self.upstream = None
        # Set, then replaced, whenever the endpoint changes, to wake the requests waiting on it.
        self.changed = asyncio.Event()
        self.held = 0
        self.inflight = 0
        self.resent = 0
        self.tls = None
        self.clients = set()

    # -- endpoint, hold and status ---------------------------------------------------------------------------

    def refresh(self):
        endpoint = read_endpoint(self.endpoint_path)
        if endpoint is None or endpoint == (self.gen, self.upstream):
            return
        if self.gen is not None:
            LOG.info("endpoint gen %s -> %s: %s", self.gen, endpoint[0], endpoint[1])
        self.gen, self.upstream = endpoint
        self.changed.set()
        self.changed = asyncio.Event()

    def paused(self):
        return read_seq(self.pause_path) > read_seq(self.continue_path)

    def write_status(self):
        status = {
            "pid": os.getpid(),
            "start": self.start,
            "port": self.port,
            "gen": self.gen,
            "held": self.held,
            "inflight": self.inflight,
            "resent": self.resent,
        }
        temporary = "%s.%d.tmp" % (self.status_path, os.getpid())
        with open(temporary, "w") as handle:
            json.dump(status, handle)
        os.replace(temporary, self.status_path)

    async def watch(self):
        last_status = 0.0
        while True:
            self.refresh()
            now = time.monotonic()
            if now - last_status >= STATUS_INTERVAL_S:
                try:
                    self.write_status()
                except OSError as error:
                    LOG.warning("status not written: %r", error)
                last_status = now
            await asyncio.sleep(POLL_S)

    async def wait_released(self):
        if not self.paused():
            return
        self.held += 1
        try:
            while self.paused():
                await asyncio.sleep(POLL_S)
        finally:
            self.held -= 1

    async def wait_changed(self, changed, delay_s):
        """Sleep ``delay_s``, or less if the endpoint changes first."""
        try:
            await asyncio.wait_for(changed.wait(), delay_s)
        except asyncio.TimeoutError:
            pass

    # -- one request -----------------------------------------------------------------------------------------

    async def open(self, upstream, request):
        """Send ``request`` to ``upstream`` and return its response once the first body bytes arrived."""
        host, port, secure, authority, target = upstream_target(upstream, request.target)
        if secure and self.tls is None:
            self.tls = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(
            host, port, ssl=self.tls if secure else None, server_hostname=host if secure else None
        )
        try:
            writer.write(request.encode(authority, target))
            await writer.drain()
            while True:
                head = await read_head(reader)
                if head is None:
                    raise ConnectionResetError("upstream closed before its response")
                status_line, headers = head
                status = int(status_line.split(" ")[1])
                if status >= 200 or status == 101:
                    break  # an interim 1xx response precedes the real one
            response = UpstreamResponse(reader, writer, status_line, headers, "close", None)
            values = dict((key.lower(), value) for key, value in headers)
            if request.method == "HEAD" or status in (204, 304):
                response.framing = "none"
            elif values.get("transfer-encoding", "").lower().endswith("chunked"):
                response.framing = "chunked"
            elif "content-length" in values:
                response.framing, response.remaining = "length", int(values["content-length"])
            response.first = await response.next_chunk()
            return response
        except BaseException:
            writer.close()
            raise

    async def forward(self, request, writer):
        self.inflight += 1
        try:
            await self._forward(request, writer)
        finally:
            self.inflight -= 1

    async def _forward(self, request, writer):
        failures = 0
        failing_since = None
        while True:
            await self.wait_released()
            self.refresh()
            changed, gen, upstream = self.changed, self.gen, self.upstream
            if upstream is None:
                attempt = asyncio.ensure_future(self._no_endpoint())
            else:
                attempt = asyncio.ensure_future(self.open(upstream, request))
            waiter = asyncio.ensure_future(changed.wait())
            await asyncio.wait({attempt, waiter}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            if not attempt.done():
                attempt.cancel()
                await asyncio.gather(attempt, return_exceptions=True)
                self.resent += 1
                failures, failing_since = 0, None
                LOG.info("%s %s sent under gen %s re-sent under gen %s", request.method, request.target, gen, self.gen)
                continue
            try:
                response = attempt.result()
            except (OSError, EOFError, asyncio.IncompleteReadError, RelayError, ValueError, IndexError) as error:
                now = time.monotonic()
                failing_since = now if failing_since is None else failing_since
                if now - failing_since >= self.hold_timeout_s:
                    LOG.warning(
                        "%s %s: upstream unreachable for %.0fs: %r",
                        request.method,
                        request.target,
                        now - failing_since,
                        error,
                    )
                    await self.reply_error(writer, 502, "model server unreachable: %r" % (error,))
                    return
                delay = BACKOFF_S[min(failures, len(BACKOFF_S) - 1)]
                failures += 1
                await self.wait_changed(changed, min(delay, self.hold_timeout_s - (now - failing_since)))
                if self.changed is not changed:
                    failures, failing_since = 0, None
                continue
            await self.respond(response, request, writer, changed)
            return

    async def _no_endpoint(self):
        raise RelayError("no endpoint in %s" % self.endpoint_path)

    async def respond(self, response, request, writer, changed):
        """Pass ``response`` on; a response the endpoint change overtakes is cut with a reset."""
        headers = [(key, value) for key, value in response.headers if key.lower() not in HOP_BY_HOP]
        if response.framing == "none" and request.method == "HEAD":
            headers.extend((key, value) for key, value in response.headers if key.lower() == "content-length")
        elif response.framing == "length":
            headers.append(("Content-Length", str(len(response.first) + response.remaining)))
        elif response.framing in ("chunked", "close"):
            headers.append(("Transfer-Encoding", "chunked"))
        chunked = response.framing in ("chunked", "close")
        head = "\r\n".join(
            ["HTTP/1.1 " + response.status_line.split(" ", 1)[1]] + ["%s: %s" % item for item in headers]
        )
        try:
            writer.write((head + "\r\n\r\n").encode("latin-1") + self.frame(response.first, chunked))
            await writer.drain()
            data = response.first
            while data:
                reading = asyncio.ensure_future(response.next_chunk())
                waiter = asyncio.ensure_future(changed.wait())
                await asyncio.wait({reading, waiter}, return_when=asyncio.FIRST_COMPLETED)
                waiter.cancel()
                if not reading.done():
                    reading.cancel()
                    await asyncio.gather(reading, return_exceptions=True)
                    LOG.warning(
                        "%s %s: endpoint changed after its response started; cutting it",
                        request.method,
                        request.target,
                    )
                    writer.transport.abort()
                    raise ConnectionResetError("response cut by an endpoint change")
                data = reading.result()
                writer.write(self.frame(data, chunked))
                await writer.drain()
            if chunked:
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        finally:
            response.close()

    @staticmethod
    def frame(data, chunked):
        if not chunked or not data:
            return data
        return ("%x\r\n" % len(data)).encode("ascii") + data + b"\r\n"

    async def reply_error(self, writer, status, message):
        body = json.dumps({"error": {"message": message, "type": "model_relay"}}).encode("utf-8")
        reason = {502: "Bad Gateway", 400: "Bad Request"}.get(status, "Error")
        head = "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (
            status,
            reason,
            len(body),
        )
        writer.write(head.encode("latin-1") + body)
        await writer.drain()

    # -- connections -----------------------------------------------------------------------------------------

    async def serve(self, reader, writer):
        self.clients.add(writer)
        try:
            while True:
                try:
                    request = await read_request(reader)
                except (RelayError, ValueError) as error:
                    await self.reply_error(writer, 400, str(error))
                    return
                if request is None:
                    return
                await self.forward(request, writer)
                if not request.keep_alive():
                    return
        except (OSError, EOFError, asyncio.IncompleteReadError):
            pass
        finally:
            self.clients.discard(writer)
            writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--hold-timeout-s", type=float, default=1800.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s model_relay %(levelname)s %(message)s")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    relay = Relay(args.endpoint, args.session_dir, args.status, args.hold_timeout_s)
    server = loop.run_until_complete(asyncio.start_server(relay.serve, "127.0.0.1", args.port))
    relay.port = server.sockets[0].getsockname()[1]
    relay.refresh()
    relay.write_status()
    LOG.info("listening on 127.0.0.1:%d, endpoint gen %s", relay.port, relay.gen)
    stop = asyncio.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    watcher = loop.create_task(relay.watch())
    loop.run_until_complete(stop.wait())
    watcher.cancel()
    server.close()
    for writer in list(relay.clients):
        writer.transport.abort()
    loop.run_until_complete(asyncio.gather(watcher, server.wait_closed(), return_exceptions=True))
    loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
