# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model relay as a sandbox runs it: a separate process between a client and real local upstream servers.

model_relay.mjs runs on a JavaScript runtime: ``NEMO_GYM_TEST_JS_RUNTIME``, else ``node`` or ``bun`` on PATH. Without
one, these tests are skipped; a ``NEMO_GYM_TEST_JS_RUNTIME`` that names no executable is an error.
"""

import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from nemo_gym.agents.utils.sandbox_session import HARNESS_RUNNER_SCRIPT
from nemo_gym.agents.utils.supervisor_client import PAUSED_FILE, NotResumable, RestoredLaunch, check_restored
from tests.unit_tests.agents.test_harness_pause import LocalSandbox, _start_time


RELAY = HARNESS_RUNNER_SCRIPT.with_name("model_relay.mjs")


def _js_runtime() -> str | None:
    named = os.environ.get("NEMO_GYM_TEST_JS_RUNTIME")
    if named:
        assert os.access(named, os.X_OK), f"NEMO_GYM_TEST_JS_RUNTIME={named!r} is not an executable"
        return named
    return shutil.which("node") or shutil.which("bun")


JS_RUNTIME = _js_runtime()

pytestmark = pytest.mark.skipif(JS_RUNTIME is None, reason="no JavaScript runtime (node or bun) on this machine")


# Replies no retry can mend, each sent whole; the upstream then keeps the connection open.
MALFORMED_REPLIES = {
    "bad_status_line": b"HTTP/1.1 fine OK\r\nContent-Length: 2\r\n\r\n{}",
    "bad_content_length": b"HTTP/1.1 200 OK\r\nContent-Length: two\r\n\r\n{}",
    "bad_chunk_size": b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n{}\r\n",
}


class Upstream:
    """A local HTTP/1.1 model server whose behaviour each test picks: it records every request it reads."""

    def __init__(self, mode: str = "respond", status: int = 200) -> None:
        self.mode = mode
        self.status = status
        self.requests: list[dict] = []
        self.release = asyncio.Event()
        self.server: asyncio.AbstractServer | None = None
        self.writers: set[asyncio.StreamWriter] = set()
        self.port = 0

    async def start(self, port: int = 0) -> "Upstream":
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", port)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        self.server.close()
        for writer in list(self.writers):
            writer.transport.abort()
        await self.server.wait_closed()

    def url(self, prefix: str = "/rollout/key-a1/v1") -> str:
        return f"http://127.0.0.1:{self.port}{prefix}"

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.add(writer)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            headers = dict(
                (key.strip().lower(), value.strip())
                for key, _, value in (line.partition(":") for line in lines[1:] if line)
            )
            body = await reader.readexactly(int(headers.get("content-length", "0")))
            method, path, _ = lines[0].split(" ")
            self.requests.append({"method": method, "path": path, "headers": headers, "body": body})
            digest = hashlib.sha256(body).hexdigest()
            if self.mode == "reset_once" and len(self.requests) == 1:
                writer.transport.abort()
                return
            if self.mode == "drop_mid_stream":
                data = b"data: 0\n\n"
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                    + b"%x\r\n%s\r\n" % (len(data), data)
                )
                await writer.drain()
                await asyncio.sleep(0.2)
                writer.transport.abort()
                return
            if self.mode in MALFORMED_REPLIES:
                writer.write(MALFORMED_REPLIES[self.mode])
                await writer.drain()
                await asyncio.Event().wait()
            if self.mode == "hold":
                await asyncio.Event().wait()
            if self.mode == "hold_after_headers":
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n")
                await writer.drain()
                await asyncio.Event().wait()
            if self.mode == "stream":
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                )
                for index in range(3):
                    if index == 1:
                        await self.release.wait()
                    data = f"data: {index}\n\n".encode()
                    writer.write(b"%x\r\n%s\r\n" % (len(data), data))
                    await writer.drain()
                writer.write(b"0\r\n\r\n")
                await writer.drain()
                return
            if self.mode == "delay":
                await asyncio.sleep(json.loads(body)["delay"])
            payload = json.dumps({"port": self.port, "path": path, "sha256": digest, "body": body.decode()}).encode()
            writer.write(
                b"HTTP/1.1 %d X\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
                % (self.status, len(payload), payload)
            )
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.writers.discard(writer)
            writer.close()


class RelayProcess:
    def __init__(self, tmp_path: Path, hold_timeout_s: float = 30.0) -> None:
        self.endpoint = tmp_path / "endpoint.json"
        self.session = tmp_path / "session"
        self.session.mkdir(exist_ok=True)
        self.status_path = tmp_path / "status.json"
        self.hold_timeout_s = hold_timeout_s
        self.process: subprocess.Popen | None = None

    def point(self, gen: int, upstream: str) -> None:
        temporary = self.endpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps({"gen": gen, "upstream": upstream}))
        temporary.replace(self.endpoint)

    def request_file(self, name: str, seq: int) -> None:
        (self.session / name).write_text(str(seq))

    def status(self) -> dict:
        return json.loads(self.status_path.read_text())

    async def start(self) -> "RelayProcess":
        argv = [JS_RUNTIME, str(RELAY), "--port", "0", "--endpoint", str(self.endpoint)]
        argv += ["--session-dir", str(self.session), "--status", str(self.status_path)]
        argv += ["--hold-timeout-s", str(self.hold_timeout_s)]
        self.process = subprocess.Popen(argv)
        # The runtime may start cold from a network file system while other test workers do the same.
        await wait_until(lambda: self.status_path.exists() and self.status()["port"] > 0, timeout=30)
        self.port = self.status()["port"]
        return self

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


async def wait_until(condition, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.05)


async def read_response(reader: asyncio.StreamReader) -> tuple[int, dict, bytes]:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    headers = dict((k.strip().lower(), v.strip()) for k, _, v in (line.partition(":") for line in lines[1:] if line))
    if headers.get("transfer-encoding") == "chunked":
        body = b""
        while True:
            size = int((await reader.readline()).strip(), 16)
            if size == 0:
                await reader.readline()
                break
            body += await reader.readexactly(size)
            await reader.readexactly(2)
    else:
        body = await reader.readexactly(int(headers.get("content-length", "0")))
    return int(lines[0].split(" ")[1]), headers, body


def encode_request(path: str, body: bytes, headers: dict | None = None, close: bool = True) -> bytes:
    lines = [f"POST {path} HTTP/1.1", "Host: 127.0.0.1", f"Content-Length: {len(body)}"]
    lines += [f"{key}: {value}" for key, value in (headers or {}).items()]
    if close:
        lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + body


async def call(
    port: int, path: str, body: bytes, headers: dict | None = None, timeout: float = 30.0
) -> tuple[int, dict, bytes]:
    async with asyncio.timeout(timeout):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(encode_request(path, body, headers))
            await writer.drain()
            return await read_response(reader)
        finally:
            writer.close()


@pytest.fixture
async def relay(tmp_path: Path):
    process = RelayProcess(tmp_path)
    yield process
    process.stop()


async def test_a_v1_call_goes_to_the_upstream_base_with_its_headers_and_body(relay: RelayProcess) -> None:
    upstream = await Upstream().start()
    relay.point(1, upstream.url())
    await relay.start()
    body = b'{"model": "policy", "input": "hi"}'

    status, headers, reply = await call(relay.port, "/v1/responses?x=1", body, {"X-Session-Id": "sess:launch-1"})

    assert status == 200 and headers["content-type"] == "application/json"
    seen = upstream.requests[0]
    assert seen["path"] == "/rollout/key-a1/v1/responses?x=1"
    assert seen["body"] == body and seen["headers"]["x-session-id"] == "sess:launch-1"
    assert json.loads(reply)["sha256"] == hashlib.sha256(body).hexdigest()
    await upstream.stop()


async def test_new_calls_wait_unsent_while_a_pause_is_open(relay: RelayProcess) -> None:
    upstream = await Upstream().start()
    relay.point(1, upstream.url())
    relay.request_file("pause.request", 2)
    relay.request_file("continue.request", 1)
    await relay.start()

    pending = asyncio.ensure_future(call(relay.port, "/v1/responses", b"{}"))
    await wait_until(lambda: relay.status()["held"] == 1)
    await asyncio.sleep(0.5)
    assert upstream.requests == [] and not pending.done()

    relay.request_file("continue.request", 2)
    status, _, _ = await asyncio.wait_for(pending, 10)
    assert status == 200 and len(upstream.requests) == 1
    await wait_until(lambda: relay.status()["held"] == 0)
    await upstream.stop()


async def test_an_endpoint_change_resends_an_unanswered_call_byte_for_byte(relay: RelayProcess) -> None:
    dead = await Upstream("hold").start()  # a model server that will never answer, like one that crashed
    relay.point(1, dead.url())
    await relay.start()
    body = json.dumps({"input": [{"role": "user", "content": "x" * 5000}]}).encode()
    pending = asyncio.ensure_future(call(relay.port, "/v1/responses", body, {"X-Session-Id": "s"}))
    await wait_until(lambda: len(dead.requests) == 1)

    fresh = await Upstream().start()
    relay.point(2, fresh.url("/rollout/key-a2/v1"))
    status, _, reply = await asyncio.wait_for(pending, 10)

    assert status == 200 and json.loads(reply)["port"] == fresh.port
    first, again = dead.requests[0], fresh.requests[0]
    assert again["body"] == first["body"] and again["method"] == first["method"]
    assert again["path"] == "/rollout/key-a2/v1/responses"
    assert {k: v for k, v in again["headers"].items() if k != "host"} == {
        k: v for k, v in first["headers"].items() if k != "host"
    }
    await wait_until(lambda: relay.status()["resent"] == 1 and relay.status()["gen"] == 2)
    await dead.stop()
    await fresh.stop()


async def test_response_headers_wait_for_the_first_body_byte(relay: RelayProcess) -> None:
    stalled = await Upstream("hold_after_headers").start()
    relay.point(1, stalled.url())
    await relay.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
    writer.write(encode_request("/v1/responses", b"{}"))
    await writer.drain()
    await wait_until(lambda: len(stalled.requests) == 1)

    # The upstream sent its headers; the client must not see them, or the call could not be sent again.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(reader.read(1), 1.0)
    fresh = await Upstream().start()
    relay.point(2, fresh.url())
    status, _, reply = await asyncio.wait_for(read_response(reader), 10)

    assert status == 200 and json.loads(reply)["port"] == fresh.port
    writer.close()
    await stalled.stop()
    await fresh.stop()


async def test_a_response_already_started_is_cut_by_an_endpoint_change(relay: RelayProcess) -> None:
    streaming = await Upstream("stream").start()
    relay.point(1, streaming.url())
    await relay.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
    writer.write(encode_request("/v1/responses", b"{}"))
    await writer.drain()
    await asyncio.wait_for(reader.readuntil(b"data: 0"), 10)

    relay.point(2, streaming.url())
    with pytest.raises((ConnectionError, asyncio.IncompleteReadError)):
        await asyncio.wait_for(reader.readuntil(b"0\r\n\r\n"), 10)
    await asyncio.sleep(0.5)
    assert len(streaming.requests) == 1, "a response that started is never sent again"
    writer.close()
    await streaming.stop()


async def test_a_streamed_response_passes_through_in_order(relay: RelayProcess) -> None:
    streaming = await Upstream("stream").start()
    streaming.release.set()
    relay.point(1, streaming.url())
    await relay.start()

    status, headers, body = await call(relay.port, "/v1/responses", b"{}")

    assert status == 200 and headers["content-type"] == "text/event-stream"
    assert body == b"data: 0\n\ndata: 1\n\ndata: 2\n\n"
    await streaming.stop()


async def test_an_unreachable_upstream_is_retried_until_it_comes_up(relay: RelayProcess) -> None:
    probe = await Upstream().start()
    port = probe.port
    await probe.stop()  # nothing listens on port now
    relay.point(1, f"http://127.0.0.1:{port}/v1")
    await relay.start()
    pending = asyncio.ensure_future(call(relay.port, "/v1/responses", b'{"a": 1}'))
    await asyncio.sleep(1.0)
    assert not pending.done()

    upstream = await Upstream().start(port)
    status, _, reply = await asyncio.wait_for(pending, 15)
    assert status == 200 and json.loads(reply)["body"] == '{"a": 1}'
    await upstream.stop()


async def test_an_upstream_unreachable_past_the_hold_timeout_gets_a_502(tmp_path: Path) -> None:
    probe = await Upstream().start()
    port = probe.port
    await probe.stop()
    relay = RelayProcess(tmp_path, hold_timeout_s=1.5)
    relay.point(1, f"http://127.0.0.1:{port}/v1")
    try:
        await relay.start()
        status, _, body = await asyncio.wait_for(call(relay.port, "/v1/responses", b"{}"), 15)
        assert status == 502 and "unreachable" in json.loads(body)["error"]["message"]
    finally:
        relay.stop()


async def test_an_upstream_that_accepts_but_never_completes_tls_gets_a_502_within_the_hold_timeout(
    tmp_path: Path,
) -> None:
    accepted: list[asyncio.StreamWriter] = []

    async def stall(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted.append(writer)
        await reader.read()  # takes the ClientHello and answers nothing

    silent = await asyncio.start_server(stall, "127.0.0.1", 0)
    relay = RelayProcess(tmp_path, hold_timeout_s=1.0)
    relay.point(1, f"https://127.0.0.1:{silent.sockets[0].getsockname()[1]}/v1")
    try:
        await relay.start()
        started = asyncio.get_running_loop().time()
        status, _, body = await asyncio.wait_for(call(relay.port, "/v1/responses", b"{}"), 15)
        waited = asyncio.get_running_loop().time() - started
        assert status == 502 and "no connection to the upstream" in json.loads(body)["error"]["message"]
        # The time the attempt waited to connect counts against the hold timeout, so the call ends about there.
        assert 0.9 <= waited < 8 and len(accepted) == 1
        await wait_until(lambda: relay.status()["inflight"] == 0)
    finally:
        relay.stop()
        for writer in accepted:
            writer.transport.abort()
        silent.close()
        await silent.wait_closed()


@pytest.mark.parametrize(
    ("mode", "cause"),
    [
        ("bad_status_line", "bad status line"),
        ("bad_content_length", "bad content-length"),
        ("bad_chunk_size", "bad chunk size"),
    ],
)
async def test_a_reply_the_relay_cannot_use_gets_a_502_at_once_and_is_not_asked_for_again(
    relay: RelayProcess, mode: str, cause: str
) -> None:
    upstream = await Upstream(mode).start()
    relay.point(1, upstream.url())
    await relay.start()
    # Well inside the 30 s an unreachable upstream is retried for.
    status, _, body = await asyncio.wait_for(call(relay.port, "/v1/responses", b"{}"), 10)
    message = json.loads(body)["error"]["message"]
    assert status == 502 and "cannot use the upstream" in message and cause in message
    await asyncio.sleep(0.5)
    assert len(upstream.requests) == 1
    await upstream.stop()


@pytest.mark.parametrize(
    ("address", "cause"),
    [
        ("ftp://127.0.0.1:9/v1", "unsupported upstream"),
        ("http:///v1", "unsupported upstream"),
        ("http://127.0.0.1:99999/v1", "bad port"),
    ],
)
async def test_an_upstream_address_the_relay_cannot_use_gets_a_502_at_once_and_a_later_endpoint_is_served(
    relay: RelayProcess, address: str, cause: str
) -> None:
    relay.point(1, address)
    await relay.start()
    status, _, body = await asyncio.wait_for(call(relay.port, "/v1/responses", b"{}"), 10)
    message = json.loads(body)["error"]["message"]
    assert status == 502 and "cannot use the upstream" in message and cause in message

    upstream = await Upstream().start()
    relay.point(2, upstream.url())
    status, _, _ = await asyncio.wait_for(call(relay.port, "/v1/responses", b"{}"), 10)
    assert status == 200 and len(upstream.requests) == 1
    await upstream.stop()


async def test_a_chunk_size_cut_short_by_a_closed_upstream_is_sent_again(relay: RelayProcess) -> None:
    # A closed stream leaves part of a line: that is an interrupted response, not a reply the relay cannot use.
    served = 0

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal served
        await reader.readuntil(b"\r\n\r\n")
        served += 1
        head = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        writer.write(head + (b"z" if served == 1 else b"2\r\n{}\r\n0\r\n\r\n"))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    relay.point(1, f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1")
    await relay.start()
    status, _, body = await asyncio.wait_for(call(relay.port, "/v1/responses", b""), 15)
    assert (status, body, served) == (200, b"{}", 2)
    server.close()
    await server.wait_closed()


async def test_an_error_reply_of_the_model_server_is_passed_on_not_retried(relay: RelayProcess) -> None:
    upstream = await Upstream(status=409).start()
    relay.point(1, upstream.url())
    await relay.start()

    status, _, _ = await call(relay.port, "/v1/responses", b"{}")

    assert status == 409 and len(upstream.requests) == 1
    await upstream.stop()


async def test_concurrent_calls_each_get_their_own_reply(relay: RelayProcess) -> None:
    upstream = await Upstream("delay").start()
    relay.point(1, upstream.url())
    await relay.start()
    bodies = [json.dumps({"delay": 0.05 * (7 - index), "n": index}).encode() for index in range(8)]

    replies = await asyncio.gather(*(call(relay.port, "/v1/responses", body) for body in bodies))

    assert [json.loads(json.loads(reply)["body"])["n"] for _, _, reply in replies] == list(range(8))
    await upstream.stop()


async def test_one_connection_serves_several_calls(relay: RelayProcess) -> None:
    upstream = await Upstream().start()
    relay.point(1, upstream.url())
    await relay.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
    for index in range(3):
        writer.write(encode_request("/v1/responses", b'{"n": %d}' % index, close=False))
        await writer.drain()
        status, _, reply = await asyncio.wait_for(read_response(reader), 15)
        assert status == 200 and json.loads(reply)["body"] == '{"n": %d}' % index
    writer.close()
    await upstream.stop()


async def test_the_status_file_names_the_process_and_its_counters(relay: RelayProcess) -> None:
    relay.point(3, "http://127.0.0.1:9/v1")
    await relay.start()

    status = relay.status()
    assert status["pid"] == relay.process.pid and status["gen"] == 3
    assert status["held"] == status["inflight"] == status["resent"] == 0
    stat = Path(f"/proc/{relay.process.pid}/stat").read_text()
    assert status["start"] == int(stat[stat.rindex(")") + 2 :].split()[19])


async def test_a_restore_sends_cut_and_held_calls_to_the_new_endpoint_only_after_the_continue(
    relay: RelayProcess,
) -> None:
    dead, fresh = await Upstream("hold").start(), await Upstream().start()
    try:
        relay.point(1, dead.url())
        await relay.start()
        cut_body, held_body = b'{"call": "in flight at the snapshot"}', b'{"call": "made while held"}'
        cut = asyncio.ensure_future(call(relay.port, "/v1/responses", cut_body))
        await wait_until(lambda: len(dead.requests) == 1)
        # The checkpoint: a hold pause, and a call the harness makes while it is open.
        relay.request_file("pause.request", 1)
        held = asyncio.ensure_future(call(relay.port, "/v1/responses", held_body))
        await wait_until(lambda: relay.status()["held"] == 1)
        # The restore: the new job's model server first, the continue second.
        relay.point(2, fresh.url("/rollout/key-b/v1"))
        await wait_until(lambda: relay.status()["gen"] == 2 and relay.status()["held"] == 2)
        assert fresh.requests == [] and len(dead.requests) == 1 and not cut.done() and not held.done()

        relay.request_file("continue.request", 1)
        replies = await asyncio.wait_for(asyncio.gather(cut, held), 15)

        assert [status for status, _, _ in replies] == [200, 200]
        assert sorted(request["body"] for request in fresh.requests) == sorted([cut_body, held_body])
        assert all(request["path"] == "/rollout/key-b/v1/responses" for request in fresh.requests)
        assert len(dead.requests) == 1
    finally:
        await dead.stop()
        await fresh.stop()


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="check_restored reads /proc")
async def test_check_restored_reads_the_status_file_the_relay_writes(relay: RelayProcess) -> None:
    relay.point(1, "http://127.0.0.1:9/v1")
    await relay.start()
    me = f"{os.getpid()}.{_start_time(os.getpid())}"
    (relay.session / PAUSED_FILE).write_text(
        json.dumps({"seq": 1, "confirmed": True, "pids": [], "error": None, "mode": "hold", "identity": me})
    )

    async def check():
        return await check_restored(
            LocalSandbox(),  # type: ignore[arg-type]
            str(relay.session),
            pause_seq=1,
            identity=me,
            relay_status_path=str(relay.status_path),
            timeout=15,
        )

    assert await check() == RestoredLaunch(pause_seq=1, identity=me)
    relay.process.kill()
    relay.process.wait(timeout=10)
    assert await check() == NotResumable("relay_gone")


async def test_an_upstream_that_closes_before_its_response_gets_the_same_call_again(relay: RelayProcess) -> None:
    upstream = await Upstream("reset_once").start()
    try:
        relay.point(1, upstream.url())
        await relay.start()
        body = json.dumps({"input": "x" * 3000}).encode()

        status, _, reply = await call(relay.port, "/v1/responses", body, {"X-Session-Id": "s"})

        assert status == 200 and json.loads(reply)["sha256"] == hashlib.sha256(body).hexdigest()
        first, again = upstream.requests
        assert first["body"] == again["body"] == body
        assert first["headers"] == again["headers"] and first["path"] == again["path"]
    finally:
        await upstream.stop()


async def test_an_upstream_that_drops_a_started_response_ends_the_call_and_is_not_asked_again(
    relay: RelayProcess,
) -> None:
    upstream = await Upstream("drop_mid_stream").start()
    try:
        relay.point(1, upstream.url())
        await relay.start()
        async with asyncio.timeout(15):
            reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
            writer.write(encode_request("/v1/responses", b"{}"))
            await writer.drain()
            try:
                received = await reader.read()
            except ConnectionError:
                received = b""
            writer.close()

        assert not received.endswith(b"0\r\n\r\n"), "a dropped stream must not look complete to the harness"
        await asyncio.sleep(1.0)  # longer than the first retry delays
        assert len(upstream.requests) == 1
    finally:
        await upstream.stop()
