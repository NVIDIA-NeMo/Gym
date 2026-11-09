# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Node launcher's 429 forwarder and native-session snapshot against real HTTP servers."""

import base64
import json
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from responses_api_agents.codex_agent.sandbox import LAUNCHER_FILE


LAUNCHER = Path(__file__).parents[1] / LAUNCHER_FILE
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is required for the Codex launcher")

# Imports the launcher without running it: its first argument is a URL, not the launcher's path.
_PROXY_DRIVER = """
const [launcher, baseUrl, attempts, timeoutMs] = process.argv.slice(1);
const { rateLimitProxy } = await import(launcher);
const server = await rateLimitProxy(baseUrl, "invocation", attempts, { timeoutMs: Number(timeoutMs) });
console.log(`http://127.0.0.1:${server.address().port}`);
process.stdin.on("end", () => process.exit(0)).resume();
"""


def _timeout_after_headers_driver(timeout: float) -> str:
    # Isolate a body deadline from scheduling before headers without replacing the HTTP exchange or timeout handler.
    return f"""
import assert from "node:assert/strict";
import http from "node:http";
import {{ syncBuiltinESMExports }} from "node:module";
const realRequest = http.request.bind(http);
http.request = (url, options, onResponse) => {{
  assert.equal(options.timeout, {timeout * 1000});
  const request = realRequest(url, {{ ...options, timeout: 0 }}, onResponse);
  request.once("response", (response) => queueMicrotask(() => response.emit("timeout")));
  return request;
}};
syncBuiltinESMExports();
"""


@contextmanager
def rate_limit_proxy(base_url, attempts_path, *, timeout, timeout_after_headers: bool = False):
    source = (_timeout_after_headers_driver(timeout) if timeout_after_headers else "") + _PROXY_DRIVER
    process = subprocess.Popen(
        ["node", "--input-type=module", "-e", source, LAUNCHER.as_uri(), base_url, str(attempts_path)]
        + [str(timeout * 1000)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        yield process.stdout.readline().strip()
    finally:
        process.stdin.close()
        process.wait(timeout=10)


@contextmanager
def upstream_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("statuses", [[429, 429, 200], [429, 429, 429], [400], [500]])
def test_forwarder_retries_only_429_and_retains_every_attempt(tmp_path, statuses):
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(
                (self.path, self.headers["x-session-id"], self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(statuses[len(requests) - 1])
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b"data: first\n\ndata: last\n\n")

    attempts_path = tmp_path / "attempts.jsonl"
    with upstream_server(Upstream) as upstream, rate_limit_proxy(f"{upstream}/v1", attempts_path, timeout=10) as proxy:
        try:
            response = urlopen(
                Request(f"{proxy}/responses", data=b'{"input":"task"}', headers={"x-session-id": "native-session"})
            )
        except HTTPError as error:
            response = error
        with response:
            assert response.status == statuses[-1]
            assert response.headers["Content-Type"] == "text/event-stream"
            assert response.read() == b"data: first\n\ndata: last\n\n"
    assert requests == [("/v1/responses", "invocation", b'{"input":"task"}')] * len(statuses)
    assert [json.loads(line)["status"] for line in attempts_path.read_text().splitlines()] == statuses


@pytest.mark.parametrize(
    "damage", [None, "policy", "prompt_only", "failed", "missing_id", "duplicate", "oversize", "truncated"]
)
def test_forwarder_retains_only_completed_compaction_identity_without_changing_stream(tmp_path, damage):
    response = {
        "id": "helper-id",
        "status": "completed",
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Summary café"}]}
        ],
    }
    if damage == "missing_id":
        del response["id"]
    if damage == "failed":
        response["status"] = "failed"
    event = json.dumps({"type": "response.completed", "response": response}, ensure_ascii=False).encode()
    wire = b": keepalive\r\n\r\nevent: response.completed\r\ndata: " + event + b"\r\n\r\n"
    if damage == "truncated":
        wire = wire.rstrip(b"\r\n")
    if damage == "duplicate":
        wire += wire
    # A comment frame larger than the launcher's 8 MiB evidence cap; the stream must still arrive whole.
    padding = b": " + b"x" * (8 * 1024 * 1024) + b"\n\n" if damage == "oversize" else b""

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(padding)
            for start in range(0, len(wire), 7):
                self.wfile.write(wire[start : start + 7])
                self.wfile.flush()

    prompt = (
        "You are performing a CONTEXT CHECKPOINT COMPACTION."
        if damage in {"policy", "prompt_only"}
        else "Summarize the current conversation using a different prompt."
    )
    payload = {"input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
    if damage != "prompt_only":
        payload["client_metadata"] = {
            "x-codex-turn-metadata": json.dumps({"request_kind": "turn" if damage == "policy" else "compaction"})
        }
    attempts_path = tmp_path / "attempts.jsonl"
    with upstream_server(Upstream) as upstream, rate_limit_proxy(upstream, attempts_path, timeout=10) as proxy:
        with urlopen(Request(f"{proxy}/responses", data=json.dumps(payload).encode())) as forwarded:
            assert forwarded.read() == padding + wire
    [attempt] = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    assert attempt["status"] == 200
    assert attempt.get("compaction_response") == (
        {"response_id": "helper-id", "summary": "Summary café"} if damage is None else None
    )


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"\xff",
        b"null",
        b"[]",
        b"{}",
        b'{"client_metadata":null}',
        b'{"client_metadata":[]}',
        b'{"client_metadata":{}}',
        *[
            json.dumps({"client_metadata": {"x-codex-turn-metadata": value}}).encode()
            for value in (
                None,
                {},
                1,
                '{"request_kind":',
                "null",
                "[]",
                '"compaction"',
                "{}",
                '{"request_kind":"turn"}',
                '{"request_kind":"prewarm"}',
                '{"request_kind":["compaction"]}',
            )
        ],
    ],
)
def test_compaction_request_rejects_missing_or_malformed_metadata(body):
    script = (
        "const { compactionRequest } = await import(process.argv[1]);"
        "console.log(JSON.stringify(compactionRequest(Buffer.from(process.argv[2], 'base64'))));"
    )
    checked = subprocess.run(
        ["node", "--input-type=module", "-e", script, LAUNCHER.as_uri(), base64.b64encode(body).decode()],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(checked.stdout) is False


@pytest.mark.parametrize(
    ("retry_after", "delays"),
    [
        (None, [1, 2]),
        ("0", [0, 0]),
        (" 2.5 ", [2.5, 2.5]),
        ("45", [30, 30]),
        ("-3", [0, 0]),
        ("1e400", [1, 2]),
        ("nan", [1, 2]),
        ("inf", [1, 2]),
        ("-inf", [1, 2]),
        ("", [1, 2]),
        ("0x10", [1, 2]),
        ("Wed, 21 Oct 2026 07:28:00 GMT", [1, 2]),
    ],
)
def test_retry_after_is_bounded_and_invalid_values_back_off(retry_after, delays):
    script = (
        "const { retryDelay } = await import(process.argv[1]);"
        "console.log(JSON.stringify([0, 1].map((attempt) => retryDelay(JSON.parse(process.argv[2]), attempt))));"
    )
    computed = subprocess.run(
        ["node", "--input-type=module", "-e", script, LAUNCHER.as_uri(), json.dumps(retry_after)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(computed.stdout) == delays


def test_connection_error_answers_502_and_records_the_error(tmp_path):
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text('{"status":429}\n')
    with rate_limit_proxy(f"http://127.0.0.1:{port}", attempts, timeout=1) as proxy:
        with pytest.raises(HTTPError) as reported:
            urlopen(Request(f"{proxy}/responses", data=b"{}"))
    assert reported.value.code == 502
    records = [json.loads(line) for line in attempts.read_text().splitlines()]
    assert records[-1]["status"] is None
    assert "ECONNREFUSED" in records[-1]["error"]


def test_launcher_starts_codex_behind_the_forwarder(tmp_path):
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append((self.headers.get("x-session-id"), self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(429 if len(requests) == 1 else 200)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b"done")

    fake = tmp_path / "fake_codex.mjs"
    fake.write_text(
        "const [flag, setting] = process.argv.slice(2);\n"
        "if (flag !== '-c') process.exit(2);\n"
        "const url = JSON.parse(setting.split('=').slice(1).join('='));\n"
        "const response = await fetch(`${url}/responses`, { method: 'POST', body: '{}' });\n"
        "console.log(JSON.stringify({ type: 'turn.completed', result: await response.text() }));\n"
    )
    with upstream_server(Upstream) as upstream:
        params = {
            "directory": str(tmp_path),
            "cwd": str(tmp_path),
            "env": {},
            "command": [shutil.which("node"), str(fake)],
            "timeout": 5,
            "model_base_url": upstream,
            "invocation_id": "launcher-invocation",
        }
        (tmp_path / "input.json").write_text(json.dumps(params))
        launched = subprocess.run(
            ["node", str(LAUNCHER), str(tmp_path / "input.json")], capture_output=True, text=True, timeout=15
        )
    assert launched.returncode == 0, launched.stderr
    assert requests == [("launcher-invocation", b"{}")] * 2
    assert [json.loads(line)["status"] for line in (tmp_path / "model_attempts.jsonl").read_text().splitlines()] == [
        429,
        200,
    ]
    assert json.loads(launched.stdout) == {"type": "turn.completed", "result": "done"}


@pytest.mark.parametrize("fault", ["timeout", "truncated_chunk", "truncated_length"])
def test_broken_429_body_is_recorded_and_retried(tmp_path, fault):
    release = Event()
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(self.path)
            if len(requests) > 1:
                self.send_response(200)
                self.send_header("Content-Length", "4")
                self.end_headers()
                self.wfile.write(b"done")
                return
            self.send_response(429)
            self.send_header("Retry-After", "0")
            if fault == "truncated_chunk":
                self.send_header("Transfer-Encoding", "chunked")
            elif fault == "truncated_length":
                self.send_header("Content-Length", "1000")
            self.end_headers()
            if fault == "timeout":
                release.wait(2)
            elif fault == "truncated_chunk":
                self.wfile.write(b"20\r\nslow down")
            else:
                self.wfile.write(b"slow down")

    attempts_path = tmp_path / "attempts.jsonl"
    with upstream_server(Upstream) as upstream:
        try:
            with rate_limit_proxy(upstream, attempts_path, timeout=0.3) as proxy:
                with urlopen(Request(f"{proxy}/responses", data=b"{}"), timeout=10) as response:
                    assert (response.status, response.read()) == (200, b"done")
        finally:
            release.set()
    records = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    assert [record["status"] for record in records] == [429, 200]
    assert records[0]["transport_error"]
    assert "transport_error" not in records[1]


@pytest.mark.parametrize("fault", ["timeout", "truncated_chunk", "truncated_length", "server_error"])
def test_body_failure_retains_http_status_and_transport_error(tmp_path, fault):
    release = Event()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if fault == "truncated_chunk":
                self.send_header("Transfer-Encoding", "chunked")
            elif fault == "truncated_length":
                self.send_header("Content-Length", "1000")
            self.end_headers()
            if fault == "timeout":
                release.wait(2)
            elif fault == "truncated_chunk":
                self.wfile.write(b"20\r\ndata: partial")
            elif fault == "truncated_length":
                self.wfile.write(b"data: partial")
            else:
                self.wfile.write(
                    b'data: {"type":"response.failed","response":{"error":{"message":"invalid input"}}}\n\n'
                )

    attempts_path = tmp_path / "attempts.jsonl"
    with upstream_server(Upstream) as upstream:
        try:
            with rate_limit_proxy(
                upstream,
                attempts_path,
                timeout=0.1 if fault == "timeout" else 5,
                timeout_after_headers=fault == "timeout",
            ) as proxy:
                with urlopen(Request(f"{proxy}/responses", data=b"{}"), timeout=5) as response:
                    assert response.status == 200
                    body = response.read()
        finally:
            release.set()
    records = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == 200
    if fault == "server_error":
        assert b"invalid input" in body
        assert "transport_error" not in records[0]
    else:
        assert records[0]["transport_error"]


def test_codex_hanging_up_before_the_response_cancels_the_upstream_stream(tmp_path):
    received, hung_up, upstream_closed = Event(), Event(), Event()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            received.set()
            hung_up.wait(5)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # A model stream that would last the rest of the turn unless the forwarder closes it.
            try:
                for _ in range(100):
                    self.wfile.write(b"data: x\n\n")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                upstream_closed.set()

    attempts_path = tmp_path / "attempts.jsonl"
    with upstream_server(Upstream) as upstream, rate_limit_proxy(upstream, attempts_path, timeout=10) as proxy:
        with socket.create_connection(("127.0.0.1", int(proxy.rsplit(":", 1)[1]))) as codex:
            codex.sendall(b"POST /responses HTTP/1.1\r\nHost: proxy\r\nContent-Length: 2\r\n\r\n{}")
            assert received.wait(5)
        # Let the forwarder see the hang-up before the upstream response starts.
        time.sleep(0.3)
        hung_up.set()
        assert upstream_closed.wait(3)
    assert [json.loads(line) for line in attempts_path.read_text().splitlines()] == [{"status": 200}]


def test_snapshot_joins_native_sessions_in_path_order(tmp_path):
    sessions = tmp_path / "home/.codex/sessions"
    # Path order compares components: "2026/9" sorts after "2026/10", and a/b.jsonl before a.b/c.jsonl.
    files = {
        "2026/9/1/late.jsonl": '{"day":"9"}\n',
        "2026/10/19/first.jsonl": '{"id":"first","text":"line separator"}\r\n{"id":"second"}',
        "2026/10/19/second.jsonl": '{"id":"third"}\n',
        "a/b.jsonl": '{"id":"a/b"}\n',
        "a.b/c.jsonl": '{"id":"a.b/c"}\n',
        "2026/10/19/notes.txt": "not a session\n",
    }
    for name, text in files.items():
        (sessions / name).parent.mkdir(parents=True, exist_ok=True)
        (sessions / name).write_bytes(text.encode())
    subprocess.run(["node", str(LAUNCHER), "--snapshot", str(tmp_path)], check=True, capture_output=True, timeout=10)
    assert (tmp_path / "native_session.jsonl").read_bytes().decode().split("\n") == [
        '{"id":"first","text":"line separator"}',
        '{"id":"second"}',
        '{"id":"third"}',
        '{"day":"9"}',
        '{"id":"a/b"}',
        '{"id":"a.b/c"}',
    ]


def test_launcher_reached_through_a_symlink_still_runs(tmp_path):
    (tmp_path / "home/.codex/sessions").mkdir(parents=True)
    (tmp_path / "home/.codex/sessions/rollout.jsonl").write_text('{"id":"only"}\n')
    # For example, a task image whose /tmp is a symlink.
    linked = tmp_path / "linked"
    linked.symlink_to(LAUNCHER.parent, target_is_directory=True)
    subprocess.run(
        ["node", str(linked / LAUNCHER_FILE), "--snapshot", str(tmp_path)], check=True, capture_output=True, timeout=10
    )
    assert (tmp_path / "native_session.jsonl").read_text() == '{"id":"only"}'


def test_snapshot_keeps_every_line_of_a_long_rollout_file(tmp_path):
    (tmp_path / "home/.codex/sessions").mkdir(parents=True)
    lines = [json.dumps({"index": index}) for index in range(200_000)]
    (tmp_path / "home/.codex/sessions/rollout.jsonl").write_text("\n".join(lines) + "\n")
    subprocess.run(["node", str(LAUNCHER), "--snapshot", str(tmp_path)], check=True, capture_output=True, timeout=30)
    assert (tmp_path / "native_session.jsonl").read_text().split("\n") == lines


def test_snapshot_without_native_sessions_writes_nothing(tmp_path):
    subprocess.run(["node", str(LAUNCHER), "--snapshot", str(tmp_path)], check=True, capture_output=True, timeout=10)
    assert not (tmp_path / "native_session.jsonl").exists()
