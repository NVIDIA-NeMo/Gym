# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from responses_api_agents.codex_agent.sandbox_transport import rate_limit_proxy


@pytest.mark.parametrize("statuses", [[429, 429, 200], [429, 429, 429], [400], [500]])
def test_sandbox_forwarder_retries_only_429_and_retains_every_attempt(tmp_path, statuses):
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

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    attempts_path = tmp_path / "attempts.jsonl"
    try:
        with rate_limit_proxy(
            f"http://127.0.0.1:{upstream.server_port}/v1", "invocation", attempts_path, timeout=10
        ) as proxy:
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
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join()
    assert requests == [("/v1/responses", "invocation", b'{"input":"task"}')] * len(statuses)
    assert [json.loads(line)["status"] for line in attempts_path.read_text().splitlines()] == statuses


@pytest.mark.parametrize("retry_after", ["nan", "inf", "-inf", "invalid"])
def test_invalid_retry_after_uses_finite_backoff(tmp_path, retry_after):
    errors = [
        HTTPError("http://model/responses", 429, "retry", {"Retry-After": retry_after}, io.BytesIO(b"retry"))
        for _ in range(3)
    ]
    with (
        patch("responses_api_agents.codex_agent.sandbox_transport.urlopen", side_effect=errors),
        patch("responses_api_agents.codex_agent.sandbox_transport.sleep") as sleep,
    ):
        with rate_limit_proxy("http://model", "invocation", tmp_path / "attempts.jsonl", timeout=1) as proxy:
            with pytest.raises(HTTPError) as reported:
                urlopen(Request(f"{proxy}/responses", data=b"{}"))
        assert reported.value.code == 429
        assert [call.args[0] for call in sleep.call_args_list] == [1, 2]


def test_connection_error_replaces_prior_http_status(tmp_path):
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text('{"status":429}\n')
    with patch(
        "responses_api_agents.codex_agent.sandbox_transport.urlopen", side_effect=URLError("connection refused")
    ):
        with rate_limit_proxy("http://model", "invocation", attempts, timeout=1) as proxy:
            with pytest.raises(HTTPError) as reported:
                urlopen(Request(f"{proxy}/responses", data=b"{}"))
        assert reported.value.code == 502
    records = [json.loads(line) for line in attempts.read_text().splitlines()]
    assert records[-1]["status"] is None
    assert "connection refused" in records[-1]["error"]


def test_runner_installs_proxy_and_retains_retry(tmp_path):
    from responses_api_agents.codex_agent import sandbox_runner

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

    fake = tmp_path / "fake_codex.py"
    fake.write_text("""import json, sys
from urllib.request import Request, urlopen
assert sys.argv[1] == '-c'
url = json.loads(sys.argv[2].split('=', 1)[1])
with urlopen(Request(url + '/responses', data=b'{}')) as response:
    print(json.dumps({'type': 'turn.completed', 'result': response.read().decode()}))
""")
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    params = {
        "directory": str(tmp_path),
        "cwd": str(tmp_path),
        "prompt": "task",
        "env": {},
        "command": [sys.executable, str(fake)],
        "timeout": 5,
        "model_base_url": f"http://127.0.0.1:{upstream.server_port}",
        "invocation_id": "runner-invocation",
    }
    request_file = tmp_path / "input.json"
    request_file.write_text(json.dumps(params))
    try:
        result = subprocess.run(
            [sys.executable, "-I", sandbox_runner.__file__, str(request_file)], capture_output=True, timeout=15
        )
        assert result.returncode == 0, result.stderr.decode()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join()
    assert requests == [("runner-invocation", b"{}")] * 2
    assert [json.loads(line)["status"] for line in (tmp_path / "model_attempts.jsonl").read_text().splitlines()] == [
        429,
        200,
    ]
    assert json.loads((tmp_path / "events.jsonl").read_text())[1]["result"] == "done"


@pytest.mark.parametrize("fault", ["timeout", "truncated_chunk", "truncated_length", "server_error"])
def test_body_failure_retains_http_status_and_transport_error(tmp_path, fault):
    from threading import Event

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

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    attempts_path = tmp_path / "attempts.jsonl"
    try:
        with rate_limit_proxy(
            f"http://127.0.0.1:{upstream.server_port}", "invocation", attempts_path, timeout=0.1
        ) as proxy:
            with urlopen(Request(f"{proxy}/responses", data=b"{}"), timeout=5) as response:
                assert response.status == 200
                body = response.read()
        records = [json.loads(line) for line in attempts_path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["status"] == 200
        if fault == "server_error":
            assert b"invalid input" in body
            assert "transport_error" not in records[0]
        else:
            assert records[0]["transport_error"]
    finally:
        release.set()
        upstream.shutdown()
        upstream.server_close()
        thread.join()
