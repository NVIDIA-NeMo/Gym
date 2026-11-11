# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge resilience preserves native verifier payloads and unrelated app requests."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import requests
from responses_api_agents.saas_bench_agent.verifier_transport import install_transport


def test_judge_transient_failure_retried_with_identical_payload(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(503 if len(seen) == 1 else 200)
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"YES"}}]}')

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.delenv("MINDRA_BASE_URL", raising=False)
    monkeypatch.setenv("SAAS_BENCH_JUDGE_HTTP_TIMEOUT", "120")
    monkeypatch.setattr(requests.Session, "request", requests.Session.request)
    install_transport()
    try:
        response = requests.post(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            json={"model": "judge", "messages": [{"role": "user", "content": "original criterion"}]},
            timeout=30,
        )
        assert response.status_code == 200
        assert len(seen) == 2 and seen[0] == seen[1]
        assert json.loads(seen[0])["messages"][0]["content"] == "original criterion"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_nonjudge_app_timeout_is_unchanged(monkeypatch):
    observed = []
    monkeypatch.setenv("LLM_BASE_URL", "https://judge.example/v1")
    monkeypatch.delenv("MINDRA_BASE_URL", raising=False)
    monkeypatch.setenv("SAAS_BENCH_JUDGE_HTTP_TIMEOUT", "120")
    monkeypatch.setattr(requests.Session, "request", lambda self, method, url, **kw: observed.append(kw))
    install_transport()
    requests.post("http://localhost:30000/api/task", timeout=7, json={"task": "unchanged"})
    assert observed == [{"data": None, "json": {"task": "unchanged"}, "timeout": 7}]
