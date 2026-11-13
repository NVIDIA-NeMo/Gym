# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge resilience preserves native verifier payloads and unrelated app requests."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
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
    monkeypatch.setattr(
        __import__("urllib.request", fromlist=["urlopen"]),
        "urlopen",
        __import__("urllib.request", fromlist=["urlopen"]).urlopen,
    )
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
    monkeypatch.setattr(
        __import__("urllib.request", fromlist=["urlopen"]),
        "urlopen",
        __import__("urllib.request", fromlist=["urlopen"]).urlopen,
    )
    install_transport()
    requests.post("http://localhost:30000/api/task", timeout=7, json={"task": "unchanged"})
    assert observed == [{"data": None, "json": {"task": "unchanged"}, "timeout": 7}]


@pytest.mark.parametrize("client", ["requests", "urllib"])
@pytest.mark.parametrize("base_suffix", ["", "/"])
def test_caught_judge_fault_is_not_a_task_failure(tmp_path, monkeypatch, client, base_suffix):
    """An actual verifier subprocess may swallow HTTP errors; its grade must not."""
    import importlib.util
    from pathlib import Path

    from responses_api_agents.saas_bench_agent import verifier_transport

    root = Path(__file__).parents[1] / "benchmark"
    spec = importlib.util.spec_from_file_location("fault_test_runner", root / "saas_bench/verify_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    seen = []
    reply = [503, b"{}"]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(reply[0])
            self.end_headers()
            self.wfile.write(reply[1])

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1{base_suffix}")
    monkeypatch.delenv("MINDRA_BASE_URL", raising=False)
    monkeypatch.setenv("SAAS_BENCH_JUDGE_HTTP_TIMEOUT", "30")
    monkeypatch.setenv("SAAS_VERIFIER_LAUNCHER", verifier_transport.__file__)
    script = tmp_path / "verify.py"
    script.write_text(
        """import os, requests, sys, json, urllib.request
try:
    if 'CLIENT' == 'requests':
        reply = requests.post(os.environ['LLM_BASE_URL']+'/chat/completions', json={'model':'judge'})
        data = reply.json()
    else:
        req = urllib.request.Request(os.environ['LLM_BASE_URL']+'/chat/completions', data=b'{"model":"judge"}', headers={'Content-Type':'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=10) as reply:
            data = json.loads(reply.read())
    answer = data['choices'][0]['message']['content']
except Exception:
    answer = 'NO'  # Deliberately discard all exception text, like a task-local fallback.
passed = answer == 'YES'
print('[PASS] (1pt) criterion' if passed else '[FAIL] (1pt) criterion', file=sys.stderr)
print(f'SCORE: {float(passed):.3f} PASS: {passed} ({int(passed)}/1)', file=sys.stderr)
""".replace("CLIENT", client)
    )
    try:
        for status, payload, kind, calls in [
            (503, b"{}", "judge_failed", 2),
            (429, b"{}", "judge_failed", 2),
            (200, b'{"choices":[]}', "judge_unparseable", 1),
            (200, b"not JSON", "judge_unparseable", 1),
            (200, b'{"choices":[{"message":{"content":""}}]}', "judge_unparseable", 1),
            (200, b'{"choices":[{"message":{"content":"NO"}}]}', None, 1),
            (200, b'{"choices":[{"message":{"content":"YES"}}]}', None, 1),
        ]:
            reply[:] = [status, payload]
            seen.clear()
            result = runner.run_verify(
                {"task_id": "fixture", "verify_py_path": str(script)}, 0, {}, "localhost", str(tmp_path)
            )
            assert len(seen) == calls
            if kind:
                assert result["status"] == "ERROR" and result["raw_score"] == 0
                assert result["judge_failures"][0]["failure_kind"] == kind
            else:
                assert result["status"] == ("PASS" if b"YES" in payload else "FAIL")
                assert result["judge_failures"] == []
                assert result["score"] == (1 if b"YES" in payload else 0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
