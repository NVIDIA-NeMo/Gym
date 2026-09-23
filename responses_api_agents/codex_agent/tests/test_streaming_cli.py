# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional offline boundary check using the actual pinned CLI and a localhost fake API."""

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nemo_gym.responses_streaming import synthesize_responses_sse
from responses_api_agents.codex_agent.app import parse_exec_jsonl, toml_dumps
from responses_api_agents.codex_agent.tests.test_native_sessions import close_body, seed
from responses_api_agents.codex_agent.tests.test_native_sessions import setup as setup


@pytest.mark.parametrize("recover", [False, True])
def test_pinned_cli_incomplete_stream_retries_preserve_usage_gaps(tmp_path: Path, setup, recover: bool) -> None:
    binary = os.environ.get("CODEX_TEST_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("Set CODEX_TEST_BIN to an existing Codex 0.144.4 executable; never auto-install for this test")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, errors="replace", timeout=10)
    if version.returncode or version.stdout.strip() != "codex-cli 0.144.4":
        pytest.skip("This boundary regression targets Codex 0.144.4")
    reasoning = "Inspect arithmetic before editing."
    response = {
        "id": "resp_incomplete",
        "object": "response",
        "created_at": 1,
        "model": "gym-policy-model",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {"id": "rs_partial", "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]}
        ],
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    }
    requests = []
    completed = {
        **response,
        "id": "resp_complete",
        "status": "completed",
        "incomplete_details": None,
        "output": [
            {
                "id": "msg_done",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
            }
        ],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            terminal = completed if recover and len(requests) > 1 else response
            wire = "".join(synthesize_responses_sse(terminal)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "home"
    home.mkdir()
    agent, sandbox = setup
    config = agent._build_config(f"http://127.0.0.1:{server.server_port}/v1")
    (home / "config.toml").write_text(toml_dumps(config))
    try:
        result = subprocess.run(
            [
                binary,
                "exec",
                "--json",
                "--ephemeral",
                "--skip-git-repo-check",
                "--cd",
                str(tmp_path),
                "--",
                "Say done.",
            ],
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(home),
                "CODEX_HOME": str(home),
                "OPENAI_API_KEY": "synthetic",
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.returncode == (0 if recover else 1), result.stdout + result.stderr
    assert len(requests) == (2 if recover else 6), "Pinned CLI defaults allow five reconnects, then must terminate"
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert any(event["type"] == "turn.completed" for event in events) == recover
    failed = [event for event in events if event["type"] == "turn.failed"]
    output, usage = parse_exec_jsonl(result.stdout, structured_reasoning=True, include_partial=True)
    if recover:
        assert not failed
        assert [item.type for item in output] == ["reasoning", "message"]
        assert output[0].summary[0].text == reasoning
        assert len({item.id for item in output}) == 2
    else:
        assert len(failed) == 1
        assert "max_output_tokens" in failed[0]["error"]["message"]
        assert output and all(item.type == "reasoning" for item in output)
        assert all(item.summary[0].text == reasoning for item in output)
        assert any("max_output_tokens" in error for error in usage["errors"])

    # Replay actual pinned-CLI events through native response + close. No model call from
    # an incomplete attempt appears in turn.completed, even when a later retry succeeds.
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(events))
    sandbox.result["return_code"] = result.returncode
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        activated = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Say done."})
        assert activated.status_code == 200, activated.text
        native = activated.json()
        assert native["status"] == ("completed" if recover else "failed")
        assert native["usage"]["total_tokens"] == (10 if recover else 0)
        assert native["usage"]["total_tokens"] < sum(response["usage"]["total_tokens"] for _ in requests)
        # The fake backend omitted both details; CLI zeros cannot prove known zero usage.
        assert native["usage"]["input_tokens_details"]["cached_tokens"] is None
        assert native["usage"]["output_tokens_details"]["reasoning_tokens"] is None
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
        assert "partial_model_usage_unavailable" in {gap["code"] for gap in closed["agent_observations"]["gaps"]}
