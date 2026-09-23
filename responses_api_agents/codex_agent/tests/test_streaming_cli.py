# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional offline boundary check using the actual pinned CLI and a localhost fake API."""

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.responses_streaming import synthesize_responses_sse
from nemo_gym.server_utils import ServerClient
from responses_api_agents.codex_agent.app import CodexAgent, CodexAgentConfig, parse_exec_jsonl, toml_dumps


def test_pinned_cli_rejects_incomplete_reasoning_with_bounded_retries(tmp_path: Path) -> None:
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

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            wire = "".join(synthesize_responses_sse(response)).encode()
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
    agent = CodexAgent(
        config=CodexAgentConfig(
            host="127.0.0.1", port=1, name="test", entrypoint="", codex_version="0.144.4", model="gym-policy-model"
        ),
        server_client=MagicMock(spec=ServerClient),
    )
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
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(requests) == 6, "Pinned CLI defaults allow five reconnects, then must terminate"
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert not any(event["type"] == "turn.completed" for event in events)
    failed = [event for event in events if event["type"] == "turn.failed"]
    assert len(failed) == 1
    assert "max_output_tokens" in failed[0]["error"]["message"]
    output, usage = parse_exec_jsonl(result.stdout, structured_reasoning=True, include_partial=True)
    assert output and all(item.type == "reasoning" for item in output)
    assert all(item.summary[0].text == reasoning for item in output)
    assert any("max_output_tokens" in error for error in usage["errors"])
