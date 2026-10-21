# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve the request OpenCode was awaiting, using its real pinned runtime."""

import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from responses_api_agents.opencode_sandboxed_agent.resume_opencode import resume_payload


def test_resume_payload_reuses_the_user_id_and_retains_settings() -> None:
    user = {
        "id": "msg_002",
        "role": "user",
        "agent": "build",
        "model": {"providerID": "gym", "modelID": "policy", "variant": "high"},
        "system": "system instruction",
        "tools": {"bash": True},
        "format": {"type": "text"},
    }
    assert resume_payload(
        [
            {"info": user},
            {"info": {"id": "msg_003", "role": "assistant"}},
            {"info": {"id": "msg_001", "role": "user"}},
        ]
    ) == {
        "messageID": "msg_002",
        "parts": [],
        "agent": "build",
        "model": {"providerID": "gym", "modelID": "policy"},
        "variant": "high",
        "system": "system instruction",
        "tools": {"bash": True},
        "format": {"type": "text"},
    }


def test_resume_rejects_a_missing_user_turn() -> None:
    with pytest.raises(ValueError, match="saved user turn"):
        resume_payload([])


@pytest.mark.skipif(not os.environ.get("OPENCODE_TEST_BINARY"), reason="Set OPENCODE_TEST_BINARY to OpenCode 1.17.11")
@pytest.mark.parametrize("interrupt_resume", [False, True])
def test_real_opencode_reissues_the_interrupted_request_without_a_new_user_turn(
    tmp_path: Path, interrupt_resume: bool
) -> None:
    """Complete one tool call, interrupt the next generation, then compare its retry's exact messages."""
    binary = Path(os.environ["OPENCODE_TEST_BINARY"])
    resume_python = os.environ.get("OPENCODE_TEST_PYTHON", sys.executable)
    project = tmp_path / "project"
    project.mkdir()
    (project / "table.py").write_text("def example():\n    return 42\n")
    calls: list[dict] = []
    pending = threading.Event()
    pending_resume = threading.Event()
    release = threading.Event()

    class ModelHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            number = len(calls)
            if number == 2:
                pending.set()
                release.wait(timeout=90)
            if number == 3 and interrupt_resume:
                pending_resume.set()
                release.wait(timeout=90)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if number == 1:
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps({"filePath": str(project / "table.py")}),
                            },
                        }
                    ]
                }
                reason = "tool_calls"
            else:
                delta, reason = {"content": "Done."}, "stop"
            chunk = {
                "id": "chatcmpl_test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            end = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}
            try:
                self.wfile.write(
                    (f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(end)}\n\ndata: [DONE]\n\n").encode()
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # The checkpoint deliberately disconnects the second request.

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "opencode").symlink_to(binary)
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "PWD": str(project),
        "HOME": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
        "OPENCODE_CONFIG_CONTENT": json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "model": "checkpoint-test/test-model",
                "enabled_providers": ["checkpoint-test"],
                "compaction": {"auto": False, "prune": False},
                "permission": "allow",
                "provider": {
                    "checkpoint-test": {
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": f"http://127.0.0.1:{server.server_port}/v1", "apiKey": "test"},
                        "models": {"test-model": {"name": "test-model", "limit": {"context": 32768, "output": 1024}}},
                    }
                },
            }
        ),
    }
    initial = None
    resumed = None
    try:
        with (tmp_path / "initial.log").open("w") as log:
            initial = subprocess.Popen(
                [
                    str(binary),
                    "run",
                    "--print-logs",
                    "--log-level",
                    "DEBUG",
                    "--title",
                    "Checkpoint test",
                    "Read table.py and finish.",
                ],
                cwd=project,
                env=env,
                stdout=log,
                stderr=log,
            )
        assert pending.wait(timeout=60), (tmp_path / "initial.log").read_text()[-5000:]
        initial.send_signal(signal.SIGINT)
        initial.wait(timeout=20)
        saved_messages = calls[1]["messages"]
        assert any(message["role"] == "tool" for message in saved_messages)
        helper = Path(__file__).resolve().parents[1] / "resume_opencode.py"
        if interrupt_resume:
            with (tmp_path / "interrupted-resume.log").open("w") as log:
                resumed = subprocess.Popen([resume_python, str(helper)], cwd=project, env=env, stdout=log, stderr=log)
            assert pending_resume.wait(timeout=60), (tmp_path / "interrupted-resume.log").read_text()
            resumed.send_signal(signal.SIGINT)
            assert resumed.wait(timeout=20) == 130
        with (tmp_path / "resume.log").open("w") as log:
            completed = subprocess.run(
                [resume_python, str(helper)], cwd=project, env=env, stdout=log, stderr=log, timeout=60
            )
        assert completed.returncode == 0, (tmp_path / "resume.log").read_text()
        assert len(calls) == (4 if interrupt_resume else 3)
        for call in calls[2:]:
            assert call["messages"] == saved_messages, (
                "The resumed generation must have exactly the saved conversation"
            )
            assert sum(message["role"] == "user" for message in call["messages"]) == 1
    finally:
        release.set()
        if initial is not None and initial.poll() is None:
            initial.kill()
            initial.wait()
        if resumed is not None and resumed.poll() is None:
            resumed.send_signal(signal.SIGINT)
            resumed.wait(timeout=20)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
