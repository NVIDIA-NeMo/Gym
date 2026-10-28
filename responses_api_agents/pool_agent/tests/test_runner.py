# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone runner checks, including opt-in execution of the real pinned Pool binary."""

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from responses_api_agents.pool_agent import sandbox_runner
from responses_api_agents.pool_agent.configuration import pool_config
from responses_api_agents.pool_agent.trajectory import parse_trajectory


@pytest.mark.parametrize("count", [0, 2])
def test_snapshot_refuses_missing_or_ambiguous_root(tmp_path, count):
    for index in range(count):
        (tmp_path / f"trajectory-agent{index}_gym-root.ndjson").write_text("{}\n")
    with pytest.raises(RuntimeError, match="Expected one Pool root trajectory"):
        sandbox_runner.snapshot(tmp_path)


def test_snapshot_ignores_child_session_and_keeps_partial_root(tmp_path):
    root = tmp_path / "home/.pool/agent/trajectories"
    root.mkdir(parents=True)
    (root / "trajectory-agent_gym-root.ndjson").write_text('{"type":"session.start"}\n{"type":')
    (root / "trajectory-agent_child.ndjson").write_text("child output")
    sandbox_runner.snapshot(tmp_path)
    assert (tmp_path / "trajectory.ndjson").read_text() == '{"type":"session.start"}\n{"type":'


def test_task_shell_restores_task_home_and_xdg(tmp_path):
    (tmp_path / "task_directories.json").write_text(
        json.dumps({"HOME": "/task/home", "XDG_CONFIG_HOME": "/task/config"})
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(sandbox_runner.__file__)),
            "--task-shell",
            str(tmp_path),
            'printf "%s\\n%s\\n" "$HOME" "$XDG_CONFIG_HOME"',
        ],
        env={**os.environ, "HOME": "/adapter/home", "XDG_CONFIG_HOME": "/adapter/config"},
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.splitlines() == ["/task/home", "/task/config"]


def test_cache_hints_preserve_tool_schema_and_content():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "cache_control", "cache_control": {"type": "ephemeral"}}],
            }
        ],
        "tools": [
            {
                "type": "function",
                "cache_control": {},
                "function": {"name": "example", "parameters": {"properties": {"cache_control": {"type": "string"}}}},
            }
        ],
    }
    sandbox_runner.strip_cache_hints(body)
    assert body["messages"][0]["content"] == [{"type": "text", "text": "cache_control"}]
    assert "cache_control" not in body["tools"][0]
    assert body["tools"][0]["function"]["parameters"]["properties"] == {"cache_control": {"type": "string"}}


@pytest.mark.parametrize("status", [200, 429, 503])
def test_proxy_preserves_upstream_status_and_counts_only_successes(tmp_path, status):
    class Reject(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            # Drain the request before closing the socket to avoid a TCP reset.
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"capacity"}')

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Reject)
    calls = tmp_path / "model_calls.json"
    proxy = sandbox_runner.model_proxy(f"http://127.0.0.1:{upstream.server_port}/v1", calls_path=calls)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (upstream, proxy)]
    for thread in threads:
        thread.start()
    try:
        request = Request(f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions", data=b'{"messages":[]}')
        if status == 200:
            with urlopen(request, timeout=5) as response:
                assert response.status == 200
                assert response.read() == b'{"error":"capacity"}'
        else:
            with pytest.raises(HTTPError) as caught:
                urlopen(request, timeout=5)
            assert caught.value.code == status
            assert caught.value.read() == b'{"error":"capacity"}'
            caught.value.close()
        assert json.loads(calls.read_text()) == {"completed": int(status == 200)}
    finally:
        for server in (proxy, upstream):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


@pytest.mark.skipif(
    not os.environ.get("POOL_TEST_BINARY"), reason="Set POOL_TEST_BINARY to a local Pool v1.0.16 binary"
)
@pytest.mark.parametrize("reject_first_call", [False, True])
def test_real_pool_shell_and_persisted_trajectory(tmp_path, reject_first_call):
    binary = Path(os.environ["POOL_TEST_BINARY"])
    version = subprocess.run([str(binary), "--version"], text=True, capture_output=True, timeout=10, check=True)
    assert version.stdout.strip() == "1.0.16"
    directory, workdir = tmp_path / "session", tmp_path / "repo"
    (directory / "bin").mkdir(parents=True)
    workdir.mkdir()
    shutil.copy(binary, directory / "bin/pool")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if len(requests) == 1 or (reject_first_call and len(requests) == 2):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-shell",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps(
                                    {
                                        "cmd": 'printf "%s\\n%s\\n%s\\n" "$HOME" "$XDG_CONFIG_HOME" "$TASK_TEST" > proof.txt',
                                        "cwd": str(workdir),
                                    }
                                ),
                            },
                        }
                    ],
                }
                if reject_first_call and len(requests) == 1:
                    message["tool_calls"][0]["id"] = "call-invalid"
                    message["tool_calls"][0]["function"]["arguments"] = '{"cmd":"true","timeout":"120"}'
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": "File written."}
                finish = "stop"
            response = json.dumps(
                {
                    "id": f"chatcmpl-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/ng-rollout/local-a1/v1"
    (directory / "agent_config.json").write_text(
        json.dumps(pool_config(base_url=base_url, model="test", context_window=32768))
    )
    (directory / "prompt.txt").write_text("Write the requested file, then finish.")
    (directory / "task_env.json").write_text(
        json.dumps(
            {
                "HOME": str(tmp_path / "task-home"),
                "XDG_CONFIG_HOME": str(tmp_path / "task-config"),
                "TASK_TEST": "present",
            }
        )
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(sandbox_runner.__file__)), str(directory), str(workdir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (directory / "stderr.log").read_text() + result.stderr
        sandbox_runner.snapshot(directory)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    for _, body in requests:
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(body)
    assert len(requests) == (3 if reject_first_call else 2)
    assert all(path == "/ng-rollout/local-a1/v1/chat/completions" for path, _ in requests)
    assert all(body.get("stream", False) is False for _, body in requests)
    assert (workdir / "proof.txt").read_text().splitlines() == [
        str(tmp_path / "task-home"),
        str(tmp_path / "task-config"),
        "present",
    ]
    assert sorted(p.name for p in workdir.iterdir()) == ["proof.txt"]
    parsed = parse_trajectory((directory / "trajectory.ndjson").read_text())
    assert parsed.exit_reason == "exit_tool_called"
    assert not parsed.errors and not parsed.gaps
    calls = [item for item in parsed.output if item.type == "function_call"]
    results = [item for item in parsed.output if item.type == "function_call_output"]
    assert len(calls) == len(results) == (2 if reject_first_call else 1)
    assert calls[-1].call_id == results[-1].call_id == "call-shell"
    if reject_first_call:
        assert calls[0].call_id == results[0].call_id == "call-invalid"
        assert "unexpected additional properties" in results[0].output
    assert parsed.output[-1].content[0].text == "File written."


def test_task_home_output_survives_private_session_cleanup(tmp_path):
    directory = tmp_path / "session"
    home = tmp_path / "task-home"
    directory.mkdir()
    home.mkdir()
    directories = {"HOME": str(home), "XDG_CONFIG_HOME": str(home / "config")}
    (directory / "task_directories.json").write_text(json.dumps(directories))
    subprocess.run(
        [
            sys.executable,
            str(Path(sandbox_runner.__file__)),
            "--task-shell",
            str(directory),
            'mkdir -p "$XDG_CONFIG_HOME"; printf result > "$HOME/proof"; printf config > "$XDG_CONFIG_HOME/proof"',
        ],
        env=os.environ | {"HOME": str(directory / "home")},
        check=True,
        timeout=10,
    )
    shutil.rmtree(directory)
    assert (home / "proof").read_text() == "result"
    assert (home / "config/proof").read_text() == "config"
