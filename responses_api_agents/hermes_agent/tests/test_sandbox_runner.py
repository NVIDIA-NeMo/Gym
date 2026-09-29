# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import openai
import pytest
from run_agent import AIAgent

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from responses_api_agents.hermes_agent.sandbox_runner import _run, _use_model_server


def _completion(message: dict) -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", **message}}],
        "created": 0,
        "model": "model",
        "object": "chat.completion",
    }


class _ModelServer:
    """Answer chat completions over HTTP in order and record each request body."""

    def __init__(self, answers: list[dict]) -> None:
        self.answers = answers
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                answer = server.answers[min(len(server.requests), len(server.answers)) - 1]
                payload = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_ModelServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def restore_process_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner changes process-wide state; restore it so other tests see the originals."""
    monkeypatch.setattr(AIAgent, "__init__", AIAgent.__init__)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("template_enabled", [False, True])
def test_iteration_limit_summary_reaches_the_model_server(tmp_path, restore_process_globals, template_enabled) -> None:
    tool_call = {
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "terminal", "arguments": json.dumps({"command": "echo hi"})},
            }
        ],
    }
    answers = [_completion(tool_call), _completion({"content": "summary of the work"})]

    with _ModelServer(answers) as model_server:
        output = _run(
            {
                "agent_session_id": "session",
                "chat_template_kwargs_enabled": template_enabled,
                "config_yaml": "model: policy_model\nprovider: auto\n",
                "disabled_toolsets": None,
                "enabled_toolsets": ["terminal"],
                "history": [],
                "max_tokens": 128,
                "max_turns": 1,
                "model": "policy_model",
                "model_base_url": model_server.base_url,
                "system_message": None,
                "temperature": 0.0,
                "terminal_timeout": 30,
                "user_message": "fix bug",
            },
            tmp_path,
        )

    assert len(model_server.requests) == 2
    assert all(not request.get("stream") for request in model_server.requests)
    first = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(model_server.requests[0])
    if template_enabled:
        assert json.loads(first.metadata["chat_template_kwargs"]) == {
            "enable_thinking": True,
            "truncate_history_thinking": False,
        }
    assert output["result"]["final_response"] == "summary of the work"


def test_clients_hermes_builds_itself_use_the_model_server(restore_process_globals) -> None:
    answers = [_completion({"content": "sync"}), _completion({"content": "async"})]

    with _ModelServer(answers) as model_server:
        _use_model_server(model_server.base_url)
        # Auxiliary clients find the endpoint through the environment, as Hermes's custom runtime does.
        sync = openai.OpenAI().chat.completions.create(model="m", messages=[{"role": "user", "content": "a"}])
        asynchronous = asyncio.run(
            openai.AsyncOpenAI().chat.completions.create(model="m", messages=[{"role": "user", "content": "b"}])
        )

    assert [sync.choices[0].message.content, asynchronous.choices[0].message.content] == ["sync", "async"]
    assert len(model_server.requests) == 2
    # Delegated children are AIAgents Hermes constructs itself; the Model Server rejects streaming.
    child = AIAgent(base_url=model_server.base_url, api_key="gym", model="m", quiet_mode=True)
    assert child.use_streaming is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc are required")
@pytest.mark.parametrize("ending", ["normal", "cancel"])
def test_supervisor_reaps_detached_tools_before_acknowledging_close(tmp_path, ending):
    worker = (
        "import subprocess,sys,pathlib,time,signal; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "pathlib.Path('child.pid').write_text(str(p.pid)); " + ("time.sleep(60)" if ending == "cancel" else "pass")
    )
    root = str(Path(__file__).resolve().parents[3])
    supervisor = (
        "import sys,json,pathlib; "
        f"sys.path.insert(0, {root!r}); "
        "import responses_api_agents; "
        f"responses_api_agents.__path__ = [{str(Path(root) / 'responses_api_agents')!r}]; "
        "from responses_api_agents.hermes_agent.sandbox_runner import _supervise; "
        "receipt=_supervise(json.loads(sys.argv[1]),cleanup_timeout=2); "
        "pathlib.Path('cleanup.json').write_text(json.dumps(receipt))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", supervisor, json.dumps([sys.executable, "-c", worker])],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        if ending == "cancel":
            for _ in range(500):
                if (tmp_path / "child.pid").exists():
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.01)
            assert (tmp_path / "child.pid").exists()
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, (stdout, stderr)
        receipt = json.loads((tmp_path / "cleanup.json").read_text())
        assert receipt == {"cleanup_confirmed": True, "error": None}
        pid = int((tmp_path / "child.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        # A failing regression must not leave the test's detached child running.
        if (tmp_path / "child.pid").exists():
            try:
                os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
