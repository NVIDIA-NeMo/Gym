# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import openai
import pytest
import uvicorn
from run_agent import AIAgent
from tools.mcp_tool import shutdown_mcp_servers

from nemo_gym.mcp_auto_exposure import TOKEN_HEADER, maybe_auto_expose
from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.example_mcp_weather.app import (
    ExampleMCPWeatherResourcesServer,
    ExampleMCPWeatherResourcesServerConfig,
)
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


class _ResourcesServer:
    """Serve a Resources Server app, with its tools exposed over MCP, on a local port."""

    def __init__(self, app) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "_ResourcesServer":
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("Resources Server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *_exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


@pytest.fixture
def restore_process_globals(monkeypatch: pytest.MonkeyPatch):
    """The runner changes process-wide state; restore it so other tests see the originals."""
    monkeypatch.setattr(AIAgent, "__init__", AIAgent.__init__)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    yield
    shutdown_mcp_servers()


def _payload(model_base_url: str, **overrides) -> dict:
    return {
        "agent_session_id": "session",
        "chat_template_kwargs_enabled": False,
        "config_yaml": "model: policy_model\nprovider: auto\n",
        "disabled_toolsets": None,
        "enabled_toolsets": ["terminal"],
        "history": [],
        "max_tokens": 128,
        "max_turns": 1,
        "mcp_servers": [],
        "model": "policy_model",
        "model_base_url": model_base_url,
        "required_mcp_servers": [],
        "system_message": None,
        "temperature": 0.0,
        "terminal_timeout": 30,
        "user_message": "fix bug",
        **overrides,
    }


def test_granted_mcp_tools_reach_the_seeded_resources_session(tmp_path, restore_process_globals) -> None:
    resources = ExampleMCPWeatherResourcesServer(
        config=ExampleMCPWeatherResourcesServerConfig(
            host="127.0.0.1", port=0, entrypoint="app.py", name="example_mcp_weather", expose_tools_over_mcp=True
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    app = resources.setup_webserver()
    maybe_auto_expose(resources, app)
    tool_call = {
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "mcp_example_mcp_weather_get_weather",
                    "arguments": json.dumps({"city": "Paris"}),
                },
            }
        ],
    }
    answers = [_completion(tool_call), _completion({"content": "The weather in Paris is sunny and 72 F."})]

    with _ResourcesServer(app) as resources_server, _ModelServer(answers) as model_server:
        seed_request = urllib.request.Request(
            f"{resources_server.base_url}/seed_session",
            data=json.dumps({"verifier_metadata": {"expected_city": "Paris"}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(seed_request) as seed_response:
            token = json.loads(seed_response.read())["mcp"]["headers"][TOKEN_HEADER]
        config_yaml = (
            "model: policy_model\nprovider: auto\nmcp_servers:\n"
            "  example_mcp_weather:\n"
            f"    url: {resources_server.base_url}/mcp\n"
            f"    headers: {{{TOKEN_HEADER}: {token}}}\n"
            "    tools: {resources: false, prompts: false}\n"
        )
        output = _run(
            _payload(
                model_server.base_url,
                config_yaml=config_yaml,
                enabled_toolsets=["example_mcp_weather"],
                max_turns=2,
                mcp_servers=["example_mcp_weather"],
                required_mcp_servers=["example_mcp_weather"],
            ),
            tmp_path,
        )

    offered = [tool["function"]["name"] for tool in model_server.requests[0]["tools"]]
    assert offered == ["mcp_example_mcp_weather_get_weather"]
    assert output["result"]["final_response"] == "The weather in Paris is sunny and 72 F."
    # The call carried the seed's token, so it landed in the session the seed created.
    assert [state["weather_calls"] for state in resources.session_id_to_state.values()] == [
        [{"city": "Paris", "weather": "The weather in Paris is sunny and 72 F."}]
    ]


def test_required_mcp_server_that_does_not_connect_fails_before_the_model(tmp_path, restore_process_globals) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        unused_port = probe.getsockname()[1]
    config_yaml = (
        "model: policy_model\nprovider: auto\nmcp_servers:\n"
        f"  weather:\n    url: http://127.0.0.1:{unused_port}/mcp\n    connect_timeout: 1\n"
    )

    with _ModelServer([_completion({"content": "done"})]) as model_server:
        with pytest.raises(RuntimeError, match="Required MCP servers did not connect: weather"):
            _run(
                _payload(
                    model_server.base_url,
                    config_yaml=config_yaml,
                    mcp_servers=["weather"],
                    required_mcp_servers=["weather"],
                ),
                tmp_path,
            )

    assert model_server.requests == []


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
            _payload(model_server.base_url, chat_template_kwargs_enabled=template_enabled),
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
