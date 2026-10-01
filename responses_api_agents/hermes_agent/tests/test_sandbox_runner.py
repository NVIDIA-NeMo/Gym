# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
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
from nemo_gym.sandbox import process_supervisor
from nemo_gym.server_utils import ServerClient
from resources_servers.example_mcp_weather.app import (
    ExampleMCPWeatherResourcesServer,
    ExampleMCPWeatherResourcesServerConfig,
)
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig
from responses_api_agents.hermes_agent.sandbox_runner import _run, _use_model_server
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


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
@pytest.mark.parametrize("execution", ["sandbox", "local"])
@pytest.mark.parametrize("conflicting_override", [False, True])
def test_iteration_limit_summary_reaches_the_model_server(
    tmp_path, restore_process_globals, monkeypatch, caplog, template_enabled, execution, conflicting_override
) -> None:
    if conflicting_override:
        original = AIAgent._build_api_kwargs

        def with_override(self, messages):
            kwargs = original(self, messages)
            kwargs.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})["enable_thinking"] = True
            return kwargs

        monkeypatch.setattr(AIAgent, "_build_api_kwargs", with_override)
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
        if execution == "sandbox":
            output = _run(
                _payload(
                    model_server.base_url, chat_template_kwargs_enabled=template_enabled, model_enable_thinking=False
                ),
                tmp_path,
            )
            assert output["result"]["final_response"] == "summary of the work"
        else:
            from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

            agent = HermesAgent(
                config=HermesAgentConfig(
                    host="127.0.0.1",
                    port=0,
                    name="hermes",
                    entrypoint="app.py",
                    model_server={"type": "responses_api_models", "name": "model"},
                    resources_server={"type": "resources_servers", "name": "resources"},
                    max_turns=1,
                    max_tokens=128,
                    enabled_toolsets=["terminal"],
                    chat_template_kwargs_enabled=template_enabled,
                ),
                server_client=MagicMock(
                    spec=ServerClient,
                    global_config_dict={
                        "model": {
                            "responses_api_models": {
                                "vllm_model": {
                                    "chat_template_kwargs": {
                                        "enable_thinking": False,
                                    }
                                }
                            }
                        }
                    },
                ),
            )
            monkeypatch.setattr(HermesAgent, "resolve_model_base_url", lambda *_args: model_server.base_url)
            output = asyncio.run(agent._create_response(NeMoGymResponseCreateParamsNonStreaming(input="fix bug")))
            assert output.output[-1].content[0].text == "summary of the work"

    assert len(model_server.requests) == 2
    if conflicting_override:
        assert "conflicts with Model Server enable_thinking=False" in caplog.text
    assert all(not request.get("stream") for request in model_server.requests)
    first = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(model_server.requests[0])
    assert all("chat_template_kwargs" not in body for body in model_server.requests)
    if template_enabled:
        assert json.loads(first.metadata["chat_template_kwargs"]) == {
            "truncate_history_thinking": False,
        }
    # Exercise Gym's actual merge order: neither path may override a server-configured thinking mode.
    for thinking in (False, True):
        server = VLLMModel(
            config=VLLMModelConfig(
                host="127.0.0.1",
                port=0,
                name="model",
                entrypoint="app.py",
                model="policy_model",
                base_url="http://unused/v1",
                api_key="gym",
                chat_template_kwargs={"enable_thinking": thinking},
                return_token_id_information=False,
                uses_reasoning_parser=False,
            ),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )
        forwarded = server._preprocess_chat_completion_create_params(
            request=None, body_dict=first.model_dump(exclude_unset=True)
        )
        assert forwarded["chat_template_kwargs"]["enable_thinking"] is thinking


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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
@pytest.mark.parametrize("cooperative", [True, False])
def test_worker_deadline_checkpoints_partial_work_before_hard_cleanup(tmp_path, cooperative):
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from responses_api_agents.hermes_agent import sandbox_runner

    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(_payload("http://unused/v1")))
    # Run the real worker/observer under the real supervisor, with a deterministic slow harness.
    driver = """
import os, pathlib, sys, time, types
sys.path.insert(0, sys.argv[1])
class Agent:
    def __init__(self, **kwargs):
        self._session_messages = []
        self.stopping = False
    def _build_api_kwargs(self, messages):
        return {}
    def interrupt(self, message):
        pathlib.Path('interrupted').write_text(message)
        self.stopping = True
    def run_conversation(self, *args, **kwargs):
        self._session_messages = [
            {'role': 'user', 'content': 'fix bug'},
            {'role': 'assistant', 'content': 'Partial work', 'prompt_token_ids': [1], 'generation_token_ids': [2]},
        ]
        pathlib.Path('model.patch').write_bytes(b'partial patch\\n')
        while not self.stopping or sys.argv[4] == 'False':
            time.sleep(0.01)
        return {'completed': False, 'messages': self._session_messages}
module = types.ModuleType('run_agent')
module.AIAgent = Agent
sys.modules['run_agent'] = module
import sandbox_runner
raise SystemExit(sandbox_runner._run_worker(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            process_supervisor.__file__,
            "--timeout",
            "2",
            "--cleanup-timeout",
            "0.5",
            "--receipt",
            str(tmp_path / "cleanup.json"),
            "--",
            sys.executable,
            "-I",
            "-c",
            driver,
            str(Path(sandbox_runner.__file__).parent),
            str(input_path),
            str(output_path),
            str(cooperative),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((tmp_path / "cleanup.json").read_text())
    assert receipt["timed_out"] is True
    assert receipt["cleanup_confirmed"] is True
    output = json.loads(output_path.read_text())
    assert (tmp_path / "interrupted").read_text() == "sandbox timeout"
    assert (tmp_path / "model.patch").read_bytes() == b"partial patch\n"
    response = HermesAgent._response_from_result(
        None,
        body=NeMoGymResponseCreateParamsNonStreaming(input="fix bug"),
        result=output["result"],
        model_name="model",
        n_input=1,
    )
    assert response.status == "incomplete"
    assert response.error is None
    assert response.metadata["stop_reason"] == "wall_time"
    assert response.output[0].content[0].text == "Partial work"
    assert response.output[0].generation_token_ids == [2]
    assert output["observations"]["invocations"][0]["status"] == "incomplete"
    with pytest.raises(ProcessLookupError):
        os.kill(output["runtime"]["pid"], 0)
