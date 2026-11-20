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
from types import ModuleType
from unittest.mock import MagicMock

import openai
import pytest
import uvicorn

from nemo_gym.agents import STOP_REQUEST_FILE, SUPERVISOR_SCRIPT
from nemo_gym.client import ServerClient
from nemo_gym.core import (
    AGENT_LIMIT,
    MODEL_CONTEXT_LIMIT,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_runtime import TOKEN_HEADER, maybe_auto_expose
from nemo_gym.trajectory import AgentInvocation
from resources_servers.example_mcp_weather.app import (
    ExampleMCPWeatherResourcesServer,
    ExampleMCPWeatherResourcesServerConfig,
)
from responses_api_agents.hermes_agent import sandbox_runner
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig
from responses_api_agents.hermes_agent.runtime import host_python
from responses_api_agents.hermes_agent.sandbox_runner import _use_model_server
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def _run(payload: dict, session: Path) -> dict:
    input_file = session / "input.json"
    output_file = session / "output.json"
    input_file.write_text(json.dumps(payload))
    process = subprocess.run(
        [str(host_python()), "-E", "-s", sandbox_runner.__file__, str(input_file), str(output_file)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    if not output_file.exists():
        raise RuntimeError(process.stderr)
    output = json.loads(output_file.read_text())
    if process.returncode:
        raise RuntimeError(output.get("error") or process.stderr)
    return output


def test_sandbox_iteration_limit_stays_on_child_when_observations_are_imported() -> None:
    agent = HermesAgent(
        config=HermesAgentConfig(
            host="127.0.0.1",
            port=0,
            name="hermes",
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "model"},
            resources_server={"type": "resources_servers", "name": "resources"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    raw = {
        "invocations": [
            {"invocation_id": "root", "status": "completed"},
            {
                "invocation_id": "root.child-1",
                "parent_invocation_id": "root",
                "spawned_by_tool_call_id": "call-delegate",
                "status": "incomplete",
                "stop_reason": "max_iterations",
            },
        ]
    }

    invocations = [
        record
        for record in agent._sandbox_observations({}, raw, runtime_info=None).records
        if isinstance(record, AgentInvocation)
    ]

    assert len(invocations) == 2
    assert invocations[0].status == "completed"
    assert invocations[0].stop_reason is None
    assert invocations[1].parent_invocation_id == "root"
    # The delegating tool call is what links a child invocation to the parent's trajectory.
    assert invocations[0].spawned_by_tool_call_id is None
    assert invocations[1].spawned_by_tool_call_id == "call-delegate"
    assert invocations[1].status == "incomplete"
    assert invocations[1].stop_reason == "max_iterations"


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

    def __init__(self, answers: list[dict], *, status_codes: list[int] | None = None) -> None:
        self.answers = answers
        self.requests: list[dict] = []
        self.request_headers: list[dict[str, str]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if not self.path.endswith("/chat/completions"):
                    self.send_error(404)
                    return
                server.requests.append(body)
                server.request_headers.append(dict(self.headers))
                answer = server.answers[min(len(server.requests), len(server.answers)) - 1]
                payload = json.dumps(answer).encode()
                index = min(len(server.requests), len(server.answers)) - 1
                self.send_response(status_codes[index] if status_codes is not None else 200)
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
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    yield


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


@pytest.mark.parametrize("inherited_hermes_home", [False, True])
def test_worker_configures_session_state_before_hermes_imports(tmp_path, inherited_hermes_home) -> None:
    import yaml

    from responses_api_agents.hermes_agent import sandbox_runner

    task_home = tmp_path / "task-home"
    task_hermes_home = task_home / ("custom-hermes" if inherited_hermes_home else ".hermes")
    task_hermes_home.mkdir(parents=True)
    (task_hermes_home / ".env").write_text("HERMES_TASK_ENV_LOADED=1\n")
    (task_hermes_home / "config.yaml").write_text("model: task_model\n")
    original_task_files = {path.name: path.read_bytes() for path in task_hermes_home.iterdir()}
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    input_path, output_path = session_dir / "input.json", session_dir / "output.json"
    report_path = tmp_path / "report.json"
    task_env = {
        "HOME": str(task_home),
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(tmp_path / "task-packages"),
        "PYTHONHOME": str(tmp_path / "task-python"),
    }
    env = {
        **os.environ,
        **task_env,
        "OPENAI_BASE_URL": "http://stale-task-model.invalid/v1",
        "OPENAI_API_KEY": "stale-task-key",
    }
    env.pop("HERMES_HOME", None)
    env.pop("HERMES_TASK_ENV_LOADED", None)
    if inherited_hermes_home:
        env["HERMES_HOME"] = str(task_hermes_home)
    # A fresh process must import the real dependency: this test module has already imported Hermes.
    # Pass the test environment's trusted import paths explicitly, independently of the task's PYTHONPATH.
    driver = """
import json, os, pathlib, subprocess, sys
sys.path[:0] = json.loads(sys.argv[1])
imports = {}
class WatchImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'run_agent', 'hermes_constants'}:
            home = pathlib.Path(os.environ.get('HERMES_HOME', pathlib.Path.home() / '.hermes'))
            config = home / 'config.yaml'
            imports[fullname] = {
                'home': str(home),
                'config': config.read_text() if config.exists() else None,
                'model_base_url': os.environ.get('OPENAI_BASE_URL'),
                'model_api_key': os.environ.get('OPENAI_API_KEY'),
            }
sys.meta_path.insert(0, WatchImports())
import sandbox_runner
status = sandbox_runner._run_worker(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
import run_agent
from tools.memory_tool import get_memory_dir
from tools.process_registry import _checkpoint_path
keys = ['HOME', 'PATH', 'PYTHONPATH', 'PYTHONHOME']
child_code = 'import json, os, sys; print(json.dumps({k: os.environ.get(k) for k in sys.argv[1:]}))'
child = subprocess.check_output(
    [sys.executable, '-I', '-c', child_code, *keys], text=True,
)
pathlib.Path(sys.argv[4]).write_text(json.dumps({
    'imports': imports,
    'import_home': str(run_agent._hermes_home),
    'memory_dir': str(get_memory_dir()),
    'checkpoint_path': str(_checkpoint_path()),
    'task_dotenv_loaded': os.environ.get('HERMES_TASK_ENV_LOADED'),
    'environment': {key: os.environ.get(key) for key in keys},
    'child_environment': json.loads(child),
}))
raise SystemExit(status)
"""
    with _ModelServer([_completion({"content": "done"})]) as server:
        payload = _payload(server.base_url)
        input_path.write_text(json.dumps(payload))
        result = subprocess.run(
            [
                str(host_python()),
                "-I",
                "-c",
                driver,
                json.dumps([str(Path(sandbox_runner.__file__).parent)]),
                str(input_path),
                str(output_path),
                str(report_path),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )

    details = output_path.read_text() if output_path.exists() else result.stdout
    assert result.returncode == 0, f"{result.stderr}\n{details}"
    assert json.loads(output_path.read_text())["result"]["final_response"] == "done"
    report = json.loads(report_path.read_text())
    hermes_home = session_dir / "hermes-home"
    expected_config = {
        "model": "policy_model",
        "provider": "auto",
        "tools": {"tool_search": {"enabled": "off"}},
        "auxiliary": {
            "compression": {
                "provider": "custom",
                "base_url": server.base_url,
                "api_key": "gym",
                "model": "policy_model",
            }
        },
    }
    for imported in report["imports"].values():
        imported["config"] = yaml.safe_load(imported["config"])
    assert report["imports"] == {
        name: {
            "home": str(hermes_home),
            "config": expected_config,
            "model_base_url": server.base_url,
            "model_api_key": "gym",
        }
        for name in ("run_agent", "hermes_constants")
    }
    assert report["import_home"] == str(hermes_home)
    assert report["memory_dir"] == str(hermes_home / "memories")
    assert report["checkpoint_path"] == str(hermes_home / "processes.json")
    assert report["task_dotenv_loaded"] is None
    assert report["environment"] == report["child_environment"] == task_env
    assert {path.name: path.read_bytes() for path in task_hermes_home.iterdir()} == original_task_files


def test_sandbox_retry_attempts_retain_invocation_headers(tmp_path, restore_process_globals) -> None:
    rejected = {"error": {"message": "retry this request", "type": "server_error"}}
    with _ModelServer([rejected, rejected, _completion({"content": "done"})], status_codes=[429, 500, 200]) as server:
        output = _run(_payload(server.base_url, max_turns=4), tmp_path)

    assert len(server.requests) == 3
    assert [headers.get("x-session-id") for headers in server.request_headers] == ["root"] * 3
    assert output["observations"]["invocations"][0]["model_response_ids"] == ["chatcmpl-test"]
    assert output["result"]["completed"]


@pytest.mark.parametrize("has_completed_call", [False, True])
def test_sandbox_rejection_preserves_runtime_token_totals(
    tmp_path, restore_process_globals, has_completed_call
) -> None:
    answers = [{"error": {"message": "rejected request", "type": "invalid_request_error"}}]
    statuses = [400]
    if has_completed_call:
        response = _completion(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": json.dumps({"command": "printf 391"})},
                    }
                ],
            }
        )
        response["usage"] = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        answers.insert(0, response)
        statuses.insert(0, 200)

    with _ModelServer(answers, status_codes=statuses) as server:
        output = _run(_payload(server.base_url, max_turns=4), tmp_path)

    result = output["result"]
    assert result["failed"] and not result["completed"]
    assert "400" in result["error"]
    assert result["prompt_tokens"] == (10 if has_completed_call else 0)
    assert result["completion_tokens"] == (2 if has_completed_call else 0)
    # This release treats HTTP 400 as terminal on the first rejected request.
    assert len(server.requests) == (2 if has_completed_call else 1)
    assert output["observations"]["invocations"][0]["status"] == "failed"


@pytest.mark.parametrize("marker", ["present", "absent", "omitted"])
def test_interruption_uses_supplied_stop_request_path(tmp_path, monkeypatch, restore_process_globals, marker):
    from responses_api_agents.hermes_agent import sandbox_runner

    class Agent:
        def __init__(self, **kwargs):
            self._session_messages = [{"role": "assistant", "content": "partial answer"}]

        def _build_api_kwargs(self, messages):
            return {}

        def run_conversation(self, *args, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return {"completed": True, "messages": self._session_messages}

        def interrupt(self, message):
            assert message == ("sandbox cancellation" if marker == "present" else "sandbox timeout")

    import yaml

    from responses_api_agents.hermes_agent import model_kwargs, sandbox_observer

    dependency = ModuleType("run_agent")
    dependency.AIAgent = Agent
    monkeypatch.setitem(sys.modules, "run_agent", dependency)
    monkeypatch.setitem(sys.modules, "hermes_yaml", yaml)
    monkeypatch.setattr(model_kwargs, "configure_model_calls", lambda *_args, **_kwargs: None)
    instrument = sandbox_observer.SandboxHermesObserver.instrument
    monkeypatch.setattr(
        sandbox_observer.SandboxHermesObserver, "instrument", lambda self, agent, **_: instrument(self, agent)
    )
    monkeypatch.setattr(sandbox_runner, "_use_model_server", lambda *_args: None)
    payload = _payload("http://unused/v1")
    if marker != "omitted":
        marker_path = tmp_path / "custom-stop-marker"
        payload["stop_request_path"] = str(marker_path)
        if marker == "present":
            marker_path.touch()
    output_path = tmp_path / "output.json"
    result = sandbox_runner._run(payload, tmp_path, output_path=output_path)["result"]
    expected = "cancelled" if marker == "present" else "wall_time"
    assert result["stop_reason"] == expected
    assert result["interrupted"] and not result["completed"]
    checkpoint = json.loads(output_path.read_text())["result"]
    assert checkpoint["stop_reason"] == expected
    assert checkpoint["messages"][-1]["content"] == "partial answer"


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
                    "name": "mcp__example_mcp_weather__get_weather",
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
    assert offered == ["mcp__example_mcp_weather__get_weather"]
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
def test_iteration_limit_summary_reaches_the_model_server(
    tmp_path, restore_process_globals, monkeypatch, template_enabled, execution
) -> None:
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
    answers[0]["id"] = "chatcmpl-tool"
    answers[1]["id"] = "chatcmpl-summary"

    with _ModelServer(answers) as model_server:
        if execution == "sandbox":
            output = _run(
                _payload(model_server.base_url, chat_template_kwargs_enabled=template_enabled),
                tmp_path,
            )
            assert output["result"]["final_response"].splitlines()[0] == "summary of the work"
            assert output["result"]["messages"][-1]["content"] == "summary of the work"
            assert output["result"]["stop_reason"] == "max_iterations"
            assert output["observations"]["invocations"][0]["stop_reason"] == "max_iterations"
            assert output["observations"]["invocations"][0]["status"] == "incomplete"
            assert output["observations"]["invocations"][0]["model_response_ids"] == [
                "chatcmpl-tool",
                "chatcmpl-summary",
            ]
        else:
            from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

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
            monkeypatch.setattr(HermesAgent, "resolve_model_base_url", lambda *_args, **_kwargs: model_server.base_url)
            observations = []
            output = asyncio.run(
                agent._create_response(
                    NeMoGymResponseCreateParamsNonStreaming(input="fix bug"),
                    observation_collector=observations.append,
                )
            )
            assert output.output[-1].content[0].text == "summary of the work"
            assert output.status == "incomplete"
            assert output.metadata["stop_reason"] == "max_iterations"
            assert observations[0].records[0].stop_reason == "max_iterations"
            assert observations[0].records[0].status == "incomplete"
            assert [call.response_id for call in observations[0].records[0].model_calls] == [
                "chatcmpl-tool",
                "chatcmpl-summary",
            ]

    assert len(model_server.requests) == 2
    assert all(not request.get("stream") for request in model_server.requests)
    first = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(model_server.requests[0])
    NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(model_server.requests[1])
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
@pytest.mark.parametrize("cooperative", [True, False])
@pytest.mark.parametrize("ending", ["deadline", "close"])
def test_worker_stop_checkpoints_partial_work_before_hard_cleanup(
    tmp_path: Path, cooperative: bool, ending: str
) -> None:
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from responses_api_agents.hermes_agent import sandbox_runner

    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    stop_request_path = tmp_path / STOP_REQUEST_FILE
    expire_path = tmp_path / "expire.request"
    input_path.write_text(json.dumps({**_payload("http://unused/v1"), "stop_request_path": str(stop_request_path)}))
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
import sandbox_runner, model_kwargs
model_kwargs.configure_model_calls = lambda *_args, **_kwargs: None
raise SystemExit(sandbox_runner._run_worker(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])))
"""
    process = subprocess.Popen(
        [
            "bash",
            str(SUPERVISOR_SCRIPT),
            "--timeout",
            "60",
            "--cleanup-timeout",
            "5",
            "--stop-file",
            str(stop_request_path),
            "--expire-file",
            str(expire_path),
            "--receipt",
            str(tmp_path / "cleanup.json"),
            "--",
            str(host_python()),
            "-I",
            "-c",
            driver,
            str(Path(sandbox_runner.__file__).parent),
            str(input_path),
            str(output_path),
            str(cooperative),
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    try:
        # Imports and worker startup are not part of the stop/checkpoint contract. Trigger either
        # stop only after the handler is installed and the harness has partial work to preserve.
        deadline = time.monotonic() + 30
        while not (tmp_path / "model.patch").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "model.patch").exists(), "the worker never produced partial work"
        assert not output_path.exists(), "the worker must still be running before the stop"
        # Expiry takes the supervisor's deadline path without racing a loaded runner's startup.
        # A close uses only the production stop file, not a signal from the test.
        (expire_path if ending == "deadline" else stop_request_path).touch()
        _, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=30)
    receipt = json.loads((tmp_path / "cleanup.json").read_text())
    assert receipt["timed_out"] is (ending == "deadline")
    assert receipt["cleanup_confirmed"] is True
    output = json.loads(output_path.read_text())
    reason = "wall_time" if ending == "deadline" else "cancelled"
    assert (tmp_path / "interrupted").read_text() == (
        "sandbox timeout" if ending == "deadline" else "sandbox cancellation"
    )
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
    assert response.metadata["stop_reason"] == reason
    assert response.output[0].content[0].text == "Partial work"
    assert response.output[0].generation_token_ids == [2]
    assert output["observations"]["invocations"][0]["status"] == "incomplete"
    with pytest.raises(ProcessLookupError):
        os.kill(output["runtime"]["pid"], 0)


def test_compression_auxiliary_uses_episode_model_route(tmp_path, restore_process_globals):
    import yaml

    with _ModelServer([_completion({"content": "done"})]) as server:
        _run(_payload(server.base_url, max_turns=4), tmp_path)
        probe = "from agent.auxiliary_client import call_llm; print(call_llm(task='compression', messages=[{'role':'user','content':'Summarize'}]).choices[0].message.content)"
        response = subprocess.check_output(
            [str(host_python()), "-I", "-c", probe],
            text=True,
            env={**os.environ, "HERMES_HOME": str(tmp_path / "hermes-home")},
        )
    assert response.strip().endswith("done")
    assert server.requests[-1]["messages"] == [{"role": "user", "content": "Summarize"}]
    config = yaml.safe_load((tmp_path / "hermes-home/config.yaml").read_text())
    assert config["auxiliary"]["compression"]["base_url"] == server.base_url
    assert config["auxiliary"]["compression"]["model"] == "policy_model"


@pytest.mark.parametrize("image_only", [False, True])
@pytest.mark.parametrize(
    "detail,sent_detail", [("auto", "auto"), ("low", "low"), ("high", "high"), ("original", "high")]
)
def test_prompt_images_reach_pinned_hermes_model_requests(
    tmp_path, restore_process_globals, image_only, detail, sent_detail
) -> None:
    """Exercise the installed Hermes loop, not a mock of its multimodal input handling."""
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

    image_uri = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="
    )
    content = [
        {"type": "input_text", "text": "before"},
        {"type": "input_image", "image_url": image_uri, "detail": detail},
        {"type": "input_text", "text": "after"},
    ]
    if image_only:
        content = [content[1]]
    adapter = HermesAgent(
        config=HermesAgentConfig(
            host="127.0.0.1",
            port=0,
            name="hermes",
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "model"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    body = adapter._validate_request(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "user", "content": content},
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": content},
            ]
        )
    )
    with _ModelServer([_completion({"content": "done"})]) as server:
        # Hermes marks a response at its iteration ceiling incomplete, even with a final answer.
        output = _run(_payload(server.base_url, max_turns=4, **adapter._conversation_params(body)), tmp_path)
    expected = [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": image_uri, "detail": sent_detail}},
        {"type": "text", "text": "after"},
    ]
    if image_only:
        expected = [expected[1]]
    assert output["result"]["completed"], output["result"]
    assert len(server.requests) == 1
    user_messages = [message for message in server.requests[0]["messages"] if message["role"] == "user"]
    assert [message["content"] for message in user_messages] == [expected, expected]
    assert any(message["content"] == "prior answer" for message in server.requests[0]["messages"])


def test_compression_auxiliary_uses_its_own_route_when_the_payload_has_one(tmp_path, restore_process_globals):
    """The agent gives compression a URL that keeps its own sampling; the runner points Hermes at it."""
    import yaml

    compression_url = "http://model:8000/ng-rollout/episode-a1/ng-sampling/null/v1"
    with _ModelServer([_completion({"content": "done"})]) as server:
        _run(_payload(server.base_url, compression_model_base_url=compression_url), tmp_path)
    config = yaml.safe_load((tmp_path / "hermes-home/config.yaml").read_text())
    assert config["auxiliary"]["compression"]["base_url"] == compression_url


@pytest.mark.parametrize("max_turns", [1, 3], ids=["summary", "normal"])
def test_training_token_bundles_follow_their_exact_turn(tmp_path, max_turns):
    bundles = [
        {"prompt_token_ids": [17], "generation_token_ids": [23], "generation_log_probs": [-0.4]},
        {"prompt_token_ids": [17, 23, 30], "generation_token_ids": [41, 42], "generation_log_probs": [-0.5, -0.2]},
    ]
    answers = [
        _completion(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "t",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command":"echo ok"}'},
                    }
                ],
                **bundles[0],
            }
        ),
        _completion({"content": "finished", **bundles[1]}),
    ]
    with _ModelServer(answers) as model:
        output = _run(_payload(model.base_url, max_turns=max_turns), tmp_path)
    assert len(model.requests) == 2
    for body in model.requests:
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(body)
    replay = next(m for m in model.requests[1]["messages"] if m["role"] == "assistant")
    assert {key: replay.get(key) for key in bundles[0]} == bundles[0]
    turns = [m for m in output["result"]["messages"] if m["role"] == "assistant"]
    assert [{key: m.get(key) for key in bundles[0]} for m in turns] == bundles


def _length_cut(index: int, **message) -> dict:
    """A chat completion cut at the output limit, carrying one distinct sampled token bundle."""
    response = _completion(
        {"prompt_token_ids": [100 + index], "generation_token_ids": [200 + index], "generation_log_probs": [-0.5]}
        | message
    )
    response["choices"][0]["finish_reason"] = "length"
    return response


def _hermes(**overrides) -> HermesAgent:
    return HermesAgent(
        config=HermesAgentConfig(
            host="127.0.0.1",
            port=0,
            name="hermes",
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            **overrides,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def test_continuation_ceiling_keeps_every_sampled_fragment_for_training(tmp_path) -> None:
    """At its continuation ceiling Hermes stitches the fragments into one row without token IDs; Gym keeps them."""
    with _ModelServer([_length_cut(i, content=f"part-{i} ") for i in range(6)]) as model:
        result = _run(_payload(model.base_url, max_turns=8, max_tokens=64), tmp_path)["result"]
    response = _hermes()._response_from_result(
        body=NeMoGymResponseCreateParamsNonStreaming(input="fix bug"), result=result, model_name="model", n_input=1
    )

    assert len(model.requests) == 4
    assert result["error"] == "Response remained truncated after 4 continuation attempts"
    assert _hermes().classify_outcome(result).failure_kind == AGENT_LIMIT
    messages = [item for item in response.output if item.type == "message"]
    assert [item.generation_token_ids for item in messages] == [[200], [201], [202], [203]]
    assert [item.content[0].text for item in messages] == [f"part-{i}" for i in range(4)]
    assert not any("_length_continuation" in key for row in result["messages"] for key in row)


def test_truncation_retries_stay_within_configured_max_tokens(tmp_path) -> None:
    """Hermes doubles the output cap on each continuation; Gym's max_tokens stays the per-call budget."""
    with _ModelServer([_length_cut(i, content=f"part-{i} ") for i in range(6)]) as model:
        _run(_payload(model.base_url, max_turns=8, max_tokens=64), tmp_path)

    assert [request["max_tokens"] for request in model.requests] == [64] * 4


def test_truncated_tool_call_limit_is_an_agent_limit(tmp_path) -> None:
    """Hermes retries a length-cut tool call four times, then ends with `failure_reason: truncated`."""
    call = {"id": "call", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "ec'}}
    with _ModelServer([_length_cut(i, content=None, tool_calls=[call]) for i in range(6)]) as model:
        result = _run(_payload(model.base_url, max_turns=8, max_tokens=64), tmp_path)["result"]

    assert len(model.requests) == 5
    assert result["failure_reason"] == "truncated"
    assert _hermes().classify_outcome(result).failure_kind == AGENT_LIMIT


def test_configured_context_length_reaches_the_pinned_worker(tmp_path) -> None:
    """The configured window reaches Hermes: a prompt that fills it is a model context limit, not a truncation."""
    agent = _hermes(context_length=65536)
    answers = []
    for index in range(6):
        answer = _length_cut(index, content=f"part-{index} ")
        answer["usage"] = {"prompt_tokens": 65500, "completion_tokens": 64, "total_tokens": 65564}
        answers.append(answer)
    with _ModelServer(answers) as model:
        result = _run(
            _payload(model.base_url, config_yaml=agent._build_config(), max_turns=8, max_tokens=64), tmp_path
        )["result"]

    assert result["error"] == "Prompt used 65500 of 65536 context tokens; no room to answer"
    assert agent.classify_outcome(result).failure_kind == MODEL_CONTEXT_LIMIT


def test_delegated_child_records_the_delegate_task_call_that_spawned_it(tmp_path) -> None:
    """Hermes runs delegate_task on its sequential tool path; the child still names the call that spawned it.

    Hermes 0.23.2 runs a top-level delegation in the background, so the child may not answer before the parent
    finishes; it is registered when spawned.
    """
    delegate = {
        "id": "call-delegate",
        "type": "function",
        "function": {"name": "delegate_task", "arguments": json.dumps({"goal": "Reply with the word done."})},
    }
    answers = [
        _completion({"content": None, "tool_calls": [delegate]}) | {"id": "chatcmpl-parent-1"},
        _completion({"content": "done"}) | {"id": "chatcmpl-child"},
        _completion({"content": "The child finished."}) | {"id": "chatcmpl-parent-2"},
    ]
    answers[0]["choices"][0]["finish_reason"] = "tool_calls"
    with _ModelServer(answers) as model:
        output = _run(_payload(model.base_url, enabled_toolsets=["delegation"], max_turns=4), tmp_path)

    children = [
        invocation
        for invocation in output["observations"]["invocations"]
        if invocation.get("parent_invocation_id") == "root"
    ]
    assert [child["spawned_by_tool_call_id"] for child in children] == ["call-delegate"]
