# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_sandboxed_agent import app as app_module
from responses_api_agents.terminus_2_sandboxed_agent.app import (
    NeMoGymLLM,
    NeMoGymSandboxEnvironment,
    NeMoGymTerminus2,
    Terminus2Agent,
    Terminus2AgentConfig,
    _instruction,
)


def test_instruction_joins_text_content():
    assert _instruction([{"content": [{"text": "first"}]}, {"content": "second"}]) == "first\n\nsecond"


def test_missing_usage_falls_back_to_counting_current_chat(monkeypatch):
    import litellm.utils

    counted = []

    def count_tokens(*, model, messages):
        counted.append((model, messages))
        return 42

    monkeypatch.setattr(litellm.utils, "token_counter", count_tokens)
    agent = object.__new__(NeMoGymTerminus2)
    agent._model_name = "policy_model"
    agent._is_check_proactive_summarization = True
    agent._nemo_gym_llm = SimpleNamespace(usages=[SimpleNamespace(total_tokens=1000), None])
    chat = SimpleNamespace(messages=[{"role": "user", "content": "current prompt"}])
    assert agent._count_total_tokens(chat) == 42
    assert counted == [("policy_model", chat.messages)]
    assert agent._nemo_gym_llm.usages[-1] is None


@pytest.mark.asyncio
async def test_shell_recovery_skips_remaining_commands_and_resets_completion():
    from responses_api_agents.terminus_2_sandboxed_agent.terminal import ShellExitedError

    sent = []

    async def send_keys(keys, **kwargs):
        sent.append(keys)
        if keys == "second":
            raise ShellExitedError("shell exited after first command")

    async def recover_shell():
        return "The shell exited. Its state has reset; remaining commands were skipped."

    agent = object.__new__(NeMoGymTerminus2)
    agent._pending_completion = True
    agent._completed_command_batches = 0
    agent._times_spent = []
    commands = [SimpleNamespace(keystrokes=key, duration_sec=0) for key in ("first", "second", "third")]
    result = await agent._execute_commands(commands, SimpleNamespace(send_keys=send_keys, recover_shell=recover_shell))
    assert sent == ["first", "second"]
    assert result == (False, "The shell exited. Its state has reset; remaining commands were skipped.")
    assert agent._pending_completion is False
    assert agent._completed_command_batches == 1


@pytest.mark.asyncio
async def test_sandbox_environment_adapts_exec_and_is_dir():
    sandbox_calls = []

    async def sandbox_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        return SimpleNamespace(stdout="output", stderr=None, return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)
    environment = NeMoGymSandboxEnvironment(sandbox, logs_dir=SimpleNamespace(), session_id="session-1")

    result = await environment.exec("pwd", timeout_sec=12, user="root", cwd="/work")

    assert result.stdout == "output"
    assert result.stderr == ""
    assert result.return_code == 0
    assert await environment.is_dir("/workspace")
    assert sandbox_calls == [
        ("pwd", {"timeout_s": 12, "cwd": "/work", "user": "root", "env": None}),
        ('test -d "/workspace"', {"user": None}),
    ]


@pytest.mark.asyncio
async def test_sandbox_environment_uses_sandbox_exec_for_stateful_commands():
    calls = []

    async def sandbox_exec(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="output", stderr=None, return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)
    environment = NeMoGymSandboxEnvironment(sandbox, logs_dir=SimpleNamespace(), session_id="session-1")

    await environment.exec("tmux new-session")

    assert calls == [("tmux new-session", {"timeout_s": None, "cwd": None, "user": None, "env": None})]


def test_agent_implements_required_responses_endpoint():
    assert not getattr(Terminus2Agent, "__abstractmethods__", set())


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_content", [None, "reasoning before answer 1"])
async def test_nemo_gym_llm_records_every_responses_request_and_output(reasoning_content):
    class Client:
        def __init__(self):
            self.requests = []

        async def create_response(self, **kwargs):
            self.requests.append(kwargs)
            index = len(self.requests)
            return NeMoGymResponse(
                id=f"resp_{index}",
                created_at=0,
                model="policy_model",
                object="response",
                output=[
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{index}",
                        content=[
                            NeMoGymResponseOutputText(type="output_text", text=f"answer {index}", annotations=[])
                        ],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=True,
                usage=NeMoGymResponseUsage(
                    input_tokens=10,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=2),
                    output_tokens=3,
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                    total_tokens=13,
                ),
            )

    client = Client()
    llm = NeMoGymLLM(
        client=client,
        model_name="policy_model",
        model_context_limit=32_000,
        model_output_limit=4_000,
        llm_request_timeout=60,
    )

    first = await llm.call("first")
    second = await llm.call(
        "second",
        message_history=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer 1", "reasoning_content": reasoning_content},
        ],
        previous_response_id="resp_1",
    )
    third = await llm.call(
        "third",
        message_history=[{"role": "user", "content": "compacted summary"}],
        previous_response_id="resp_2",
    )

    assert first.content == "answer 1"
    assert first.usage.prompt_tokens == 10
    assert second.content == "answer 2"
    assert third.content == "answer 3"
    expected_reasoning = (
        [{"summary": [{"text": reasoning_content, "type": "summary_text"}], "type": "reasoning"}]
        if reasoning_content
        else []
    )
    # Replayed assistant and reasoning items carry fresh well-formed ids; compare everything else.
    for request_body in client.requests:
        for item in request_body["input"]:
            if item.get("type") in ("message", "reasoning") and item.get("id", "").split("_")[0] in ("msg", "rs"):
                item.pop("id")
    assert client.requests == [
        {"model": "policy_model", "input": [{"content": "first", "role": "user", "type": "message"}]},
        {
            "model": "policy_model",
            "input": [
                {"content": "first", "role": "user", "type": "message"},
                *expected_reasoning,
                {
                    "content": [{"annotations": [], "text": "answer 1", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
                {"content": "second", "role": "user", "type": "message"},
            ],
        },
        {
            "model": "policy_model",
            "input": [
                {"content": "compacted summary", "role": "user", "type": "message"},
                {"content": "third", "role": "user", "type": "message"},
            ],
        },
    ]
    assert [item.content for item in llm.trajectory if isinstance(item, NeMoGymEasyInputMessage)] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("dump_trajectory", [False, True])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("interleaved_thinking", [False, True])
async def test_execute_runs_terminus_in_seeded_sandbox(monkeypatch, dump_trajectory, debug, interleaved_thinking):
    config = Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_1_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="swebench_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_turns=100,
        enable_summarize=True,
        proactive_summarization_threshold=8000,
        tmux_pane_width=160,
        tmux_pane_height=40,
        dump_trajectory=dump_trajectory,
        debug=debug,
        model_context_limit=32_000,
        model_output_limit=4_000,
        interleaved_thinking=interleaved_thinking,
        llm_request_timeout=60,
        sandbox_provider="opensandbox",
        sandbox_timeout=10,
        remote_tmux_binary_path=None,
    )
    set_level = MagicMock()
    monkeypatch.setattr(app_module.harbor_logger, "setLevel", set_level)
    server = Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox_calls = []

    async def sandbox_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)

    class FakeTerminus:
        session = SimpleNamespace()

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._session = SimpleNamespace(stop=self.stop)
            self._times_spent = [1.0, 3.0]
            self._num_proactive_compactions = 0
            self._num_compactions = 2

        async def stop(self):
            return None

        async def setup(self, environment):
            await environment.exec("tmux setup")

        async def run(self, instruction, environment, context):
            assert instruction == "solve this"
            assert self.kwargs["dump_trajectory"] is dump_trajectory
            assert self.kwargs["interleaved_thinking"] is interleaved_thinking
            await environment.exec("tmux run")
            self.kwargs["llm"]._times_spent.extend([2.0, 4.0])
            self.kwargs["llm"]._num_compactions = 2
            context.n_input_tokens = 4
            context.n_output_tokens = 3
            self.kwargs["llm"].trajectory.append(
                NeMoGymResponseOutputMessage(
                    id="msg_done",
                    content=[NeMoGymResponseOutputText(type="output_text", text="done", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

    class FakeContext:
        n_input_tokens = None
        n_cache_tokens = None
        n_output_tokens = None
        metadata = None

    monkeypatch.setattr(app_module, "NeMoGymTerminus2", FakeTerminus)
    monkeypatch.setattr(app_module, "AgentContext", FakeContext)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    elapsed_times = iter([10.0, 20.0])
    monkeypatch.setattr(app_module, "perf_counter", lambda: next(elapsed_times))

    async def request_json():
        return {"task_id": "task"}

    request = SimpleNamespace(json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    response, metrics = await server._execute(
        request,
        NeMoGymResponseCreateParamsNonStreaming(input="solve this"),
        sandbox,
    )

    assert metrics == {
        "terminus2_completed": True,
        "command_exec_times": [1.0, 3.0],
        "model_call_times": [2.0, 4.0],
        "average_command_exec_time": 2.0,
        "average_model_call_time": 3.0,
        "total_command_exec_time": 4.0,
        "total_model_call_time": 6.0,
        "command_exec_time_pct": 40.0,
        "model_call_time_pct": 60.0,
        "terminus2_time_taken": 10.0,
        "model_calls_gt_10min": 0,
        "num_proactive_compactions": 0,
        "num_compactions": 2,
        "error": None,
        "usages": [],
    }
    assert response.output[-1].content[0].text == "done"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 3
    if not debug:
        set_level.assert_called_once_with(logging.WARNING)
    else:
        set_level.assert_not_called()
    assert sandbox_calls == [
        ("mkdir -p /logs/agent", {"timeout_s": None, "cwd": None, "user": "root", "env": None}),
        ("tmux setup", {"timeout_s": None, "cwd": None, "user": None, "env": None}),
        ("tmux run", {"timeout_s": None, "cwd": None, "user": None, "env": None}),
    ]


def _native_config() -> Terminus2AgentConfig:
    return Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_sandboxed_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="harbor_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_turns=1,
        enable_summarize=False,
        proactive_summarization_threshold=8000,
        tmux_pane_width=160,
        tmux_pane_height=40,
        model_context_limit=32_000,
        model_output_limit=None,
        interleaved_thinking=False,
        llm_request_timeout=60,
        sandbox_provider="sandbox",
        sandbox_timeout=10,
        remote_tmux_binary_path=None,
    )


def _seed_body(*, session="ag-1", rollout="r1", with_sandbox=True) -> dict:
    body = {
        "agent_session_id": session,
        "episode_id": {"rollout_id": rollout, "attempt": 0},
        "task_id": {"taskset": "tb", "task_id": "path-tracing"},
        "tool_accesses": [],
    }
    if with_sandbox:
        body["sandbox_access"] = {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "sb-1"}},
            "workdir": "/app",
        }
    return body


class TestNativeSessions:
    """The environment-server protocol: borrow the resources server's sandbox, run, disconnect."""

    def _client(self, monkeypatch):
        from unittest.mock import AsyncMock

        from fastapi.testclient import TestClient

        agent = Terminus2Agent(config=_native_config(), server_client=MagicMock(spec=ServerClient))
        sandbox = SimpleNamespace(disconnect=AsyncMock(), exec=AsyncMock())
        connect = AsyncMock(return_value=sandbox)
        monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
        monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, cfg: {"opensandbox": {}})
        monkeypatch.setattr(app_module, "create_provider", lambda config: MagicMock())
        monkeypatch.setattr(app_module.AsyncSandbox, "connect", connect)
        return agent, TestClient(agent.setup_webserver()), sandbox, connect

    def test_seed_connects_to_borrowed_sandbox(self, monkeypatch):
        agent, client, sandbox, connect = self._client(monkeypatch)

        response = client.post("/v1/agent_sessions", json=_seed_body())

        assert response.status_code == 200, response.text
        connect.assert_awaited_once()
        assert connect.await_args.args[0] == {"sandbox_id": "sb-1"}
        assert agent._session_records["ag-1"].state.workdir == "/app"
        # Idempotent re-seed, and a different episode cannot take the id.
        assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 200
        assert connect.await_count == 1
        assert client.post("/v1/agent_sessions", json=_seed_body(rollout="other")).status_code == 409

    def test_seed_refuses_multiple_workers(self, monkeypatch):
        from fastapi.testclient import TestClient

        config = _native_config()
        config.num_workers = 4
        agent = Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))
        # The base agent refuses multi-worker sessions before any Terminus state exists.
        with pytest.raises(ValueError, match="num_workers=1"):
            TestClient(agent.setup_webserver()).post("/v1/agent_sessions", json=_seed_body())

    def test_seed_requires_sandbox_access(self, monkeypatch):
        _, client, _, _ = self._client(monkeypatch)
        assert client.post("/v1/agent_sessions", json=_seed_body(with_sandbox=False)).status_code == 422

    def test_responses_uses_the_session_sandbox(self, monkeypatch):
        from unittest.mock import AsyncMock

        agent, client, sandbox, _ = self._client(monkeypatch)
        assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 200
        seen = {}

        async def fake_execute(request, body, used_sandbox):
            seen["sandbox"] = used_sandbox
            return app_module.NeMoGymResponse(
                id="resp",
                created_at=0,
                model="m",
                object="response",
                output=[],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            ), {}

        monkeypatch.setattr(agent, "_execute", fake_execute)
        response = client.post("/v1/responses", json={"input": [{"role": "user", "content": "do it"}]})
        assert response.status_code == 200, response.text
        assert seen["sandbox"] is sandbox
        _ = AsyncMock

    def test_close_disconnects_without_stopping(self, monkeypatch):
        agent, client, sandbox, _ = self._client(monkeypatch)
        assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 200
        wrong = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "other", "attempt": 0}},
        )
        assert wrong.status_code == 409 and not sandbox.disconnect.await_count
        right = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
        )
        assert right.status_code == 200
        sandbox.disconnect.assert_awaited_once()
        assert not hasattr(sandbox, "stop") or not getattr(sandbox.stop, "await_count", 0)
        assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 409


def _harness_config(**overrides) -> Terminus2AgentConfig:
    config = _native_config()
    config.sandbox_timeout = 10
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


async def _execute_with_harness(monkeypatch, run, *, config=None, metadata=None, sandbox_exec=None):
    """Run ``_execute`` with a fake Harbor loop whose ``run`` is the given coroutine function."""
    server = Terminus2Agent(config=config or _harness_config(), server_client=MagicMock(spec=ServerClient))

    async def default_exec(command, **kwargs):
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec or default_exec, stop=MagicMock())

    class FakeTerminus:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._times_spent = []
            self._num_proactive_compactions = 0

        async def setup(self, environment):
            return None

        async def run(self, instruction, environment, context):
            await run(self, instruction, environment, context)

    class FakeContext:
        n_input_tokens = None
        n_cache_tokens = None
        n_output_tokens = None
        metadata = None

    monkeypatch.setattr(app_module, "NeMoGymTerminus2", FakeTerminus)
    monkeypatch.setattr(app_module, "AgentContext", FakeContext)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")

    async def request_json():
        return {"task_id": "task"}

    request = SimpleNamespace(json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    body = NeMoGymResponseCreateParamsNonStreaming(input="solve this", metadata=metadata)
    return server, sandbox, await server._execute(request, body, sandbox)


class TestHarnessOutcomes:
    """Infrastructure failures are refused with a 5xx; the agent's own outcomes are verified."""

    @pytest.mark.asyncio
    async def test_model_unreachable_is_a_502(self, monkeypatch):
        from aiohttp import ClientConnectionError

        class DeadClient:
            async def create_response(self, **kwargs):
                raise ClientConnectionError("Cannot connect to host model:8000")

        async def run(agent, instruction, environment, context):
            # The real adapter translates the transport failure; Harbor would have retried on top.
            agent.kwargs["llm"]._client = DeadClient()
            await agent.kwargs["llm"].call("prompt")

        with pytest.raises(app_module.Terminus2InfrastructureError) as excinfo:
            await _execute_with_harness(monkeypatch, run)
        assert excinfo.value.status_code == 502
        assert "model unreachable" in excinfo.value.detail
        assert "Cannot connect to host" in excinfo.value.detail
        assert excinfo.value.metrics["terminus2_completed"] is False
        assert "ModelUnreachableError: ClientConnectionError" in excinfo.value.metrics["error"]

    @pytest.mark.asyncio
    async def test_sandbox_lost_is_a_503(self, monkeypatch):
        async def sandbox_exec(command, **kwargs):
            if command == "tmux send-keys":
                raise ConnectionResetError("sandbox connection dropped")
            return SimpleNamespace(stdout="", stderr="", return_code=0)

        async def run(agent, instruction, environment, context):
            await environment.exec("tmux send-keys")

        with pytest.raises(app_module.Terminus2InfrastructureError) as excinfo:
            await _execute_with_harness(monkeypatch, run, sandbox_exec=sandbox_exec)
        assert excinfo.value.status_code == 503
        assert "sandbox lost" in excinfo.value.detail
        assert "ConnectionResetError: sandbox connection dropped" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_cancelled_run_is_a_503(self, monkeypatch):
        import asyncio

        async def run(agent, instruction, environment, context):
            raise asyncio.CancelledError

        with pytest.raises(app_module.Terminus2InfrastructureError) as excinfo:
            await _execute_with_harness(monkeypatch, run)
        assert excinfo.value.status_code == 503
        assert "cancelled" in excinfo.value.detail
        assert excinfo.value.metrics["terminus2_completed"] is False

    @pytest.mark.asyncio
    async def test_timeout_is_verified_with_metadata(self, monkeypatch):
        import asyncio

        async def run(agent, instruction, environment, context):
            await asyncio.sleep(5)

        _, _, (response, metrics) = await _execute_with_harness(
            monkeypatch, run, config=_harness_config(sandbox_timeout=0.01)
        )
        assert response.metadata["terminus2_outcome"] == "timeout"
        assert response.metadata["terminus2_completed"] == "false"
        assert response.metadata["terminus2_error_type"] == "TimeoutError"
        assert metrics["terminus2_completed"] is False

    @pytest.mark.asyncio
    async def test_step_limit_is_verified(self, monkeypatch):
        async def run(agent, instruction, environment, context):
            context.metadata = {"n_episodes": 1}

        _, _, (response, metrics) = await _execute_with_harness(monkeypatch, run, config=_harness_config(max_turns=1))
        assert response.metadata == {"terminus2_outcome": "step_limit", "terminus2_completed": "true"}
        assert metrics["terminus2_completed"] is True and metrics["error"] is None

    @pytest.mark.asyncio
    async def test_completed_run_is_verified(self, monkeypatch):
        async def run(agent, instruction, environment, context):
            context.metadata = {"n_episodes": 1}

        _, _, (response, _) = await _execute_with_harness(monkeypatch, run, config=_harness_config(max_turns=5))
        assert response.metadata == {"terminus2_outcome": "completed", "terminus2_completed": "true"}

    @pytest.mark.asyncio
    async def test_harness_bug_stays_a_verified_failure(self, monkeypatch):
        async def run(agent, instruction, environment, context):
            raise ValueError("parser blew up")

        _, _, (response, metrics) = await _execute_with_harness(monkeypatch, run)
        assert response.metadata["terminus2_outcome"] == "failed"
        assert response.metadata["terminus2_error_type"] == "ValueError"
        assert "parser blew up" in response.metadata["terminus2_error"]

    def test_responses_endpoint_returns_the_status(self, monkeypatch):
        from fastapi.testclient import TestClient

        agent = Terminus2Agent(config=_harness_config(), server_client=MagicMock(spec=ServerClient))

        async def failing_execute(request, body, sandbox):
            raise app_module.Terminus2InfrastructureError(502, detail="model unreachable: refused", metrics={})

        monkeypatch.setattr(agent, "_execute", failing_execute)
        monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
        monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, cfg: {"opensandbox": {}})
        monkeypatch.setattr(app_module, "create_provider", lambda config: MagicMock())
        from unittest.mock import AsyncMock

        monkeypatch.setattr(app_module.AsyncSandbox, "connect", AsyncMock(return_value=SimpleNamespace()))
        client = TestClient(agent.setup_webserver())
        assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 200
        response = client.post("/v1/responses", json={"input": [{"role": "user", "content": "do it"}]})
        assert response.status_code == 502
        assert response.json() == {"detail": "model unreachable: refused"}

    @pytest.mark.asyncio
    async def test_legacy_run_stops_the_sandbox_on_infrastructure_failure(self, monkeypatch):
        from unittest.mock import AsyncMock

        from responses_api_agents.terminus_2_sandboxed_agent.app import Terminus2AgentRunRequest

        agent = Terminus2Agent(config=_harness_config(), server_client=MagicMock(spec=ServerClient))
        sandbox = SimpleNamespace(stop=AsyncMock())
        monkeypatch.setattr(agent, "_connect_sandbox", AsyncMock(return_value=sandbox))
        seed = SimpleNamespace(ok=True, status=200, cookies={}, json=AsyncMock(return_value={"sandbox_handle": "sb"}))
        agent.server_client.post = AsyncMock(return_value=seed)

        async def failing_execute(request, body, used_sandbox):
            raise app_module.Terminus2InfrastructureError(503, detail="sandbox lost: gone", metrics={})

        monkeypatch.setattr(agent, "_execute", failing_execute)
        request = SimpleNamespace(cookies={}, session={app_module.SESSION_ID_KEY: "session-1"})
        body = Terminus2AgentRunRequest(responses_create_params={"input": "solve this"})
        with pytest.raises(app_module.Terminus2InfrastructureError):
            await agent.run(request, body)
        sandbox.stop.assert_awaited_once()
        assert "session-1" not in agent._session_sandboxes
        # Verification never ran: only the seed call reached the resources server.
        assert agent.server_client.post.await_count == 1


def _response_error(status, message):
    from aiohttp import ClientResponseError

    request_info = SimpleNamespace(real_url="http://model/v1/responses")
    return ClientResponseError(request_info, (), status=status, message=message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc,expected",
    [
        (lambda: _response_error(status=401, message="Unauthorized"), "model"),
        (lambda: _response_error(status=503, message="Unavailable"), "model"),
        (lambda: __import__("aiohttp").ClientPayloadError("dropped mid-body"), "model"),
        (lambda: _response_error(status=400, message="Bad Request"), "raw"),
        (lambda: ValueError("not a transport failure"), "raw"),
    ],
)
async def test_nemo_gym_llm_classifies_endpoint_failures(exc, expected):
    error = exc()

    class Client:
        async def create_response(self, **kwargs):
            raise error

    llm = NeMoGymLLM(
        client=Client(),
        model_name="policy_model",
        model_context_limit=1000,
        model_output_limit=None,
        llm_request_timeout=1,
    )
    if expected == "model":
        with pytest.raises(app_module.ModelUnreachableError) as excinfo:
            await llm.call("prompt")
        assert excinfo.value.__cause__ is error
    else:
        with pytest.raises(type(error)):
            await llm.call("prompt")


@pytest.mark.asyncio
async def test_nemo_gym_llm_exhausted_timeouts_mean_the_model_is_unreachable():
    class Client:
        async def create_response(self, **kwargs):
            raise TimeoutError

    llm = NeMoGymLLM(
        client=Client(),
        model_name="policy_model",
        model_context_limit=1000,
        model_output_limit=None,
        llm_request_timeout=1,
    )
    with pytest.raises(app_module.ModelUnreachableError, match="after 10 attempts"):
        await llm.call("prompt")
    assert llm._model_calls_gt_10min == 10


class TestAgentTimeout:
    """The task's ``[agent].timeout_sec`` bounds the configured budget."""

    def _agent(self, sandbox_timeout=10800) -> Terminus2Agent:
        return Terminus2Agent(
            config=_harness_config(sandbox_timeout=sandbox_timeout), server_client=MagicMock(spec=ServerClient)
        )

    def test_task_timeout_bounds_the_config_budget(self, caplog):
        body = NeMoGymResponseCreateParamsNonStreaming(input="x", metadata={"agent_timeout_sec": "900"})
        with caplog.at_level(logging.DEBUG, logger=app_module.__name__):
            assert self._agent()._agent_timeout_sec(body) == 900.0
        assert "Terminus 2 agent timeout: 900.0s" in caplog.text

    def test_config_budget_wins_when_shorter(self):
        body = NeMoGymResponseCreateParamsNonStreaming(input="x", metadata={"agent_timeout_sec": "900"})
        assert self._agent(sandbox_timeout=600)._agent_timeout_sec(body) == 600.0

    @pytest.mark.parametrize(
        "metadata", [None, {}, {"other": "1"}, {"agent_timeout_sec": "0"}, {"agent_timeout_sec": "-5"}]
    )
    def test_missing_or_non_positive_task_timeout_keeps_the_config(self, metadata):
        body = NeMoGymResponseCreateParamsNonStreaming(input="x", metadata=metadata)
        assert self._agent()._agent_timeout_sec(body) == 10800.0

    def test_non_numeric_task_timeout_is_ignored_with_a_warning(self, caplog):
        body = NeMoGymResponseCreateParamsNonStreaming(input="x", metadata={"agent_timeout_sec": "soon"})
        with caplog.at_level(logging.WARNING, logger=app_module.__name__):
            assert self._agent()._agent_timeout_sec(body) == 10800.0
        assert "Ignoring non-numeric agent_timeout_sec='soon'" in caplog.text

    @pytest.mark.asyncio
    async def test_execute_runs_under_the_task_budget(self, monkeypatch):
        import asyncio

        async def run(agent, instruction, environment, context):
            await asyncio.sleep(5)

        # Config allows 10 s; the task declares 0.01 s, so the run times out under the task's budget.
        _, _, (response, _) = await _execute_with_harness(monkeypatch, run, metadata={"agent_timeout_sec": "0.01"})
        assert response.metadata["terminus2_outcome"] == "timeout"
