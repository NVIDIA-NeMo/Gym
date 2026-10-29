# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import socket
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import nemo_gym.client.http_client
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.client import GlobalAIOHTTPAsyncClientConfig, ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    EpisodeId,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.trajectory import TrajectoryRecord, TrajectoryTurn
from responses_api_agents.terminus_2_sandboxed_agent import app as app_module
from responses_api_agents.terminus_2_sandboxed_agent.app import (
    NeMoGymLLM,
    NeMoGymSandboxEnvironment,
    NeMoGymTerminus2,
    Terminus2Agent,
    Terminus2AgentConfig,
    Terminus2AgentRunRequest,
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
    # This turn confirmed completion, but the shell reset below withdraws the pending claim, as in Harbor.
    agent._last_turn_claimed_completion = True
    agent._completed_by_agent = False
    commands = [SimpleNamespace(keystrokes=key, duration_sec=0) for key in ("first", "second", "third")]
    result = await agent._execute_commands(commands, SimpleNamespace(send_keys=send_keys, recover_shell=recover_shell))
    assert sent == ["first", "second"]
    assert result == (False, "The shell exited. Its state has reset; remaining commands were skipped.")
    assert agent._pending_completion is False
    assert agent._completed_command_batches == 1
    assert agent._completed_by_agent is False


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
async def test_sandbox_environment_raises_on_structured_sandbox_failures_only():
    """A provider may report a lost sandbox as a result rather than an exception; a command timeout is not one."""
    results = {
        "lost": SimpleNamespace(stdout=None, stderr="gone", return_code=125, error_type="sandbox"),
        "slow": SimpleNamespace(stdout="partial", stderr="", return_code=124, error_type="timeout"),
        "failed": SimpleNamespace(stdout="", stderr="no such file", return_code=1, error_type=None),
    }

    async def sandbox_exec(command, **kwargs):
        return results[command]

    environment = NeMoGymSandboxEnvironment(SimpleNamespace(exec=sandbox_exec), logs_dir=None, session_id="s")
    with pytest.raises(app_module.SandboxLostError, match=r"sandbox exec failed: sandbox \(exit 125\)"):
        await environment.exec("lost")
    assert (await environment.exec("slow")).return_code == 124
    assert (await environment.exec("failed")).return_code == 1


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
@pytest.mark.parametrize("terminal_hidden_mounts", [[], ["/mnt/s3-data", "/mnt/.s3-gate"]])
async def test_execute_runs_terminus_in_seeded_sandbox(
    monkeypatch, dump_trajectory, debug, interleaved_thinking, terminal_hidden_mounts
):
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
        terminal_hidden_mounts=terminal_hidden_mounts,
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
            assert self.kwargs["terminal_hidden_mounts"] == terminal_hidden_mounts
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
    assert response.metadata == {"terminus2_completed": "true", "terminus2_outcome": "completed"}
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


def _run_fixture(monkeypatch, *, failure: str | None):
    """Build an agent whose run() uses fakes; `failure` is "setup", "verify" or None."""
    config = Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_1_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="terminal_bench_2_1_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_turns=100,
        enable_summarize=True,
        proactive_summarization_threshold=8000,
        tmux_pane_width=160,
        tmux_pane_height=40,
        dump_trajectory=False,
        debug=False,
        model_context_limit=32_000,
        model_output_limit=4_000,
        interleaved_thinking=True,
        terminal_hidden_mounts=["/mnt/s3-data", "/mnt/.s3-gate"],
        llm_request_timeout=60,
        sandbox_provider="opensandbox",
        sandbox_timeout=10,
        remote_tmux_binary_path=None,
    )
    monkeypatch.setattr(app_module.harbor_logger, "setLevel", MagicMock())
    server_client = MagicMock(spec=ServerClient)
    seed_response = SimpleNamespace(cookies={}, json=AsyncMock(return_value={"sandbox_handle": "sandbox-1"}))
    server_client.post = AsyncMock(side_effect=[seed_response, SimpleNamespace(cookies={})])
    server = Terminus2Agent(config=config, server_client=server_client)

    async def sandbox_exec(command, **kwargs):
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec, stop=AsyncMock())

    class FakeTerminus:
        def __init__(self, **kwargs):
            self._times_spent = []
            self._num_proactive_compactions = 0

        async def setup(self, environment):
            if failure == "setup":
                raise RuntimeError("Private terminal mount bootstrap failed: Private terminal requires tmux")

        async def run(self, instruction, environment, context):
            context.n_input_tokens = 1
            context.n_output_tokens = 1

    class FakeContext:
        n_input_tokens = None
        n_cache_tokens = None
        n_output_tokens = None

    verify_failure = RuntimeError("verifier unavailable")
    monkeypatch.setattr(
        app_module, "raise_for_status", AsyncMock(side_effect=[None, verify_failure] if failure == "verify" else None)
    )
    monkeypatch.setattr(app_module, "NeMoGymTerminus2", FakeTerminus)
    monkeypatch.setattr(app_module, "AgentContext", FakeContext)
    monkeypatch.setattr(app_module, "get_response_json", AsyncMock(return_value={"reward": 1.0}))
    monkeypatch.setattr(app_module.Terminus2AgentVerifyResponse, "model_validate", lambda value: value)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(Terminus2Agent, "_connect_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")

    async def request_json():
        return {"task_id": "task"}

    request = SimpleNamespace(cookies={}, json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    body = Terminus2AgentRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="solve"))
    return server, server_client, sandbox, request, body


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["setup", "verify"])
async def test_run_releases_sandbox_and_session_reference_when_a_step_before_grading_fails(monkeypatch, failure):
    server, server_client, sandbox, request, body = _run_fixture(monkeypatch, failure=failure)

    expected = "Private terminal requires tmux" if failure == "setup" else "verifier unavailable"
    with pytest.raises(RuntimeError, match=expected):
        await server.run(request, body)

    sandbox.stop.assert_awaited_once()
    assert server._session_sandboxes == {}
    called = [call.kwargs["url_path"] for call in server_client.post.await_args_list]
    assert called == (["/seed_session"] if failure == "setup" else ["/seed_session", "/verify"])


@pytest.mark.asyncio
async def test_run_stops_sandbox_once_and_returns_merged_result_on_success(monkeypatch):
    server, _, sandbox, request, body = _run_fixture(monkeypatch, failure=None)

    result = await server.run(request, body)

    sandbox.stop.assert_awaited_once()
    assert server._session_sandboxes == {}
    assert result["reward"] == 1.0
    assert result["terminus2_completed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_params, forwarded",
    [
        pytest.param(
            {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 2048},
            {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 2048},
            id="task_sets_sampling",
        ),
        pytest.param({"top_p": None}, {}, id="task_leaves_sampling_unset"),
    ],
)
async def test_execute_forwards_task_sampling_params_to_every_model_call(monkeypatch, task_params, forwarded):
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
        model_context_limit=32_000,
        model_output_limit=4_000,
        interleaved_thinking=False,
        llm_request_timeout=60,
        sandbox_provider="opensandbox",
        sandbox_timeout=10,
        remote_tmux_binary_path=None,
    )
    server = Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))
    requests = []

    class FakeModelClient:
        def __init__(self, **kwargs):
            pass

        async def create_response(self, **kwargs):
            requests.append(kwargs)
            return NeMoGymResponse(
                id=f"resp_{len(requests)}",
                created_at=0,
                model="policy_model",
                object="response",
                output=[
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{len(requests)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text="answer", annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=True,
            )

    class FakeTerminus:
        def __init__(self, **kwargs):
            self.llm = kwargs["llm"]
            self._times_spent = []
            self._num_proactive_compactions = 0

        async def setup(self, environment):
            return None

        async def run(self, instruction, environment, context):
            await self.llm.call(instruction)
            # Harbor's summarization subagents call the same LLM with their own prompt and history.
            await self.llm.call("summarize", message_history=[{"role": "user", "content": instruction}])

    async def sandbox_exec(command, **kwargs):
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(app_module, "NeMoGymAsyncOpenAI", FakeModelClient)
    monkeypatch.setattr(app_module, "NeMoGymTerminus2", FakeTerminus)
    monkeypatch.setattr(
        app_module,
        "AgentContext",
        lambda: SimpleNamespace(n_input_tokens=None, n_cache_tokens=None, n_output_tokens=None),
    )
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")

    async def request_json():
        return {"task_id": "task"}

    request = SimpleNamespace(json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    _, metrics = await server._execute(
        request,
        NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "solve this", **task_params}),
        SimpleNamespace(exec=sandbox_exec),
    )

    assert metrics["error"] is None
    task_message = {"content": "solve this", "role": "user", "type": "message"}
    assert requests == [
        {"model": "policy_model", "input": [task_message], **forwarded},
        {
            "model": "policy_model",
            "input": [task_message, {"content": "summarize", "role": "user", "type": "message"}],
            **forwarded,
        },
    ]


def _native_config() -> Terminus2AgentConfig:
    return Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_sandboxed_agent",
        # Episode sessions borrow the environment server's sandbox; no resources server is involved.
        resources_server=None,
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
        # Terminus 2 never sees a task's MCP servers or skills: the environment server refuses such tasks.
        assert response.json() == {"agent_session_id": "ag-1", "supports_task_tool_context": False}
        connect.assert_awaited_once()
        assert connect.await_args.args[0] == {"sandbox_id": "sb-1"}
        state = agent._session_records["ag-1"].state
        assert state.sandbox is sandbox and state.task is None and state.metrics is None
        # Idempotent re-seed, and a different episode cannot take the id.
        assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 200
        assert connect.await_count == 1
        assert client.post("/v1/agent_sessions", json=_seed_body(rollout="other")).status_code == 409

    def test_multiple_workers_are_refused_at_startup(self):
        # Session state is process-local: a hand-written config fails at `gym env start`, not per episode.
        config = _native_config()
        config.num_workers = 4
        with pytest.raises(ValueError, match="num_workers=1"):
            Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))
        config.num_workers = 1
        Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_benchmark_composition_keeps_its_legacy_workers(self):
        """The shipped template (num_workers: 4) composed onto a benchmark still starts: /run is per request."""
        from pathlib import Path

        from omegaconf import DictConfig, OmegaConf

        from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig

        root = Path(__file__).parents[3]
        global_config = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.merge(
                    GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                    DictConfig({"config_paths": [str(root / "benchmarks/terminal_bench_2_1/terminus_2.yaml")]}),
                ),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
        composed = global_config["terminal_bench_2_1_terminus_2_sandboxed_agent"]["responses_api_agents"][
            "terminus_2_sandboxed_agent"
        ]
        config = Terminus2AgentConfig.model_validate(
            OmegaConf.to_container(composed, resolve=True)
            | {"name": "terminal_bench_2_1_terminus_2_sandboxed_agent", "host": "0.0.0.0", "port": 8080}
        )
        assert config.num_workers == 4
        assert config.resources_server.name == "terminal_bench_2_1_terminus_2_sandboxed_resources_server"
        Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))
        # The same block without a resources server is session-only, and four workers are refused.
        config.resources_server = None
        with pytest.raises(ValueError, match="num_workers=1"):
            Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_close_carries_the_runs_observations(self, monkeypatch):
        """seed -> responses -> close: the harness bundle (invocation record, compactions) rides on the close."""
        agent, client, sandbox, _ = self._client(monkeypatch)
        sandbox.exec = AsyncMock(return_value=SimpleNamespace(stdout="", stderr="", return_code=0))

        class FakeTerminus:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self._times_spent = []
                self._num_proactive_compactions = 0

            async def setup(self, environment):
                return None

            async def run(self, instruction, environment, context):
                observations = self.kwargs["llm"].observations
                observations.compactions.append(
                    app_module.ContextCompactionObservation(
                        invocation_id=observations.invocation_id, observed_at=1.0, trigger="harbor_summarize"
                    )
                )
                # One decision with two command batches behind it: what the collector cannot rebuild from calls.
                observations.trajectory.turns.append(
                    TrajectoryTurn(
                        invocation_id=observations.invocation_id,
                        task_id=observations.trajectory.task_id,
                        rollout_id=observations.trajectory.rollout_id,
                        turn_no=1,
                        timestamp=1.0,
                        question=instruction,
                        answer=[],
                        step_count=2,
                    )
                )

        monkeypatch.setattr(app_module, "NeMoGymTerminus2", FakeTerminus)
        monkeypatch.setattr(Terminus2Agent, "resolve_model_base_url", lambda *_args, **_kwargs: "http://model/v1")
        monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
        assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 200

        response = client.post("/ng-rollout/r1/v1/responses", json={"input": [{"role": "user", "content": "do it"}]})
        assert response.status_code == 200, response.text
        assert response.json()["metadata"]["terminus2_outcome"] == "completed"

        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
        )
        assert close.status_code == 200, close.text
        observations = close.json()["agent_observations"]
        assert observations["source"] == "terminus2"
        kinds = [record["kind"] for record in observations["records"]]
        assert kinds == ["agent_invocation", "context_compaction"]
        assert observations["records"][0]["invocation_id"].startswith("terminus2_")
        assert observations["records"][0]["status"] == "completed"
        # The run's own decisions ride on the close too, next to the bundle: the environment server forwards
        # them so rollout collection does not rebuild turns from every captured model call.
        trajectory = close.json()["trajectory"]
        assert trajectory["rollout_id"] == "r1"
        # The turn body carries no task identity; the trajectory names the seeded task the way the collector
        # names a materialized row (``str(row["task_id"])``), not the rollout id.
        assert trajectory["task_id"] == str(_seed_body()["task_id"]) == "{'taskset': 'tb', 'task_id': 'path-tracing'}"
        [turn] = trajectory["turns"]
        assert turn["task_id"] == trajectory["task_id"]
        assert turn["invocation_id"] == observations["records"][0]["invocation_id"]
        assert (turn["turn_no"], turn["step_count"]) == (1, 2)

    def test_close_carries_observations_of_a_refused_run(self, monkeypatch):
        agent, client, _, _ = self._client(monkeypatch)
        bundle = app_module.AgentObservationBundle(
            source="terminus2",
            records=[
                app_module.AgentInvocation(
                    invocation_id="terminus2_x", status="failed", duration_ms=1.0, error_type="SandboxLostError"
                )
            ],
        )

        trajectory = TrajectoryRecord(task_id="path-tracing", rollout_id="r1")
        identities = []

        async def failing_execute(request, body, sandbox, task_id=None):
            identities.append(task_id)
            raise app_module.Terminus2InfrastructureError(
                503,
                detail="sandbox lost: gone",
                metrics={"ng_agent_observations": bundle, "ng_trajectory": trajectory},
            )

        monkeypatch.setattr(agent, "_execute", failing_execute)
        assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 200
        turn = client.post("/v1/responses", json={"input": [{"role": "user", "content": "do it"}]})
        assert turn.status_code == 503
        assert turn.json() == {"detail": "sandbox lost: gone"}  # the body carries only the detail, never the metrics
        # A retry of the same request replays the refusal; it does not start a second run.
        assert client.post("/v1/responses", json={"input": [{"role": "user", "content": "do it"}]}).status_code == 503
        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
        )
        assert close.status_code == 200
        assert close.json()["agent_observations"]["records"][0]["invocation_id"] == "terminus2_x"
        assert close.json()["trajectory"] == trajectory.model_dump(mode="json")
        # The one run was handed the seeded task's collector identity.
        assert identities == [str(_seed_body()["task_id"])]

    def test_seed_requires_sandbox_access(self, monkeypatch):
        _, client, _, _ = self._client(monkeypatch)
        assert client.post("/v1/agent_sessions", json=_seed_body(with_sandbox=False)).status_code == 422

    def test_responses_uses_the_session_sandbox(self, monkeypatch):
        from unittest.mock import AsyncMock

        agent, client, sandbox, _ = self._client(monkeypatch)
        assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 200
        seen = {}

        async def fake_execute(request, body, used_sandbox, task_id=None):
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


class TestSessionActivation:
    """One run per session: identical retries join it, a different body is refused, close cancels it."""

    async def _seeded(self, monkeypatch, execute):
        agent = Terminus2Agent(config=_native_config(), server_client=MagicMock(spec=ServerClient))
        order: list[str] = []

        async def disconnect():
            order.append("disconnect")

        sandbox = SimpleNamespace(disconnect=disconnect)
        monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
        monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, cfg: {"opensandbox": {}})
        monkeypatch.setattr(app_module, "create_provider", lambda config: MagicMock())
        monkeypatch.setattr(app_module.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
        monkeypatch.setattr(agent, "_execute", execute)
        request = SimpleNamespace(session={}, path_params={"rollout_id": "r1"})
        await agent.seed_agent_session(request, AgentSeedSessionRequest.model_validate(_seed_body()))
        return agent, request, order

    @staticmethod
    def _response(text: str) -> app_module.NeMoGymResponse:
        return app_module.NeMoGymResponse(
            id=f"resp_{text}",
            created_at=0,
            model="m",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )

    @pytest.mark.asyncio
    async def test_identical_retries_join_one_run(self, monkeypatch):
        started, release, runs = asyncio.Event(), asyncio.Event(), []

        async def slow_execute(request, body, sandbox, task_id=None):
            runs.append(body.input)
            started.set()
            await release.wait()
            return self._response("one"), {"ng_agent_observations": None}

        agent, request, _ = await self._seeded(monkeypatch, slow_execute)
        body = NeMoGymResponseCreateParamsNonStreaming(input="do it")
        first = asyncio.create_task(agent.responses(request, body))
        await asyncio.wait_for(started.wait(), 5)
        # The client resends the same POST after a lost reply: it joins the run already in flight.
        second = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="do it")))
        await asyncio.sleep(0)
        with pytest.raises(HTTPException) as excinfo:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="something else"))
        assert excinfo.value.status_code == 409
        release.set()
        responses = await asyncio.gather(first, second)
        assert [response.id for response in responses] == ["resp_one", "resp_one"]
        assert runs == ["do it"]
        # After the run, the same request replays its result without starting another.
        assert (await agent.responses(request, body)).id == "resp_one" and runs == ["do it"]

    @pytest.mark.asyncio
    async def test_close_cancels_the_active_run_before_releasing_the_sandbox(self, monkeypatch):
        started = asyncio.Event()
        order_ref: list[list[str]] = []

        async def hanging_execute(request, body, sandbox, task_id=None):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                order_ref[0].append("cancelled")
                raise

        agent, request, order = await self._seeded(monkeypatch, hanging_execute)
        order_ref.append(order)
        activation = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="x")))
        await asyncio.wait_for(started.wait(), 5)
        close = await asyncio.wait_for(
            agent.close_agent_session(
                request, AgentCloseSessionRequest(agent_session_id="ag-1", episode_id=EpisodeId(rollout_id="r1"))
            ),
            5,
        )
        assert close.agent_session_id == "ag-1" and close.agent_observations is None
        assert order == ["cancelled", "disconnect"]
        with pytest.raises(asyncio.CancelledError):
            await activation
        assert agent._session_records["ag-1"].state is None


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
    async def test_model_client_retries_are_bounded(self, monkeypatch):
        """An internal client would retry refused connections forever and extend its HTTP budget on every 5xx."""
        clients = []

        async def run(agent, instruction, environment, context):
            clients.append(agent.kwargs["llm"]._client)

        await _execute_with_harness(monkeypatch, run)

        (client,) = clients
        assert isinstance(client, app_module.NeMoGymAsyncOpenAI)
        assert client.internal is False
        assert client.max_connection_retries == app_module.MODEL_CONNECTION_RETRIES == 20
        assert client.max_http_attempts == app_module.MODEL_HTTP_ATTEMPTS == 20
        assert client.max_num_tries is None

    @pytest.mark.asyncio
    async def test_refused_connections_surface_within_the_bound_not_as_a_timeout(self, monkeypatch):
        """Through the real client against a closed port: the outage reaches the handler well inside the task budget."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]
        monkeypatch.setattr(app_module, "MODEL_CONNECTION_RETRIES", 2)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
        monkeypatch.setattr("nemo_gym.utils.server_process.get_nemo_gym_fastapi_num_workers", lambda: 1)
        monkeypatch.setattr(nemo_gym.client.http_client, "is_nemo_gym_fastapi_worker", lambda: True)
        http_client = nemo_gym.client.http_client.set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())

        async def run(agent, instruction, environment, context):
            llm = agent.kwargs["llm"]
            llm._client.base_url = f"http://127.0.0.1:{closed_port}/v1"
            await llm.call("prompt")

        started = perf_counter()
        try:
            with pytest.raises(app_module.Terminus2InfrastructureError) as excinfo:
                # A 30 s task budget: unbounded retries would run into it and report the agent's own "timeout".
                await _execute_with_harness(monkeypatch, run, config=_harness_config(sandbox_timeout=30))
        finally:
            await http_client.close()
            monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
        elapsed = perf_counter() - started

        # Two connection retries half a second apart, then ModelUnreachableError: a 502, not a verified timeout.
        assert elapsed < 10, elapsed
        assert excinfo.value.status_code == 502
        assert "model unreachable" in excinfo.value.detail
        assert "ClientConnectorError" in excinfo.value.metrics["error"]
        assert "TimeoutError" not in excinfo.value.metrics["error"]

    @pytest.mark.asyncio
    async def test_structured_sandbox_failure_is_a_503_before_any_model_call(self, monkeypatch):
        """OpenSandbox reports a dead sandbox as `error_type="sandbox"` on the tmux liveness check."""
        model_calls = []

        async def sandbox_exec(command, **kwargs):
            if command == "tmux has-session":
                return SimpleNamespace(stdout=None, stderr="", return_code=125, error_type="sandbox")
            return SimpleNamespace(stdout="", stderr="", return_code=0)

        async def run(agent, instruction, environment, context):
            await environment.exec("tmux has-session")
            model_calls.append(await agent.kwargs["llm"].call("prompt"))

        with pytest.raises(app_module.Terminus2InfrastructureError) as excinfo:
            await _execute_with_harness(monkeypatch, run, sandbox_exec=sandbox_exec)
        assert excinfo.value.status_code == 503
        assert "sandbox lost" in excinfo.value.detail
        assert "sandbox exec failed: sandbox (exit 125)" in excinfo.value.detail
        assert model_calls == [] and excinfo.value.metrics["model_call_times"] == []

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
    async def test_completion_on_the_last_allowed_turn_is_not_a_step_limit(self, monkeypatch):
        async def run(agent, instruction, environment, context):
            context.metadata = {"n_episodes": 1}
            agent._completed_by_agent = True

        _, _, (response, _) = await _execute_with_harness(monkeypatch, run, config=_harness_config(max_turns=1))
        assert response.metadata == {"terminus2_outcome": "completed", "terminus2_completed": "true"}

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

    @pytest.mark.asyncio
    async def test_legacy_run_stops_the_sandbox_on_infrastructure_failure(self, monkeypatch):
        from unittest.mock import AsyncMock

        from responses_api_agents.terminus_2_sandboxed_agent.app import Terminus2AgentRunRequest

        # The legacy /run path is the one place that still seeds and verifies through a resources server.
        legacy = _harness_config(
            resources_server=ResourcesServerRef(type="resources_servers", name="legacy_resources_server")
        )
        agent = Terminus2Agent(config=legacy, server_client=MagicMock(spec=ServerClient))
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
        (lambda: _response_error(status=502, message="Bad Gateway"), "model"),
        (lambda: _response_error(status=503, message="Unavailable"), "model"),
        (lambda: _response_error(status=504, message="Gateway Timeout"), "model"),
        (lambda: __import__("aiohttp").ClientPayloadError("dropped mid-body"), "model"),
        (lambda: _response_error(status=400, message="Bad Request"), "raw"),
        # The openai_model adapter answers 500 for an upstream 400: a malformed prompt is a visible harness failure.
        (lambda: _response_error(status=500, message="Internal Server Error"), "raw"),
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
