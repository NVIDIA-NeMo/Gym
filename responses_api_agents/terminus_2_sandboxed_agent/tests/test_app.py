# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
        [{"id": "", "summary": [{"text": reasoning_content, "type": "summary_text"}], "type": "reasoning"}]
        if reasoning_content
        else []
    )
    assert client.requests == [
        {"model": "policy_model", "input": [{"content": "first", "role": "user", "type": "message"}]},
        {
            "model": "policy_model",
            "input": [
                {"content": "first", "role": "user", "type": "message"},
                *expected_reasoning,
                {
                    "id": "",
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
