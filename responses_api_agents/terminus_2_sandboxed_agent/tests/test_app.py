# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

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
from nemo_gym.sandbox.agent_user import check_agent_user
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_sandboxed_agent import app as app_module
from responses_api_agents.terminus_2_sandboxed_agent.app import (
    NeMoGymLLM,
    NeMoGymSandboxEnvironment,
    Terminus2Agent,
    Terminus2AgentConfig,
    Terminus2AgentRunRequest,
    _instruction,
)


ROOT_EXEC_KWARGS = {"timeout_s": None, "cwd": None, "user": "root", "env": None}


def _exec_kwargs(user):
    return {"timeout_s": None, "cwd": None, "user": user, "env": None}


def _make_config(**overrides):
    kwargs = dict(
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
        dump_trajectory=False,
        debug=False,
        sandbox_provider="opensandbox",
        sandbox_timeout=10,
        remote_tmux_binary_path=None,
    )
    kwargs.update(overrides)
    return Terminus2AgentConfig(**kwargs)


def _identity_sandbox_exec(sandbox_calls, agent_result=None):
    """Fake AsyncSandbox.exec whose `id` outputs describe a root-default image with a uid/gid 1000 account."""

    async def sandbox_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        if command == "id -u":
            return SimpleNamespace(stdout="0\n", stderr="", return_code=0)
        if command == "id -u && id -g":
            return agent_result or SimpleNamespace(stdout="1000\n1000\n", stderr="", return_code=0)
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    return sandbox_exec


def test_instruction_joins_text_content():
    assert _instruction([{"content": [{"text": "first"}]}, {"content": "second"}]) == "first\n\nsecond"


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
async def test_sandbox_environment_resolves_default_user():
    sandbox_calls = []

    async def sandbox_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr=None, return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)
    environment = NeMoGymSandboxEnvironment(
        sandbox, logs_dir=SimpleNamespace(), session_id="session-1", default_user="agent"
    )

    assert environment.default_user == "agent"
    await environment.exec("tmux new-session")
    await environment.exec("mkdir -p /logs/agent", user="root")
    await environment.exec("id -u", user=0)
    assert await environment.is_dir("/workspace")
    assert await environment.is_dir("/tests", user="root")

    assert sandbox_calls == [
        ("tmux new-session", _exec_kwargs("agent")),
        ("mkdir -p /logs/agent", _exec_kwargs("root")),
        ("id -u", _exec_kwargs(0)),
        ('test -d "/workspace"', {"user": "agent"}),
        ('test -d "/tests"', {"user": "root"}),
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
async def test_nemo_gym_llm_records_every_responses_request_and_output():
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
    llm = NeMoGymLLM(client=client, model_name="policy_model", model_context_limit=32_000, model_output_limit=4_000)

    first = await llm.call("first")
    second = await llm.call(
        "second",
        message_history=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer 1"}],
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
    assert client.requests == [
        {"model": "policy_model", "input": [{"content": "first", "role": "user", "type": "message"}]},
        {
            "model": "policy_model",
            "input": [
                {"content": "first", "role": "user", "type": "message"},
                {"content": "answer 1", "role": "assistant", "type": "message"},
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
        "compacted summary",
        "third",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_user", [None, "agent", 1000, "root"])
@pytest.mark.parametrize("dump_trajectory", [False, True])
@pytest.mark.parametrize("debug", [False, True])
async def test_execute_runs_terminus_in_seeded_sandbox(monkeypatch, dump_trajectory, debug, agent_user):
    config = _make_config(dump_trajectory=dump_trajectory, debug=debug)
    set_level = MagicMock()
    monkeypatch.setattr(app_module.harbor_logger, "setLevel", set_level)
    server = Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox_calls = []
    sandbox = SimpleNamespace(exec=_identity_sandbox_exec(sandbox_calls))

    class FakeTerminus:
        session = SimpleNamespace()

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._session = SimpleNamespace(stop=self.stop)
            self._times_spent = [1.0, 3.0]
            self._num_compactions = 2

        async def stop(self):
            return None

        async def setup(self, environment):
            # Harbor's Terminus2.setup builds TmuxSession(..., user=environment.default_user).
            assert environment.default_user == agent_user
            await environment.exec("tmux setup", user=environment.default_user)

        async def run(self, instruction, environment, context):
            assert instruction == "solve this"
            assert self.kwargs["dump_trajectory"] is dump_trajectory
            # No user: exercises the default_user resolution inside NeMoGymSandboxEnvironment.exec.
            await environment.exec("tmux run")
            self.kwargs["llm"]._times_spent.extend([2.0, 4.0])
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
        agent_user=agent_user,
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
        "num_compactions": 2,
    }
    assert response.output[-1].content[0].text == "done"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 3
    if not debug:
        set_level.assert_called_once_with(logging.WARNING)
    else:
        set_level.assert_not_called()
    if agent_user in (None, "root"):
        # Image default (root on the supported images): no chmod, no identity check.
        expected_calls = [("mkdir -p /logs/agent", ROOT_EXEC_KWARGS)]
    else:
        expected_calls = [
            ("mkdir -p /logs/agent", ROOT_EXEC_KWARGS),
            ("chmod 777 /logs/agent", ROOT_EXEC_KWARGS),
            ("id -u", ROOT_EXEC_KWARGS),
            ("id -u && id -g", _exec_kwargs(agent_user)),
        ]
    expected_calls += [("tmux setup", _exec_kwargs(agent_user)), ("tmux run", _exec_kwargs(agent_user))]
    assert sandbox_calls == expected_calls


@pytest.mark.asyncio
async def test_execute_installs_remote_tmux_as_root(monkeypatch):
    server = Terminus2Agent(
        config=_make_config(remote_tmux_binary_path="/mnt/tmux-3.7c"), server_client=MagicMock(spec=ServerClient)
    )
    sandbox_calls = []
    sandbox = SimpleNamespace(exec=_identity_sandbox_exec(sandbox_calls))

    class FakeTerminus:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._times_spent = []
            self._num_compactions = 0

        async def setup(self, environment):
            await environment.exec("tmux setup", user=environment.default_user)

        async def run(self, instruction, environment, context):
            return None

    monkeypatch.setattr(app_module, "NeMoGymTerminus2", FakeTerminus)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    elapsed_times = iter([10.0, 20.0])
    monkeypatch.setattr(app_module, "perf_counter", lambda: next(elapsed_times))

    async def request_json():
        return {}

    request = SimpleNamespace(json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    await server._execute(request, NeMoGymResponseCreateParamsNonStreaming(input="solve"), sandbox, agent_user="agent")

    install_command, install_kwargs = sandbox_calls[2]
    assert install_command.startswith("mkdir -p /usr/local/bin")
    assert "cp /mnt/tmux-3.7c /usr/local/bin/tmux" in install_command
    assert install_kwargs == {"user": "root"}
    assert [(command, kwargs["user"]) for command, kwargs in sandbox_calls if command != install_command] == [
        ("mkdir -p /logs/agent", "root"),
        ("chmod 777 /logs/agent", "root"),
        ("id -u", "root"),
        ("id -u && id -g", "agent"),
        ("tmux setup", "agent"),
    ]


@pytest.mark.asyncio
async def test_execute_fails_closed_when_identity_check_fails(monkeypatch):
    server = Terminus2Agent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
    sandbox_calls = []
    failing_check = SimpleNamespace(stdout="", stderr="su: user agent does not exist", return_code=1)
    sandbox = SimpleNamespace(exec=_identity_sandbox_exec(sandbox_calls, agent_result=failing_check))
    setup_calls = []

    class FakeTerminus:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._times_spent = []
            self._num_compactions = 0

        async def setup(self, environment):
            setup_calls.append(environment)

        async def run(self, instruction, environment, context):
            raise AssertionError("run must not be reached")

    monkeypatch.setattr(app_module, "NeMoGymTerminus2", FakeTerminus)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")

    async def request_json():
        return {}

    request = SimpleNamespace(json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    with pytest.raises(RuntimeError, match="agent_user='agent' identity check failed") as excinfo:
        await server._execute(
            request, NeMoGymResponseCreateParamsNonStreaming(input="solve"), sandbox, agent_user="agent"
        )

    assert "su: user agent does not exist" in str(excinfo.value)
    assert setup_calls == []
    assert sandbox_calls == [
        ("mkdir -p /logs/agent", ROOT_EXEC_KWARGS),
        ("chmod 777 /logs/agent", ROOT_EXEC_KWARGS),
        ("id -u", ROOT_EXEC_KWARGS),
        ("id -u && id -g", _exec_kwargs("agent")),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_user", ["agent", 1000])
async def test_sandbox_environment_serves_as_check_agent_user_executor(agent_user):
    # The helper-level tables live in tests/unit_tests/test_sandbox_agent_user.py; this covers the adapter side:
    # the shared check drives NeMoGymSandboxEnvironment.exec with the exact user kwargs the sandbox receives.
    sandbox_calls = []
    environment = NeMoGymSandboxEnvironment(
        SimpleNamespace(exec=_identity_sandbox_exec(sandbox_calls)), logs_dir=SimpleNamespace(), session_id="s"
    )

    await check_agent_user(environment, agent_user)

    assert sandbox_calls == [("id -u", ROOT_EXEC_KWARGS), ("id -u && id -g", _exec_kwargs(agent_user))]


def test_agent_config_defaults_agent_user_to_none():
    assert _make_config().agent_user is None


NORMALIZED_AGENT_USERS = [
    ("1000", 1000),
    ("0", 0),
    ("agent", "agent"),
    ("root", "root"),
    (1000, 1000),
    (0, 0),
    (None, None),
    # isdecimal, not isdigit: a superscript digit is not a uid and must not make int() raise.
    ("\u00b2", "\u00b2"),
]
# Booleans (pydantic lax mode would coerce `true` to uid 1) and empty / option-like names (`su` would parse them
# as options; shlex.quote leaves "-m" unquoted).
REJECTED_AGENT_USERS = [True, False, "", "-m", "--login", "-"]
AGENT_USER_ERROR = "agent_user must be an account name, a uid, or null"


@pytest.mark.parametrize(("value", "expected"), NORMALIZED_AGENT_USERS)
def test_agent_config_normalizes_agent_user(value, expected):
    config = _make_config(agent_user=value)
    assert config.agent_user == expected
    assert type(config.agent_user) is type(expected)


@pytest.mark.parametrize("value", REJECTED_AGENT_USERS)
def test_agent_config_rejects_bools_and_option_like_agent_user(value):
    with pytest.raises(ValidationError, match=AGENT_USER_ERROR):
        _make_config(agent_user=value)


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload
        self.cookies = {}

    async def json(self):
        return self._payload


class _FakeServerClient:
    def __init__(self, seed_payload):
        self.seed_payload = seed_payload
        self.posts = []

    async def post(self, **kwargs):
        self.posts.append(kwargs)
        if kwargs["url_path"] == "/seed_session":
            return _FakeHTTPResponse(self.seed_payload)
        return _FakeHTTPResponse({"reward": 1.0})


def _run_harness(monkeypatch, config, seed_payload, execute=None, stop=None):
    """Wire a Terminus2Agent whose network and sandbox edges are recorded fakes."""
    server_client = _FakeServerClient(seed_payload)
    mock_server_client = MagicMock(spec=ServerClient)
    mock_server_client.post = server_client.post
    server = Terminus2Agent(config=config, server_client=mock_server_client)
    stop_calls = []
    execute_calls = []

    async def default_stop():
        stop_calls.append("stopped")

    sandbox = SimpleNamespace(stop=stop or default_stop, sandbox_id="sb-1")

    async def connect_sandbox(sandbox_id):
        assert sandbox_id == "sb-1"
        return sandbox

    async def default_execute(request, body, sandbox_arg, agent_user=None):
        assert sandbox_arg is sandbox
        execute_calls.append(agent_user)
        response = NeMoGymResponse(
            id="resp_1",
            created_at=0,
            model="policy_model",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )
        return response, {
            "terminus2_completed": True,
            "command_exec_times": [],
            "model_call_times": [],
            "average_command_exec_time": 0.0,
            "average_model_call_time": 0.0,
            "total_command_exec_time": 0.0,
            "total_model_call_time": 0.0,
            "command_exec_time_pct": 0.0,
            "model_call_time_pct": 0.0,
            "terminus2_time_taken": 1.0,
            "model_calls_gt_10min": 0,
            "num_compactions": 0,
        }

    async def fake_raise_for_status(_response):
        return None

    async def fake_get_response_json(response):
        payload = dict(response._payload)
        payload["responses_create_params"] = {"input": "solve this"}
        payload["response"] = {
            "id": "resp_1",
            "created_at": 0,
            "model": "policy_model",
            "object": "response",
            "output": [],
            "tool_choice": "auto",
            "tools": [],
            "parallel_tool_calls": True,
        }
        return payload

    monkeypatch.setattr(server, "_connect_sandbox", connect_sandbox)
    monkeypatch.setattr(server, "_execute", execute or default_execute)
    monkeypatch.setattr(app_module, "raise_for_status", fake_raise_for_status)
    monkeypatch.setattr(app_module, "get_response_json", fake_get_response_json)
    request = SimpleNamespace(cookies={}, session={app_module.SESSION_ID_KEY: "session-1"})
    body = Terminus2AgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="solve this"),
        task_name="hello-world",
    )
    return server, server_client, sandbox, request, body, execute_calls, stop_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_agent_user", "seed_payload", "expected"),
    [
        # Row wins over lane config.
        (None, {"sandbox_handle": "sb-1", "agent_user": "agent"}, "agent"),
        ("lane-user", {"sandbox_handle": "sb-1", "agent_user": "agent"}, "agent"),
        # Lane config used when the row omits it (or sends null).
        ("agent", {"sandbox_handle": "sb-1"}, "agent"),
        ("agent", {"sandbox_handle": "sb-1", "agent_user": None}, "agent"),
        # Neither -> image default.
        (None, {"sandbox_handle": "sb-1"}, None),
        # Explicit per-row image-default escape hatch beats a lane `agent`.
        ("agent", {"sandbox_handle": "sb-1", "agent_user": "root"}, "root"),
        ("agent", {"sandbox_handle": "sb-1", "agent_user": 0}, 0),
        # Digit strings echoed by a resources server are normalized like config values.
        (None, {"sandbox_handle": "sb-1", "agent_user": "1000"}, 1000),
    ],
)
async def test_run_resolves_agent_user_precedence_and_reuses_it_in_responses(
    monkeypatch, config_agent_user, seed_payload, expected
):
    config = _make_config(agent_user=config_agent_user)
    server, server_client, sandbox, request, body, execute_calls, stop_calls = _run_harness(
        monkeypatch, config, seed_payload
    )
    reused_agent_users = []
    original_execute = server._execute

    async def recording_execute(request_arg, body_arg, sandbox_arg, agent_user=None):
        # While run() is inside _execute, responses() must reuse the stored per-session identity.
        if body_arg.input == "solve this":
            response = await server.responses(request_arg, NeMoGymResponseCreateParamsNonStreaming(input="follow-up"))
            assert response.id == "resp_1"
            reused_agent_users.append(server._session_agent_users["session-1"])
        return await original_execute(request_arg, body_arg, sandbox_arg, agent_user=agent_user)

    monkeypatch.setattr(server, "_execute", recording_execute)

    result = await server.run(request, body)

    assert result.reward == 1.0
    assert result.terminus2_completed is True
    # Outer /run call plus the nested responses() reuse, both with the effective identity.
    assert execute_calls == [expected, expected]
    assert type(execute_calls[0]) is type(expected)
    assert reused_agent_users == [expected]
    assert [post["url_path"] for post in server_client.posts] == ["/seed_session", "/verify"]
    seed_post, verify_post = server_client.posts
    assert seed_post["json"] == {
        "responses_create_params": body.responses_create_params.model_dump(),
        "task_name": "hello-world",
    }
    assert verify_post["json"]["agent_user"] == expected
    assert verify_post["json"]["task_name"] == "hello-world"
    assert verify_post["json"]["response"]["id"] == "resp_1"
    assert verify_post["json"]["responses_create_params"] == body.responses_create_params.model_dump()
    assert stop_calls == ["stopped"]
    assert server._session_sandboxes == {}
    assert server._session_agent_users == {}


@pytest.mark.asyncio
async def test_responses_reuses_per_session_sandbox_and_agent_user(monkeypatch):
    server = Terminus2Agent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
    sandbox = SimpleNamespace()
    server._session_sandboxes["session-1"] = sandbox
    server._session_agent_users["session-1"] = "agent"
    execute_calls = []

    async def fake_execute(request, body, sandbox_arg, agent_user=None):
        execute_calls.append((body.input, sandbox_arg, agent_user))
        return "response", {}

    monkeypatch.setattr(server, "_execute", fake_execute)
    request = SimpleNamespace(session={app_module.SESSION_ID_KEY: "session-1"})

    assert await server.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="again")) == "response"
    assert execute_calls == [("again", sandbox, "agent")]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_fails", [False, True])
async def test_run_stops_sandbox_and_clears_session_when_execute_raises(monkeypatch, stop_fails, capsys):
    stop_calls = []

    async def stop():
        stop_calls.append("stopped")
        if stop_fails:
            raise ConnectionError("sandbox gateway unreachable")

    async def failing_execute(request, body, sandbox_arg, agent_user=None):
        raise RuntimeError("agent_user='agent' identity check failed")

    server, server_client, sandbox, request, body, _execute_calls, _ = _run_harness(
        monkeypatch,
        _make_config(agent_user="agent"),
        {"sandbox_handle": "sb-1"},
        execute=failing_execute,
        stop=stop,
    )

    with pytest.raises(RuntimeError, match="identity check failed"):
        await server.run(request, body)

    assert stop_calls == ["stopped"]
    assert server._session_sandboxes == {}
    assert server._session_agent_users == {}
    assert [post["url_path"] for post in server_client.posts] == ["/seed_session"]
    if stop_fails:
        assert "Failed to stop sandbox after error" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("echoed_agent_user", [True, "-m", ""])
async def test_run_stops_sandbox_when_echoed_agent_user_is_malformed(monkeypatch, echoed_agent_user):
    server, server_client, sandbox, request, body, execute_calls, stop_calls = _run_harness(
        monkeypatch, _make_config(agent_user="agent"), {"sandbox_handle": "sb-1", "agent_user": echoed_agent_user}
    )

    # The echo is normalized only after the sandbox is connected, inside the cleanup wrapper.
    with pytest.raises(ValueError, match=AGENT_USER_ERROR):
        await server.run(request, body)

    assert execute_calls == []
    assert [post["url_path"] for post in server_client.posts] == ["/seed_session"]
    assert stop_calls == ["stopped"]
    assert server._session_sandboxes == {}
    assert server._session_agent_users == {}


@pytest.mark.asyncio
async def test_run_stops_sandbox_when_verify_fails(monkeypatch):
    server, server_client, sandbox, request, body, execute_calls, stop_calls = _run_harness(
        monkeypatch, _make_config(), {"sandbox_handle": "sb-1", "agent_user": "agent"}
    )

    async def failing_raise_for_status(response):
        if response._payload == {"reward": 1.0}:
            raise ValueError("verify returned 500")

    monkeypatch.setattr(app_module, "raise_for_status", failing_raise_for_status)

    with pytest.raises(ValueError, match="verify returned 500"):
        await server.run(request, body)

    assert execute_calls == ["agent"]
    assert [post["url_path"] for post in server_client.posts] == ["/seed_session", "/verify"]
    assert stop_calls == ["stopped"]
    assert server._session_sandboxes == {}
    assert server._session_agent_users == {}
