# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_sandboxed_agent import app as app_module
from responses_api_agents.terminus_2_sandboxed_agent.app import (
    NeMoGymSandboxEnvironment,
    Terminus2Agent,
    Terminus2AgentConfig,
    _instruction,
)


def test_instruction_joins_text_content():
    assert _instruction([{"content": [{"text": "first"}]}, {"content": "second"}]) == "first\n\nsecond"


@pytest.mark.asyncio
async def test_sandbox_environment_adapts_exec_and_is_dir():
    sandbox = SimpleNamespace()
    calls = []

    async def exec(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="output", stderr=None, return_code=0)

    sandbox.exec = exec
    environment = NeMoGymSandboxEnvironment(sandbox, logs_dir=SimpleNamespace())

    result = await environment.exec("pwd", timeout_sec=12, user="root", cwd="/work", env={"A": "B"})

    assert result.stdout == "output"
    assert result.stderr == ""
    assert result.return_code == 0
    assert await environment.is_dir("/workspace")
    assert calls == [
        ("pwd", {"cwd": "/work", "env": {"A": "B"}, "timeout_s": 12, "user": "root"}),
        ('test -d "/workspace"', {"user": None}),
    ]


@pytest.mark.asyncio
async def test_sandbox_environment_uses_seeded_pty_for_stateful_commands():
    sandbox = SimpleNamespace(pty=SimpleNamespace())
    calls = []

    async def pty_exec(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="output", stderr=None, return_code=0)

    sandbox.pty.exec = pty_exec
    environment = NeMoGymSandboxEnvironment(sandbox, logs_dir=SimpleNamespace(), pty_session="seeded-pty")

    await environment.exec("tmux new-session")

    assert calls == [("tmux new-session", {"session": "seeded-pty", "timeout_s": None})]


def test_agent_implements_required_responses_endpoint():
    assert not getattr(Terminus2Agent, "__abstractmethods__", set())


@pytest.mark.asyncio
async def test_execute_runs_terminus_in_seeded_sandbox(monkeypatch):
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
        use_responses_api=True,
        tmux_pane_width=160,
        tmux_pane_height=40,
        sandbox_provider="opensandbox",
        sandbox_timeout=10,
    )
    server = Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox_calls = []

    async def sandbox_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec, pty=SimpleNamespace())

    class FakeTerminus:
        session = SimpleNamespace()

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._session = SimpleNamespace(stop=self.stop)

        async def stop(self):
            return None

        async def setup(self, environment):
            await environment.exec("tmux setup")

        async def run(self, instruction, environment, context):
            assert instruction == "solve this"
            await environment.exec("tmux run")
            context.n_input_tokens = 4
            context.n_output_tokens = 3
            context.metadata = {"all_messages": [{"role": "assistant", "content": "done"}]}

    class FakeContext:
        n_input_tokens = None
        n_cache_tokens = None
        n_output_tokens = None
        metadata = None

    async def pty_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    sandbox.pty.exec = pty_exec
    monkeypatch.setattr(app_module, "Terminus2", FakeTerminus)
    monkeypatch.setattr(app_module, "AgentContext", FakeContext)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")

    async def request_json():
        return {"task_id": "task"}

    request = SimpleNamespace(json=request_json)
    response = await server._execute(
        request,
        NeMoGymResponseCreateParamsNonStreaming(input="solve this"),
        sandbox,
        "seeded-pty",
    )

    assert response.output[0].content[0].text == "done"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 3
    assert sandbox_calls == [
        ("mkdir -p /logs/agent", {"cwd": None, "env": None, "timeout_s": None, "user": "root"}),
        ("tmux setup", {"session": "seeded-pty", "timeout_s": None}),
        ("tmux run", {"session": "seeded-pty", "timeout_s": None}),
    ]
