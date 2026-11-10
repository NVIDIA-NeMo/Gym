# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in sandbox-session settings for running Claude Code as a harness started in the task container."""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from nemo_gym.agents import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.claude_code_agent import app
from responses_api_agents.claude_code_agent import sandbox as adapter
from responses_api_agents.claude_code_agent.system_dependencies import system_dependencies_command
from responses_api_agents.claude_code_agent.tests.test_sandbox import agent, result, seed


RESULT = json.dumps({"type": "result", "subtype": "success", "is_error": False, "usage": {"input_tokens": 1}})


async def launch(monkeypatch, instance, prompt="Fix the code"):
    """Run one sandbox activation with a stubbed execution; return the runner input and response."""
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(
            session_dir="/tmp/session",
            workdir="/app",
            cleanup={"timed_out": False, "return_code": 0, "cleanup_confirmed": True},
            artifacts=None,
        ),
    )
    execute = AsyncMock(return_value=adapter.ClaudeCodeArtifacts(RESULT))
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", execute)
    body = NeMoGymResponseCreateParamsNonStreaming(input=prompt)
    response = await instance._sandbox_response(state, body, prompt=prompt, system=None)
    return execute.await_args.args[0], response


@pytest.mark.asyncio
@pytest.mark.parametrize("on_stdin", [False, True])
async def test_instruction_reaches_the_cli_on_stdin_when_configured(monkeypatch, on_stdin):
    instance = agent(anthropic_base_url="https://model.example", instruction_on_stdin=on_stdin)
    payload, response = await launch(monkeypatch, instance, prompt="Fix the 'parser'")
    assert response.status == "completed"
    command = payload["command"]
    assert command[0] == "/runtime/claude"
    if on_stdin:
        # `claude -p` with no instruction argument reads it from stdin.
        assert "--" not in command
        assert not any("parser" in argument for argument in command)
        assert payload["prompt"] == "Fix the 'parser'"
    else:
        assert command[-2:] == ["--", "Fix the 'parser'"]
        assert "prompt" not in payload


def run_runner(tmp_path, prompt=None):
    """Stage a stub CLI through stage_activation and run it with the shared bash runner, as the sandbox does."""

    async def upload(source, target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    shutil.copyfile(HARNESS_RUNNER_SCRIPT, tmp_path / HARNESS_RUNNER_FILE)
    child = tmp_path / "child.py"
    child.write_text("import json,sys; print(json.dumps({'stdin': sys.stdin.read()}))")
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(session_dir=str(tmp_path), sandbox=SimpleNamespace(upload=upload)),
    )
    payload = {
        "command": [sys.executable, str(child)],
        "cwd": str(tmp_path),
        "env": {},
        "settings": {},
        "mcp_config": None,
        "skills_path": None,
    }
    if prompt is not None:
        payload["prompt"] = prompt
    command = asyncio.run(state.stage_activation(payload)).argv
    completed = subprocess.run(command, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    return json.loads((tmp_path / "events.jsonl").read_text())["stdin"]


def test_runner_passes_the_prompt_on_stdin(tmp_path):
    assert run_runner(tmp_path, prompt="line one\nline two\n") == "line one\nline two\n"


def test_runner_stdin_is_empty_without_a_prompt(tmp_path):
    assert run_runner(tmp_path) == ""
    assert not (tmp_path / "prompt.txt").exists()


def test_stdin_setting_is_off_by_default():
    assert (
        app.ClaudeCodeAgentConfig(name="claude", host="h", port=1, entrypoint="app.py").instruction_on_stdin is False
    )


@pytest.mark.asyncio
async def test_effort_output_cap_and_client_env_reach_the_sandbox_launch(monkeypatch):
    instance = agent(
        anthropic_base_url="https://model.example",
        reasoning_effort="max",
        max_output_tokens=128000,
        extra_env={"API_TIMEOUT_MS": 3600000, "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000"},
    )
    payload, _ = await launch(monkeypatch, instance)
    command = payload["command"]
    assert command[command.index("--effort") + 1] == "max"
    env = payload["env"]
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "128000"
    # YAML integers become strings, as the CLI's environment needs.
    assert (env["API_TIMEOUT_MS"], env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]) == ("3600000", "1000000")
    assert env["ANTHROPIC_BASE_URL"] == "https://model.example"


@pytest.mark.asyncio
async def test_no_effort_output_cap_or_client_env_by_default(monkeypatch):
    payload, _ = await launch(monkeypatch, agent(anthropic_base_url="https://model.example"))
    assert "--effort" not in payload["command"]
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in payload["env"]
    assert "API_TIMEOUT_MS" not in payload["env"]


def test_local_launch_uses_the_same_client_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "ensure_claude_code", lambda version: None)
    monkeypatch.setattr(app.Path, "home", lambda: tmp_path)
    captured = {}

    class Process:
        returncode = 0

        async def communicate(self):
            return RESULT.encode() + b"\n", b""

    async def spawn(*command, **kwargs):
        captured.update(command=list(command), env=kwargs["env"])
        return Process()

    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", spawn)
    instance = agent(reasoning_effort="high", max_output_tokens=64000, extra_env={"API_TIMEOUT_MS": "60000"})
    asyncio.run(instance._run_claude_code("hello"))
    assert captured["command"][captured["command"].index("--effort") + 1] == "high"
    assert (captured["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], captured["env"]["API_TIMEOUT_MS"]) == ("64000", "60000")


@pytest.mark.parametrize(
    "name",
    [
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS",
    ],
)
def test_extra_env_cannot_replace_what_the_agent_sets(name):
    with pytest.raises(ValidationError, match=f"extra_env cannot set {name}"):
        app.ClaudeCodeAgentConfig(name="claude", host="h", port=1, entrypoint="app.py", extra_env={name: "x"})


def test_output_cap_must_be_positive():
    with pytest.raises(ValidationError):
        app.ClaudeCodeAgentConfig(name="claude", host="h", port=1, entrypoint="app.py", max_output_tokens=0)


@pytest.mark.asyncio
async def test_root_workdir_is_accepted_when_allowed(monkeypatch):
    # Task images without a WORKDIR start in "/": the runtime and session files stay under /tmp.
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    monkeypatch.setattr(app, "install_sandbox_runtime", AsyncMock())
    state = await agent(allow_root_workdir=True)._seed_agent_session_state(seed(workdir="/"))
    assert state.session.workdir == "/"
    assert state.session.session_dir.startswith("/tmp/nemo-gym-claude-sessions/")
    assert "\ntest -d / && " in remote.exec.await_args_list[0].args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("workdir", ["/tmp", "/tmp/nemo-gym-claude-runtime-2.1.108", "/app/../tmp"])
async def test_root_workdir_setting_still_rejects_adapter_storage(monkeypatch, workdir):
    connector = AsyncMock()
    monkeypatch.setattr(app.AsyncSandbox, "connect", connector)
    with pytest.raises(HTTPException, match="separate from runtime/session storage"):
        await agent(allow_root_workdir=True)._seed_agent_session_state(seed(workdir=workdir))
    connector.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("exit_code", [0, 100])
async def test_harbor_dependency_step_runs_first_and_never_blocks_setup(monkeypatch, enabled, exit_code):
    script = system_dependencies_command()

    async def run(command, **kwargs):
        return result(code=exit_code) if command == script else result()

    remote = SimpleNamespace(exec=AsyncMock(side_effect=run), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    install = AsyncMock()
    monkeypatch.setattr(app, "install_sandbox_runtime", install)
    instance = agent(harbor_system_dependencies=enabled, system_dependencies_timeout_seconds=900)
    await instance._seed_agent_session_state(seed())
    calls = remote.exec.await_args_list
    if enabled:
        # Before the session directory and the Claude Code install, with its own time limit.
        assert calls[0].args[0] == script
        assert calls[0].kwargs["timeout_s"] == 900
    else:
        assert all(call.args[0] != script for call in calls)
    install.assert_awaited_once()
