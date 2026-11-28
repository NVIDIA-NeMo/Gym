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
    remote = SimpleNamespace(sandbox_id="sandbox-actual", exec=AsyncMock(return_value=result()), upload=AsyncMock())
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
async def test_harbor_dependency_step_runs_first_and_a_failed_install_fails_setup(monkeypatch, enabled, exit_code):
    script = system_dependencies_command()

    async def run(command, **kwargs):
        return result(code=exit_code) if command == script else result()

    remote = SimpleNamespace(sandbox_id="sandbox-actual", exec=AsyncMock(side_effect=run), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    install = AsyncMock()
    monkeypatch.setattr(app, "install_sandbox_runtime", install)
    close = AsyncMock()
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "close", close)
    instance = agent(harbor_system_dependencies=enabled, system_dependencies_timeout_seconds=900)
    failed = enabled and exit_code != 0
    if failed:
        # As in Harbor, the task never runs without the tools the step guarantees: setup fails, the session is
        # released, and the episode is retried.
        with pytest.raises(RuntimeError, match=r"system dependency step failed \(exit=100, error=None\)"):
            await instance._seed_agent_session_state(seed())
    else:
        await instance._seed_agent_session_state(seed())
    assert close.await_count == int(failed)
    calls = remote.exec.await_args_list
    if enabled:
        # Before the session directory and the Claude Code install, with its own time limit.
        assert calls[0].args[0] == script
        assert calls[0].kwargs["timeout_s"] == 900
    else:
        assert all(call.args[0] != script for call in calls)
    assert install.await_count == int(not failed)


def collecting_session(tmp_path, monkeypatch, *, logs_dir, failing=(), slow=()):
    """A session whose directory and commands are local, so that ``collect_artifacts`` copies real files.

    The session directory holds the transcript of one session and of one of its subagents. A command that holds a
    word of ``failing`` ends with exit code 1, and one that holds a word of ``slow`` does not end.
    """
    session_dir = tmp_path / "session"
    subagents = session_dir / "config" / "projects" / "-app" / "session-1" / "subagents"
    subagents.mkdir(parents=True)
    (subagents.parents[1] / "session-1.jsonl").write_text('{"type":"user"}\n')
    (subagents / "agent-1.jsonl").write_text('{"type":"assistant"}\n')
    commands = []

    async def run(command, **kwargs):
        commands.append(command)
        if any(word in command for word in slow):
            await asyncio.Event().wait()
        if any(word in command for word in failing):
            return result(code=1)
        completed = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=30)
        return result(completed.stdout, code=completed.returncode)

    async def download(source, target):
        shutil.copyfile(source, target)

    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(session_dir=str(session_dir), sandbox=SimpleNamespace(exec=run, download=download)),
        logs_dir=logs_dir,
    )
    files = {"events.jsonl": RESULT + "\n", "stderr.log": "", "runtime.json": '{"hostname":"worker","pid":1}'}
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "read_text", AsyncMock(side_effect=files.__getitem__))
    return state, commands


@pytest.mark.asyncio
async def test_transcripts_are_copied_to_the_logs_dir_after_the_artifacts(tmp_path, monkeypatch):
    logs = tmp_path / "logs" / "agent"
    state, commands = collecting_session(tmp_path, monkeypatch, logs_dir=str(logs))
    artifacts = await state.collect_artifacts()
    assert artifacts.events == RESULT + "\n" and artifacts.transcripts
    copied = logs / "sessions" / "projects" / "-app"
    assert (copied / "session-1.jsonl").read_text() == '{"type":"user"}\n'
    assert (copied / "session-1" / "subagents" / "agent-1.jsonl").read_text() == '{"type":"assistant"}\n'
    # The copy is the last command, after the archive of the transcripts that the agent server reads.
    assert ["tar -czf" in command for command in commands] == [True, False]


@pytest.mark.asyncio
async def test_failed_copy_keeps_the_artifacts(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    state, _ = collecting_session(tmp_path, monkeypatch, logs_dir=str(logs), failing=("cp -R",))
    artifacts = await state.collect_artifacts()
    assert artifacts.events == RESULT + "\n" and artifacts.transcripts
    assert not logs.exists()


@pytest.mark.asyncio
async def test_copy_that_does_not_end_keeps_the_artifacts(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    state, _ = collecting_session(tmp_path, monkeypatch, logs_dir=str(logs), slow=("cp -R",))
    monkeypatch.setattr(adapter, "_COPY_RESERVE_S", 0.2)
    state.capture_budget_s = 0.5
    # SandboxSession cancels a collection that outlasts the close timeout, which is the capture budget.
    async with asyncio.timeout(state.capture_budget_s):
        artifacts = await state.collect_artifacts()
    assert artifacts.events == RESULT + "\n" and artifacts.transcripts
    assert not logs.exists()


@pytest.mark.asyncio
async def test_no_copy_starts_without_capture_time(tmp_path, monkeypatch):
    state, commands = collecting_session(tmp_path, monkeypatch, logs_dir=str(tmp_path / "logs"))
    state.capture_budget_s = 0
    artifacts = await state.collect_artifacts()
    assert artifacts.events == RESULT + "\n" and artifacts.transcripts
    assert len(commands) == 1 and "tar -czf" in commands[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("through", ["execute", "close"])
async def test_capture_budget_is_the_close_timeout(through):
    session = SimpleNamespace(execute=AsyncMock(), close=AsyncMock(), closing=False, artifacts=None)
    state = adapter.ClaudeCodeSandboxSession(request=seed(), runtime="/runtime", session=session)
    if through == "execute":
        await state.execute({}, timeout=600, close_timeout=45)
    else:
        await state.close(timeout=45)
    assert state.capture_budget_s == 45


@pytest.mark.asyncio
async def test_transcripts_are_not_copied_without_a_logs_dir(tmp_path, monkeypatch):
    state, commands = collecting_session(tmp_path, monkeypatch, logs_dir=None)
    await state.collect_artifacts()
    assert len(commands) == 1 and "tar -czf" in commands[0]


@pytest.mark.asyncio
async def test_session_takes_the_logs_dir_of_the_configuration(monkeypatch):
    remote = SimpleNamespace(sandbox_id="sandbox-actual", exec=AsyncMock(return_value=result()), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    monkeypatch.setattr(app, "install_sandbox_runtime", AsyncMock())
    assert (await agent()._seed_agent_session_state(seed())).logs_dir is None
    state = await agent(sandbox_logs_dir="/logs/agent/")._seed_agent_session_state(seed())
    assert state.logs_dir == "/logs/agent"


@pytest.mark.parametrize("value", ["/", "relative", "/logs/../tmp"])
def test_sandbox_logs_dir_must_be_an_absolute_directory_other_than_root(value):
    with pytest.raises(ValidationError, match="sandbox_logs_dir must be an absolute sandbox directory"):
        app.ClaudeCodeAgentConfig(name="claude", host="h", port=1, entrypoint="app.py", sandbox_logs_dir=value)
