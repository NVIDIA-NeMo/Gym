# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in sandbox-session settings for running Claude Code as a harness started in the task container."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.claude_code_agent import app, sandbox_runner
from responses_api_agents.claude_code_agent import sandbox as adapter
from responses_api_agents.claude_code_agent.tests.test_sandbox import agent, seed


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
    child = tmp_path / "child.py"
    child.write_text("import json,sys; print(json.dumps({'stdin': sys.stdin.read()}))")
    payload = {"directory": str(tmp_path), "command": [sys.executable, str(child)], "cwd": str(tmp_path), "env": {}}
    if prompt is not None:
        payload["prompt"] = prompt
    params = tmp_path / "input.json"
    params.write_text(json.dumps(payload))
    completed = subprocess.run(
        [sys.executable, str(Path(sandbox_runner.__file__)), str(params)], capture_output=True, timeout=30
    )
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
