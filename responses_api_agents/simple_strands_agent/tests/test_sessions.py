# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Strands agent sessions: seed refusals, the runner handoff, close, and the runner's sandbox environment."""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request

from nemo_gym.agents import AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.simple_strands_agent.app import (
    SimpleStrandsAgent,
    SimpleStrandsAgentConfig,
    SimpleStrandsSession,
)


def agent(**kwargs) -> SimpleStrandsAgent:
    return SimpleStrandsAgent(
        config=SimpleStrandsAgentConfig(
            host="localhost",
            port=8080,
            name="simple_strands_agent",
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            **kwargs,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def seed(*, workdir="/repo", tools=None, user=None, borrowed=True) -> AgentSeedSessionRequest:
    access = {
        "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"id": "task"}},
        "workdir": workdir,
        "env": {"TASK_TEST": "task-value"},
    }
    if user is not None:
        access["user"] = user
    return AgentSeedSessionRequest(
        agent_session_id="session-1",
        episode_id={"rollout_id": "rollout-1"},
        task_id={"taskset": "test", "task_id": "task"},
        tool_accesses=tools or [],
        sandbox_access=access if borrowed else None,
    )


@pytest.mark.parametrize(
    ("request_seed", "reason"),
    [
        (
            seed(
                tools=[
                    {
                        "kind": "mcp",
                        "name": "task_tools",
                        "required": True,
                        "connection": {"transport": "streamable_http", "url": "http://r/mcp"},
                    }
                ]
            ),
            "Simple Strands Agent uses its own tools; required tool grants are unsupported",
        ),
        (seed(user="nobody"), "Simple Strands Agent does not support a SandboxAccess user override"),
        (seed(workdir="repo"), "Simple Strands Agent task workdir must be an absolute path"),
        (seed(borrowed=False), "Simple Strands Agent requires sandbox_access or a configured sandbox_provider"),
    ],
)
def test_unsupported_seeds_are_final_refusals(request_seed, reason):
    server = agent()
    assert check_agent_seed_rejected(server.setup_webserver(), request_seed) == reason
    assert not server._session_records


@pytest.mark.asyncio
async def test_session_runs_tools_in_the_task_sandbox_and_releases_it(monkeypatch, tmp_path):
    server = agent(workspace_root=str(tmp_path / "workspaces"))
    sandbox = MagicMock(sandbox_id="task")
    sandbox.serialize = AsyncMock(return_value={"id": "task"})
    sandbox.disconnect = AsyncMock()
    state = SimpleStrandsSession(
        request=seed(),
        sandbox=sandbox,
        owns_sandbox=False,
        provider_name="opensandbox",
        provider_config={"opensandbox": {"api_key": "secret"}},  # pragma: allowlist secret
        workdir="/repo",
    )
    seen = {}

    class Process:
        pid, returncode = 123, 0

        def __init__(self, args):
            self.args = args

        async def communicate(self, input=None):
            seen["args"], seen["stdin"] = self.args, json.loads(input)
            request_path, result_path = Path(self.args[2]), Path(self.args[3])
            seen["payload"] = json.loads(request_path.read_text())
            messages = [{"role": "assistant", "content": [{"text": "Done."}]}]
            result_path.write_text(json.dumps({"messages": messages, "usage": {"inputTokens": 5, "outputTokens": 2}}))
            return b"", b""

    async def create_subprocess_exec(*args, **kwargs):
        seen["stdin_pipe"] = kwargs.get("stdin")
        return Process(args)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(
        SimpleStrandsAgent, "resolve_model_base_url", lambda self, name, rollout: f"http://m/{rollout}/v1"
    )
    server._session_records["session-1"] = MagicMock(state=state, closing=False)
    request = Request(
        {"type": "http", "session": {"agent_session_id": "session-1"}, "path_params": {"rollout_id": "rollout-1"}}
    )
    response = await server.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="Fix the bug."))
    assert response.output[-1].content[0].text == "Done."
    # Tools run in the task workdir; the provider config (with its credentials) reaches the runner only on stdin.
    assert seen["payload"]["work_dir"] == "/repo" and "secret" not in json.dumps(seen["payload"])
    assert seen["args"][-1] == "--sandbox-stdin" and seen["stdin_pipe"] is not None
    assert seen["stdin"] == {
        "provider": {"opensandbox": {"api_key": "secret"}},  # pragma: allowlist secret
        "descriptor": {"id": "task"},
        "env": {"TASK_TEST": "task-value"},
    }
    assert seen["payload"]["model_base_url"] == "http://m/rollout-1/v1"
    assert not list((tmp_path / "workspaces").iterdir())
    close = await server._close_agent_session_state(state)
    assert close.cleanup_confirmed and close.agent_observations.records[0].status == "completed"
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
async def test_runner_sandbox_environment_runs_commands_through_the_provider(tmp_path):
    runner = pytest.importorskip("responses_api_agents.simple_strands_agent.ssa_runner")

    class LocalSandbox:
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None):
            done = subprocess.run(
                command, shell=True, cwd=cwd, env={**env} if env else None, capture_output=True, text=True
            )
            return SandboxExecResult(done.stdout, done.stderr, done.returncode)

        async def disconnect(self):
            pass

    with (
        patch("nemo_gym.sandbox.AsyncSandbox.connect", AsyncMock(return_value=LocalSandbox())),
        patch("nemo_gym.sandbox.create_provider", MagicMock()),
        patch("nemo_gym.sandbox.upload_text", AsyncMock()) as upload,
    ):
        with runner.SandboxEnvironment(
            {"provider": {"local": {}}, "descriptor": {}, "env": {"TASK_TEST": "v", "PATH": "/usr/bin:/bin"}},
            str(tmp_path),
            30,
        ) as environment:
            result = environment.execute_bash("echo $TASK_TEST; pwd; exit 3")
            assert (result["exit_code"], result["output"].split()) == (3, ["v", str(tmp_path)])
            assert environment.dir_exists(str(tmp_path)) and not environment.file_exists(str(tmp_path / "x"))
            environment.write_file("text", str(tmp_path / "x"))
    assert upload.await_args.kwargs == {"path": str(tmp_path / "x"), "text": "text"}


def test_iteration_limit_is_an_incomplete_graded_response():
    body = NeMoGymResponseCreateParamsNonStreaming(input="Fix the bug.")
    messages = [{"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "bash", "input": {}}}]}]
    limited = agent()._response(body, {"messages": messages, "stop_reason": "max_turns"}, "policy")
    finished = agent()._response(body, {"messages": messages, "stop_reason": None}, "policy")
    assert (limited.status, finished.status) == ("incomplete", "completed")
    assert limited.output[0].type == "function_call"
