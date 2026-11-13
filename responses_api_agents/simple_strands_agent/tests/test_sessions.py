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

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.simple_strands_agent.app import (
    SimpleStrandsAgent,
    SimpleStrandsAgentConfig,
    SimpleStrandsSession,
    SSARunError,
)


APP = "responses_api_agents.simple_strands_agent.app"


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
    limited = agent()._response(body, {"messages": messages, "stop_reason": "max_iterations"}, "policy")
    finished = agent()._response(body, {"messages": messages, "stop_reason": None}, "policy")
    assert (limited.status, finished.status) == ("incomplete", "completed")
    assert limited.output[0].type == "function_call"


@pytest.mark.parametrize(
    "exec_mock",
    [
        AsyncMock(side_effect=ConnectionError("backend unavailable")),
        AsyncMock(return_value=SandboxExecResult(None, "backend unavailable", 125, error_type="sandbox")),
    ],
)
def test_runner_sandbox_outage_fails_the_run_instead_of_an_exit_code(exec_mock):
    runner = pytest.importorskip("responses_api_agents.simple_strands_agent.ssa_runner")
    sandbox = MagicMock(exec=exec_mock, disconnect=AsyncMock())
    with (
        patch("nemo_gym.sandbox.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
        patch("nemo_gym.sandbox.create_provider", MagicMock()),
        runner.SandboxEnvironment({"provider": {}, "descriptor": {}}, "/repo", 30) as environment,
    ):
        with pytest.raises(Exception, match="backend unavailable"):
            environment.dir_exists("/repo")
        assert "backend unavailable" in environment.infra_error


@pytest.mark.asyncio
@pytest.mark.parametrize(("owns_sandbox", "confirmed"), [(False, False), (True, True)])
async def test_close_during_a_run_confirms_cleanup_only_for_an_owned_sandbox(owns_sandbox, confirmed):
    sandbox = MagicMock(sandbox_id="task", stop=AsyncMock(), disconnect=AsyncMock())
    state = SimpleStrandsSession(
        request=seed(),
        sandbox=sandbox,
        owns_sandbox=owns_sandbox,
        provider_name="opensandbox",
        provider_config={},
        workdir="/repo",
    )
    state.task = asyncio.create_task(asyncio.sleep(60))
    close = await agent()._close_agent_session_state(state)
    # A borrowed sandbox may still run the interrupted tool command, which disconnect does not stop.
    assert close.cleanup_confirmed is confirmed and state.task.cancelled()


def test_runner_upload_outage_fails_the_run_instead_of_a_tool_error():
    runner = pytest.importorskip("responses_api_agents.simple_strands_agent.ssa_runner")
    with (
        patch("nemo_gym.sandbox.AsyncSandbox.connect", AsyncMock(return_value=MagicMock(disconnect=AsyncMock()))),
        patch("nemo_gym.sandbox.create_provider", MagicMock()),
        patch("nemo_gym.sandbox.upload_text", AsyncMock(side_effect=ConnectionError("backend unavailable"))),
        runner.SandboxEnvironment({"provider": {}, "descriptor": {}}, "/repo", 30) as environment,
    ):
        with pytest.raises(ConnectionError):
            environment.write_file("text", "/repo/x")
        assert environment.infra_error == "Sandbox upload failed: backend unavailable"


@pytest.mark.parametrize("error", [Exception("Agent has exceeded the set threshold"), RuntimeError("model down")])
def test_runner_iteration_limit_raised_raw_ends_the_run(tmp_path, error):
    runner = pytest.importorskip("responses_api_agents.simple_strands_agent.ssa_runner")
    strands_agent = MagicMock(side_effect=error, messages=[{"role": "user", "content": []}])
    strands_agent.event_loop_metrics.accumulated_usage = {}
    environment = MagicMock()
    environment.__enter__.return_value = environment
    payload = {
        "max_turns": 3,
        "work_dir": str(tmp_path),
        "shell_timeout": 30,
        "rollout_id": "rollout-1",
        "instruction": "Fix the bug.",
        "model": "policy",
        "prompt_tag": "swe_generic_v2",
        "max_output_tokens": 16,
        "model_base_url": "http://m/v1",
        "model_timeout": 30,
        "tools": [],
        "output_dir": str(tmp_path / "output"),
        "conversation_window": 10,
        "native_user_prompt": False,
    }
    with (
        patch.object(runner, "StrandsResolverAgent", return_value=strands_agent),
        patch.object(runner, "sr_model"),
        patch.object(runner, "MetricsCollector"),
        patch.object(runner, "load_tools", return_value=([], {})),
        patch.object(runner, "create_environment", return_value=environment),
    ):
        if "threshold" in str(error):
            assert runner._run(payload)["stop_reason"] == "max_iterations"
        else:
            with pytest.raises(RuntimeError, match="model down"):
                runner._run(payload)


def test_runner_error_result_carries_the_model_request_status(tmp_path, monkeypatch):
    runner = pytest.importorskip("responses_api_agents.simple_strands_agent.ssa_runner")
    upstream = type("APIStatusError", (Exception,), {"status_code": 413})("too large")
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text("{}")
    monkeypatch.setattr("sys.argv", ["ssa_runner.py", str(request), str(result)])

    def run(payload, sandbox):
        try:
            raise upstream
        except Exception as error:
            raise RuntimeError("event loop failed") from error

    with patch.object(runner, "_run", run), pytest.raises(RuntimeError):
        runner.main()
    assert json.loads(result.read_text())["status_code"] == 413


def borrowed_state(**kwargs) -> SimpleStrandsSession:
    sandbox = MagicMock(sandbox_id="task", disconnect=AsyncMock(), stop=AsyncMock())
    sandbox.serialize = AsyncMock(return_value={"id": "task"})
    return SimpleStrandsSession(
        request=seed(), sandbox=sandbox, provider_name="opensandbox", provider_config={}, workdir="/repo", **kwargs
    )


async def activate(server, state, run_ssa, body=None):
    server._session_records["session-1"] = MagicMock(state=state, closing=False)
    request = Request(
        {"type": "http", "session": {"agent_session_id": "session-1"}, "path_params": {"rollout_id": "rollout-1"}}
    )
    with (
        patch.object(SimpleStrandsAgent, "_run_ssa", run_ssa),
        patch.object(SimpleStrandsAgent, "resolve_model_base_url", return_value="http://m/v1"),
    ):
        return await server.responses(request, body or NeMoGymResponseCreateParamsNonStreaming(input="Fix it."))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "outcome", "confirmed"),
    [
        (SSARunError("SSA failed: 400", 400), {"failure_kind": "model_request_error", "status_code": 400}, True),
        (SSARunError("SSA failed: sandbox upload failed"), {"failure_kind": "unknown"}, True),
        (RuntimeError("SSA failed: crashed"), {"failure_kind": "unknown"}, False),
    ],
)
async def test_runner_failures_are_classified_failed_responses(tmp_path, error, outcome, confirmed):
    server, state = agent(workspace_root=str(tmp_path)), borrowed_state(owns_sandbox=False)
    response = await activate(server, state, AsyncMock(side_effect=error))
    assert response.status == "failed"
    assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == outcome
    close = await server._close_agent_session_state(state)
    invocation, observation = close.agent_observations.records
    assert (invocation.status, invocation.error_type, observation.outcome) == (
        "failed",
        type(error).__name__,
        "failed",
    )
    # Only a runner that wrote its own error result had no tool command left running.
    assert close.cleanup_confirmed is confirmed


@pytest.mark.asyncio
async def test_deadline_grades_the_saved_transcript_as_incomplete(tmp_path):
    server, state = agent(workspace_root=str(tmp_path)), borrowed_state(owns_sandbox=False)

    async def run_ssa(self, payload, host_dir, sandbox):
        messages = [{"role": "assistant", "content": [{"text": "Partial."}]}]
        (host_dir / "output").mkdir()
        (host_dir / "output" / "trajectory.json").write_text(json.dumps(messages))
        raise TimeoutError("SSA timed out")

    response = await activate(server, state, run_ssa)
    assert (response.status, response.output[0].content[0].text) == ("incomplete", "Partial.")
    close = await server._close_agent_session_state(state)
    invocation, observation = close.agent_observations.records
    assert (invocation.stop_reason, invocation.error_type, observation.outcome) == ("deadline", None, "timeout")
    assert not close.cleanup_confirmed


@pytest.mark.asyncio
async def test_iteration_limit_is_a_stop_reason_not_an_error(tmp_path):
    server, state = agent(workspace_root=str(tmp_path)), borrowed_state(owns_sandbox=False)
    run_ssa = AsyncMock(return_value={"messages": [], "stop_reason": "max_iterations"})
    assert (await activate(server, state, run_ssa)).status == "incomplete"
    invocation, observation = (await server._close_agent_session_state(state)).agent_observations.records
    assert (invocation.stop_reason, invocation.error_type, observation.error_type) == ("max_iterations", None, None)


@pytest.mark.asyncio
async def test_run_ssa_raises_the_error_result_of_a_failed_runner(tmp_path, monkeypatch):
    class Process:
        pid, returncode = 123, 1

        def __init__(self, args):
            self.args = args

        async def communicate(self, input=None):
            Path(self.args[3]).write_text(json.dumps({"error": "Error code: 413", "status_code": 413}))
            return b"", b"Traceback"

    async def create_subprocess_exec(*args, **kwargs):
        return Process(args)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    with pytest.raises(SSARunError) as error:
        await agent()._run_ssa({}, tmp_path)
    assert error.value.status_code == 413


@pytest.mark.asyncio
async def test_one_activation_per_session(tmp_path):
    server, state = agent(workspace_root=str(tmp_path)), borrowed_state(owns_sandbox=False)
    release = asyncio.Event()

    calls = []

    async def run(self, payload, host_dir, sandbox):
        calls.append(payload)
        await release.wait()
        return {"messages": [{"role": "assistant", "content": [{"text": "Done."}]}]}

    first = asyncio.create_task(activate(server, state, run))
    retry = asyncio.create_task(activate(server, state, run))
    await asyncio.sleep(0.05)
    with pytest.raises(Exception) as other:
        await activate(server, state, run, NeMoGymResponseCreateParamsNonStreaming(input="Other task."))
    assert other.value.status_code == 409
    release.set()
    assert (await first).output[0].content[0].text == (await retry).output[0].content[0].text == "Done."
    assert len(calls) == 1
    wrong_route = Request(
        {"type": "http", "session": {"agent_session_id": "session-1"}, "path_params": {"rollout_id": "other"}}
    )
    with pytest.raises(Exception) as route:
        await server.responses(wrong_route, NeMoGymResponseCreateParamsNonStreaming(input="Fix it."))
    assert route.value.status_code == 409


@pytest.mark.asyncio
async def test_close_before_activation_reports_no_observations():
    state = borrowed_state(owns_sandbox=False)
    close = await agent()._close_agent_session_state(state)
    assert close.cleanup_confirmed
    assert [gap.code for gap in close.agent_observations.gaps] == ["observation_capture_failed"]
    state.sandbox.disconnect.assert_awaited_once()


class ConnectableProvider:
    name = "docker"

    def __init__(self):
        self.aclose = AsyncMock()

    async def serialize_handle(self, handle, *, scope=None):
        return {}

    async def connect(self, descriptor):
        return None


def seed_patches(provider, sandbox):
    return (
        patch(f"{APP}.get_global_config_dict", return_value={}),
        patch(f"{APP}.resolve_provider_config", return_value={"docker": {}}),
        patch(f"{APP}.resolve_provider_metadata", return_value={"team": "gym"}),
        patch(f"{APP}.create_provider", return_value=provider),
        patch(f"{APP}.AsyncSandbox", MagicMock(return_value=sandbox, connect=AsyncMock(return_value=sandbox))),
    )


def seed_sandbox(return_code=0):
    sandbox = MagicMock(sandbox_id="task", start=AsyncMock(), stop=AsyncMock(), disconnect=AsyncMock())
    sandbox.exec = AsyncMock(return_value=SandboxExecResult("", "no workdir", return_code))
    return sandbox


@pytest.mark.asyncio
async def test_seed_connects_to_a_borrowed_sandbox_and_checks_the_workdir():
    provider, sandbox = MagicMock(name="opensandbox", aclose=AsyncMock()), seed_sandbox()
    patches = seed_patches(provider, sandbox)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        state = await agent()._seed_agent_session_state(seed())
    assert (state.owns_sandbox, state.workdir, state.provider_config) == (False, "/repo", {"docker": {}})
    assert "test -d /repo" in sandbox.exec.await_args.args[0]


@pytest.mark.asyncio
async def test_seed_starts_an_owned_sandbox_with_merged_metadata():
    provider, sandbox = ConnectableProvider(), seed_sandbox()
    server = agent(sandbox_provider="docker", sandbox_config={"image": "img", "metadata": {"task": "t"}})
    patches = seed_patches(provider, sandbox)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        state = await server._seed_agent_session_state(seed(borrowed=False))
    spec = sandbox.start.await_args.args[0]
    assert spec.metadata == {"team": "gym", "task": "t", "agent_session_id": "session-1"}
    assert (spec.workdir, spec.ttl_s, state.owns_sandbox) == ("/app", 1800 + 600, True)


@pytest.mark.asyncio
async def test_seed_refuses_an_owned_sandbox_on_a_provider_that_cannot_reconnect():
    provider = MagicMock(spec=["name", "aclose"], aclose=AsyncMock())
    provider.name = "local"
    patches = seed_patches(provider, seed_sandbox())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with pytest.raises(AgentSeedRejectedError, match="connectable sandbox provider, not local"):
            await agent(sandbox_provider="local")._seed_agent_session_state(seed(borrowed=False))
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_seed_failures_release_the_sandbox():
    # A failed workdir check releases the borrowed sandbox.
    sandbox = seed_sandbox(return_code=1)
    patches = seed_patches(MagicMock(aclose=AsyncMock()), sandbox)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with pytest.raises(RuntimeError, match="no bash or no workdir"):
            await agent()._seed_agent_session_state(seed())
    sandbox.disconnect.assert_awaited_once()

    # A failed connect closes the provider.
    provider = MagicMock(aclose=AsyncMock())
    patches = seed_patches(provider, sandbox)
    with patches[0], patches[1], patches[2], patches[3]:
        with patch(f"{APP}.AsyncSandbox.connect", AsyncMock(side_effect=ConnectionError("down"))):
            with pytest.raises(ConnectionError):
                await agent()._seed_agent_session_state(seed())
    provider.aclose.assert_awaited_once()

    # A failed start whose release also fails keeps the state for close.
    sandbox = seed_sandbox()
    sandbox.start.side_effect, sandbox.stop.side_effect = RuntimeError("no capacity"), RuntimeError("gone")
    patches = seed_patches(ConnectableProvider(), sandbox)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with pytest.raises(AgentSessionSetupError) as error:
            await agent(sandbox_provider="docker")._seed_agent_session_state(seed(borrowed=False))
    assert str(error.value.error) == "no capacity" and error.value.state.owns_sandbox
