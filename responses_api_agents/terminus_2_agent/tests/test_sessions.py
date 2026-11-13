# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions: Terminus-2 in a borrowed or owned task sandbox."""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.agents import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    RolloutStopRequest,
    SandboxPythonRuntime,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.agents.base_responses_api_agent import _AgentSessionRecord
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess, SandboxExecResult, SandboxSpec
from nemo_gym.server_runtime import DirectHTTPToolAccess
from nemo_gym.trajectory import AgentInvocation, SandboxObservation, ToolCallObservation
from responses_api_agents.terminus_2_agent import sandbox as terminus_sandbox
from responses_api_agents.terminus_2_agent.app import Terminus2Agent, Terminus2AgentConfig, Terminus2AgentRunRequest
from responses_api_agents.terminus_2_agent.sandbox import Terminus2SandboxSession, terminus_runtime


APP = "responses_api_agents.terminus_2_agent.app"


def _model_url(agent: Terminus2Agent, name: str, rollout_id: str | None) -> str:
    """The Model Server URL for one rollout, as the agent base builds it."""
    return f"http://model:8000/ng-rollout/{rollout_id}/v1"


def _config(**kwargs) -> Terminus2AgentConfig:
    return Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        model="test-model",
        **kwargs,
    )


def _agent(**kwargs) -> Terminus2Agent:
    return Terminus2Agent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))


def _seed(*, borrowed: bool = True, workdir: str = "/app", **kwargs) -> AgentSeedSessionRequest:
    access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="runtime", descriptor={"sandbox_id": "task"}),
        workdir=workdir,
    )
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="test", task_id="task"),
        sandbox_access=access if borrowed else None,
        **kwargs,
    )


def _ok(stdout: str = "") -> SandboxExecResult:
    return SandboxExecResult(stdout=stdout, stderr="", return_code=0)


def _sandbox(*results: SandboxExecResult) -> AsyncMock:
    """A sandbox whose commands succeed, after answering the given results in order."""
    sandbox = AsyncMock()
    sandbox.sandbox_id = "sandbox-1"
    pending = list(results)

    async def execute(command: str, **kwargs) -> SandboxExecResult:
        return pending.pop(0) if pending else _ok("tmux 3.4")

    sandbox.exec.side_effect = execute
    return sandbox


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch provider lookup and sandbox connection; runtime installation succeeds unless a test changes it."""
    sandbox = _sandbox()
    patched = SimpleNamespace(
        sandbox=sandbox,
        provider=AsyncMock(),
        connect=AsyncMock(return_value=sandbox),
        resolve=MagicMock(return_value={"docker": {}}),
        metadata=MagicMock(return_value={}),
        is_installed=AsyncMock(return_value=False),
        install=AsyncMock(),
    )
    monkeypatch.setattr(f"{APP}.get_global_config_dict", lambda: {"runtime": {}})
    monkeypatch.setattr(f"{APP}.resolve_provider_config", patched.resolve)
    monkeypatch.setattr(f"{APP}.resolve_provider_metadata", patched.metadata)
    monkeypatch.setattr(f"{APP}.create_provider", lambda config: patched.provider)
    monkeypatch.setattr(f"{APP}.AsyncSandbox.connect", patched.connect)
    monkeypatch.setattr(Terminus2Agent, "resolve_model_base_url", _model_url)
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", patched.is_installed)
    monkeypatch.setattr(SandboxPythonRuntime, "install", patched.install)
    return patched


class TestSeed:
    @pytest.mark.parametrize(
        ("seed", "reason"),
        [
            (_seed(borrowed=False), "Terminus 2 needs sandbox_access or a configured sandbox_provider"),
            (_seed(workdir="app"), "Terminus 2 needs an absolute sandbox workdir, got 'app'"),
            (
                _seed(
                    tool_accesses=[DirectHTTPToolAccess(name="tools", required=True, base_url="http://resources:1")]
                ),
                "Terminus 2 runs only its terminal; required tool grants it cannot use: tools",
            ),
        ],
    )
    def test_unsupported_seed_is_refused_with_a_final_422(self, seed: AgentSeedSessionRequest, reason: str) -> None:
        agent = _agent()

        assert check_agent_seed_rejected(agent.setup_webserver(), seed).startswith(reason)
        assert not agent._session_records

    async def test_optional_grants_are_skipped(self, providers: SimpleNamespace) -> None:
        seed = _seed(tool_accesses=[DirectHTTPToolAccess(name="tools", required=False, base_url="http://r:1")])

        state = await _agent()._seed_agent_session_state(seed)

        assert state.request is seed

    async def test_borrowed_sandbox_installs_the_runtime_and_stages_the_harness(
        self, providers: SimpleNamespace
    ) -> None:
        state = await _agent(sandbox_install_timeout_seconds=123)._seed_agent_session_state(_seed())

        providers.resolve.assert_called_once_with("runtime", {"runtime": {}})
        providers.connect.assert_awaited_once_with({"sandbox_id": "task"}, provider=providers.provider)
        providers.install.assert_awaited_once_with(providers.sandbox, timeout_s=123)
        session = state.session
        assert session.owns_sandbox is False and session.workdir == "/app"
        assert session.session_dir.startswith("/tmp/nemo-gym-terminus-2-sessions/")
        assert (state.sandbox_id, state.provider) == ("sandbox-1", "runtime")
        uploads = {call.args[1] for call in providers.sandbox.upload.await_args_list}
        assert uploads == {f"{session.session_dir}/harness/{name}" for name in terminus_sandbox._HARNESS_FILES}
        # Setup never runs in, or writes to, the task's working directory.
        assert {call.kwargs["cwd"] for call in providers.sandbox.exec.await_args_list} == {"/"}

    async def test_a_runtime_that_passes_its_check_is_reused(self, providers: SimpleNamespace) -> None:
        providers.is_installed.return_value = True

        await _agent()._seed_agent_session_state(_seed())

        providers.is_installed.assert_awaited_once_with(providers.sandbox)
        providers.install.assert_not_awaited()

    async def test_missing_tmux_is_installed_during_setup(self, providers: SimpleNamespace) -> None:
        sandbox = _sandbox(_ok(), SandboxExecResult(stdout="", stderr="tmux: not found", return_code=127))
        providers.connect.return_value = sandbox

        await _agent()._seed_agent_session_state(_seed())

        commands = [call.args[0] for call in sandbox.exec.await_args_list]
        assert commands[-1] == (
            f"{terminus_runtime().python} -E -s {commands[0].split()[2]}/sandbox_runner.py --ensure-tmux"
        )

    async def test_a_failed_tmux_install_fails_setup_and_releases_the_sandbox(
        self, providers: SimpleNamespace
    ) -> None:
        failed = SandboxExecResult(stdout="E: Unable to locate package tmux", stderr="", return_code=100)
        sandbox = _sandbox(_ok(), SandboxExecResult(stdout="", stderr="", return_code=127), failed)
        providers.connect.return_value = sandbox

        with pytest.raises(RuntimeError, match="Installing tmux for Terminus 2 failed with 100"):
            await _agent()._seed_agent_session_state(_seed())

        sandbox.disconnect.assert_awaited_once()
        sandbox.stop.assert_not_awaited()

    @pytest.mark.parametrize("missing", ["missing bash", "missing script (util-linux)"])
    def test_an_image_without_bash_or_script_is_refused_with_a_final_422(
        self, providers: SimpleNamespace, missing: str
    ) -> None:
        sandbox = _sandbox(_ok(), SandboxExecResult(stdout=f"{missing}\n", stderr="", return_code=3))
        providers.connect.return_value = sandbox
        agent = _agent()

        assert check_agent_seed_rejected(agent.setup_webserver(), _seed()).endswith(missing)
        assert not agent._session_records
        sandbox.disconnect.assert_awaited_once()

    @pytest.mark.skipif(shutil.which("script") is None, reason="needs script")
    @pytest.mark.parametrize(
        ("present", "reported"), [(("script",), "missing bash"), (("bash",), "missing script (util-linux)")]
    )
    def test_the_terminal_probe_names_the_first_missing_tool(
        self, tmp_path: Path, present: tuple[str, ...], reported: str
    ) -> None:
        for tool in present:
            (tmp_path / tool).symlink_to(shutil.which(tool))

        probe = subprocess.run(
            ["/bin/sh", "-c", terminus_sandbox._TERMINAL_PROBE],
            env={**os.environ, "PATH": str(tmp_path)},
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert (probe.returncode, probe.stdout.strip()) == (3, reported)

    async def test_a_failed_install_releases_the_borrowed_sandbox(self, providers: SimpleNamespace) -> None:
        providers.install.side_effect = RuntimeError("no wheel for this platform")

        with pytest.raises(RuntimeError, match="no wheel"):
            await _agent()._seed_agent_session_state(_seed())

        providers.sandbox.disconnect.assert_awaited_once()
        providers.sandbox.upload.assert_not_awaited()

    async def test_unfinished_cleanup_keeps_the_state_for_close(self, providers: SimpleNamespace) -> None:
        providers.install.side_effect = RuntimeError("install failed")
        providers.sandbox.disconnect.side_effect = ConnectionError("provider unreachable")

        with pytest.raises(AgentSessionSetupError) as error:
            await _agent()._seed_agent_session_state(_seed())

        assert isinstance(error.value.state, Terminus2SandboxSession)
        assert str(error.value.error) == "install failed"

    async def test_a_seed_without_sandbox_access_starts_and_later_stops_its_own(
        self, providers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = MagicMock(return_value=providers.sandbox)
        monkeypatch.setattr(f"{APP}.AsyncSandbox", factory)
        providers.metadata.return_value = {"sandbox-api": "opensandbox-sdk", "team": "provider-default"}
        sandbox_config = {"image": "python:3.13-slim", "workdir": "/work", "metadata": {"team": "configured"}}
        agent = _agent(
            sandbox_provider="sandbox",
            sandbox_config=sandbox_config,
            sandbox_install_timeout_seconds=100,
            timeout=200,
        )
        seed = _seed(borrowed=False)

        state = await agent._seed_agent_session_state(seed)

        factory.assert_called_once_with(providers.provider)
        providers.metadata.assert_called_once_with("sandbox", {"runtime": {}})
        providers.sandbox.start.assert_awaited_once_with(
            SandboxSpec(
                image="python:3.13-slim",
                workdir="/work",
                # Both setup steps (runtime and tmux install), the run, and ten minutes for cleanup.
                ttl_s=1000,
                # The provider's defaults, then the configured metadata, then this session's labels.
                metadata={
                    "sandbox-api": "opensandbox-sdk",
                    "team": "configured",
                    "gym-agent": "terminus_2",
                    "gym-session": "session",
                    "gym-rollout": seed.episode_id.capture_key,
                },
            )
        )
        assert agent.config.sandbox_config == sandbox_config
        assert state.session.owns_sandbox is True and state.provider == "sandbox"
        await asyncio.wait_for(agent._close_agent_session_state(state), 10)
        providers.sandbox.stop.assert_awaited_once()
        providers.sandbox.disconnect.assert_not_awaited()

    @pytest.mark.parametrize("borrowed", [True, False], ids=["borrowed", "owned"])
    async def test_setup_creates_the_workdir_only_in_its_own_sandbox(
        self, providers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, borrowed: bool
    ) -> None:
        factory = MagicMock(return_value=providers.sandbox)
        factory.connect = providers.connect
        monkeypatch.setattr(f"{APP}.AsyncSandbox", factory)
        agent = _agent(sandbox_provider="sandbox", sandbox_config={"image": "i", "workdir": "/work"})

        state = await agent._seed_agent_session_state(_seed(borrowed=borrowed, workdir="/work"))

        prepare = providers.sandbox.exec.await_args_list[0]
        directory = state.session.session_dir
        # A borrowed sandbox's task directory belongs to the Resources server that lent it.
        expected = [f"{directory}/harness", f"{directory}/logs", *([] if borrowed else ["/work"])]
        assert prepare.args[0].split() == ["mkdir", "-p", *expected]
        assert prepare.kwargs["cwd"] == "/"

    async def test_a_failed_install_stops_an_owned_sandbox(
        self, providers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f"{APP}.AsyncSandbox", MagicMock(return_value=providers.sandbox))
        providers.install.side_effect = RuntimeError("no wheel")
        agent = _agent(sandbox_provider="sandbox", sandbox_config={"image": "python:3.13-slim", "workdir": "/work"})

        with pytest.raises(RuntimeError, match="no wheel"):
            await agent._seed_agent_session_state(_seed(borrowed=False))

        providers.sandbox.start.assert_awaited_once()
        providers.sandbox.stop.assert_awaited_once()
        providers.sandbox.disconnect.assert_not_awaited()

    async def test_an_owned_sandbox_that_fails_to_start_closes_the_provider(
        self, providers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        providers.sandbox.start.side_effect = RuntimeError("image pull failed")
        monkeypatch.setattr(f"{APP}.AsyncSandbox", MagicMock(return_value=providers.sandbox))
        agent = _agent(sandbox_provider="sandbox", sandbox_config={"image": "python:3.13-slim", "workdir": "/work"})

        with pytest.raises(RuntimeError, match="image pull failed"):
            await agent._seed_agent_session_state(_seed(borrowed=False))

        providers.provider.aclose.assert_awaited_once()

    async def test_a_configured_ttl_is_kept(self, providers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(f"{APP}.AsyncSandbox", MagicMock(return_value=providers.sandbox))
        agent = _agent(sandbox_provider="sandbox", sandbox_config={"image": "i", "workdir": "/work", "ttl_s": 42})

        await agent._seed_agent_session_state(_seed(borrowed=False))

        assert providers.sandbox.start.await_args.args[0].ttl_s == 42

    async def test_a_loopback_model_url_is_refused_before_any_sandbox_work(
        self, providers: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Terminus2Agent, "resolve_model_base_url", lambda self, name, rollout_id: "http://127.0.0.1:8000/v1"
        )
        providers.resolve.return_value = {"opensandbox": {}}

        with pytest.raises(ValueError, match="loopback"):
            await _agent()._seed_agent_session_state(_seed())

        providers.connect.assert_not_awaited()
        providers.sandbox.exec.assert_not_awaited()


def _output(**kwargs) -> dict:
    """A runner result for a run that sent one command and confirmed completion."""
    trajectory = {
        "steps": [
            {"source": "user", "message": "task"},
            {
                "source": "agent",
                "message": '{"commands": [{"keystrokes": "ls\\n", "duration": 0.1}]}',
                "tool_calls": [
                    {
                        "tool_call_id": "call_0_1",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": "ls\n", "duration": 0.1},
                    }
                ],
                "observation": {"results": [{"content": "file.txt"}]},
            },
        ],
        "final_metrics": {"total_prompt_tokens": 10, "total_completion_tokens": 4},
    }
    return {
        "stop_reason": "completed",
        "trajectory": trajectory,
        "context": {"n_input_tokens": 10, "n_output_tokens": 4, "metadata": {"n_episodes": 2}},
        "finished_naturally": True,
        "context_length_exceeded": False,
        "error": None,
        "error_type": None,
        "error_kind": None,
        "error_status": None,
        "terminal_response_id": "chatcmpl-2",
        "response_ids": ["chatcmpl-1", "chatcmpl-2"],
        "calls_without_usage": 0,
        "compactions": [],
        **kwargs,
    }


_BODY = NeMoGymResponseCreateParamsNonStreaming(input="solve it", metadata={"agent_timeout_sec": "60"})


class _Activation:
    """A seeded session whose sandbox execution returns ``output`` without a sandbox."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output: dict | None = None,
        *,
        deadline_receipt: bool | None = None,
        **config,
    ) -> None:
        """``deadline_receipt`` is whether the supervisor's receipt records its deadline; by default only a result
        that ended at its wall time has one."""
        self.agent = _agent(**config)
        monkeypatch.setattr(Terminus2Agent, "resolve_model_base_url", _model_url)
        self.seed = _seed()
        sandbox = _sandbox()
        self.state = Terminus2SandboxSession(
            request=self.seed,
            session=SandboxSession(sandbox=sandbox, workdir="/app", session_dir="/tmp/session", harness="Terminus 2"),
            sandbox_id="sandbox-1",
            provider="runtime",
        )
        self.release = asyncio.Event()
        self.release.set()
        self.payloads: list[dict] = []

        async def execute(payload: dict, *, timeout: float, close_timeout: float) -> dict:
            self.payloads.append({**payload, "timeout": timeout})
            await self.release.wait()
            result = output if output is not None else _output()
            timed_out = result.get("stop_reason") == "wall_time" if deadline_receipt is None else deadline_receipt
            if timed_out:
                self.state.session.cleanup = {
                    "timed_out": True,
                    "return_code": 143,
                    "cleanup_confirmed": True,
                    "error": None,
                }
            return result

        self.state.execute = execute
        self.agent._session_records["session"] = _AgentSessionRecord(state=self.state, episode_id=self.seed.episode_id)
        self.request = SimpleNamespace(
            session={"agent_session_id": "session"}, path_params={"rollout_id": self.seed.episode_id.capture_key}
        )

    async def respond(self, body: NeMoGymResponseCreateParamsNonStreaming | None = None, request=None):
        return await asyncio.wait_for(self.agent.responses(request or self.request, body or _BODY), 30)


class TestActivation:
    async def test_a_session_runs_once_and_identical_retries_join(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch)
        activation.release.clear()
        body = NeMoGymResponseCreateParamsNonStreaming(input="solve it")

        first = asyncio.create_task(activation.respond(body))
        await asyncio.sleep(0)
        retry = asyncio.create_task(activation.respond(body))
        with pytest.raises(HTTPException, match="support one activation") as different:
            await activation.respond(NeMoGymResponseCreateParamsNonStreaming(input="another task"))
        activation.release.set()

        assert (await asyncio.wait_for(first, 10)).id == (await asyncio.wait_for(retry, 10)).id == "chatcmpl-2"
        assert different.value.status_code == 409
        assert len(activation.payloads) == 1

    @pytest.mark.parametrize("path_params", [{}, {"rollout_id": "other"}])
    async def test_an_activation_must_name_the_seeded_rollout(
        self, monkeypatch: pytest.MonkeyPatch, path_params: dict
    ) -> None:
        activation = _Activation(monkeypatch)
        request = SimpleNamespace(session={"agent_session_id": "session"}, path_params=path_params)

        with pytest.raises(HTTPException) as error:
            await activation.respond(request=request)

        assert error.value.status_code == 409
        assert not activation.payloads

    async def test_an_unknown_session_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch)
        request = SimpleNamespace(session={"agent_session_id": "missing"}, path_params=activation.request.path_params)

        with pytest.raises(HTTPException) as error:
            await activation.respond(request=request)

        assert error.value.status_code == 409

    async def test_the_runner_gets_the_rollout_model_route_and_the_task_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        activation = _Activation(monkeypatch, timeout=600, max_turns=7, temperature=0.3)

        await activation.respond()

        [payload] = activation.payloads
        assert payload["model"] == "test-model"
        # The runner calls the Model Server through the seeded rollout's route, so its calls join this rollout.
        assert payload["model_base_url"] == f"http://model:8000/ng-rollout/{activation.seed.episode_id.capture_key}/v1"
        assert payload["instruction"] == "solve it"
        # The task's agent timeout lowers the configured ceiling.
        assert payload["timeout"] == 60
        assert payload["chat_params"]["temperature"] == 0.3
        assert payload["terminus"]["max_turns"] == 7
        assert payload["invocation_id"] == activation.state.invocation_id

    async def test_a_completed_run_returns_the_trajectory_and_its_evidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        activation = _Activation(monkeypatch)

        response = await activation.respond()
        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        assert response.status == "completed"
        assert response.metadata["harness_execution"] == "sandbox"
        assert response.metadata["stop_reason"] == "completed"
        assert [item.type for item in response.output] == ["message", "function_call", "function_call_output"]
        assert response.usage.total_tokens == 14
        records = closed.agent_observations.records
        [invocation] = [record for record in records if isinstance(record, AgentInvocation)]
        assert invocation.status == "completed"
        assert [call.response_id for call in invocation.model_calls] == ["chatcmpl-1", "chatcmpl-2"]
        assert [item.type for item in invocation.conversation] == [item.type for item in response.output]
        [tool] = [record for record in records if isinstance(record, ToolCallObservation)]
        assert (tool.tool_call_id, tool.tool_name, tool.status) == ("call_0_1", "bash_command", "completed")
        [sandbox] = [record for record in records if isinstance(record, SandboxObservation)]
        assert (sandbox.outcome, sandbox.sandbox_id, sandbox.provider) == ("completed", "sandbox-1", "runtime")

    @pytest.mark.parametrize(
        ("result", "status", "outcome", "sandbox_outcome"),
        [
            (
                {"stop_reason": "error", "error_kind": "model_request_error", "error_status": 429},
                "failed",
                {"failure_kind": "model_request_error", "status_code": 429},
                "failed",
            ),
            (
                {"stop_reason": "error", "error_kind": "model_transport_error"},
                "failed",
                {"failure_kind": "model_transport_error"},
                "failed",
            ),
            # A model error without an HTTP status is not a request the model refused.
            (
                {"stop_reason": "error", "error_kind": "model_request_error"},
                "failed",
                {"failure_kind": "unknown"},
                "failed",
            ),
            (
                {"stop_reason": "error", "error_kind": "output_limit"},
                "failed",
                {"failure_kind": "agent_invalid_output"},
                "failed",
            ),
            ({"stop_reason": "error", "error_kind": "harness_error"}, "failed", {"failure_kind": "unknown"}, "failed"),
            ({"stop_reason": "error", "error_kind": "context_limit"}, "incomplete", None, "completed"),
            ({"stop_reason": "wall_time", "finished_naturally": False}, "incomplete", None, "timeout"),
            ({"stop_reason": "max_turns", "finished_naturally": False}, "incomplete", None, "completed"),
            ({"stop_reason": "cancelled", "finished_naturally": False}, "incomplete", None, "cancelled"),
            ({"stop_reason": "terminal_exited", "finished_naturally": False}, "incomplete", None, "completed"),
            # A stop reason this adapter does not know is a failure, never a graded ending.
            ({"stop_reason": "something_new"}, "failed", {"failure_kind": "unknown"}, "failed"),
        ],
    )
    async def test_how_a_run_ended_reaches_the_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        result: dict,
        status: str,
        outcome: dict | None,
        sandbox_outcome: str,
    ) -> None:
        failed = result["stop_reason"] == "error" or status == "failed"
        error = {"error": "Boom", "error_type": "HTTPStatusError"} if failed else {}
        activation = _Activation(monkeypatch, _output(**result, **error))

        response = await activation.respond()
        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        assert response.status == status
        expected_stop = "context_limit" if result.get("error_kind") == "context_limit" else result["stop_reason"]
        assert response.metadata["stop_reason"] == expected_stop
        reported = response.metadata.get("ng_agent_outcome")
        assert (json.loads(reported) if reported else None) == outcome
        assert (response.error is not None) == (status == "failed")
        [sandbox] = [record for record in closed.agent_observations.records if isinstance(record, SandboxObservation)]
        assert sandbox.outcome == sandbox_outcome

    async def test_an_unfinished_turn_of_a_failed_run_is_recorded_as_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = _output(stop_reason="error", error_kind="harness_error", error="Boom", finished_naturally=False)
        output["trajectory"]["steps"][1]["observation"] = {}
        activation = _Activation(monkeypatch, output)

        await activation.respond()
        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        [tool] = [record for record in closed.agent_observations.records if isinstance(record, ToolCallObservation)]
        assert tool.status == "failed"

    async def test_the_request_instructions_join_the_instruction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch, system_prompt="You are Terminus.")
        body = NeMoGymResponseCreateParamsNonStreaming(
            instructions="Answer in the file answer.txt.",
            input=[
                {"role": "system", "content": "Work in /app."},
                {"role": "user", "content": "solve it"},
            ],
        )

        await activation.respond(body)

        [payload] = activation.payloads
        assert payload["instruction"] == (
            "You are Terminus.\n\nAnswer in the file answer.txt.\n\nWork in /app.\n\nsolve it"
        )
        # Terminus builds its own messages from the instruction; none reach the model through the chat params.
        assert "messages" not in payload["chat_params"]

    async def test_the_request_tools_stay_out_of_the_model_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch)
        # A Terminal-Bench row: tool fields Terminus never uses and task metadata.
        body = NeMoGymResponseCreateParamsNonStreaming(
            input="solve it",
            tools=[
                {"type": "function", "name": "f", "parameters": {"type": "object", "properties": {}}, "strict": True}
            ],
            tool_choice="auto",
            parallel_tool_calls=True,
            metadata={"agent_timeout_sec": "60", "instance_id": "task-1", "docker_image": "registry/task:1"},
        )

        response = await activation.respond(body)

        [payload] = activation.payloads
        assert not {"tools", "tool_choice", "parallel_tool_calls"} & payload["chat_params"].keys()
        assert "max_tokens" not in payload["chat_params"]
        assert response.status == "completed"

    async def test_an_output_budget_is_refused_without_binding_the_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        activation = _Activation(monkeypatch, model_info={"max_output_tokens": 8192})
        body = NeMoGymResponseCreateParamsNonStreaming(
            input="solve it", max_output_tokens=50, temperature=0.2, top_p=0.9, reasoning={"effort": "low"}
        )

        with pytest.raises(HTTPException, match="max_output_tokens") as refused:
            await activation.respond(body)

        assert refused.value.status_code == 422
        assert activation.state.task is None and activation.state.activation_request is None
        assert not activation.payloads
        activation.state.session.sandbox.exec.assert_not_awaited()

        corrected = body.model_copy(update={"max_output_tokens": None})
        response = await activation.respond(corrected)

        assert response.status == "completed"
        [payload] = activation.payloads
        assert "max_tokens" not in payload["chat_params"]
        assert payload["chat_params"]["temperature"] == 0.2
        assert payload["chat_params"]["top_p"] == 0.9
        assert payload["chat_params"]["reasoning_effort"] == "low"
        assert payload["terminus"]["model_info"] == {"max_output_tokens": 8192}
        assert activation.state.activation_request == corrected

    @pytest.mark.parametrize("field", [{"previous_response_id": "resp_1"}, {"truncation": "auto"}])
    async def test_a_request_the_converter_cannot_represent_is_refused_before_any_sandbox_work(
        self, monkeypatch: pytest.MonkeyPatch, field: dict
    ) -> None:
        activation = _Activation(monkeypatch)

        with pytest.raises(HTTPException) as refused:
            await activation.respond(NeMoGymResponseCreateParamsNonStreaming(input="solve it", **field))

        assert refused.value.status_code == 422
        assert refused.value.detail.startswith("Terminus 2 cannot run this request:")
        assert activation.state.task is None and activation.state.activation_request is None
        assert not activation.payloads
        activation.state.session.sandbox.exec.assert_not_awaited()

    async def test_a_result_that_does_not_validate_is_an_evidence_gap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The runner's result file is writable by task commands; a malformed one must not fail the activation.
        activation = _Activation(monkeypatch, _output(compactions=[1]))

        response = await activation.respond()
        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        assert response.status == "completed"
        [gap] = closed.agent_observations.gaps
        assert (gap.code, gap.invocation_id, gap.detail) == (
            "observation_capture_failed",
            activation.state.invocation_id,
            "AttributeError",
        )
        assert not closed.agent_observations.records

    async def test_a_timed_out_command_batch_is_recorded_as_timed_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = _output(stop_reason="wall_time", finished_naturally=False)
        output["trajectory"]["steps"][1]["observation"] = {}
        activation = _Activation(monkeypatch, output)

        await activation.respond()
        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        [tool] = [record for record in closed.agent_observations.records if isinstance(record, ToolCallObservation)]
        [sandbox] = [record for record in closed.agent_observations.records if isinstance(record, SandboxObservation)]
        assert (tool.status, sandbox.outcome) == ("timeout", "timeout")

    async def test_every_command_of_a_turn_with_a_result_ran(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = _output()
        # Terminus records one terminal result for a turn's whole command batch.
        output["trajectory"]["steps"][1]["tool_calls"].append(
            {
                "tool_call_id": "call_0_2",
                "function_name": "bash_command",
                "arguments": {"keystrokes": "pwd\n", "duration": 0.1},
            }
        )
        activation = _Activation(monkeypatch, output)

        await activation.respond()
        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        tools = [record for record in closed.agent_observations.records if isinstance(record, ToolCallObservation)]
        assert [(tool.tool_call_id, tool.status) for tool in tools] == [
            ("call_0_1", "completed"),
            ("call_0_2", "completed"),
        ]

    async def test_usage_is_unknown_when_a_model_response_had_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch, _output(calls_without_usage=1))

        response = await activation.respond()
        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        assert response.usage is None
        assert "model_usage_unavailable" in {gap.code for gap in closed.agent_observations.gaps}

    async def test_usage_is_unknown_when_the_result_has_no_usage_accounting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The runner's report of its own crash carries no usage counts, though model calls may have run.
        output = _output(stop_reason="error", error="Boom", error_type="RuntimeError", error_kind="harness_error")
        del output["calls_without_usage"]
        activation = _Activation(monkeypatch, output)

        response = await activation.respond()
        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        assert response.usage is None
        [gap] = [gap for gap in closed.agent_observations.gaps if gap.code == "model_usage_unavailable"]
        assert gap.detail == "the runner reported no usage accounting"

    async def test_a_sigterm_from_the_task_is_no_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The runner saved `wall_time` on a SIGTERM, but the supervisor's receipt shows no deadline: a task command
        # such as `pkill python3` sent it.
        activation = _Activation(
            monkeypatch, _output(stop_reason="wall_time", finished_naturally=False), deadline_receipt=False
        )

        response = await activation.respond()
        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        assert (response.status, response.metadata["stop_reason"], response.error) == (
            "incomplete",
            "terminated",
            None,
        )
        assert '"timed_out": false' in response.metadata["terminus_2"]
        [invocation] = [record for record in closed.agent_observations.records if isinstance(record, AgentInvocation)]
        assert (invocation.status, invocation.stop_reason, invocation.error_type) == ("incomplete", "terminated", None)
        [sandbox] = [record for record in closed.agent_observations.records if isinstance(record, SandboxObservation)]
        assert sandbox.outcome == "completed"

    async def test_a_provider_timeout_keeps_partial_work_after_a_confirmed_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        activation = _Activation(monkeypatch)

        async def overran(payload: dict, *, timeout: float, close_timeout: float) -> dict:
            activation.state.session.artifacts = _output(stop_reason="completed")
            activation.state.session.cleanup = {
                "timed_out": True,
                "return_code": 143,
                "cleanup_confirmed": True,
                "error": None,
            }
            raise TimeoutError("supervisor exceeded its deadline")

        activation.state.execute = overran
        clock = iter([0.0, 61.0, 61.0])
        monkeypatch.setattr(f"{APP}.monotonic", lambda: next(clock))

        response = await activation.respond()

        assert response.metadata["stop_reason"] == "wall_time"
        assert response.status == "incomplete"

    async def test_an_early_provider_timeout_stays_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch)

        async def lost(payload: dict, *, timeout: float, close_timeout: float) -> dict:
            activation.state.session.artifacts = _output()
            raise TimeoutError("transport timed out")

        activation.state.execute = lost

        with pytest.raises(TimeoutError, match="transport"):
            await activation.respond()

    async def test_waiting_for_a_run_slot_does_not_count_toward_the_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        activation = _Activation(monkeypatch)
        now = [0.0]
        monkeypatch.setattr(f"{APP}.monotonic", lambda: now[0])

        async def transport_timeout(payload: dict, *, timeout: float, close_timeout: float) -> dict:
            # Close stopped the runner after the provider's transport timed out, and the runner saved its turns.
            activation.state.session.artifacts = _output(stop_reason="cancelled", finished_naturally=False)
            raise TimeoutError("provider transport timed out")

        activation.state.execute = transport_timeout
        # Another session's run holds the only slot for longer than this task's 60-second budget.
        await activation.agent.sem.acquire()
        running = asyncio.create_task(activation.respond())
        for _ in range(5):
            await asyncio.sleep(0)
        now[0] = 100.0
        activation.agent.sem.release()

        # The run itself took no time, so its transport timeout is a failure, not the end of its budget.
        with pytest.raises(TimeoutError, match="transport"):
            await asyncio.wait_for(running, 10)


class TestClose:
    async def test_close_before_activation_reports_a_gap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch)

        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        assert [gap.code for gap in closed.agent_observations.gaps] == ["observation_capture_failed"]
        activation.state.session.sandbox.disconnect.assert_awaited_once()
        activation.state.session.sandbox.stop.assert_not_awaited()

    async def test_close_converts_a_result_captured_before_the_activation_was_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        activation = _Activation(monkeypatch)
        activation.release.clear()

        async def captured(payload: dict, *, timeout: float, close_timeout: float) -> dict:
            activation.state.session.artifacts = _output(stop_reason="cancelled", finished_naturally=False)
            await activation.release.wait()
            return _output()

        activation.state.execute = captured
        running = asyncio.create_task(activation.respond())
        await asyncio.sleep(0)

        closed = await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        assert running.cancelled()
        [invocation] = [record for record in closed.agent_observations.records if isinstance(record, AgentInvocation)]
        assert (invocation.status, invocation.stop_reason) == ("incomplete", "cancelled")
        assert [call.response_id for call in invocation.model_calls] == ["chatcmpl-1", "chatcmpl-2"]

    async def test_close_releases_the_sandbox_before_cancelling_a_running_activation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        activation = _Activation(monkeypatch)
        activation.release.clear()
        order: list[str] = []
        release = activation.state.session.close

        async def close(*, timeout: float) -> None:
            order.append(f"session closed, activation done={running.done()}")
            await release(timeout=timeout)

        activation.state.session.close = close
        running = asyncio.create_task(activation.respond())
        await asyncio.sleep(0)

        await asyncio.wait_for(activation.agent._close_agent_session_state(activation.state), 10)

        assert order == ["session closed, activation done=False"]
        assert running.cancelled()

    async def test_close_waits_for_a_setup_still_installing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch)
        installed = asyncio.Event()

        async def install() -> None:
            await installed.wait()

        activation.state.installation_task = asyncio.create_task(install())
        closing = asyncio.create_task(activation.agent._close_agent_session_state(activation.state))
        await asyncio.sleep(0)
        assert not closing.done()
        activation.state.session.sandbox.disconnect.assert_not_awaited()

        installed.set()
        await asyncio.wait_for(closing, 10)

        activation.state.session.sandbox.disconnect.assert_awaited_once()

    def test_rollout_stops_reach_the_sandbox_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch)

        assert activation.agent._rollout_sandbox_session(activation.state) is activation.state.session

    async def test_an_activation_after_close_is_refused(self, wired: SimpleNamespace) -> None:
        agent = _agent()
        seed = _seed()
        request = _request(seed)
        await asyncio.wait_for(agent.seed_agent_session(request, seed), 10)
        await asyncio.wait_for(
            agent.close_agent_session(
                request, AgentCloseSessionRequest(agent_session_id="session", episode_id=seed.episode_id)
            ),
            10,
        )

        with pytest.raises(HTTPException) as refused:
            await asyncio.wait_for(agent.responses(request, _BODY), 10)

        assert refused.value.status_code == 409
        assert wired.sandbox.launches == 0


class FakeSandbox:
    """Files in memory below the real SandboxSession; the supervised launch plays the runner.

    ``runner`` decides what the launched runner does: "complete" saves a result and exits; "dies_early" is ended
    by a rollout stop or its deadline before it installs its SIGTERM handler, so it saves nothing; "crashes" exits
    on its own without saving anything; "blocks" runs until a stop or
    expiry marker arrives, then saves its turns as the real runner's handler does.
    """

    sandbox_id = "sandbox-1"
    uploads_independent_of_execs = True

    def __init__(self, runner: str = "complete") -> None:
        self.files: dict[str, str] = {}
        self.runner = runner
        self.launched = asyncio.Event()
        self.ended = asyncio.Event()
        self.launches = 0
        self.stop_requests: set[str] = set()
        # Set to make the check for a missing output.json fail, as an unreachable sandbox would.
        self.absence_check_error: Exception | None = None
        self.disconnect = AsyncMock()
        self.stop = AsyncMock()

    async def upload(self, source, destination) -> None:
        self.files[str(destination)] = Path(source).read_text()
        if str(destination).endswith("/expire.request"):
            self.ended.set()

    async def download(self, source, destination) -> None:
        if source not in self.files:
            raise FileNotFoundError(source)
        Path(destination).write_text(self.files[source])

    async def exec(self, command: str, **kwargs) -> SandboxExecResult:
        if "process_supervisor.sh" in command and "--receipt" in command:
            return await self._launch(command)
        if "echo absent" in command:
            if self.absence_check_error is not None:
                raise self.absence_check_error
            target = command.split("[ -e ", 1)[1].split(" ]", 1)[0].strip("'")
            return SandboxExecResult(stdout="" if target in self.files else "absent\n", stderr="", return_code=0)
        if "touch " in command and "stop.request" in command:
            # Close writes the supervisor's stop file and, like the real script, waits a bounded time for the receipt.
            receipt = command.split("[ -f ", 1)[1].split(" ]", 1)[0].strip("'")
            self.stop_requests.add(receipt.removesuffix("cleanup.json") + "stop.request")
            self.ended.set()
            for _ in range(100):
                if receipt in self.files:
                    break
                await asyncio.sleep(0.01)
            else:
                # Nothing launched: the stop wins the claim and writes the no-launch receipt.
                self.files[receipt] = json.dumps(
                    {"cleanup_confirmed": True, "return_code": None, "error": None, "timed_out": False}
                )
        return SandboxExecResult(stdout="tmux 3.4", stderr="", return_code=0)

    async def _launch(self, command: str) -> SandboxExecResult:
        directory = command.split("--receipt ", 1)[1].split(" ", 1)[0].strip("'").removesuffix("/cleanup.json")
        self.launches += 1
        self.launched.set()

        def receipt(**fields) -> None:
            base = {"cleanup_confirmed": True, "return_code": 0, "error": None, "timed_out": False}
            self.files[f"{directory}/cleanup.json"] = json.dumps({**base, **fields})

        if f"{directory}/expire.request" in self.files:
            # The supervisor never starts a harness whose expiry was requested before launch.
            receipt(return_code=None, timed_out=True)
        elif self.runner == "complete":
            self.files[f"{directory}/output.json"] = json.dumps(_output())
            receipt()
        else:
            await self.ended.wait()
            if self.runner == "blocks":
                # The runner's handler: close writes stop.request first (cancelled); an expiry does not (wall_time).
                reason = "cancelled" if f"{directory}/stop.request" in self.stop_requests else "wall_time"
                self.files[f"{directory}/output.json"] = json.dumps(
                    _output(stop_reason=reason, finished_naturally=False)
                )
                receipt(timed_out=True)
            elif self.runner == "crashes":
                # The runner fails on its own before it saves anything; the supervisor stopped nothing.
                receipt(return_code=1)
            else:  # dies_early: SIGTERM's default action, before the runner's handler exists, saves nothing.
                receipt(return_code=143, timed_out=True)
        return SandboxExecResult(stdout="", stderr="", return_code=0)


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Seeds connect to the FakeSandbox in ``sandbox``, which a test may replace; the runtime counts as installed."""
    holder = SimpleNamespace(sandbox=FakeSandbox(), provider=AsyncMock())

    async def connect(descriptor, *, provider):
        return holder.sandbox

    monkeypatch.setattr(f"{APP}.get_global_config_dict", lambda: {"runtime": {}})
    monkeypatch.setattr(f"{APP}.resolve_provider_config", lambda ref, config: {"docker": {}})
    monkeypatch.setattr(f"{APP}.create_provider", lambda config: holder.provider)
    monkeypatch.setattr(f"{APP}.AsyncSandbox.connect", connect)
    monkeypatch.setattr(Terminus2Agent, "resolve_model_base_url", _model_url)
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", AsyncMock(return_value=True))
    monkeypatch.setattr(SandboxPythonRuntime, "install", AsyncMock())
    return holder


def _request(seed: AgentSeedSessionRequest) -> Request:
    return Request(
        {"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": seed.episode_id.capture_key}}
    )


class TestSandboxLifecycle:
    """The real stage, supervised launch, receipt, collection and close, over an in-memory sandbox."""

    @staticmethod
    async def _seeded(wired: SimpleNamespace) -> tuple[Terminus2Agent, AgentSeedSessionRequest, Request]:
        agent = _agent(session_close_timeout_seconds=3)
        seed = _seed()
        request = _request(seed)
        await asyncio.wait_for(agent.seed_agent_session(request, seed), 10)
        return agent, seed, request

    @staticmethod
    async def _close(agent: Terminus2Agent, seed: AgentSeedSessionRequest, request: Request):
        return await asyncio.wait_for(
            agent.close_agent_session(
                request, AgentCloseSessionRequest(agent_session_id="session", episode_id=seed.episode_id)
            ),
            10,
        )

    async def test_a_run_goes_through_the_supervised_lifecycle(self, wired: SimpleNamespace) -> None:
        agent, seed, request = await self._seeded(wired)

        response = await asyncio.wait_for(agent.responses(request, _BODY), 30)
        closed = await self._close(agent, seed, request)

        assert (response.status, response.metadata["stop_reason"]) == ("completed", "completed")
        staged = json.loads(next(text for path, text in wired.sandbox.files.items() if path.endswith("/input.json")))
        assert staged["workdir"] == "/app"
        assert staged["model_base_url"] == f"http://model:8000/ng-rollout/{seed.episode_id.capture_key}/v1"
        [invocation] = [record for record in closed.agent_observations.records if isinstance(record, AgentInvocation)]
        assert invocation.status == "completed"
        # Terminus uses no Resources tools, so close hands no Resources cookies back.
        assert closed.resources_cookies is None
        wired.sandbox.disconnect.assert_awaited_once()
        wired.sandbox.stop.assert_not_awaited()

    async def test_a_rollout_stop_during_seed_ends_the_activation_as_a_timeout(self, wired: SimpleNamespace) -> None:
        agent, seed, request = await self._seeded(wired)
        # The supervisor's receipt alone shows that Terminus never started; nothing else is asked of the sandbox.
        wired.sandbox.absence_check_error = ConnectionError("sandbox unreachable")
        assert (await agent.stop_rollout(seed.episode_id.capture_key, RolloutStopRequest(reason="budget"))).stopped

        response = await asyncio.wait_for(agent.responses(request, _BODY), 30)

        # Terminus never ran: an empty run graded like a timeout, never an activation error that is retried.
        assert (response.status, response.metadata["stop_reason"], response.error) == ("incomplete", "wall_time", None)
        assert json.loads(response.metadata["ng_rollout_stop"]) == {"reason": "budget"}
        assert "ng_agent_outcome" not in response.metadata
        assert not any(path.endswith("/output.json") for path in wired.sandbox.files)

    async def test_a_rollout_stop_before_the_runner_saves_anything_ends_as_a_timeout(
        self, wired: SimpleNamespace
    ) -> None:
        # A stop that lands while the runner still imports Harbor leaves no output.json; it is still a stop.
        wired.sandbox = FakeSandbox(runner="dies_early")
        agent, seed, request = await self._seeded(wired)
        running = asyncio.create_task(agent.responses(request, _BODY))
        await asyncio.wait_for(wired.sandbox.launched.wait(), 10)

        assert (await agent.stop_rollout(seed.episode_id.capture_key, RolloutStopRequest(reason="budget"))).stopped
        response = await asyncio.wait_for(running, 30)

        assert (response.status, response.metadata["stop_reason"], response.error) == ("incomplete", "wall_time", None)
        assert json.loads(response.metadata["ng_rollout_stop"]) == {"reason": "budget"}

    async def test_a_deadline_before_the_runner_saves_anything_ends_as_a_timeout(self, wired: SimpleNamespace) -> None:
        wired.sandbox = FakeSandbox(runner="dies_early")
        agent, seed, request = await self._seeded(wired)
        running = asyncio.create_task(agent.responses(request, _BODY))
        await asyncio.wait_for(wired.sandbox.launched.wait(), 10)

        # The supervisor's own deadline ends the runner while it still imports Harbor; no rollout stop was sent.
        wired.sandbox.ended.set()
        response = await asyncio.wait_for(running, 30)

        assert (response.status, response.metadata["stop_reason"], response.error) == ("incomplete", "wall_time", None)
        assert "ng_rollout_stop" not in response.metadata

    async def test_a_runner_that_saved_nothing_without_a_stop_is_still_an_error(self, wired: SimpleNamespace) -> None:
        wired.sandbox = FakeSandbox(runner="crashes")
        agent, seed, request = await self._seeded(wired)
        running = asyncio.create_task(agent.responses(request, _BODY))
        await asyncio.wait_for(wired.sandbox.launched.wait(), 10)

        # The runner ends on its own, without a rollout stop, and leaves no result.
        wired.sandbox.ended.set()

        with pytest.raises(RuntimeError, match="exited without output"):
            await asyncio.wait_for(running, 30)

    async def test_a_rollout_stop_while_running_keeps_the_saved_turns(self, wired: SimpleNamespace) -> None:
        wired.sandbox = FakeSandbox(runner="blocks")
        agent, seed, request = await self._seeded(wired)
        running = asyncio.create_task(agent.responses(request, _BODY))
        await asyncio.wait_for(wired.sandbox.launched.wait(), 10)

        assert (await agent.stop_rollout(seed.episode_id.capture_key, RolloutStopRequest(reason="budget"))).stopped
        response = await asyncio.wait_for(running, 30)

        assert (response.status, response.metadata["stop_reason"]) == ("incomplete", "wall_time")
        assert json.loads(response.metadata["ng_rollout_stop"]) == {"reason": "budget"}
        assert "ng_agent_outcome" not in response.metadata
        assert [item.type for item in response.output] == ["message", "function_call", "function_call_output"]

    async def test_close_during_a_run_keeps_the_turns_the_runner_saved(self, wired: SimpleNamespace) -> None:
        wired.sandbox = FakeSandbox(runner="blocks")
        agent, seed, request = await self._seeded(wired)
        running = asyncio.create_task(agent.responses(request, _BODY))
        await asyncio.wait_for(wired.sandbox.launched.wait(), 10)

        closed = await self._close(agent, seed, request)

        # Close stopped the runner and captured its saved turns before it cancelled the activation.
        response = await asyncio.wait_for(running, 30)
        assert (response.status, response.metadata["stop_reason"]) == ("incomplete", "cancelled")
        assert "ng_rollout_stop" not in response.metadata
        assert "observation_capture_failed" not in {gap.code for gap in closed.agent_observations.gaps}
        [invocation] = [record for record in closed.agent_observations.records if isinstance(record, AgentInvocation)]
        assert (invocation.status, invocation.stop_reason) == ("incomplete", "cancelled")
        assert [call.response_id for call in invocation.model_calls] == ["chatcmpl-1", "chatcmpl-2"]
        wired.sandbox.disconnect.assert_awaited_once()


class TestRun:
    """``/run`` is the host path's compatibility endpoint; it needs a Resources server and no agent session."""

    @staticmethod
    def _agent(**kwargs) -> Terminus2Agent:
        agent = _agent(**kwargs)
        agent.server_client.post = AsyncMock(side_effect=AssertionError("a refused /run must not call any server"))
        return agent

    async def test_run_inside_an_agent_session_is_refused(self) -> None:
        agent = self._agent(resources_server=ResourcesServerRef(type="resources_servers", name="resources"))
        request = Request({"type": "http", "headers": [], "session": {"agent_session_id": "session"}})

        with pytest.raises(HTTPException) as refused:
            await agent.run(request, Terminus2AgentRunRequest(responses_create_params={"input": "task"}))

        assert refused.value.status_code == 409

    async def test_run_without_a_resources_server_is_refused(self) -> None:
        agent = self._agent()
        request = Request({"type": "http", "headers": [], "session": {}})

        with pytest.raises(HTTPException) as refused:
            await agent.run(request, Terminus2AgentRunRequest(responses_create_params={"input": "task"}))

        assert refused.value.status_code == 422
        assert "resources_server" in refused.value.detail

    async def test_run_with_an_output_budget_is_refused_before_seeding_resources(self) -> None:
        agent = self._agent(resources_server=ResourcesServerRef(type="resources_servers", name="resources"))
        request = Request({"type": "http", "headers": [], "session": {}})
        body = Terminus2AgentRunRequest(responses_create_params={"input": "task", "max_output_tokens": 32})

        with pytest.raises(HTTPException, match="max_output_tokens") as refused:
            await agent.run(request, body)

        assert refused.value.status_code == 422
        agent.server_client.post.assert_not_awaited()


class TestStageAndCollect:
    async def test_stage_uploads_the_input_and_runs_the_private_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        uploaded: dict[str, str] = {}

        async def upload_text(sandbox, *, path: str, text: str) -> None:
            uploaded[path] = text

        monkeypatch.setattr(terminus_sandbox, "upload_text", upload_text)
        state = _Activation(monkeypatch).state

        command = await state.stage_activation({"instruction": "solve it"})

        staged = json.loads(uploaded["/tmp/session/input.json"])
        assert staged == {
            "instruction": "solve it",
            "logs_dir": "/tmp/session/logs",
            "stop_request_path": state.session.stop_request_path,
            "workdir": "/app",
        }
        assert command.argv == [
            terminus_runtime().python,
            "-E",
            "-s",
            "/tmp/session/harness/sandbox_runner.py",
            "/tmp/session/input.json",
            "/tmp/session/output.json",
        ]

    async def test_a_runner_without_output_reports_its_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(terminus_sandbox, "read_text", AsyncMock(side_effect=FileNotFoundError("output.json")))
        state = _Activation(monkeypatch).state
        state.session.read_output_log = AsyncMock(return_value="ModuleNotFoundError: harbor")

        with pytest.raises(RuntimeError, match="exited without output: ModuleNotFoundError: harbor"):
            await state.collect_artifacts()


class TestRuntime:
    def test_the_runtime_is_built_from_this_servers_harbor(self) -> None:
        runtime = terminus_runtime()

        [requirement] = runtime.requirements
        assert requirement.startswith("harbor")
        # Harbor's sandbox backends and CLI never load in Terminus-2, so they stay out of the runtime.
        assert "e2b; sys_platform == 'never'" in runtime.overrides
        assert runtime.directory.startswith("/tmp/nemo-gym-terminus-2-runtime-")

    @pytest.mark.parametrize(
        ("direct_url", "requirement"),
        [
            (
                {"url": "https://github.com/org/harbor.git", "vcs_info": {"vcs": "git", "commit_id": "abc"}},
                "harbor @ https://github.com/org/harbor/archive/abc.tar.gz",
            ),
            (None, "harbor==1.2.3"),
        ],
    )
    def test_the_harbor_requirement_follows_its_installation(
        self, monkeypatch: pytest.MonkeyPatch, direct_url: dict | None, requirement: str
    ) -> None:
        distribution = MagicMock(version="1.2.3")
        distribution.read_text.return_value = json.dumps(direct_url) if direct_url else None
        monkeypatch.setattr(terminus_sandbox.importlib.metadata, "distribution", lambda name: distribution)

        assert terminus_sandbox._sandbox_harbor_requirement() == requirement

    def test_a_git_install_outside_github_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        distribution = MagicMock()
        distribution.read_text.return_value = json.dumps(
            {"url": "https://gitlab.example/harbor.git", "vcs_info": {"commit_id": "abc"}}
        )
        monkeypatch.setattr(terminus_sandbox.importlib.metadata, "distribution", lambda name: distribution)

        with pytest.raises(RuntimeError, match="Cannot build a sandbox install URL"):
            terminus_sandbox._sandbox_harbor_requirement()


@pytest.mark.parametrize("provider_config", [{"unknown_provider": {}}, {"local": 17}])
@pytest.mark.parametrize("borrowed", [True, False])
def test_invalid_provider_configuration_is_a_final_seed_refusal(providers, monkeypatch, provider_config, borrowed):
    from nemo_gym.sandbox import create_provider

    agent = _agent(sandbox_provider="runtime", sandbox_config={"workdir": "/app"})
    providers.resolve.return_value = provider_config
    monkeypatch.setattr(f"{APP}.create_provider", create_provider)
    reason = check_agent_seed_rejected(agent.setup_webserver(), _seed(borrowed=borrowed))
    assert "provider" in reason.lower()
    providers.connect.assert_not_awaited()
    providers.install.assert_not_awaited()
    assert not agent._session_records


@pytest.mark.parametrize("typed_cleanup", [True, False])
async def test_connect_failure_survives_cleanup_failure(providers, typed_cleanup):
    from nemo_gym.core import NgFailureEvent, RequestFailureLog, failure_report_for

    original = NgFailureEvent(
        boundary="sandbox",
        failure_kind="provider_unavailable",
        failure_reason="connect failed",
        operation="connect",
        stage="seed",
        event_id="original-connect-event",
    )
    cleanup = (
        NgFailureEvent(
            boundary="sandbox",
            failure_kind="session_release_failed",
            failure_reason="cleanup failed",
            operation="aclose",
            stage="cleanup",
            event_id="cleanup-event",
        )
        if typed_cleanup
        else RuntimeError("cleanup failed")
    )
    providers.connect.side_effect = original
    providers.provider.aclose.side_effect = cleanup
    with RequestFailureLog.capture() as log:
        with pytest.raises(NgFailureEvent) as caught:
            await _agent()._seed_agent_session_state(_seed())
        assert caught.value is original
        report = failure_report_for(caught.value)
        event = report.events[report.failure_id]
        assert (event.event_id, event.stage, event.operation) == ("original-connect-event", "seed", "connect")
        (cleanup_event,) = log.report.events.values()
        assert (cleanup_event.stage, cleanup_event.operation) == ("cleanup", "aclose")
        if typed_cleanup:
            assert cleanup_event.event_id == "cleanup-event"
    providers.provider.aclose.assert_awaited_once()


@pytest.mark.parametrize("cached_known", [True, False])
async def test_session_usage_keeps_unreported_details_unknown(monkeypatch, cached_known):
    output = _output(cached_usage_known=cached_known)
    output["trajectory"]["final_metrics"]["total_cached_tokens"] = 3
    activation = _Activation(monkeypatch, output)

    response = await activation.respond()
    closed = await activation.agent._close_agent_session_state(activation.state)

    assert response.usage is not None
    assert response.usage.input_tokens == output["trajectory"]["final_metrics"]["total_prompt_tokens"]
    assert response.usage.output_tokens == output["trajectory"]["final_metrics"]["total_completion_tokens"]
    assert response.usage.output_tokens_details.reasoning_tokens is None
    expected_cached = output["trajectory"]["final_metrics"]["total_cached_tokens"] if cached_known else None
    assert response.usage.input_tokens_details.cached_tokens == expected_cached
    gaps = {gap.code for gap in closed.agent_observations.gaps}
    assert "reasoning_token_usage_unavailable" in gaps
    assert ("cached_token_usage_unavailable" in gaps) is (not cached_known)


@pytest.mark.parametrize("settings", [{"metadata": 17}, {"unknown_setting": True}])
def test_invalid_owned_sandbox_spec_is_refused_before_provider_allocation(providers, monkeypatch, settings):
    agent = _agent(sandbox_provider="runtime", sandbox_config={"workdir": "/app", **settings})
    create = MagicMock()
    monkeypatch.setattr(f"{APP}.create_provider", create)
    assert check_agent_seed_rejected(agent.setup_webserver(), _seed(borrowed=False))
    create.assert_not_called()
    providers.connect.assert_not_awaited()
    assert not agent._session_records


@pytest.mark.parametrize("confirmed_deadline", [True, False])
async def test_provider_timeout_evidence_survives_only_confirmed_deadline_suppression(monkeypatch, confirmed_deadline):
    from nemo_gym.core import NgFailureEvent, RequestFailureLog

    activation = _Activation(monkeypatch)
    original = NgFailureEvent(
        boundary="sandbox", failure_kind="transport_timeout", failure_reason="provider wait expired", stage="agent"
    )
    failure = TimeoutError("provider timeout details")
    failure.__cause__ = original

    async def expired(payload, *, timeout, close_timeout):
        activation.state.session.artifacts = _output()
        activation.state.session.cleanup = {
            "timed_out": confirmed_deadline,
            "return_code": 143,
            "cleanup_confirmed": True,
            "error": None,
        }
        raise failure

    activation.state.execute = expired
    clock = iter([0.0, 61.0, 61.0])
    monkeypatch.setattr(f"{APP}.monotonic", lambda: next(clock))
    with RequestFailureLog.capture() as log:
        if not confirmed_deadline:
            with pytest.raises(TimeoutError) as caught:
                await activation.respond()
            assert caught.value is failure
            return
        response = await activation.respond()
        report = log.reply_report(None)
    assert response.status == "incomplete"
    assert report.failure_id is None
    wait = report.events[activation.state.provider_wait_failure.event_id]
    assert wait.foreign_error.message == "provider timeout details"
    assert wait.cause_id == original.event_id
    assert report.events[original.event_id].stage == "agent"
    activation.state.close = AsyncMock()
    with RequestFailureLog.capture() as close_log:
        await activation.agent._close_agent_session_state(activation.state)
    assert close_log.reply_report(None) == report
