# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions: Terminus-2 in a borrowed or owned task sandbox."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.agents import (
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    SandboxPythonRuntime,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.agents.base_responses_api_agent import _AgentSessionRecord
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess, SandboxExecResult, SandboxSpec
from nemo_gym.server_runtime import DirectHTTPToolAccess
from nemo_gym.trajectory import AgentInvocation, SandboxObservation, ToolCallObservation
from responses_api_agents.terminus_2_agent import sandbox as terminus_sandbox
from responses_api_agents.terminus_2_agent.app import Terminus2Agent, Terminus2AgentConfig
from responses_api_agents.terminus_2_agent.sandbox import Terminus2SandboxSession, terminus_runtime


APP = "responses_api_agents.terminus_2_agent.app"


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
        is_installed=AsyncMock(return_value=False),
        install=AsyncMock(),
    )
    monkeypatch.setattr(f"{APP}.get_global_config_dict", lambda: {"runtime": {}})
    monkeypatch.setattr(f"{APP}.resolve_provider_config", patched.resolve)
    monkeypatch.setattr(f"{APP}.create_provider", lambda config: patched.provider)
    monkeypatch.setattr(f"{APP}.AsyncSandbox.connect", patched.connect)
    monkeypatch.setattr(Terminus2Agent, "resolve_model_base_url", lambda *args: "http://model:8000/ng-rollout/r/v1")
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

    async def test_an_image_without_script_is_refused(self, providers: SimpleNamespace) -> None:
        sandbox = _sandbox(_ok(), SandboxExecResult(stdout="missing script (util-linux)\n", stderr="", return_code=3))
        providers.connect.return_value = sandbox

        with pytest.raises(AgentSeedRejectedError, match="missing script"):
            await _agent()._seed_agent_session_state(_seed())

        sandbox.disconnect.assert_awaited_once()

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
        agent = _agent(
            sandbox_provider="sandbox",
            sandbox_config={"image": "python:3.13-slim", "workdir": "/work"},
            sandbox_install_timeout_seconds=100,
            timeout=200,
        )

        state = await agent._seed_agent_session_state(_seed(borrowed=False))

        factory.assert_called_once_with(providers.provider)
        providers.sandbox.start.assert_awaited_once_with(
            SandboxSpec(image="python:3.13-slim", workdir="/work", ttl_s=900)
        )
        assert agent.config.sandbox_config == {"image": "python:3.13-slim", "workdir": "/work"}
        assert state.session.owns_sandbox is True and state.provider == "sandbox"
        await agent._close_agent_session_state(state)
        providers.sandbox.stop.assert_awaited_once()
        providers.sandbox.disconnect.assert_not_awaited()


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


class _Activation:
    """A seeded session whose sandbox execution returns ``output`` without a sandbox."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, output: dict | None = None, **config) -> None:
        self.agent = _agent(**config)
        monkeypatch.setattr(Terminus2Agent, "resolve_model_base_url", lambda *args: "http://model:1/ng-rollout/r/v1")
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
            return output if output is not None else _output()

        self.state.execute = execute
        self.agent._session_records["session"] = _AgentSessionRecord(state=self.state, episode_id=self.seed.episode_id)
        self.request = SimpleNamespace(
            session={"agent_session_id": "session"}, path_params={"rollout_id": self.seed.episode_id.capture_key}
        )

    async def respond(self, body: NeMoGymResponseCreateParamsNonStreaming | None = None, request=None):
        body = body or NeMoGymResponseCreateParamsNonStreaming(input="solve it", metadata={"agent_timeout_sec": "60"})
        return await self.agent.responses(request or self.request, body)


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

        assert (await first).id == (await retry).id == "chatcmpl-2"
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
        assert payload["model_base_url"] == "http://model:1/ng-rollout/r/v1"
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
        closed = await activation.agent._close_agent_session_state(activation.state)

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
        ("result", "status", "outcome"),
        [
            (
                {"stop_reason": "error", "error_kind": "model_request_error", "error_status": 429},
                "failed",
                {"failure_kind": "model_request_error", "status_code": 429},
            ),
            (
                {"stop_reason": "error", "error_kind": "model_transport_error"},
                "failed",
                {"failure_kind": "model_transport_error"},
            ),
            ({"stop_reason": "error", "error_kind": "harness_error"}, "failed", {"failure_kind": "unknown"}),
            ({"stop_reason": "error", "error_kind": "context_limit"}, "incomplete", None),
            ({"stop_reason": "wall_time", "finished_naturally": False}, "incomplete", None),
            ({"stop_reason": "max_turns", "finished_naturally": False}, "incomplete", None),
        ],
    )
    async def test_how_a_run_ended_reaches_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, result: dict, status: str, outcome: dict | None
    ) -> None:
        error = {"error": "Boom", "error_type": "HTTPStatusError"} if result["stop_reason"] == "error" else {}
        activation = _Activation(monkeypatch, _output(**result, **error))

        response = await activation.respond()

        assert response.status == status
        expected_stop = "context_limit" if result.get("error_kind") == "context_limit" else result["stop_reason"]
        assert response.metadata["stop_reason"] == expected_stop
        reported = response.metadata.get("ng_agent_outcome")
        assert (json.loads(reported) if reported else None) == outcome
        assert (response.error is not None) == (status == "failed")

    async def test_a_timed_out_command_batch_is_recorded_as_timed_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        output = _output(stop_reason="wall_time", finished_naturally=False)
        output["trajectory"]["steps"][1]["observation"] = {}
        activation = _Activation(monkeypatch, output)

        await activation.respond()
        closed = await activation.agent._close_agent_session_state(activation.state)

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
        closed = await activation.agent._close_agent_session_state(activation.state)

        tools = [record for record in closed.agent_observations.records if isinstance(record, ToolCallObservation)]
        assert [(tool.tool_call_id, tool.status) for tool in tools] == [
            ("call_0_1", "completed"),
            ("call_0_2", "completed"),
        ]

    async def test_usage_is_unknown_when_a_model_response_had_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch, _output(calls_without_usage=1))

        response = await activation.respond()
        closed = await activation.agent._close_agent_session_state(activation.state)

        assert response.usage is None
        assert "model_usage_unavailable" in {gap.code for gap in closed.agent_observations.gaps}

    async def test_a_late_result_after_a_provider_timeout_is_kept_only_past_the_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        activation = _Activation(monkeypatch)

        async def overran(payload: dict, *, timeout: float, close_timeout: float) -> dict:
            activation.state.session.artifacts = _output(stop_reason="completed")
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


class TestClose:
    async def test_close_before_activation_reports_a_gap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch)

        closed = await activation.agent._close_agent_session_state(activation.state)

        assert [gap.code for gap in closed.agent_observations.gaps] == ["observation_capture_failed"]
        activation.state.session.sandbox.disconnect.assert_awaited_once()
        activation.state.session.sandbox.stop.assert_not_awaited()

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

        await activation.agent._close_agent_session_state(activation.state)

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
        await closing

        activation.state.session.sandbox.disconnect.assert_awaited_once()

    def test_rollout_stops_reach_the_sandbox_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        activation = _Activation(monkeypatch)

        assert activation.agent._rollout_sandbox_session(activation.state) is activation.state.session


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
