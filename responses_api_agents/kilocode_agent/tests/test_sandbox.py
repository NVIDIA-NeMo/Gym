# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local sandbox-contract and standalone-runner tests; no provider or model compute."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    SandboxSession,
)
from nemo_gym.core import EpisodeId, NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.rollouts import run_health_checks
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess, SandboxExecResult
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ModelCallRecord
from nemo_gym.trajectory.trajectory_record import join_agent_model_calls
from responses_api_agents.kilocode_agent.app import KiloCodeAgent, KiloCodeAgentRunRequest, _runtime_failure
from responses_api_agents.kilocode_agent.sandbox import KiloArtifacts, KiloSandboxSession, _sandbox_prepare_command
from responses_api_agents.kilocode_agent.tests.test_app import _make_model_server_agent


APP = "responses_api_agents.kilocode_agent.app"
TRANSPORT = "responses_api_agents.kilocode_agent.sandbox"


def seed(*, borrowed: bool = True, workdir: str = "/repo") -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="session-1",
        episode_id=EpisodeId(rollout_id="rollout-1", attempt=2),
        task_id=TaskId(taskset="local:test", task_id="task-1"),
        sandbox_access=SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"id": "task"}),
            workdir=workdir,
        )
        if borrowed
        else None,
    )


def request(*, marker: str | None = "session-1", rollout_id: str = "rollout-1-a2") -> Request:
    return Request(
        {
            "type": "http",
            "path_params": {"rollout_id": rollout_id},
            "session": {} if marker is None else {"agent_session_id": marker},
        }
    )


def persisted(root: str, *, status: str = "completed") -> AgentObservationBundle:
    """A database-derived session tree: one root invocation and one spawned child."""
    return AgentObservationBundle(
        source="kilocode",
        records=[
            AgentInvocation(invocation_id=root, status=status),
            AgentInvocation(invocation_id="ses-child", parent_invocation_id=root, status="completed"),
        ],
    )


def receipt(*, return_code: int | None = 0, timed_out: bool = False, error: str | None = None) -> dict:
    return {"cleanup_confirmed": True, "return_code": return_code, "error": error, "timed_out": timed_out}


def state() -> KiloSandboxSession:
    session = KiloSandboxSession(
        request=seed(),
        provider_name="docker",
        session=SandboxSession(
            sandbox=AsyncMock(),
            session_dir="/tmp/nemo-gym-kilo-sessions/test",
            workdir="/repo",
            harness="KiloCode",
        ),
    )
    session.session.cleanup = receipt()
    return session


@pytest.mark.parametrize("completion_tokens,verdict", [(2, "healthy"), (0, "unhealthy")])
def test_kilo_session_capture_join_enables_health_checks(tmp_path: Path, completion_tokens: int, verdict: str) -> None:
    session = state()
    session.session.artifacts = KiloArtifacts(
        task_environment_ready=True,
        stdout="",
        stderr="",
        exit_code=0,
        observations=persisted("ses-kilo", status="completed"),
    )
    observations = _make_model_server_agent()._sandbox_observations(session)
    call = ModelCallRecord(
        model_call_id="call-1",
        client_session_id="ses-kilo",
        call_index=0,
        status_code=200,
        tokens_in=3,
        tokens_out=completion_tokens,
        request={"messages": [{"role": "user", "content": "hi"}]},
        response={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
    )
    joined = join_agent_model_calls(observations, [call])
    assert joined.records[0].invocation_id == "ses-kilo"
    assert joined.records[0].status == "completed"
    assert joined.records[0].model_calls[0].model_call_id == "call-1"
    assert "model_call_ownership_unavailable" not in {gap.code for gap in joined.gaps}
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "response": {"usage": {"input_tokens": 3, "output_tokens": completion_tokens}},
        "ng_agent_observations": joined.model_dump(mode="json"),
        "ng_model_call_capture": {"calls": [call.model_dump(mode="json")]},
    }
    trajectory = _build_trajectory_record(record, record)
    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].model_calls[0].model_call_id == "call-1"
    assert trajectory.turns[0].answer["content"] == "answer"
    record["ng_trajectory"] = trajectory.model_dump(mode="json")
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    [digest] = run_health_checks(path, workers=1).rollouts
    assert digest.verdict == verdict
    if verdict == "unhealthy":
        assert "model_call_zero_completion_tokens" in {finding.check for finding in digest.findings}


def test_missing_database_reports_gap_and_keeps_sandbox_outcome() -> None:
    session = state()
    session.session.artifacts = KiloArtifacts(
        task_environment_ready=True, stdout="", stderr="", exit_code=0, wall_time_s=1.5
    )
    observations = _make_model_server_agent()._sandbox_observations(session)
    [sandbox] = observations.records
    assert (sandbox.provider, sandbox.outcome, sandbox.wall_time_s) == (
        "docker",
        "completed",
        1.5,
    )
    assert [gap.code for gap in observations.gaps] == ["agent_artifact_unavailable", "runtime_info_unavailable"]


def test_failed_capture_reports_capture_gap() -> None:
    observations = _make_model_server_agent()._sandbox_observations(state())
    assert observations.records[0].error_type == "artifact_capture_failed"
    assert [gap.code for gap in observations.gaps] == ["observation_capture_failed"]


@pytest.mark.parametrize("exit_code,reason,status", [(1, "stop", "failed"), (0, "length", "incomplete")])
def test_root_invocation_preserves_failure_and_length_stop(exit_code: int, reason: str, status: str) -> None:
    session = state()
    event = {"type": "step_finish", "sessionID": "ses-kilo", "part": {"reason": reason}}
    session.session.artifacts = KiloArtifacts(
        task_environment_ready=True,
        stdout=json.dumps(event),
        stderr="",
        exit_code=exit_code,
        observations=persisted("ses-kilo"),
    )
    observations = _make_model_server_agent()._sandbox_observations(session)
    root, child, sandbox = observations.records
    assert root.status == status
    assert root.error_type == ("agent_run_error" if exit_code else None)
    assert child.status == "completed"
    assert sandbox.provider == "docker"


async def test_base_seed_and_close_retry_contract_preserves_partial_output() -> None:
    agent = _make_model_server_agent()
    session = state()
    partial = persisted("ses-kilo")
    partial.records[0].conversation = [NeMoGymEasyInputMessage(role="assistant", content="partial answer")]
    session.session.artifacts = KiloArtifacts(
        task_environment_ready=True, stdout="", stderr="", exit_code=None, observations=partial
    )
    session.session.cleanup = receipt(return_code=-15, timed_out=True)
    with patch.object(agent, "_seed_agent_session_state", return_value=session) as setup:
        seeded_request = request(marker=None)
        await agent.seed_agent_session(seeded_request, seed())
        await agent.seed_agent_session(seeded_request, seed())
    setup.assert_awaited_once()
    close = AgentCloseSessionRequest(agent_session_id="session-1", episode_id=seed().episode_id)
    with patch.object(session, "close", new_callable=AsyncMock) as release:
        first = await agent.close_agent_session(seeded_request, close)
        second = await agent.close_agent_session(seeded_request, close)
    release.assert_awaited_once()
    assert first == second and first is not second
    root, _, sandbox = first.agent_observations.records
    assert root.conversation[0].content == "partial answer"
    assert root.status == "incomplete"
    assert sandbox.exit_code is None and sandbox.outcome == "timeout"
    with pytest.raises(HTTPException):
        await agent.responses(seeded_request, NeMoGymResponseCreateParamsNonStreaming(input="hi"))


@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_selects_owner_and_installs_in_sandbox(borrowed: bool) -> None:
    agent = _make_model_server_agent(
        kilo_version="7.4.15", sandbox_provider="sandbox", sandbox_config={"image": "test"}
    )
    sandbox = AsyncMock()
    sandbox.sandbox_id = "sbx-1"
    sandbox.exec.return_value = SandboxExecResult(return_code=0, stdout="", stderr="")
    provider = AsyncMock()
    with (
        patch(f"{APP}.resolve_provider_config", return_value={}),
        patch(f"{APP}.resolve_provider_metadata", return_value={"source": "provider", "gym-agent": "overridden"}),
        patch(f"{APP}.get_global_config_dict", return_value={}),
        patch(f"{APP}.create_provider", return_value=provider),
        patch(f"{APP}.AsyncSandbox", return_value=sandbox) as api,
        patch.object(KiloSandboxSession, "install_runtime", new_callable=AsyncMock) as install,
        patch(f"{APP}.ensure_kilo") as host_install,
    ):
        api.connect = AsyncMock(return_value=sandbox)
        session = await agent._seed_agent_session_state(seed(borrowed=borrowed))
    assert session.session.owns_sandbox is not borrowed
    assert session.session.workdir == ("/repo" if borrowed else "/app")
    assert session.sandbox_id == "sbx-1"
    install.assert_awaited_once_with(version="7.4.15", timeout=900)
    host_install.assert_not_called()
    if borrowed:
        api.connect.assert_awaited_once_with({"id": "task"}, provider=provider)
        sandbox.start.assert_not_awaited()
        sandbox.exec.assert_not_awaited()
    else:
        spec = sandbox.start.call_args.args[0]
        assert spec.workdir == "/app" and spec.ttl_s == 2400
        assert spec.metadata == {
            "source": "provider",
            "gym-agent": "kilocode",
            "gym-session": "session-1",
            "gym-rollout": "rollout-1-a2",
        }
        assert agent.config.sandbox_config == {"image": "test"}
        sandbox.exec.assert_awaited_once_with("mkdir -p -- /app", cwd="/", timeout_s=30)


async def test_failed_install_retains_cleanup_only_state() -> None:
    agent = _make_model_server_agent(kilo_version="7.4.15")
    with (
        patch(f"{APP}.resolve_provider_config", return_value={}),
        patch(f"{APP}.get_global_config_dict", return_value={}),
        patch(f"{APP}.create_provider", return_value=AsyncMock()),
        patch(f"{APP}.AsyncSandbox.connect", return_value=AsyncMock()),
        patch.object(KiloSandboxSession, "install_runtime", side_effect=RuntimeError("install failed")),
        patch.object(KiloSandboxSession, "close", side_effect=RuntimeError("cleanup failed")),
        pytest.raises(AgentSessionSetupError) as error,
    ):
        await agent._seed_agent_session_state(seed())
    assert isinstance(error.value.state, KiloSandboxSession)
    assert str(error.value.error) == "install failed"


async def test_install_failure_reports_installer_output() -> None:
    session = state()
    sandbox = session.session.sandbox
    sandbox.exec.side_effect = [
        SandboxExecResult(return_code=0, stdout="", stderr=""),
        SandboxExecResult(return_code=1, stdout="node: v22.19.0", stderr="Kilo version mismatch: 7.4.16"),
    ]
    with pytest.raises(RuntimeError, match="exit 1") as error:
        await session.install_runtime(version="7.4.15", timeout=60)
    assert "Kilo version mismatch: 7.4.16" in str(error.value) and "node: v22.19.0" in str(error.value)
    installer = sandbox.upload.await_args_list[0].args[1]
    assert installer == "/tmp/nemo-gym-kilo-sessions/test/install_kilo_runtime.sh"
    assert sandbox.exec.await_args_list[0].args[0] == _sandbox_prepare_command(
        "/repo", "/tmp/nemo-gym-kilo-sessions/test"
    )
    assert sandbox.exec.await_args_list[1].args[0].startswith(f"bash {installer} ")
    assert len(sandbox.upload.await_args_list) == 1  # the runner is uploaded only after a successful install


async def test_connection_failure_never_falls_back_to_host() -> None:
    agent = _make_model_server_agent(kilo_version="7.4.15")
    provider = AsyncMock()
    with (
        patch(f"{APP}.resolve_provider_config", return_value={}),
        patch(f"{APP}.get_global_config_dict", return_value={}),
        patch(f"{APP}.create_provider", return_value=provider),
        patch(f"{APP}.AsyncSandbox.connect", side_effect=RuntimeError("missing sandbox")),
        patch(f"{APP}.ensure_kilo") as install,
        pytest.raises(RuntimeError, match="missing sandbox"),
    ):
        await agent._seed_agent_session_state(seed())
    provider.aclose.assert_awaited_once()
    install.assert_not_called()


@pytest.mark.parametrize(
    "return_code, timed_out, exit_code",
    [(3, False, 3), (143, True, None), (125, False, None), (126, False, None), (127, False, None)],
)
async def test_artifacts_take_the_exit_code_from_cleanup_only_when_kilo_finished(
    return_code: int, timed_out: bool, exit_code: int | None
) -> None:
    session = state()
    session.session.cleanup = receipt(return_code=return_code, timed_out=timed_out)
    with patch(f"{TRANSPORT}.read_text", side_effect=["partial", "diagnostic", RuntimeError("not written")]):
        artifacts = await session.collect_artifacts()
    assert (artifacts.stdout, artifacts.stderr) == ("partial", "diagnostic")
    # The supervisor's TERM status after a timeout, and the runner's statuses for a Kilo that never
    # started, are not exits Kilo chose.
    assert artifacts.exit_code == exit_code


@pytest.mark.parametrize(
    "return_code, failure",
    [
        (125, "Kilo runner failed before starting Kilo (exit 125)"),
        (126, "Kilo could not be started (exit 126)"),
        (127, "Kilo could not be started (exit 127)"),
        (3, "Kilo exited 3"),
    ],
)
def test_runtime_failure_tells_the_runner_from_kilo(return_code: int, failure: str) -> None:
    artifacts = KiloArtifacts(task_environment_ready=True, stdout="", stderr="", exit_code=None)
    assert _runtime_failure(artifacts, receipt(return_code=return_code)) == failure


def test_cancelled_run_reports_no_exit_code_of_its_own() -> None:
    session = state()
    session.session.cleanup = receipt(return_code=143)
    session.session.artifacts = KiloArtifacts(task_environment_ready=True, stdout="", stderr="", exit_code=143)
    session.task = MagicMock(cancelled=MagicMock(return_value=True))
    observations = _make_model_server_agent()._sandbox_observations(session)
    sandbox = observations.records[-1]
    assert (sandbox.outcome, sandbox.exit_code) == ("cancelled", None)


async def test_identical_activations_join_and_replay_independent_copies() -> None:
    agent = _make_model_server_agent(model="m")
    session = state()
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    response = agent._build_response(body, [], {"input_tokens": 0, "output_tokens": 0}, "m")
    entered, release = asyncio.Event(), asyncio.Event()

    async def activate(*args, **kwargs):
        entered.set()
        await release.wait()
        return response

    with (
        patch.object(agent, "_require_agent_session", return_value=session),
        patch.object(agent, "_sandbox_response", side_effect=activate) as run,
    ):
        waiter = asyncio.create_task(agent.responses(request(), body))
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not session.task.cancelled()
        release.set()
        first = await agent.responses(request(), body)
        first.output.clear()
        second = await agent.responses(request(), body)
        assert second.output and second is not first
        with pytest.raises(HTTPException) as changed:
            await agent.responses(request(), body.model_copy(update={"input": "changed"}))
        assert changed.value.status_code == 409
    assert run.call_count == 1


async def test_wrong_rollout_and_stale_cookie_never_execute_locally() -> None:
    agent = _make_model_server_agent()
    session = state()
    with (
        patch.object(agent, "_require_agent_session", return_value=session),
        patch.object(agent, "_run_kilo") as local,
    ):
        with pytest.raises(HTTPException, match="rollout route"):
            await agent.responses(request(rollout_id="other"), NeMoGymResponseCreateParamsNonStreaming(input="hi"))
        local.assert_not_called()
    with pytest.raises(HTTPException, match="Unknown or closing"):
        await agent.responses(request(marker="expired"), NeMoGymResponseCreateParamsNonStreaming(input="hi"))


@pytest.mark.parametrize(
    "params",
    [
        {"temperature": 0.7},
        {"max_output_tokens": 10},
        {"reasoning": {"effort": "high"}},
        {"input": [{"role": "assistant", "content": "past"}, {"role": "user", "content": "hi"}]},
    ],
)
def test_unsupported_request_semantics_are_rejected(params: dict) -> None:
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "hi"} | params)
    with pytest.raises(HTTPException) as error:
        KiloCodeAgent._validate_sandbox_request(body)
    assert error.value.status_code == 422


TEXT = '{"type":"text","part":{"text":"partial"}}'


@pytest.mark.parametrize(
    "exit_code,cleanup,stdout,expected",
    [
        (0, receipt(), '{"type":"text","part":{"text":"55"}}', "completed"),
        (None, receipt(return_code=-15, timed_out=True), TEXT, "incomplete"),
        (None, receipt(return_code=-15, timed_out=True), "", "error"),
        (None, receipt(return_code=1), "", "error"),
        (None, receipt(return_code=1), TEXT, "error"),
        (None, receipt(error="cleanup: lost"), TEXT, "error"),
        (None, None, TEXT, "error"),
        (1, receipt(return_code=1), "", "error"),
        (0, receipt(), '{"type":"error","error":{"name":"APIError"}}', "error"),
        (0, receipt(), "", "error"),
    ],
)
async def test_sandbox_output_and_failures(
    exit_code: int | None, cleanup: dict | None, stdout: str, expected: str
) -> None:
    agent = _make_model_server_agent(model="m", system_prompt="configured")
    session = state()
    session.session.cleanup = cleanup
    artifacts = KiloArtifacts(task_environment_ready=True, stdout=stdout, stderr="diagnostic", exit_code=exit_code)
    session.session.execute = AsyncMock(return_value=artifacts)
    body = NeMoGymResponseCreateParamsNonStreaming(input="hi", instructions="requested")
    if expected == "error":
        with pytest.raises(RuntimeError):
            await agent._sandbox_response(session, body, rollout_id="rollout-1-a2")
    else:
        with patch.object(session, "stage_activation", new_callable=AsyncMock) as stage:
            response = await agent._sandbox_response(session, body, rollout_id="rollout-1-a2")
            await session.session.execute.call_args.kwargs["stage_activation"]()
        assert response.status == expected
        staged = stage.call_args.kwargs
        assert staged["command"][:2] == session.kilo_argv
        assert staged["command"][-1] == "configured\n\nrequested\n\nhi"
        config = json.loads(staged["config"])
        assert config["provider"]["nemo"]["options"]["baseURL"].endswith("/ng-rollout/rollout-1-a2/v1")
        assert config["provider"]["nemo"]["models"]["m"]["interleaved"]["field"] == "reasoning_content"
        # --pure would skip config plugins, including the one that gives tool shells the task's HOME.
        assert "--pure" not in staged["command"]
        assert config["plugin"] == [session.task_environment_plugin]


CONNECT_ERROR = 'level=ERROR message="stream error" error.error="AI_APICallError: Cannot connect to API"'


@pytest.mark.parametrize(
    "stdout",
    ["", '{"type":"step_finish","part":{"id":"p1","tokens":{"input":0,"output":0},"reason":"stop"}}'],
)
async def test_timeout_before_any_model_call_is_an_execution_failure(stdout: str) -> None:
    """Kilo retries an unreachable model until the deadline; that must not be graded as the model's zero."""
    agent = _make_model_server_agent(model="m")
    session = state()
    session.session.cleanup = receipt(return_code=-15, timed_out=True)
    artifacts = KiloArtifacts(
        task_environment_ready=True, stdout=stdout, stderr="migration done", exit_code=None, log_tail=CONNECT_ERROR
    )
    session.session.execute = AsyncMock(return_value=artifacts)
    with pytest.raises(RuntimeError, match="before any model call completed.*Cannot connect to API"):
        await agent._sandbox_response(
            session, NeMoGymResponseCreateParamsNonStreaming(input="hi"), rollout_id="rollout-1-a2"
        )


@pytest.mark.parametrize("timed_out", [True, False])
async def test_artifacts_keep_kilo_log_tail_only_after_timeout(timed_out: bool) -> None:
    session = state()
    session.session.cleanup = receipt(return_code=-15 if timed_out else 0, timed_out=timed_out)
    session.session.sandbox.exec.side_effect = [
        SandboxExecResult(stdout=CONNECT_ERROR, stderr="", return_code=0),
        SandboxExecResult(stdout="", stderr="no database", return_code=1),
    ]
    with patch(f"{TRANSPORT}.read_text", side_effect=["", "diagnostic", RuntimeError("not written")]):
        artifacts = await session.collect_artifacts()
    assert artifacts.log_tail == (CONNECT_ERROR if timed_out else "")
    if timed_out:
        log_command = session.session.sandbox.exec.await_args_list[0].args[0]
        assert "/tmp/nemo-gym-kilo-sessions/test/data/kilo/log" in log_command


async def test_runner_identity_reaches_response_metadata() -> None:
    agent = _make_model_server_agent(model="m", kilo_version="7.4.15")
    session = state()
    with patch(
        f"{TRANSPORT}.read_text",
        side_effect=[TEXT, "", json.dumps({"hostname": "sandbox-7", "pid": 42}), "ready\n"],
    ):
        artifacts = await session.collect_artifacts()
    assert (artifacts.runner_hostname, artifacts.runner_pid) == ("sandbox-7", 42)
    session.session.execute = AsyncMock(return_value=artifacts)
    session.session.artifacts = artifacts
    response = await agent._sandbox_response(
        session, NeMoGymResponseCreateParamsNonStreaming(input="hi"), rollout_id="rollout-1-a2"
    )
    assert response.metadata == {
        "harness_execution": "sandbox",
        "harness_hostname": "sandbox-7",
        "harness_pid": "42",
        "kilo_version": "7.4.15",
    }
    session.sandbox_id = "sbx-1"
    observations = agent._sandbox_observations(session)
    assert "runtime_info_unavailable" not in {gap.code for gap in observations.gaps}
    assert observations.records[-1].sandbox_id == "sbx-1"


async def test_retried_close_does_not_interrupt_a_cancelling_activation() -> None:
    session = state()
    session.session.close = AsyncMock()
    cleaned = asyncio.Event()

    async def activation() -> None:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            cleaned.set()
            raise

    session.task = asyncio.create_task(activation())
    await asyncio.sleep(0)
    session.task.cancel()
    await asyncio.sleep(0)
    await session.close(timeout=1)
    assert cleaned.is_set() and session.task.cancelled()


class LocalFiles:
    """File transfer onto the local filesystem, standing in for the sandbox."""

    async def upload(self, source: Path, destination: str) -> None:
        shutil.copyfile(source, destination)


async def stage_locally(
    directory: Path,
    workdir: Path,
    command: list[str],
    *,
    config: str = "{}",
    required_mcp_servers: tuple[str, ...] = (),
    mcp_preflight: bool = False,
) -> list[str]:
    """Stage an activation into ``directory`` as the sandbox would receive it; return the runner argv."""
    session = KiloSandboxSession(
        request=seed(workdir=str(workdir)),
        required_mcp_servers=required_mcp_servers,
        session=SandboxSession(
            sandbox=LocalFiles(), session_dir=str(directory), workdir=str(workdir), harness="KiloCode"
        ),
    )
    shutil.copyfile(HARNESS_RUNNER_SCRIPT, directory / HARNESS_RUNNER_FILE)
    return (await session.stage_activation(command=command, config=config, mcp_preflight=mcp_preflight)).argv


@pytest.mark.parametrize("exit_code", [0, 3])
async def test_runner_executes_in_task_directory(tmp_path: Path, exit_code: int) -> None:
    directory, repo = tmp_path / "session", tmp_path / "repo"
    directory.mkdir()
    repo.mkdir()
    script = (
        "import json,os,pathlib,sys; "
        "pathlib.Path('patch.txt').write_text('changed'); "
        "print(json.dumps({'cwd':os.getcwd(),'config':os.environ['KILO_CONFIG'],"
        "'home':os.environ['HOME'],'db':os.environ['KILO_DB'],"
        "'project':os.environ['KILO_DISABLE_PROJECT_CONFIG'],'key':os.getenv('OPENAI_API_KEY'),"
        "'kilo':os.getenv('KILO_MODEL'),'kept':os.getenv('KEPT'),'stdin':sys.stdin.read()})); "
        "sys.stderr.buffer.write(b'diagnostic\\xff'); "
        f"sys.exit({exit_code})"
    )
    argv = await stage_locally(directory, repo, [sys.executable, "-c", script])
    assert argv == ["bash", f"{directory}/harness_runner.sh", f"{directory}/params.sh"]
    task_home = str(tmp_path / 'task "home" \\')
    env = {key: value for key, value in os.environ.items() if not key.startswith("XDG_")}
    inherited = {"OPENAI_API_KEY": "test-credential", "KILO_MODEL": "other", "KEPT": "yes", "HOME": task_home}
    result = subprocess.run(argv, env=env | inherited, input=b"unexpected inherited input", capture_output=True)
    assert result.returncode == exit_code, result.stderr
    output = json.loads((directory / "stdout.jsonl").read_text())
    assert output == {
        "cwd": str(repo),
        "config": str(directory / "kilo.json"),
        "home": str(directory / "home"),
        "db": str(directory / "kilo.db"),
        "project": "1",
        "key": None,
        "kilo": None,
        "kept": "yes",
        "stdin": "",
    }
    assert (repo / "patch.txt").read_text() == "changed"
    assert not (repo / "kilo.json").exists()
    assert (directory / "stderr.log").read_bytes() == b"diagnostic\xff"
    # The plugin's shell.env hook receives the task's directories, recorded before Kilo's overrides.
    assert json.loads((directory / "task-env.json").read_text()) == {
        "HOME": task_home,
        "XDG_DATA_HOME": f"{task_home}/.local/share",
        "XDG_CONFIG_HOME": f"{task_home}/.config",
        "XDG_CACHE_HOME": f"{task_home}/.cache",
        "XDG_STATE_HOME": f"{task_home}/.local/state",
    }
    assert all((directory / name).is_dir() for name in ("home", "data", "config", "cache"))
    runtime = json.loads((directory / "runtime.json").read_text())
    assert runtime["hostname"] == os.uname().nodename and runtime["pid"] != os.getpid()


def test_timeout_and_cli_errors_have_explicit_sandbox_outcomes() -> None:
    session = state()
    session.session.cleanup = {"cleanup_confirmed": True, "return_code": -15, "error": None, "timed_out": True}
    session.session.artifacts = KiloArtifacts(
        task_environment_ready=True,
        stdout="",
        stderr="",
        exit_code=None,
        wall_time_s=2,
        observations=persisted("ses-kilo"),
    )
    observations = _make_model_server_agent()._sandbox_observations(session)
    assert observations.records[-1].outcome == "timeout"
    assert observations.records[-1].error_type == "agent_timeout"
    assert observations.records[0].status == "incomplete"
    session.session.cleanup = None
    session.session.artifacts = KiloArtifacts(
        task_environment_ready=True,
        stdout=json.dumps({"type": "error", "error": {"name": "APIError"}}),
        stderr="",
        exit_code=0,
        observations=persisted("ses-kilo"),
    )
    observations = _make_model_server_agent()._sandbox_observations(session)
    assert observations.records[-1].outcome == "failed"
    assert observations.records[-1].error_type == "APIError"
    assert observations.records[0].status == "failed"


@pytest.mark.parametrize("option", ["no-sandbox", "required-tool", "no-model", "unpinned"])
def test_unsupported_seed_rejected_before_connection(option: str) -> None:
    agent = _make_model_server_agent(kilo_version="7.4.15")
    agent.config.resources_server = None
    body = seed().model_dump(mode="json")
    if option == "no-sandbox":
        body["sandbox_access"] = None
    elif option == "no-model":
        agent.config.model_server = None
    elif option == "unpinned":
        agent.config.kilo_version = None
    else:
        body["tool_accesses"] = [
            {"kind": "direct_http", "name": "tools", "base_url": "http://resources", "required": True}
        ]
    with patch(f"{APP}.create_provider") as create, TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=body).status_code == 422
    create.assert_not_called()


async def test_indirect_sandbox_access_is_rejected() -> None:
    agent = _make_model_server_agent(kilo_version="7.4.15")
    body = seed()
    body.sandbox_access = SandboxAccess.model_construct(connection=object(), workdir="/repo")
    with patch(f"{APP}.create_provider") as create, pytest.raises(HTTPException) as error:
        await agent._seed_agent_session_state(body)
    assert error.value.status_code == 422
    create.assert_not_called()


async def test_closing_session_rejects_activation_and_sessions_reject_agent_run() -> None:
    agent = _make_model_server_agent()
    session = state()
    session.session._close_task = asyncio.get_running_loop().create_future()
    with patch.object(agent, "_require_agent_session", return_value=session), pytest.raises(HTTPException) as error:
        await agent.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="hi"))
    assert error.value.status_code == 409 and session.task is None
    with pytest.raises(HTTPException) as error:
        await agent.run(request(), KiloCodeAgentRunRequest(responses_create_params={"input": "hi"}))
    assert error.value.status_code == 409


async def test_snapshot_requires_confirmed_cleanup_before_any_io() -> None:
    session = state()
    session.started_at = 0.0
    session.session.cleanup = {**receipt(), "cleanup_confirmed": False}
    with patch(f"{TRANSPORT}.read_text", side_effect=["partial", "diagnostic"]):
        artifacts = await session.collect_artifacts()
    assert artifacts.stdout == "partial" and artifacts.observations is None
    session.session.sandbox.exec.assert_not_awaited()


def test_missing_worker_exit_code_is_a_coverage_gap() -> None:
    session = state()
    session.session.cleanup = receipt(return_code=None, timed_out=True)
    session.session.artifacts = KiloArtifacts(
        task_environment_ready=True, stdout="", stderr="", exit_code=None, observations=persisted("ses")
    )
    observations = _make_model_server_agent()._sandbox_observations(session)
    assert "worker_exit_code_unavailable" in {gap.code for gap in observations.gaps}


@pytest.mark.parametrize("timed_out", [False, True])
async def test_unconfirmed_task_environment_fails_instead_of_grading(timed_out: bool) -> None:
    """Kilo continues when a plugin fails; its tools then wrote to Kilo's private HOME, not the task's."""
    agent = _make_model_server_agent(model="m")
    session = state()
    session.session.cleanup = receipt(return_code=None if timed_out else 0, timed_out=timed_out)
    artifacts = KiloArtifacts(stdout=TEXT, stderr="", exit_code=None if timed_out else 0)
    session.session.execute = AsyncMock(return_value=artifacts)
    session.session.artifacts = artifacts
    with pytest.raises(RuntimeError, match="task-environment plugin"):
        await agent._sandbox_response(
            session, NeMoGymResponseCreateParamsNonStreaming(input="hi"), rollout_id="rollout-1-a2"
        )
    sandbox_record = agent._sandbox_observations(session).records[-1]
    assert (sandbox_record.outcome, sandbox_record.error_type) == ("failed", "task_environment_unconfirmed")


async def test_install_uploads_the_task_environment_plugin() -> None:
    session = state()
    sandbox = session.session.sandbox
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    await session.install_runtime(version="7.4.15", timeout=60)
    uploaded = {
        destination: Path(source).name for source, destination in (c.args for c in sandbox.upload.await_args_list)
    }
    plugin = f"{session.session.session_dir}/task_environment.mjs"
    assert uploaded[plugin] == "task_environment.mjs"
    assert session.task_environment_plugin == f"file://{plugin}"


async def test_owned_sandbox_whose_start_cleanup_fails_is_retained_for_close() -> None:
    """start() can allocate a sandbox and then fail; close must still be able to stop it."""
    agent = _make_model_server_agent(kilo_version="7.4.15", sandbox_provider="sandbox")
    sandbox = AsyncMock()
    sandbox.sandbox_id = None
    sandbox.start.side_effect = RuntimeError("initial upload failed")
    with (
        patch(f"{APP}.resolve_provider_config", return_value={}),
        patch(f"{APP}.resolve_provider_metadata", return_value={}),
        patch(f"{APP}.get_global_config_dict", return_value={}),
        patch(f"{APP}.create_provider", return_value=AsyncMock()),
        patch(f"{APP}.AsyncSandbox", return_value=sandbox),
        patch.object(KiloSandboxSession, "close", AsyncMock(side_effect=RuntimeError("stop failed"))),
        pytest.raises(AgentSessionSetupError) as retained,
    ):
        await agent._seed_agent_session_state(seed(borrowed=False))
    assert isinstance(retained.value.state, KiloSandboxSession)
    assert retained.value.state.session.sandbox is sandbox
    assert retained.value.state.session.owns_sandbox


async def test_cancelled_local_install_is_shared_not_restarted() -> None:
    """Cancelling a waiter must not let a retry start a second installer on the same paths."""
    agent = _make_model_server_agent(kilo_version="7.4.15")
    release, calls = threading.Event(), []

    def install(version: str) -> None:
        calls.append(version)
        release.wait(timeout=10)

    class Stop(Exception):
        pass

    with (
        patch(f"{APP}.ensure_kilo", side_effect=install),
        patch.object(agent, "_workspace_root", side_effect=Stop),
    ):
        first = asyncio.create_task(agent._run_kilo("task", None))
        while not calls:
            await asyncio.sleep(0.01)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(agent._run_kilo("task", None))
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(Stop):
            await asyncio.wait_for(second, timeout=10)
    assert calls == ["7.4.15"]
