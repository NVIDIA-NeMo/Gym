# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local sandbox-contract and standalone-runner tests; no provider or model compute."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
)
from nemo_gym.base_responses_api_model import ModelCallRecord
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import _build_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, join_model_call_observations
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.providers.base import SandboxExecResult
from responses_api_agents.kilocode_agent.app import KiloCodeAgent
from responses_api_agents.kilocode_agent.sandbox import KiloArtifacts, KiloSandboxSession
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


def state() -> KiloSandboxSession:
    return KiloSandboxSession(
        request=seed(),
        provider_name="docker",
        session=SandboxSession(
            sandbox=AsyncMock(),
            session_dir="/tmp/nemo-gym-kilo-sessions/test",
            workdir="/repo",
            harness="KiloCode",
        ),
    )


@pytest.mark.parametrize("completion_tokens,verdict", [(2, "healthy"), (0, "unhealthy")])
def test_native_session_capture_join_enables_health_checks(
    tmp_path: Path, completion_tokens: int, verdict: str
) -> None:
    session = state()
    session.session.artifacts = KiloArtifacts(
        stdout="", stderr="", exit_code=0, observations=persisted("ses-native", status="completed")
    )
    observations = _make_model_server_agent()._sandbox_observations(session)
    call = ModelCallRecord(
        model_call_id="call-1",
        client_session_id="ses-native",
        call_index=0,
        status_code=200,
        tokens_in=3,
        tokens_out=completion_tokens,
        request={"messages": [{"role": "user", "content": "hi"}]},
        response={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
    )
    joined = join_model_call_observations(observations, [call])
    assert joined.records[0].invocation_id == "ses-native"
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
    session.session.artifacts = KiloArtifacts(stdout="", stderr="", exit_code=0, wall_time_s=1.5)
    observations = _make_model_server_agent()._sandbox_observations(session)
    [sandbox] = observations.records
    assert (sandbox.provider, sandbox.outcome, sandbox.wall_time_s) == (
        "docker",
        "completed",
        1.5,
    )
    assert [gap.code for gap in observations.gaps] == ["agent_artifact_unavailable"]


def test_failed_capture_reports_capture_gap() -> None:
    observations = _make_model_server_agent()._sandbox_observations(state())
    assert observations.records[0].error_type == "artifact_capture_failed"
    assert [gap.code for gap in observations.gaps] == ["observation_capture_failed"]


@pytest.mark.parametrize("exit_code,reason,status", [(1, "stop", "failed"), (0, "length", "incomplete")])
def test_root_invocation_preserves_failure_and_length_stop(exit_code: int, reason: str, status: str) -> None:
    session = state()
    event = {"type": "step_finish", "sessionID": "ses-native", "part": {"reason": reason}}
    session.session.artifacts = KiloArtifacts(
        stdout=json.dumps(event), stderr="", exit_code=exit_code, observations=persisted("ses-native")
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
    partial = persisted("ses-native")
    partial.records[0].conversation = [NeMoGymEasyInputMessage(role="assistant", content="partial answer")]
    session.session.artifacts = KiloArtifacts(stdout="", stderr="", exit_code=None, observations=partial)
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
    assert sandbox.exit_code is None and sandbox.outcome == "unknown"
    with pytest.raises(HTTPException):
        await agent.responses(seeded_request, NeMoGymResponseCreateParamsNonStreaming(input="hi"))


@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_selects_owner_and_installs_in_sandbox(borrowed: bool) -> None:
    agent = _make_model_server_agent(
        kilo_version="7.4.15", sandbox_provider="sandbox", sandbox_config={"image": "test"}
    )
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(return_code=0, stdout="", stderr="")
    provider = AsyncMock()
    with (
        patch(f"{APP}.resolve_provider_config", return_value={}),
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
    install.assert_awaited_once_with(version="7.4.15", timeout=900)
    host_install.assert_not_called()
    if borrowed:
        api.connect.assert_awaited_once_with({"id": "task"}, provider=provider)
        sandbox.start.assert_not_awaited()
        sandbox.exec.assert_not_awaited()
    else:
        spec = sandbox.start.call_args.args[0]
        assert spec.workdir == "/app" and spec.ttl_s == 2400
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


async def test_artifacts_keep_partial_output_without_inventing_exit_code() -> None:
    session = state()
    with patch(f"{TRANSPORT}.read_text", side_effect=["partial", "diagnostic", RuntimeError("not written")]):
        artifacts = await session.collect_artifacts()
    assert artifacts == KiloArtifacts(stdout="partial", stderr="diagnostic", exit_code=None)


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


@pytest.mark.parametrize(
    "exit_code,stdout,expected",
    [
        (0, '{"type":"text","part":{"text":"55"}}', "completed"),
        (None, '{"type":"text","part":{"text":"partial"}}', "incomplete"),
        (1, "", "error"),
        (0, '{"type":"error","error":{"name":"APIError"}}', "error"),
        (0, "", "error"),
    ],
)
async def test_sandbox_output_and_failures(exit_code: int | None, stdout: str, expected: str) -> None:
    agent = _make_model_server_agent(model="m", system_prompt="configured")
    session = state()
    artifacts = KiloArtifacts(stdout=stdout, stderr="diagnostic", exit_code=exit_code)
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


@pytest.mark.parametrize("exit_code", [0, 3])
def test_standalone_runner_executes_in_task_directory(tmp_path: Path, exit_code: int) -> None:
    directory, repo = tmp_path / "session", tmp_path / "repo"
    directory.mkdir()
    repo.mkdir()
    script = (
        "import json,os,pathlib,sys; "
        "pathlib.Path('patch.txt').write_text('changed'); "
        "print(json.dumps({'cwd':os.getcwd(),'config':os.environ['KILO_CONFIG'],"
        "'home':os.environ['HOME'],'db':os.environ['KILO_DB'],"
        "'project':os.environ['KILO_DISABLE_PROJECT_CONFIG'],'key':os.getenv('OPENAI_API_KEY'),"
        "'stdin':sys.stdin.read()})); "
        "sys.stderr.buffer.write(b'diagnostic\\xff'); "
        f"sys.exit({exit_code})"
    )
    (directory / "input.json").write_text(json.dumps({"command": [sys.executable, "-c", script]}))
    runner = Path(__file__).parents[1] / "sandbox_runner.py"
    result = subprocess.run(
        [sys.executable, str(runner), str(directory), str(repo)],
        env={**os.environ, "OPENAI_API_KEY": "test-credential"},
        input=b"unexpected inherited input",
        capture_output=True,
    )
    assert result.returncode == exit_code
    output = json.loads((directory / "stdout.jsonl").read_text())
    assert output == {
        "cwd": str(repo),
        "config": str(directory / "kilo.json"),
        "home": str(directory / "home"),
        "db": str(directory / "kilo.db"),
        "project": "1",
        "key": None,
        "stdin": "",
    }
    assert (repo / "patch.txt").read_text() == "changed"
    assert not (repo / "kilo.json").exists()
    assert (directory / "stderr.log").read_bytes() == b"diagnostic\xff"
    assert json.loads((directory / "exit.json").read_text()) == exit_code


def test_timeout_and_cli_errors_have_explicit_sandbox_outcomes() -> None:
    session = state()
    session.session.cleanup = {"cleanup_confirmed": True, "return_code": -15, "error": None, "timed_out": True}
    session.session.artifacts = KiloArtifacts(
        stdout="", stderr="", exit_code=None, wall_time_s=2, observations=persisted("ses-native")
    )
    observations = _make_model_server_agent()._sandbox_observations(session)
    assert observations.records[-1].outcome == "timeout"
    assert observations.records[-1].error_type == "agent_timeout"
    assert observations.records[0].status == "incomplete"
    session.session.cleanup = None
    session.session.artifacts = KiloArtifacts(
        stdout=json.dumps({"type": "error", "error": {"name": "APIError"}}),
        stderr="",
        exit_code=0,
        observations=persisted("ses-native"),
    )
    observations = _make_model_server_agent()._sandbox_observations(session)
    assert observations.records[-1].outcome == "failed"
    assert observations.records[-1].error_type == "APIError"
    assert observations.records[0].status == "failed"
