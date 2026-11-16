# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenHands CodeAct agent-session contract: seed refusals, activation, outcomes, observations and close."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AGENT_SESSION_COOKIE_KEY,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPToolAccess
from nemo_gym.trajectory import AgentInvocation, ContextCompactionObservation
from responses_api_agents.openhands_codeact_agent import app
from responses_api_agents.openhands_codeact_agent import sandbox as adapter
from responses_api_agents.openhands_codeact_agent.tests.test_conversion import _write, compacting_rollout


GLOBAL_CONFIG = {
    "sandbox": {"local": {}},
    "remote_sandbox": {"opensandbox": {}},
    "misspelled_sandbox": {"typo_provider": {}},
    "policy_model": {"responses_api_models": {"vllm_model": {"host": "10.0.0.5", "port": 9000}}},
}


def mcp_grant(*, required=True):
    return MCPToolAccess(
        name="gdpval_resources_server",
        required=required,
        connection={
            "transport": "streamable_http",
            "url": "http://resources:8080/mcp",
            "headers": {"x-nemo-gym-mcp-session": "token"},
        },
    )


def seed(*, borrowed=True, workdir="/root", provider="sandbox", tool_accesses=()):
    connection = {"provider_config_ref": provider, "descriptor": {"id": "borrowed"}}
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id={"rollout_id": "rollout"},
        task_id={"taskset": "fixture", "task_id": "task"},
        tool_accesses=list(tool_accesses),
        sandbox_access={"connection": connection, "workdir": workdir} if borrowed else None,
    )


def agent(**kwargs):
    config = {
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "model": "served-model",
        "remote_openhands_install_script_path": "/staged/install.sh",
        "remote_openhands_archive_path": "/staged/runtime.tar.gz",
        **kwargs,
    }
    instance = app.OpenHandsCodeActAgent(
        config=app.OpenHandsCodeActAgentConfig(
            name="openhands_codeact", host="127.0.0.1", port=8080, entrypoint="app.py", **config
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict=GLOBAL_CONFIG),
    )
    instance.server_client._build_server_base_url = lambda server: f"http://{server['host']}:{server['port']}"
    return instance


def result(stdout="", *, code=0):
    return SimpleNamespace(stdout=stdout, stderr="diagnostic" if code else "", return_code=code, error_type=None)


def request(session_id="session", rollout="rollout"):
    session = {AGENT_SESSION_COOKIE_KEY: session_id} if session_id else {}
    return Request({"type": "http", "session": session, "path_params": {"rollout_id": rollout}})


@pytest.mark.parametrize(
    "body,reason",
    [
        (seed(borrowed=False), "task sandbox the Resources Server lends"),
        (
            seed(
                tool_accesses=[
                    DirectHTTPToolAccess(
                        name="gdpval_resources_server.direct_http", required=True, base_url="http://r"
                    )
                ]
            ),
            "only over MCP",
        ),
        (seed(workdir="relative"), "workdir"),
        (seed(workdir="/opt/openhands-codeact/venv"), "workdir"),
        (seed(workdir="/tmp/nemo-gym-openhands-codeact-sessions/x"), "workdir"),
        *[(seed(workdir=workdir), "workdir") for workdir in ("/", "/tmp", "/root/../tmp")],
        (seed(provider="unknown_sandbox"), "unknown_sandbox"),
        (seed(provider="misspelled_sandbox"), "typo_provider"),
    ],
)
def test_unservable_seeds_are_final_refusals(body, reason):
    assert reason in check_agent_seed_rejected(agent().setup_webserver(), body)


def test_loopback_model_server_is_refused_for_a_remote_sandbox():
    instance = agent()
    instance.server_client.global_config_dict["policy_model"]["responses_api_models"]["vllm_model"]["host"] = (
        "127.0.0.1"
    )
    try:
        assert "loopback" in check_agent_seed_rejected(instance.setup_webserver(), seed(provider="remote_sandbox"))
    finally:
        instance.server_client.global_config_dict["policy_model"]["responses_api_models"]["vllm_model"]["host"] = (
            "10.0.0.5"
        )


@pytest.mark.asyncio
async def test_seed_borrows_the_sandbox_installs_the_runtime_and_keeps_its_files_outside_the_task(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    install = AsyncMock()
    monkeypatch.setattr(app, "install_runtime", install)
    state = await agent()._seed_agent_session_state(seed(tool_accesses=[mcp_grant()]))
    app.AsyncSandbox.connect.assert_awaited_once_with({"id": "borrowed"}, provider=provider)
    assert install.await_args.kwargs == {
        "script_path": "/staged/install.sh",
        "archive_path": "/staged/runtime.tar.gz",
        "timeout_s": 900,
    }
    assert state.session.session_dir.startswith(adapter.SESSION_ROOT + "/")
    assert state.session.workdir == "/root" and not state.session.owns_sandbox
    uploaded = [call.args[1] for call in remote.upload.await_args_list]
    assert uploaded == [
        f"{state.session.session_dir}/harness_runner.sh",
        *[f"{state.session.session_dir}/{name}" for name in adapter.RUNNER_FILES],
    ]
    # A borrowed task is never prepared by the agent: only its own session directory is created.
    assert all(adapter.SESSION_ROOT in call.args[0] for call in remote.exec.await_args_list)


@pytest.mark.asyncio
async def test_failed_seed_releases_and_failed_release_retains_state(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    monkeypatch.setattr(app, "install_runtime", AsyncMock(side_effect=RuntimeError("install failed")))
    close = AsyncMock()
    monkeypatch.setattr(adapter.CodeActSandboxSession, "close", close)
    with pytest.raises(RuntimeError, match="install failed"):
        await agent()._seed_agent_session_state(seed())
    close.assert_awaited_once()
    close.side_effect = RuntimeError("release failed")
    with pytest.raises(AgentSessionSetupError) as error:
        await agent()._seed_agent_session_state(seed())
    assert error.value.state.request == seed()


def session_state(*, tool_accesses=(), **session):
    defaults = {
        "session_dir": "/tmp/nemo-gym-openhands-codeact-sessions/s",
        "workdir": "/root",
        "cleanup": {"timed_out": False, "return_code": 0, "cleanup_confirmed": True, "error": None},
        "artifacts": None,
        "expired": False,
        "closing": False,
        "sandbox": SimpleNamespace(sandbox_id="task-sandbox"),
    }
    return adapter.CodeActSandboxSession(
        request=seed(tool_accesses=tool_accesses),
        session=SimpleNamespace(**{**defaults, **session}),
        interpreter="/opt/openhands-codeact/venv/bin/python",
    )


def collected(tmp_path, run_result, *, payloads=None, requests=None, compactions=None):
    trace = tmp_path / "collected" / "trace"
    trace.mkdir(parents=True)
    _write(trace, compacting_rollout() if payloads is None else payloads)
    requests = (
        [{"http_status": 200, "usage": {"prompt_tokens": 10, "completion_tokens": 2}}] * 5
        if requests is None
        else requests
    )
    (trace / "model-requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    (trace / "compactions.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in (compactions or [{"response_id": "chatcmpl-2"}]))
    )
    return adapter.CodeActArtifacts(
        result=None if run_result is None else json.dumps(run_result), trace_dir=trace, stderr="trace"
    )


def body(**fields):
    return NeMoGymResponseCreateParamsNonStreaming(input="Write the report.", **fields)


@pytest.mark.asyncio
async def test_activation_runs_the_controller_through_the_rollout_prefixed_model_server(monkeypatch, tmp_path):
    instance = agent(context_summarization_cutoff=0.7, finish_tool_names=["finish", "abandon_task_finish"])
    state = session_state(tool_accesses=[mcp_grant()])
    artifacts = collected(tmp_path, {"finished": True, "agent_state": "finished"})
    state.execute = AsyncMock(return_value=artifacts)
    response = await instance._activate(
        state,
        body(temperature=0.6),
        prompt="Write the report.",
        settings=instance._validate_request(body(temperature=0.6))[1],
    )
    runner_input = state.execute.await_args.args[0]
    model = runner_input["model"]
    assert model["base_url"].startswith("http://10.0.0.5:9000/ng-rollout/") and model["base_url"].endswith("/v1")
    assert model["headers"] == {"x-session-id": state.request.episode_id.capture_key}
    assert (model["temperature"], model["top_p"], model["max_output_tokens"]) == (0.6, 1.0, 32768)
    assert runner_input["mcp_servers"] == [
        {
            "name": "gdpval_resources_server",
            "url": "http://resources:8080/mcp",
            "headers": {"x-nemo-gym-mcp-session": "token"},
            "required": True,
            "timeout_s": 900,
        }
    ]
    assert runner_input["context_summarization"] == {"cutoff": 0.7, "max_event_chars": 10_000}
    assert runner_input["finish_tool_names"] == ["finish", "abandon_task_finish"]
    assert runner_input["trace_dir"] == f"{state.session.session_dir}/trace"
    assert response.status == "completed" and response.metadata["stop_reason"] == "completed"
    assert response.output[-1].content[0].text == "Done, summary.txt written."
    assert response.usage.input_tokens == 50
    # The collected logs are removed once the response is built.
    assert not artifacts.trace_dir.exists()
    records = state.observations.records
    invocation = next(r for r in records if isinstance(r, AgentInvocation))
    assert [ref.response_id for ref in invocation.model_calls] == [f"chatcmpl-{n}" for n in (0, 1, 3, 4)]
    (compaction,) = [r for r in records if isinstance(r, ContextCompactionObservation)]
    assert [ref.response_id for ref in compaction.model_calls] == ["chatcmpl-2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "last_error,stop_reason",
    [
        ("context length limit has been exceeded", "context_limit"),
        ("Agent reached maximum iteration 200", "max_iterations"),
        ("AgentStuckInLoopError: Agent got stuck in a loop", "stuck_in_loop"),
    ],
)
async def test_policy_limits_end_incomplete_and_their_work_is_judged(tmp_path, last_error, stop_reason):
    instance = agent()
    state = session_state()
    state.execute = AsyncMock(return_value=collected(tmp_path, {"finished": False, "last_error": last_error}))
    response = await instance._activate(state, body(), prompt="p", settings=instance._validate_request(body())[1])
    assert response.status == "incomplete" and response.metadata["stop_reason"] == stop_reason
    assert AGENT_OUTCOME_METADATA_KEY not in response.metadata
    assert response.output


@pytest.mark.asyncio
async def test_deadline_keeps_partial_work_as_a_timed_out_run(tmp_path):
    instance = agent()
    artifacts = collected(tmp_path, None)
    state = session_state(
        cleanup={"timed_out": True, "return_code": 124, "cleanup_confirmed": True, "error": None}, artifacts=artifacts
    )
    state.execute = AsyncMock(side_effect=TimeoutError("deadline"))
    response = await instance._activate(state, body(), prompt="p", settings=instance._validate_request(body())[1])
    # The GDPVal Resources Server scores wall_time as the model's timeout.
    assert response.status == "incomplete" and response.metadata["stop_reason"] == "wall_time"
    assert response.output


@pytest.mark.asyncio
async def test_model_error_fails_the_run_with_a_classified_outcome(tmp_path):
    instance = agent()
    state = session_state()
    requests = [{"http_status": 200, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, {"http_status": 404}]
    state.execute = AsyncMock(
        return_value=collected(tmp_path, {"finished": False, "last_error": "NotFoundError"}, requests=requests)
    )
    response = await instance._activate(state, body(), prompt="p", settings=instance._validate_request(body())[1])
    assert response.status == "failed" and response.metadata["stop_reason"] == "error"
    outcome = json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])
    assert outcome["failure_kind"] == "model_request_error" and outcome["status_code"] == 404


@pytest.mark.asyncio
async def test_missing_result_is_a_runner_failure(tmp_path):
    instance = agent()
    state = session_state()
    state.execute = AsyncMock(return_value=collected(tmp_path, None, payloads=[]))
    with pytest.raises(RuntimeError, match="before writing a result"):
        await instance._activate(state, body(), prompt="p", settings=instance._validate_request(body())[1])
    assert state.observations is not None


@pytest.mark.asyncio
async def test_sandbox_failure_is_raised_for_a_retry_with_observations(tmp_path):
    instance = agent()
    state = session_state(cleanup=None)
    state.execute = AsyncMock(side_effect=RuntimeError("sandbox went away"))
    with pytest.raises(RuntimeError, match="sandbox went away"):
        await instance._activate(state, body(), prompt="p", settings=instance._validate_request(body())[1])
    assert [gap.code for gap in state.observations.gaps][-1] == "agent_artifact_unavailable"


@pytest.mark.parametrize(
    "fields",
    [
        {"tools": [{"type": "function", "name": "f", "parameters": {}, "strict": True}]},
        {"tool_choice": "none"},
        {"max_tool_calls": 3},
        {"metadata": {"other": "x"}},
        {"input": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]},
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"}],
                }
            ]
        },
        {"model": "other-model"},
    ],
)
def test_unsupported_request_controls_are_rejected(fields):
    request_body = NeMoGymResponseCreateParamsNonStreaming(**{"input": "x", **fields})
    with pytest.raises(HTTPException) as error:
        agent()._validate_request(request_body)
    assert error.value.status_code == 422


def test_sampling_controls_and_instructions_are_mapped():
    prompt, settings = agent()._validate_request(
        NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Do it."}],
            instructions="Follow the brief.",
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=1024,
            metadata={"agent_timeout_sec": "60"},
        )
    )
    assert prompt == "Follow the brief.\n\nBe brief.\n\nDo it."
    assert settings == {"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 1024}


@pytest.mark.asyncio
async def test_retry_shares_the_activation_and_rejects_a_changed_body(monkeypatch):
    instance = agent()
    state = session_state()
    async with instance._locked_agent_session("session") as record:
        record.state = state
    response = MagicMock()
    response.model_copy.return_value = response
    activate = AsyncMock(return_value=response)
    monkeypatch.setattr(app.OpenHandsCodeActAgent, "_activate", activate)
    assert await instance.responses(request(), body()) is response
    assert await instance.responses(request(), body()) is response
    assert activate.await_count == 1
    for changed_request, changed_body in [
        (request(), NeMoGymResponseCreateParamsNonStreaming(input="Changed")),
        (request(rollout="other"), body()),
        (request("unknown"), body()),
    ]:
        with pytest.raises(HTTPException) as error:
            await instance.responses(changed_request, changed_body)
        assert error.value.status_code == 409
    assert activate.await_count == 1


@pytest.mark.asyncio
async def test_sessionless_requests_and_run_are_refused():
    instance = agent()
    with pytest.raises(HTTPException) as error:
        await instance.responses(request(session_id=None), body())
    assert error.value.status_code == 422
    with pytest.raises(HTTPException) as error:
        await instance.run(MagicMock())
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_close_releases_the_borrowed_sandbox_without_stopping_it():
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), disconnect=AsyncMock(), stop=AsyncMock())
    session = SandboxSession(sandbox=remote, session_dir="/tmp/session", workdir="/root", harness="OpenHands CodeAct")
    session.launch_started = True
    session.cleanup = {"cleanup_confirmed": True, "timed_out": False, "return_code": 0, "error": None}
    state = adapter.CodeActSandboxSession(request=seed(), session=session)
    close = await agent()._close_agent_session_state(state)
    remote.disconnect.assert_awaited_once()
    remote.stop.assert_not_awaited()
    assert close.cleanup_confirmed is True
    assert close.agent_observations.gaps[0].code == "agent_activation_interrupted"


def test_trace_extraction_refuses_unsafe_and_oversized_members(tmp_path, monkeypatch):
    import zipfile

    archive = tmp_path / "trace.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.json", "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        adapter.extract_trace(archive, tmp_path / "out")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("completions/a.json", "x" * 200)
    monkeypatch.setattr(adapter, "MAX_TRACE_FILE_BYTES", 100)
    with pytest.raises(ValueError, match="collection limits"):
        adapter.extract_trace(archive, tmp_path / "out")
    assert not (tmp_path / "escape.json").exists()
