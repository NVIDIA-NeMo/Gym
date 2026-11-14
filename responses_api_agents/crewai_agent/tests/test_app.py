# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions end to end: a real crew in the sandbox runner, a stub Model Server, and a local sandbox.

The episode tests run the sandbox runner under Gym's process supervisor, which needs Linux. The host's CrewAI install
stands in for the runtime the agent would install in the sandbox.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.agents import (
    ROLLOUT_STOP_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    SandboxPythonRuntime,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxSpec,
)
from nemo_gym.server_runtime import DirectHTTPToolAccess
from responses_api_agents.crewai_agent import app
from responses_api_agents.crewai_agent.app import CrewAIAgent, CrewAIAgentConfig
from responses_api_agents.crewai_agent.sandbox_runner import configure_crewai
from responses_api_agents.crewai_agent.tests.fakes import LendableLocalProvider, ModelServer, completion


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Gym's process supervisor needs Linux")
_EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
_TASK = "Write hello to out.txt. Keep the {braces} as they are."


def _agent(**config: Any) -> CrewAIAgent:
    return CrewAIAgent(
        config=CrewAIAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="crewai_agent",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            **config,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def _seed(**fields: Any) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session", episode_id=_EPISODE, task_id=TaskId(taskset="tb", task_id="t"), **fields
    )


def _access(workdir: str = "/app", **fields: Any) -> SandboxAccess:
    return SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="lender", descriptor={"sandbox_id": "sb"}),
        workdir=workdir,
        **fields,
    )


def _request(rollout_id: str = _EPISODE.capture_key) -> SimpleNamespace:
    return SimpleNamespace(session={}, cookies={}, path_params={"rollout_id": rollout_id})


def _body(**fields: Any) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(**{"input": [{"role": "user", "content": _TASK}], **fields})


def _close_body() -> AgentCloseSessionRequest:
    return AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=_EPISODE)


def _records(close, kind: str) -> list:
    return [record for record in close.agent_observations.records if record.kind == kind]


# --- configuration, seed refusals and request mapping -----------------------------------------------------


def test_crewai_runs_with_a_private_home_and_sends_nothing_to_crewai_services(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/home/task")
    monkeypatch.setenv("TASK_VAR", "kept")

    task_environment = configure_crewai(tmp_path / "crewai-home")

    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true"
    assert os.environ["CREWAI_TRACING_ENABLED"] == "false"
    assert os.environ["CREWAI_DISABLE_VERSION_CHECK"] == "true"
    assert os.environ["HOME"] == str(tmp_path / "crewai-home")
    # Tool commands keep the task's environment.
    assert (task_environment["HOME"], task_environment["TASK_VAR"]) == ("/home/task", "kept")
    assert "CREWAI_STORAGE_DIR" not in task_environment


@pytest.mark.parametrize(
    ("seed", "reason"),
    [
        (
            _seed(
                sandbox_access=_access(),
                tool_accesses=[DirectHTTPToolAccess(name="task", required=True, base_url="http://r", cookies={})],
            ),
            "only MCP tool grants",
        ),
        (_seed(), "lends none and the agent has no sandbox_provider"),
        (_seed(sandbox_access=_access(user="agent")), "SandboxAccess user override"),
        (_seed(sandbox_access=_access("app")), "absolute sandbox workdir"),
    ],
)
async def test_a_seed_the_crew_cannot_serve_is_refused(seed, reason):
    with pytest.raises(AgentSeedRejectedError, match=reason):
        await _agent()._seed_agent_session_state(seed)


async def test_a_loopback_model_url_is_refused_for_a_remote_sandbox(monkeypatch):
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"lender": {"opensandbox": {}}})
    monkeypatch.setattr(
        CrewAIAgent, "resolve_model_base_url", lambda self, *args, **kwargs: "http://127.0.0.1:8000/v1"
    )

    with pytest.raises(AgentSeedRejectedError, match="loopback"):
        await _agent()._seed_agent_session_state(_seed(sandbox_access=_access()))


@pytest.mark.parametrize(
    "body, reason",
    [
        (
            {"input": "hi", "tools": [{"type": "function", "name": "f", "parameters": {}, "strict": False}]},
            "only through an MCP grant",
        ),
        ({"input": "hi", "reasoning": {"effort": "high"}}, "request field reasoning"),
        ({"input": "hi", "max_output_tokens": 32}, "request field max_output_tokens"),
        (
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_image", "image_url": "data:,", "detail": "auto"}]}
                ]
            },
            "only text",
        ),
        ({"input": [{"role": "system", "content": "Be brief."}]}, "needs a user message"),
    ],
)
def test_a_request_the_crew_cannot_apply_is_refused(body, reason):
    state = SimpleNamespace(grants=[], request=SimpleNamespace(agent_timeout_sec=None))

    with pytest.raises(HTTPException, match=reason) as refused:
        _agent()._instruction(state, NeMoGymResponseCreateParamsNonStreaming(**body))

    assert refused.value.status_code == 422


def test_system_messages_and_instructions_join_the_agents_system_prompt():
    state = SimpleNamespace(grants=[], request=SimpleNamespace(agent_timeout_sec=600))
    body = _body(
        instructions="Answer in French.",
        input=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": _TASK}],
        metadata={"agent_timeout_sec": "30"},
    )

    instruction = _agent()._instruction(state, body)

    assert instruction.system == "Answer in French.\n\nBe brief."
    assert instruction.description == _TASK
    # The seed's and the request's task timeouts both lower the configured one.
    assert instruction.timeout == 30


@linux_only
async def test_request_output_budget_is_refused_before_binding_and_a_corrected_request_can_run(
    lent_sandbox, runtime, monkeypatch
):
    _, access = lent_sandbox
    agent, request = _agent(), _request()
    model = _model(monkeypatch, completion("Done.", id="chatcmpl-corrected"))
    try:
        await agent.seed_agent_session(request, _seed(sandbox_access=access))
        with pytest.raises(HTTPException, match="request field max_output_tokens") as refused:
            await agent.responses(request, _body(max_output_tokens=32))
        state = agent._require_agent_session(agent._agent_session_id_from_request(request))
        assert refused.value.status_code == 422
        assert state.activation_request is None and state.task is None
        assert model.requests == []
        response = await agent.responses(request, _body(temperature=0.1, top_p=0.9))
        close = await agent.close_agent_session(request, _close_body())
    finally:
        model.close()
    assert response.status == "completed" and close.cleanup_confirmed
    [sent] = model.requests
    assert (sent["json"]["temperature"], sent["json"]["top_p"]) == (0.1, 0.9)
    assert "max_tokens" not in sent["json"]


async def test_an_activation_outside_the_seeded_session_is_refused():
    agent = _agent()

    with pytest.raises(HTTPException) as unseeded:
        await agent.responses(_request(), _body())
    with pytest.raises(HTTPException) as legacy:
        await agent.run(_request(), SimpleNamespace())

    assert unseeded.value.status_code == legacy.value.status_code == 409


# --- episodes on the local provider -----------------------------------------------------------------------


def _model(monkeypatch, *answers) -> ModelServer:
    model = ModelServer(list(answers))
    monkeypatch.setattr(CrewAIAgent, "resolve_model_base_url", lambda self, *args, **kwargs: model.base_url)
    return model


async def _episode(agent: CrewAIAgent, access: SandboxAccess | None, body=None, **seed: Any):
    """Seed, activate and close one session; return the response (or the activation's error) and the close."""
    request = _request()
    await agent.seed_agent_session(request, _seed(sandbox_access=access, **seed))
    try:
        response = await asyncio.wait_for(agent.responses(request, body or _body()), 120)
    except HTTPException as error:
        response = error
    close = await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 60)
    return response, close


async def test_a_failed_runtime_install_releases_the_lent_sandbox(lent_sandbox, runtime, monkeypatch):
    owner, access = lent_sandbox
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", AsyncMock(return_value=False))
    monkeypatch.setattr(SandboxPythonRuntime, "install", AsyncMock(side_effect=RuntimeError("no wheel")))

    with pytest.raises(RuntimeError, match="no wheel"):
        await _agent().seed_agent_session(_request(), _seed(sandbox_access=access))

    assert owner.sandbox_id not in LendableLocalProvider.closed
    assert (await owner.exec("cat input.txt")).stdout == "from the task image\n"


async def test_close_before_an_activation_reports_the_missing_evidence_and_leaves_the_sandbox(
    lent_sandbox, runtime, tmp_path
):
    owner, access = lent_sandbox
    agent, request = _agent(), _request()
    await agent.seed_agent_session(request, _seed(sandbox_access=access))
    close = await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 60)

    assert [(record.kind, record.outcome) for record in close.agent_observations.records] == [("sandbox", "cancelled")]
    assert [gap.code for gap in close.agent_observations.gaps] == ["observation_capture_failed"]
    assert close.cleanup_confirmed is True
    assert owner.sandbox_id not in LendableLocalProvider.closed
    assert list((tmp_path / "sessions").iterdir()) == []


@linux_only
async def test_an_episode_runs_one_crew_turn_in_the_lent_sandbox(lent_sandbox, runtime, monkeypatch, tmp_path):
    owner, access = lent_sandbox
    workdir = Path(access.workdir)
    model = _model(
        monkeypatch,
        completion(None, ("shell", {"command": "echo $TASK_VAR > out.txt; cat out.txt input.txt"}), id="chatcmpl-1"),
        completion("Wrote hello; the {braces} stay.", id="chatcmpl-2"),
    )
    body = _body(instructions="Answer in French.", temperature=0.0)
    try:
        response, close = await _episode(_agent(), access, body)
    finally:
        model.close()

    assert response.status == "completed"
    assert (workdir / "out.txt").read_text() == "from-the-owner\n"
    first, second = model.requests
    assert first["path"] == f"/ng-rollout/{_EPISODE.capture_key}/v1/chat/completions"
    assert {request["headers"]["x-session-id"] for request in model.requests} == {_EPISODE.capture_key}
    assert first["json"]["temperature"] == 0.0
    assert "max_tokens" not in first["json"]
    assert [tool["function"]["name"] for tool in first["json"]["tools"]] == ["shell"]
    system, user = first["json"]["messages"]
    assert "Answer in French." in system["content"] and _TASK in user["content"]
    # CrewAI's own message keys (cache_breakpoint) never reach the Model Server.
    assert all(
        set(message) <= {"role", "content", "tool_calls", "tool_call_id"} for message in second["json"]["messages"]
    )
    result = second["json"]["messages"][-1]
    assert result["role"] == "tool" and result["content"] == "from-the-owner\nfrom the task image\n\n[exit code: 0]"

    assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
    assert response.output[0].call_id == response.output[1].call_id == "chatcmpl-1-call-0"
    assert response.output[2].content[0].text == "Wrote hello; the {braces} stay."
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (20, 10, 30)
    assert response.usage.input_tokens_details.cached_tokens is None
    assert response.usage.output_tokens_details.reasoning_tokens is None

    assert close.cleanup_confirmed is True and close.trajectory is None
    [invocation] = _records(close, "agent_invocation")
    [tool] = _records(close, "tool_call")
    [record] = _records(close, "sandbox")
    assert invocation.status == "completed"
    assert [ref.response_id for ref in invocation.model_calls] == ["chatcmpl-1", "chatcmpl-2"]
    assert (tool.tool_call_id, tool.tool_name, tool.status, tool.timing_source) == (
        "chatcmpl-1-call-0",
        "shell",
        "completed",
        "executor",
    )
    # The agent used a lent sandbox: it observed it, and left its teardown to the owner.
    assert (record.role, record.provider, record.sandbox_id) == ("agent", "lendable_local", owner.sandbox_id)
    assert (record.outcome, record.error_type, record.teardown_confirmed) == ("completed", None, None)
    assert {gap.code for gap in close.agent_observations.gaps} == {
        "cached_token_usage_unavailable",
        "reasoning_token_usage_unavailable",
    }
    # The runtime, CrewAI's own files and the session's files stay outside the task directory, and close removed the
    # session's files. The owner still runs commands in the sandbox and stops it itself.
    assert sorted(path.name for path in workdir.iterdir()) == ["input.txt", "out.txt"]
    assert list((tmp_path / "sessions").iterdir()) == []
    assert owner.sandbox_id not in LendableLocalProvider.closed


@linux_only
async def test_a_failed_command_is_a_failed_tool_call_the_model_sees(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(
        monkeypatch,
        completion(None, ("shell", {"command": "echo missing >&2; exit 2"}), id="chatcmpl-1"),
        completion("It failed.", id="chatcmpl-2"),
    )
    try:
        _, close = await _episode(_agent(), access)
    finally:
        model.close()

    assert model.requests[1]["json"]["messages"][-1]["content"] == "missing\n\n[exit code: 2]"
    [tool] = _records(close, "tool_call")
    assert (tool.status, tool.error_type) == ("failed", "exit_code_2")


@linux_only
async def test_an_activation_runs_once_and_a_retry_gets_the_same_response(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, completion("Done.", id="chatcmpl-1"))
    agent, request = _agent(), _request()
    try:
        await agent.seed_agent_session(request, _seed(sandbox_access=access))
        first = await asyncio.wait_for(agent.responses(request, _body()), 120)
        again = await asyncio.wait_for(agent.responses(request, _body()), 10)
        with pytest.raises(HTTPException) as different:
            await agent.responses(request, _body(input="Something else."))
        with pytest.raises(HTTPException) as elsewhere:
            await agent.responses(_request("another-rollout"), _body())
        await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 60)
    finally:
        model.close()

    assert first.id == again.id
    assert len(model.requests) == 1
    assert different.value.status_code == elsewhere.value.status_code == 409


@linux_only
async def test_a_crew_that_runs_out_of_time_answers_incomplete(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    workdir = Path(access.workdir)
    model = _model(
        monkeypatch, completion(None, ("shell", {"command": "echo partial > partial.txt"}), id="chatcmpl-1"), None
    )
    # A five-second total budget sometimes expired while the cold runtime was importing,
    # before the first command. Require the intended blocked second call before checking
    # the real deadline, keeping all assertions about retained progress and cleanup.
    started = time.monotonic()
    episode = asyncio.create_task(_episode(_agent(timeout=30), access))
    try:
        assert await asyncio.to_thread(model.blocked.wait, 25), "the crew never reached its blocked second call"
        assert (workdir / "partial.txt").read_text() == "partial\n"
        response, close = await asyncio.wait_for(episode, 60)
    finally:
        if not episode.done():
            episode.cancel()
        await asyncio.gather(episode, return_exceptions=True)
        model.close()

    assert time.monotonic() - started < 60
    assert (response.status, response.metadata["crewai_stop_reason"]) == ("incomplete", "timeout")
    assert len(model.requests) == 2
    assert (workdir / "partial.txt").read_text() == "partial\n"
    # The second model call never returned, so the run's usage is unknown.
    assert response.usage is None
    assert "agent_run_timeout" in {gap.code for gap in close.agent_observations.gaps}
    assert _records(close, "sandbox")[0].outcome == "timeout"


@linux_only
async def test_close_stops_a_running_crew_and_kills_its_command(lent_sandbox, runtime, monkeypatch):
    owner, access = lent_sandbox
    workdir = Path(access.workdir)
    model = _model(
        monkeypatch,
        completion(None, ("shell", {"command": "touch started.txt; sleep 5; touch late.txt"}), id="chatcmpl-1"),
    )
    agent, request = _agent(), _request()
    try:
        await agent.seed_agent_session(request, _seed(sandbox_access=access))
        activation = asyncio.create_task(agent.responses(request, _body()))
        deadline = time.monotonic() + 60
        while not (workdir / "started.txt").exists():
            assert time.monotonic() < deadline, "the command never started"
            await asyncio.sleep(0.1)
        close = await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 60)
        [ending] = await asyncio.wait_for(asyncio.gather(activation, return_exceptions=True), 10)
    finally:
        model.close()

    [invocation] = _records(close, "agent_invocation")
    assert invocation.status == "incomplete"
    assert _records(close, "sandbox")[0].outcome == "cancelled"
    assert "agent_activation_interrupted" in {gap.code for gap in close.agent_observations.gaps}
    # The stopped runner reports its partial run, which can end the activation before close cancels it.
    if not isinstance(ending, asyncio.CancelledError):
        assert ending.status == "incomplete"
    await asyncio.sleep(6)
    assert not (workdir / "late.txt").exists()
    # The crew made no model call after close.
    assert len(model.requests) == 1
    assert owner.sandbox_id not in LendableLocalProvider.closed


@linux_only
async def test_a_rollout_stop_ends_the_run_as_a_marked_timeout(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, None)
    agent, request = _agent(), _request()
    try:
        await agent.seed_agent_session(request, _seed(sandbox_access=access))
        activation = asyncio.create_task(agent.responses(request, _body()))
        assert await asyncio.to_thread(model.blocked.wait, 60), "the crew never called the model"
        stop = await agent.stop_rollout(_EPISODE.capture_key)
        response = await asyncio.wait_for(activation, 60)
        await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 60)
    finally:
        model.close()

    assert stop.stopped
    assert response.status == "incomplete"
    assert json.loads(response.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": None}


@linux_only
async def test_a_rejected_model_request_fails_the_activation_with_its_status(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, 400)
    try:
        response, close = await _episode(_agent(), access)
    finally:
        model.close()

    assert response.status == "failed"
    assert json.loads(response.metadata["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": 400,
    }
    [invocation] = _records(close, "agent_invocation")
    [record] = _records(close, "sandbox")
    assert invocation.status == "failed"
    assert (record.outcome, record.error_type) == ("failed", invocation.error_type)


@linux_only
async def test_a_short_rate_limit_burst_recovers_with_server_retry_after(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    limited = (429, {"error": {"message": "busy"}}, {"Retry-After": "0.2"})
    model = _model(monkeypatch, limited, limited, limited, limited, completion("Recovered.", id="recovered"))
    try:
        response, close = await _episode(_agent(), access)
    finally:
        model.close()
    assert response.status == "completed"
    assert len(model.requests) == 5
    assert all(
        later["received_at"] - earlier["received_at"] >= 0.18
        for earlier, later in zip(model.requests, model.requests[1:])
    )
    assert response.metadata.get("ng_agent_outcome") is None
    assert [ref.response_id for ref in _records(close, "agent_invocation")[0].model_calls] == ["recovered"]


@linux_only
@pytest.mark.parametrize("status, attempts", [(400, 1), (429, 2)])
async def test_retry_exhaustion_keeps_actual_status_and_deterministic_rejections_do_not_retry(
    lent_sandbox, runtime, monkeypatch, status, attempts
):
    _, access = lent_sandbox
    model = _model(monkeypatch, (status, {"error": {"message": "rejected"}}, {"Retry-After": "0.01"}))
    try:
        response, _ = await _episode(_agent(max_model_retries=1), access)
    finally:
        model.close()
    assert response.status == "failed" and len(model.requests) == attempts
    assert json.loads(response.metadata["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": status,
    }


@linux_only
@pytest.mark.parametrize("stop_method", ["rollout_stop", "deadline"])
async def test_activation_stop_or_deadline_cancels_model_retry_after_sleep(
    lent_sandbox, runtime, monkeypatch, stop_method
):
    _, access = lent_sandbox
    model = _model(monkeypatch, (429, {"error": {"message": "busy"}}, {"Retry-After": "60"}))
    agent, request = _agent(timeout=10 if stop_method == "deadline" else 1800), _request()
    try:
        await agent.seed_agent_session(request, _seed(sandbox_access=access))
        activation = asyncio.create_task(agent.responses(request, _body()))
        async with asyncio.timeout(30):
            while not model.requests:
                await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)
        if stop_method == "rollout_stop":
            stop = await agent.stop_rollout(_EPISODE.capture_key)
            assert stop.stopped
        response = await asyncio.wait_for(activation, 15)
        close = await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 10)
    finally:
        model.close()
    assert close.cleanup_confirmed and response.status == "incomplete"
    assert (ROLLOUT_STOP_METADATA_KEY in response.metadata) == (stop_method == "rollout_stop")
    assert len(model.requests) == 1


@linux_only
async def test_a_transient_model_status_is_retried_with_the_session_header(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, 503, completion("Done.", id="chatcmpl-1"))
    try:
        response, close = await _episode(_agent(), access)
    finally:
        model.close()

    assert response.status == "completed"
    assert len(model.requests) == 2
    # Both attempts carried the session header. Close sends no trajectory of its own, so rollout collection joins the
    # failed attempt, which has no response ID, to the invocation by that header.
    assert [request["headers"]["x-session-id"] for request in model.requests] == [_EPISODE.capture_key] * 2
    assert close.trajectory is None


@linux_only
async def test_usage_is_unknown_when_a_model_call_reports_none(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(
        monkeypatch,
        completion(None, ("shell", {"command": "true"}), id="chatcmpl-1"),
        completion("Done.", id="chatcmpl-2", usage=False),
    )
    try:
        response, close = await _episode(_agent(), access)
    finally:
        model.close()

    assert response.usage is None
    assert "model_call_usage_unavailable" in {gap.code for gap in close.agent_observations.gaps}


@linux_only
async def test_without_a_lent_sandbox_the_agent_runs_in_its_own_and_stops_it(tmp_path, runtime, monkeypatch):
    workdir = tmp_path / "owned"
    workdir.mkdir()
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    started: list[SandboxSpec] = []
    start = AsyncSandbox.start

    async def recording_start(self, spec=None, **options):
        started.append(spec)
        return await start(self, spec, **options)

    monkeypatch.setattr(AsyncSandbox, "start", recording_start)
    model = _model(
        monkeypatch,
        completion(None, ("shell", {"command": "pwd > where.txt"}), id="chatcmpl-1"),
        completion("Done.", id="chatcmpl-2"),
    )
    agent = _agent(sandbox_provider="sandbox", sandbox_config={"workdir": str(workdir)}, timeout=300)
    try:
        response, close = await _episode(agent, None)
    finally:
        model.close()

    assert response.status == "completed"
    assert (workdir / "where.txt").read_text().strip() == str(workdir)
    [spec] = started
    assert spec.ttl_s == 900 + 300 + 600
    assert spec.metadata["nemo_gym_agent"] == "crewai_agent" and spec.metadata["agent_session"] == "agent-session"
    [record] = _records(close, "sandbox")
    assert (record.role, record.outcome, record.teardown_confirmed) == ("agent", "completed", True)
    assert close.cleanup_confirmed is True


@linux_only
async def test_a_large_tool_result_reaches_the_model_whole(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    # Without a context window CrewAI assumes 4,096 tokens, where its compaction would move this result to a file.
    model = _model(
        monkeypatch,
        completion(None, ("shell", {"command": "head -c 10000 /dev/zero | tr '\\0' x"}), id="chatcmpl-1"),
        completion("Done.", id="chatcmpl-2"),
    )
    try:
        await _episode(_agent(), access)
    finally:
        model.close()

    second = model.requests[1]["json"]
    assert second["messages"][-1]["content"] == "x" * 10000 + "\n[exit code: 0]"
    # No tool that would read a moved result is offered.
    assert [tool["function"]["name"] for tool in second["tools"]] == ["shell"]


@linux_only
async def test_each_tool_result_is_recorded_once_in_call_order(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(
        monkeypatch,
        completion(None, ("shell", {"command": "echo one"}), id="chatcmpl-1"),
        completion(None, ("shell", {"command": "echo two"}), id="chatcmpl-2"),
        completion("Done.", id="chatcmpl-3"),
    )
    try:
        response, close = await _episode(_agent(), access)
    finally:
        model.close()

    # Each later model call carries every earlier tool result again; the transcript holds each one once.
    assert [
        (item.type, getattr(item, "call_id", None), getattr(item, "output", None)) for item in response.output
    ] == [
        ("function_call", "chatcmpl-1-call-0", None),
        ("function_call_output", "chatcmpl-1-call-0", "one\n\n[exit code: 0]"),
        ("function_call", "chatcmpl-2-call-0", None),
        ("function_call_output", "chatcmpl-2-call-0", "two\n\n[exit code: 0]"),
        ("message", None, None),
    ]
    assert [tool.tool_call_id for tool in _records(close, "tool_call")] == ["chatcmpl-1-call-0", "chatcmpl-2-call-0"]


@linux_only
async def test_a_call_crewai_refused_does_not_take_the_execution_of_a_later_call(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(
        monkeypatch,
        # CrewAI answers a call with invalid arguments itself, without running the tool.
        completion(None, ("shell", {"cmd": "echo bad"}), id="chatcmpl-1"),
        completion(None, ("shell", {"command": "echo good"}), id="chatcmpl-2"),
        completion("Done.", id="chatcmpl-3"),
    )
    try:
        _, close = await _episode(_agent(), access)
    finally:
        model.close()

    assert [(tool.tool_call_id, tool.status) for tool in _records(close, "tool_call")] == [
        ("chatcmpl-2-call-0", "completed")
    ]


@linux_only
async def test_a_deadline_during_a_command_answers_incomplete_with_the_call_cancelled(
    lent_sandbox, runtime, monkeypatch
):
    _, access = lent_sandbox
    workdir = Path(access.workdir)
    model = _model(
        monkeypatch, completion(None, ("shell", {"command": "echo partial > partial.txt; sleep 30"}), id="chatcmpl-1")
    )
    try:
        response, close = await _episode(_agent(timeout=5), access)
    finally:
        model.close()

    assert response.status == "incomplete"
    assert (workdir / "partial.txt").read_text() == "partial\n"
    [invocation] = _records(close, "agent_invocation")
    [tool] = _records(close, "tool_call")
    assert (invocation.status, invocation.stop_reason) == ("incomplete", "timeout")
    assert (tool.tool_call_id, tool.status, tool.error_type) == ("chatcmpl-1-call-0", "cancelled", "cancelled")
    assert tool.completed_at >= tool.started_at
    assert _records(close, "sandbox")[0].outcome == "timeout"


@linux_only
async def test_a_final_answer_cut_off_at_the_output_cap_is_incomplete(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, completion("Partial answ", id="chatcmpl-1", finish_reason="length"))
    try:
        response, close = await _episode(_agent(), access)
    finally:
        model.close()

    assert response.status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"
    # The invocation agrees with the response; the crew's use of the sandbox ended normally.
    [invocation] = _records(close, "agent_invocation")
    assert (invocation.status, invocation.stop_reason) == ("incomplete", None)
    assert _records(close, "sandbox")[0].outcome == "completed"


@linux_only
async def test_a_rejected_model_request_after_a_command_does_not_run_the_task_again(
    lent_sandbox, runtime, monkeypatch
):
    _, access = lent_sandbox
    workdir = Path(access.workdir)
    model = _model(
        monkeypatch,
        completion(None, ("shell", {"command": "echo run >> runs.txt"}), id="chatcmpl-1"),
        400,
        # What a second attempt at the task would get.
        completion(None, ("shell", {"command": "echo run >> runs.txt"}), id="chatcmpl-2"),
        completion("Done.", id="chatcmpl-3"),
    )
    try:
        response, _ = await _episode(_agent(), access)
    finally:
        model.close()

    assert response.status == "failed"
    assert len(model.requests) == 2
    assert (workdir / "runs.txt").read_text() == "run\n"


@linux_only
async def test_an_empty_answer_is_the_models_invalid_output(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, completion("", id="chatcmpl-1"))
    try:
        response, _ = await _episode(_agent(), access)
    finally:
        model.close()

    assert response.status == "failed"
    assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "agent_invalid_output"}


@linux_only
async def test_model_server_cookies_are_sent_back_on_later_calls(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(
        monkeypatch,
        completion(None, ("shell", {"command": "true"}), id="chatcmpl-1", cookies={"route": "backend-3"}),
        completion("Done.", id="chatcmpl-2"),
    )
    try:
        await _episode(_agent(), access)
    finally:
        model.close()

    assert [request["headers"].get("Cookie") for request in model.requests] == [None, "route=backend-3"]


@linux_only
async def test_a_model_reply_without_an_id_is_a_reference_gap(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, completion("Done.", id=None))
    try:
        _, close = await _episode(_agent(), access)
    finally:
        model.close()

    assert _records(close, "agent_invocation")[0].model_calls == []
    assert "model_call_reference_unavailable" in {gap.code for gap in close.agent_observations.gaps}


@linux_only
async def test_parallel_tool_calls_and_top_p_reach_the_model_with_the_tools(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, completion("Done.", id="chatcmpl-1"))
    try:
        await _episode(_agent(), access, _body(parallel_tool_calls=False, top_p=0.5))
    finally:
        model.close()

    sent = model.requests[0]["json"]
    assert (sent["parallel_tool_calls"], sent["top_p"], len(sent["tools"])) == (False, 0.5, 1)


async def test_an_owned_sandbox_takes_the_network_policy_and_a_lent_one_refuses_it(
    lent_sandbox, runtime, monkeypatch, tmp_path
):
    _, access = lent_sandbox
    with pytest.raises(AgentSeedRejectedError, match="not a lent one"):
        await _agent(network_access="model_only")._seed_agent_session_state(_seed(sandbox_access=access))

    # The local provider has no network policies, so a restricted owned sandbox is refused before it starts.
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    agent = _agent(sandbox_provider="sandbox", sandbox_config={"workdir": str(tmp_path)}, network_access="model_only")
    with pytest.raises(AgentSeedRejectedError):
        await agent._seed_agent_session_state(_seed())


@pytest.mark.parametrize("provider_config", [{"missing_provider": {}}, {"local": 17}])
@pytest.mark.parametrize("borrowed", [False, True])
def test_invalid_provider_configuration_is_a_final_seed_refusal(monkeypatch, provider_config, borrowed):
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"lender": provider_config})
    monkeypatch.setattr(app, "assert_model_url_reachable_from_sandbox", lambda *args, **kwargs: None)
    monkeypatch.setattr(CrewAIAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model:8000/v1")
    connect, start = AsyncMock(), AsyncMock()
    monkeypatch.setattr(AsyncSandbox, "connect", connect)
    monkeypatch.setattr(AsyncSandbox, "start", start)
    agent = _agent(**({} if borrowed else {"sandbox_provider": "lender", "sandbox_config": {"workdir": "/app"}}))
    seed = _seed(sandbox_access=_access()) if borrowed else _seed()

    from nemo_gym.agents import check_agent_seed_rejected

    assert check_agent_seed_rejected(agent.setup_webserver(), seed)
    connect.assert_not_awaited()
    start.assert_not_awaited()


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_connect_failure_survives_provider_cleanup_failure(monkeypatch, cleanup_fails):
    from nemo_gym.core import NgFailureEvent, RequestFailureLog

    original = NgFailureEvent(
        boundary="sandbox", failure_kind="provider_unavailable", failure_reason="connect failed", stage="seed"
    )
    cleanup = NgFailureEvent(
        boundary="sandbox", failure_kind="session_release_failed", failure_reason="close failed", stage="cleanup"
    )
    provider = SimpleNamespace(aclose=AsyncMock(side_effect=cleanup if cleanup_fails else None))
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"lender": {"local": {}}})
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    monkeypatch.setattr(app, "assert_model_url_reachable_from_sandbox", lambda *args, **kwargs: None)
    monkeypatch.setattr(CrewAIAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model:8000/v1")
    monkeypatch.setattr(AsyncSandbox, "connect", AsyncMock(side_effect=original))

    with RequestFailureLog.capture() as failure_log:
        with pytest.raises(NgFailureEvent) as raised:
            await _agent()._seed_agent_session_state(_seed(sandbox_access=_access()))
    assert raised.value is original
    assert raised.value.stage == "seed"
    provider.aclose.assert_awaited_once()
    if cleanup_fails:
        assert cleanup.event_id in failure_log.reply_report(original).events


@linux_only
async def test_explicit_context_rejection_keeps_partial_work_gradable(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, (400, {"error": {"code": "context_length_exceeded", "message": "limit"}}))
    try:
        response, close = await _episode(_agent(), access)
    finally:
        model.close()
    assert response.status == "incomplete"
    assert response.error is None
    assert response.metadata["crewai_stop_reason"] == "context_limit"
    [invocation] = _records(close, "agent_invocation")
    assert invocation.stop_reason == "context_limit"


@pytest.mark.parametrize("with_original_event", [False, True])
async def test_confirmed_deadline_retains_provider_timeout_and_replays_it_on_close(monkeypatch, with_original_event):
    from nemo_gym.core import NgFailureEvent, RequestFailureLog
    from responses_api_agents.crewai_agent.sandbox import CrewAISessionState

    original = NgFailureEvent(
        boundary="sandbox", failure_kind="transport_timeout", failure_reason="provider wait", stage="agent"
    )
    failure = TimeoutError("provider wait expired")
    if with_original_event:
        failure.__cause__ = original
    from responses_api_agents.crewai_agent.sandbox_runner import Transcript

    report = {
        **Transcript(_EPISODE.capture_key).report(),
        "status": "incomplete",
        "stop_reason": "wall_time",
        "error_type": None,
        "error_message": None,
    }
    session = SimpleNamespace(
        artifacts=report,
        workdir="/app",
        stop_request_path="/tmp/stop.request",
        expired=False,
        execution_oom=None,
        sandbox_stopped=False,
        cleanup_confirmed=True,
    )
    state = CrewAISessionState(request=_seed(sandbox_access=_access()), session=session, grants=[])
    agent = _agent()
    monkeypatch.setattr(CrewAIAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model:8000/v1")
    monkeypatch.setattr(CrewAISessionState, "execute", AsyncMock(side_effect=failure))
    monkeypatch.setattr(CrewAISessionState, "close", AsyncMock())
    body = _body()
    state.activation_request = body
    with RequestFailureLog.capture() as activation_log:
        response = await agent._activate(state, agent._instruction(state, body))
    saved = activation_log.reply_report(None)
    assert response.status == "incomplete"
    assert saved.failure_id is None
    observed = saved.events[state.provider_wait_failure.event_id]
    assert observed.failure_kind == "transport_timeout"
    assert observed.foreign_error.message == "provider wait expired"
    if with_original_event:
        assert observed.cause_id == original.event_id
        assert original.event_id in saved.events
    with RequestFailureLog.capture() as close_log:
        await agent._close_agent_session_state(state)
    assert close_log.reply_report(None) == saved


async def test_unclassified_crew_exception_returns_failed_unknown_and_survives_close(monkeypatch, tmp_path):
    from responses_api_agents.crewai_agent import crew
    from responses_api_agents.crewai_agent.sandbox import CrewAISessionState
    from responses_api_agents.crewai_agent.sandbox_runner import run

    build = MagicMock(side_effect=RuntimeError("unexpected crew failure"))
    monkeypatch.setattr(crew, "build_crew", build)
    session = SimpleNamespace(
        artifacts=None,
        workdir=str(tmp_path),
        stop_request_path=str(tmp_path / "stop.request"),
        expired=False,
        execution_oom=None,
        sandbox_stopped=False,
        cleanup_confirmed=True,
    )
    state = CrewAISessionState(request=_seed(sandbox_access=_access(str(tmp_path))), session=session, grants=[])

    async def execute(self, payload, *, timeout, close_timeout):
        self.session.artifacts = await run(payload, scratch=tmp_path, task_environment={})
        return self.session.artifacts

    monkeypatch.setattr(CrewAISessionState, "execute", execute)
    close_state = AsyncMock()
    monkeypatch.setattr(CrewAISessionState, "close", close_state)
    monkeypatch.setattr(CrewAIAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model:8000/v1")
    agent, body = _agent(), _body()
    state.activation_request = body

    response = await agent._activate(state, agent._instruction(state, body))
    close = await agent._close_agent_session_state(state)

    build.assert_called_once()
    assert response.status == "failed"
    assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "unknown"}
    assert response.error.message == "RuntimeError: unexpected crew failure"
    assert state.session.artifacts["model_failure"] is None
    assert state.session.artifacts["infrastructure_error"] is None
    [invocation] = _records(close, "agent_invocation")
    [sandbox] = _records(close, "sandbox")
    assert (invocation.status, invocation.error_type, invocation.model_calls) == ("failed", "RuntimeError", [])
    assert (sandbox.outcome, sandbox.error_type) == ("failed", "RuntimeError")
    assert close.cleanup_confirmed is True
    close_state.assert_awaited_once_with(agent.config.session_close_timeout_seconds)
