# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentCloseSessionResponse
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import DirectHTTPToolAccess
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap
from responses_api_agents.nooa_agent.app import NOOAAgent, NOOASessionState
from responses_api_agents.nooa_agent.result import finalize_run_result
from responses_api_agents.nooa_agent.runner import NOOARunFailure
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_app import request, result, seed
from responses_api_agents.nooa_agent.tests.test_config import agent_config, invocation_config
from responses_api_agents.nooa_agent.tests.test_sandbox_runner import artifact, runner


async def test_close_releases_sandbox_before_returning_replayable_evidence(monkeypatch) -> None:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    instance = NOOAAgent(config=agent_config(nooa=invocation_config(execution_mode="sandboxed")), server_client=client)
    r, _ = runner()
    r.run = AsyncMock(return_value=result())
    r.close = AsyncMock()

    async def seed_state(self, body):
        return NOOASessionState(request=body, runner=r)

    monkeypatch.setattr(NOOAAgent, "_seed_agent_session_state", seed_state)
    req = request()
    await instance.seed_agent_session(req, seed())
    close_body = AgentCloseSessionRequest(agent_session_id="session", episode_id=seed().episode_id)
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    await instance.responses(req, body)
    wrong_episode = close_body.model_copy(update={"episode_id": seed(episode_id={"rollout_id": "other"}).episode_id})
    with pytest.raises(HTTPException, match="episode_id"):
        await instance.close_agent_session(req, wrong_episode)
    with pytest.raises(HTTPException, match="cookie"):
        await instance.close_agent_session(request(session_id="other"), close_body)
    r.close.assert_not_awaited()
    first = await instance.close_agent_session(req, close_body)
    r.close.assert_awaited_once()
    assert first.agent_observations.source == "nooa"
    assert AgentCloseSessionResponse.model_validate(first.model_dump()) == first
    assert set(first.model_dump()) == {"agent_session_id", "agent_observations", "resources_cookies"}
    assert await instance.close_agent_session(req, close_body) == first
    with pytest.raises(HTTPException, match="closing|closed"):
        await instance.responses(req, body)
    r.close.assert_awaited_once()


@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_execution_still_closes_and_returns_partial_evidence(monkeypatch, cancel: bool) -> None:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    instance = NOOAAgent(config=agent_config(nooa=invocation_config(execution_mode="sandboxed")), server_client=client)
    r, _ = runner()
    terminal = asyncio.CancelledError() if cancel else NOOARunFailure(ValueError("worker failed"), result())
    if cancel:
        terminal.nooa_result = result()
    r.run = AsyncMock(side_effect=terminal)
    r.close = AsyncMock()

    async def seed_state(self, body):
        return NOOASessionState(request=body, runner=r)

    monkeypatch.setattr(NOOAAgent, "_seed_agent_session_state", seed_state)
    req = request()
    await instance.seed_agent_session(req, seed())
    close_body = AgentCloseSessionRequest(agent_session_id="session", episode_id=seed().episode_id)
    with pytest.raises(type(terminal)):
        await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    receipt = await instance.close_agent_session(req, close_body)
    reason = NOOATerminationReason.CANCELLED if cancel else NOOATerminationReason.INFRASTRUCTURE_ERROR
    assert reason in {gap.code for gap in receipt.agent_observations.gaps}
    assert await instance.close_agent_session(req, close_body) == receipt
    r.close.assert_awaited_once()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("early_evidence", [False, True])
async def test_late_artifact_cannot_revive_failed_activation(monkeypatch, cancel: bool, early_evidence: bool) -> None:
    instance = NOOAAgent(config=agent_config(), server_client=MagicMock(spec=ServerClient))
    r, _ = runner()
    failed_result = result()
    failed_result.episode.observations.records = [AgentInvocation(invocation_id="root", status="failed")]
    terminal = (
        asyncio.CancelledError("activation cancelled")
        if cancel
        else NOOARunFailure(ValueError("worker failed"), failed_result)
        if early_evidence
        else RuntimeError("worker failed before capture")
    )
    if cancel and early_evidence:
        terminal.nooa_result = failed_result
    r.run = AsyncMock(side_effect=terminal)

    async def capture() -> None:
        r.artifact = artifact()
        r.observations = AgentObservationBundle(
            source="nooa",
            records=[AgentInvocation(invocation_id="late", status="completed")],
            gaps=[ObservationGap(code="sandbox_capture_failed", detail="late capture gap")],
        )

    r.close = AsyncMock(side_effect=capture)
    access = DirectHTTPToolAccess(kind="direct_http", name="resources", required=True, base_url="http://resources")

    async def seed_state(self, body):
        return NOOASessionState(request=body, runner=r, tool_access=access)

    monkeypatch.setattr(NOOAAgent, "_seed_agent_session_state", seed_state)
    req = request()
    await instance.seed_agent_session(req, seed())
    state = instance._require_agent_session("session")
    with pytest.raises(type(terminal)):
        await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    receipt = await instance.close_agent_session(
        req, AgentCloseSessionRequest(agent_session_id="session", episode_id=seed().episode_id)
    )
    reason = NOOATerminationReason.CANCELLED if cancel else NOOATerminationReason.INFRASTRUCTURE_ERROR
    assert state.result.termination_reason is reason
    response, _ = finalize_run_result(state.result)
    assert response.status == ("cancelled" if cancel else "failed")
    assert {reason, "sandbox_capture_failed"} <= {gap.code for gap in receipt.agent_observations.gaps}
    if early_evidence:
        assert {record.invocation_id for record in receipt.agent_observations.records} == {"root", "late"}
    assert receipt.resources_cookies == {
        **(failed_result.resource_cookies if early_evidence else {}),
        "resource": "new",
    }


async def test_sandbox_capture_completes_before_activation_cancel_and_join(monkeypatch) -> None:
    instance = NOOAAgent(config=agent_config(), server_client=MagicMock(spec=ServerClient))
    r, _ = runner()
    started, capturing, release_capture = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []

    async def run(_):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            events.append("cancel")
            error.nooa_result = result()
            await asyncio.sleep(0)
            events.append("join")
            raise

    async def cleanup() -> None:
        events.append("cleanup")
        capturing.set()
        await release_capture.wait()
        events.append("capture")
        r.artifact = artifact()
        r.observations = r.artifact.observations

    r.run, r.close = run, cleanup

    async def seed_state(self, body):
        return NOOASessionState(request=body, runner=r)

    monkeypatch.setattr(NOOAAgent, "_seed_agent_session_state", seed_state)
    req = request()
    await instance.seed_agent_session(req, seed())
    activation = asyncio.create_task(instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(started.wait(), timeout=1)
    closing = asyncio.create_task(
        instance.close_agent_session(
            req, AgentCloseSessionRequest(agent_session_id="session", episode_id=seed().episode_id)
        )
    )
    await asyncio.wait_for(capturing.wait(), timeout=1)
    assert events == ["cleanup"]
    assert not activation.done()
    release_capture.set()
    receipt = await asyncio.wait_for(closing, timeout=1)
    with pytest.raises(HTTPException, match="Agent session is closing") as caught:
        await activation
    assert caught.value.status_code == 409
    assert events == ["cleanup", "capture", "cancel", "join"]
    assert NOOATerminationReason.CANCELLED in {gap.code for gap in receipt.agent_observations.gaps}


@pytest.mark.parametrize("infrastructure_failure", [False, True])
async def test_environment_grades_limits_but_skips_failed_activation_after_late_capture(
    monkeypatch, infrastructure_failure: bool
) -> None:
    from responses_api_agents.nooa_agent.tests.test_environment import _environment, _paths, _request

    instance = NOOAAgent(config=agent_config(), server_client=MagicMock(spec=ServerClient))
    r, _ = runner()
    limited = result(reason=NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED)
    r.run = AsyncMock(
        side_effect=NOOARunFailure(RuntimeError("transport failed"), limited) if infrastructure_failure else None,
        return_value=limited,
    )

    async def capture() -> None:
        r.artifact = artifact()
        r.observations = r.artifact.observations

    r.close = capture

    async def seed_state(self, body):
        return NOOASessionState(request=body, runner=r)

    monkeypatch.setattr(NOOAAgent, "_seed_agent_session_state", seed_state)
    environment, client = _environment(instance)
    response = await environment.run_request(_request())
    assert ("/verify" in _paths(client)) is not infrastructure_failure
    receipt = next(iter(instance._closed_session_records.values())).close_response
    if infrastructure_failure:
        assert response.failure.stage == "agent"
        assert response.result is None
        assert NOOATerminationReason.INFRASTRUCTURE_ERROR in {gap.code for gap in receipt.agent_observations.gaps}
    else:
        assert response.failure is None
        assert response.result.response.status == "incomplete"
        assert response.result.response.error is None
        assert NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED in {
            gap.code for gap in receipt.agent_observations.gaps
        }
