# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from environment_servers.nooa_single_agent_turn.app import (
    NOOASingleAgentTurnEnvironmentServer,
    NOOASingleAgentTurnEnvironmentServerConfig,
    NOOASingleAgentTurnRequest,
    NOOASingleAgentTurnTaskInput,
)
from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServer
from nemo_gym.base_environment_server import BaseEnvironmentServer
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_collection import _episode_record
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.server_utils import ServerClient
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


class _Response:
    ok = True

    def __init__(self, body: dict, *, cookies: dict[str, str] | None = None) -> None:
        self.body = body
        self.cookies = {key: SimpleNamespace(value=value) for key, value in (cookies or {}).items()}

    async def read(self) -> bytes:
        return orjson.dumps(self.body)


def _response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="agent-response",
        created_at=0,
        model="model",
        object="response",
        status="completed",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "answer",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=False,
        tools=[],
    )


def _request(*, timeout: float | None = None) -> NOOASingleAgentTurnRequest:
    return NOOASingleAgentTurnRequest.model_validate(
        {
            "episode_id": {"rollout_id": "rollout", "attempt": 2},
            "task": {
                "task_id": {"taskset": "tests", "task_id": "task"},
                "task_input": {
                    "responses_create_params": {"input": "task"},
                    "task_data": {"instance_id": "task"},
                    "agent_timeout_seconds": timeout,
                },
            },
        }
    )


def _environment() -> tuple[NOOASingleAgentTurnEnvironmentServer, MagicMock]:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {
            "agent": {"responses_api_agents": {"nooa_agent": {"entrypoint": "app.py", "token_id_capture": True}}},
            "token_id_capture": {"enabled": True},
        }
    )
    client._resolve_base_url.return_value = "http://resources:8000"

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        assert current_rollout_id() == "rollout-a2"
        body = kwargs["json"]
        if url_path == "/seed_session":
            return _Response({"resources_session_id": body["resources_session_id"]}, cookies={"resource": "seed"})
        if url_path == "/v1/agent_sessions":
            AgentSeedSessionRequest.model_validate(body)
            return _Response({"agent_session_id": body["agent_session_id"]}, cookies={"agent": "seed"})
        if url_path.endswith("/v1/responses"):
            return _Response(_response().model_dump(mode="json"))
        if url_path in {"/v1/agent_sessions/finish", "/v1/agent_sessions/close"}:
            AgentCloseSessionRequest.model_validate(body)
            return _Response(
                {
                    "agent_session_id": body["agent_session_id"],
                    "resources_cookies": {"resource": "final"},
                    "agent_observations": {"source": "nooa", "gaps": [{"code": "snapshot", "detail": url_path}]},
                }
            )
        if url_path == "/verify":
            return _Response({**body, "reward": 1.0, "benchmark_field": "preserved"})
        if url_path == "/close_session":
            return _Response({"resources_session_id": body["resources_session_id"]})
        raise AssertionError(url_path)

    client.post = AsyncMock(side_effect=post)
    config = NOOASingleAgentTurnEnvironmentServerConfig(
        name="nooa_environment",
        host="localhost",
        port=8002,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "resources"},
        agent_server={"type": "responses_api_agents", "name": "agent"},
        cleanup_timeout_seconds=1,
        resources_tool_transports=["direct_http"],
    )
    return NOOASingleAgentTurnEnvironmentServer(config=config, server_client=client), client


def _paths(client: MagicMock) -> list[str]:
    return [call.kwargs["url_path"] for call in client.post.call_args_list]


async def test_finish_precedes_verification_and_close_precedes_resource_destruction() -> None:
    environment, client = _environment()
    post = client.post.side_effect
    service_alive = False

    async def guarded_post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        nonlocal service_alive
        if url_path == "/v1/agent_sessions":
            service_alive = True
        elif url_path in {"/v1/agent_sessions/finish", "/verify"}:
            assert service_alive
        elif url_path == "/v1/agent_sessions/close":
            service_alive = False
        elif url_path == "/close_session":
            assert not service_alive
        return await post(server_name=server_name, url_path=url_path, **kwargs)

    client.post.side_effect = guarded_post
    request = _request()
    response = await environment.run_request(request)
    assert _paths(client) == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/training-token-capture/v1/responses",
        "/v1/agent_sessions/finish",
        "/verify",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    assert response.failure is None
    assert response.result.response == _response()
    assert response.result.reward == 1
    assert response.result.ng_agent_observations.gaps[0].detail == "/v1/agent_sessions/finish"
    calls = client.post.call_args_list
    assert calls[1].kwargs["json"]["tool_accesses"][0]["cookies"] == {"resource": "seed"}
    assert calls[2].kwargs["cookies"] == {"agent": "seed"}
    assert calls[4].kwargs["cookies"] == {"resource": "final"}
    assert calls[6].kwargs["cookies"] == {"resource": "final"}
    assert calls[4].kwargs["json"]["instance_id"] == "task"
    record = _episode_record(response.model_dump(mode="json"))
    assert record["ng_agent_observations"]["source"] == "nooa"
    assert record["benchmark_field"] == "preserved"
    assert "ng_trajectory" not in record
    assert environment.run_request.__func__ is BaseEnvironmentServer.run_request
    assert request.task.task_input.task_data == {"instance_id": "task"}


@pytest.mark.parametrize("stage", ["agent", "finish", "verification"])
async def test_failure_uses_existing_failure_contract_and_closes_before_resources(stage: str) -> None:
    environment, client = _environment()
    original = client.post.side_effect
    failing_path = {"agent": "/v1/responses", "finish": "/finish", "verification": "/verify"}[stage]

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path.endswith(failing_path):
            raise RuntimeError(f"{stage} failed")
        return await original(server_name=server_name, url_path=url_path, **kwargs)

    client.post.side_effect = post
    response = await environment.run_request(_request())
    assert response.failure.stage == ("cleanup" if stage == "finish" else stage)
    assert response.failure.failure_reason == f"{stage} failed"
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]
    if stage != "verification":
        assert "/verify" not in _paths(client)
    assert response.failure.partial_response == (None if stage == "agent" else _response())
    record = _episode_record(response.model_dump(mode="json"))
    assert "ng_agent_observations" not in record
    assert "ng_trajectory" not in record
    assert "_ng_failure_cleanup_error" not in record


@pytest.mark.parametrize("cancel", [False, True], ids=["episode-timeout", "caller-cancellation"])
async def test_interrupted_verification_closes_agent_before_resources(cancel: bool) -> None:
    environment, client = _environment()
    environment.config.default_episode_timeout_seconds = 0.1
    original = client.post.side_effect
    verifying = asyncio.Event()

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path == "/verify":
            verifying.set()
            await asyncio.Event().wait()
        return await original(server_name=server_name, url_path=url_path, **kwargs)

    client.post.side_effect = post
    task = asyncio.create_task(environment.run_request(_request()))
    await asyncio.wait_for(verifying.wait(), 1)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        response = await asyncio.wait_for(task, 1)
        assert response.failure.failure_reason == "Episode timed out"
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]


async def test_final_close_retries_before_destroying_resources_without_erasing_verdict() -> None:
    environment, client = _environment()
    original = client.post.side_effect
    attempts = 0

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        nonlocal attempts
        if url_path == "/v1/agent_sessions/close":
            attempts += 1
            if attempts == 1:
                raise RuntimeError("cleanup temporarily unavailable")
        if url_path == "/close_session":
            assert attempts == 2
        return await original(server_name=server_name, url_path=url_path, **kwargs)

    client.post.side_effect = post
    response = await environment.run_request(_request())
    assert response.result.reward == 1
    assert _paths(client)[-3:] == ["/v1/agent_sessions/close", "/v1/agent_sessions/close", "/close_session"]


async def test_stalled_finish_is_bounded_and_never_grades() -> None:
    environment, client = _environment()
    environment.config.cleanup_timeout_seconds = 0.01
    original = client.post.side_effect

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path == "/v1/agent_sessions/finish":
            await asyncio.Event().wait()
        return await original(server_name=server_name, url_path=url_path, **kwargs)

    client.post.side_effect = post
    response = await asyncio.wait_for(environment.run_request(_request()), 0.5)
    assert response.failure.stage == "cleanup"
    assert response.failure.terminal is False
    assert "/verify" not in _paths(client)
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]


@pytest.mark.parametrize("stop_fails", [False, True])
async def test_agent_deadline_grades_artifacts_only_after_confirmed_stop(stop_fails: bool) -> None:
    environment, client = _environment()
    original = client.post.side_effect
    stopped = False

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        nonlocal stopped
        if url_path.endswith("/v1/responses"):
            await asyncio.Event().wait()
        if url_path == "/v1/agent_sessions/close":
            if stop_fails:
                raise RuntimeError("cleanup unconfirmed")
            stopped = True
        if url_path in {"/seed_session", "/verify"}:
            # Setup and verification do not consume the agent's execution budget.
            await asyncio.sleep(0.02)
        if url_path == "/verify":
            assert stopped
        return await original(server_name=server_name, url_path=url_path, **kwargs)

    client.post.side_effect = post
    response = await asyncio.wait_for(environment.run_request(_request(timeout=0.01)), 1)
    assert "/v1/agent_sessions/finish" not in _paths(client)
    if stop_fails:
        assert response.failure.stage == "cleanup"
        assert "/verify" not in _paths(client)
        return
    assert response.result.reward == 1
    assert response.result.agent_timed_out is True
    assert response.result.agent_timeout_seconds == 0.01
    partial = response.result.response
    assert partial.status == "incomplete"
    assert partial.output == []
    assert partial.usage is None
    assert partial.metadata["response_source"] == "nooa_environment_timeout_envelope"
    assert _paths(client)[-3:] == ["/v1/agent_sessions/close", "/verify", "/close_session"]


async def test_transport_timeout_does_not_authorize_artifact_grading() -> None:
    environment, client = _environment()
    original = client.post.side_effect

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path.endswith("/v1/responses"):
            raise TimeoutError("upstream socket timeout")
        return await original(server_name=server_name, url_path=url_path, **kwargs)

    client.post.side_effect = post
    response = await environment.run_request(_request(timeout=10))
    assert response.failure.stage == "agent"
    assert response.failure.failure_reason == "upstream socket timeout"
    assert response.failure.terminal is False
    assert "/verify" not in _paths(client)


@pytest.mark.parametrize("termination", ["infrastructure_error", "cancelled"])
async def test_agent_deadline_respects_fatal_evidence_from_close(termination: str) -> None:
    environment, client = _environment()
    original = client.post.side_effect

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path.endswith("/v1/responses"):
            # The activation reply can be lost after execution has already failed.
            await asyncio.Event().wait()
        response = await original(server_name=server_name, url_path=url_path, **kwargs)
        if url_path == "/v1/agent_sessions/close":
            response.body["agent_observations"] = {
                "source": "nooa",
                "gaps": [{"code": termination, "detail": "model endpoint unavailable"}],
            }
        return response

    client.post.side_effect = post
    response = await asyncio.wait_for(environment.run_request(_request(timeout=0.01)), 1)
    paths = _paths(client)
    assert paths[-1] == "/close_session"
    assert paths.count("/v1/agent_sessions/close") == 1
    if termination == "infrastructure_error":
        assert response.result is None
        assert response.failure.stage == "agent"
        assert "model endpoint unavailable" in response.failure.failure_reason
        assert "/verify" not in paths
    else:
        assert response.failure is None
        assert response.result.reward == 1
        assert response.result.agent_timed_out is True
        assert paths.index("/v1/agent_sessions/close") < paths.index("/verify")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_agent_timeout_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValidationError):
        _request(timeout=timeout)


@pytest.mark.parametrize("flat", [False, True])
def test_nooa_timeout_normalization_does_not_change_the_shared_task_contract(flat: bool) -> None:
    fields = {"instance_id": "task"}
    task = {
        "responses_create_params": {"input": "task"},
        "agent_timeout_seconds": 12.5,
        **(fields if flat else {"task_data": fields}),
    }
    nooa = NOOASingleAgentTurnTaskInput.model_validate(task)
    assert nooa.agent_timeout_seconds == 12.5
    assert nooa.task_data == fields
    assert NOOASingleAgentTurnTaskInput.model_validate_json(nooa.model_dump_json()) == nooa
    shared = SingleAgentTurnTaskInput.model_validate(task)
    assert shared.task_data == {**fields, "agent_timeout_seconds": 12.5}
    assert SingleAgentTurnEnvironmentServer.request_model is not NOOASingleAgentTurnRequest


def test_environment_only_exposes_nooa_request_contract() -> None:
    environment, _ = _environment()
    schema = environment.setup_webserver().openapi()
    assert schema["paths"]["/run"]["post"]["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/NOOASingleAgentTurnRequest"
    }
    assert AgentCloseSessionResponse.model_fields.keys() == {
        "agent_session_id",
        "agent_observations",
        "resources_cookies",
    }


@pytest.mark.parametrize("lost_path", ["/seed_session", "/v1/agent_sessions"])
async def test_lost_seed_reply_still_closes_the_caller_assigned_session(lost_path: str) -> None:
    environment, client = _environment()
    original = client.post.side_effect

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path == lost_path:
            raise TimeoutError("seed reply lost")
        return await original(server_name=server_name, url_path=url_path, **kwargs)

    client.post.side_effect = post
    response = await environment.run_request(_request())
    assert response.failure.terminal is False
    assert "/verify" not in _paths(client)
    assert _paths(client)[-1] == "/close_session"
    calls = client.post.call_args_list
    resources_id = calls[0].kwargs["json"]["resources_session_id"]
    assert calls[-1].kwargs["json"]["resources_session_id"] == resources_id
    if lost_path == "/v1/agent_sessions":
        assert _paths(client)[-2] == "/v1/agent_sessions/close"
        assert calls[-2].kwargs["json"]["agent_session_id"] == calls[1].kwargs["json"]["agent_session_id"]
    else:
        assert "/v1/agent_sessions" not in _paths(client)


async def test_mismatched_finish_receipt_blocks_verification() -> None:
    environment, client = _environment()
    original = client.post.side_effect

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path == "/v1/agent_sessions/finish":
            return _Response({"agent_session_id": "another-session"})
        return await original(server_name=server_name, url_path=url_path, **kwargs)

    client.post.side_effect = post
    response = await environment.run_request(_request())
    assert response.failure.stage == "cleanup"
    assert "/verify" not in _paths(client)
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]
