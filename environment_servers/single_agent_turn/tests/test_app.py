# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import orjson
import pytest
from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

import nemo_gym.server_utils
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
    _is_retryable_dependency_error,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
)
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import BaseServerConfig, ServerClient, SimpleServer
from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest, SingleAgentTurnResponse, SingleAgentTurnTaskInput


class _Cookie:
    value = "cookie-value"


def test_module_exports_app_for_multi_worker_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_app = object()
    monkeypatch.setattr(nemo_gym.server_utils, "is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))

    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "app.py"),
        run_name="environment_servers.single_agent_turn.worker_test",
    )

    assert namespace["app"] is worker_app


class _Response:
    ok = True
    cookies = {"session": _Cookie()}

    def __init__(self, body: dict) -> None:
        self.body = orjson.dumps(body)

    async def read(self) -> bytes:
        return self.body


def _agent_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Patch applied", "annotations": []}],
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [-0.1, -0.2],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    responses: list[_Response | Exception]
    fail_path: str | None = None
    slow_path: str | None = None
    rollout_ids: list[str | None] = []

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            body = body.model_dump(exclude_unset=True)
        if body is not None:
            encoded = orjson.dumps(body)
            if url_path == "/v1/agent_sessions":
                AgentSeedSessionRequest.model_validate_json(encoded)
            kwargs["json"] = orjson.loads(encoded)
        self.calls.append((server_name, url_path, kwargs))
        self.rollout_ids.append(current_rollout_id())
        if url_path == self.slow_path:
            await asyncio.sleep(0.2)
        if url_path == self.fail_path:
            self.fail_path = None
            raise TimeoutError(f"response lost for {url_path}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        payload = orjson.loads(response.body)
        body = kwargs.get("json")
        if url_path == "/seed_session" and "resources_session_id" in body:
            payload["resources_session_id"] = body["resources_session_id"]
        elif url_path == "/v1/agent_sessions":
            payload["agent_session_id"] = body["agent_session_id"]
        elif url_path == "/v1/agent_sessions/close":
            payload["agent_session_id"] = body["agent_session_id"]
        elif url_path == "/close_session":
            payload["resources_session_id"] = body["resources_session_id"]
        return _Response(payload)

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _environment_server(
    *,
    token_capture: bool = False,
    resources_tool_transports: list[Literal["direct_http", "mcp"]] | None = None,
) -> tuple[SingleAgentTurnEnvironmentServer, _Client]:
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"test": {"host": "resources", "port": 8000, "entrypoint": "app.py"}}},
            "agent": {
                "responses_api_agents": {
                    "test": {
                        "host": "agent",
                        "port": 8001,
                        "entrypoint": "app.py",
                        "token_id_capture": token_capture,
                    }
                }
            },
            "token_id_capture": {"enabled": token_capture},
        }
    )
    response = _agent_response()
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        calls=[],
        responses=[
            _Response({"resources_session_id": "resources-session"}),
            _Response({"agent_session_id": "agent-session"}),
            _Response(response.model_dump(mode="json")),
            _Response(
                {
                    "agent_session_id": "agent-session",
                    "resources_cookies": {"session": "updated-cookie"},
                }
            ),
            _Response(
                {
                    "responses_create_params": {"input": "task"},
                    "response": response.model_dump(mode="json"),
                    "reward": 1.0,
                    "benchmark_field": "preserved",
                }
            ),
            _Response({"resources_session_id": "resources-session"}),
        ],
    )
    config = SingleAgentTurnEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        cleanup_timeout_seconds=10,
        resources_tool_transports=resources_tool_transports or [],
    )
    return SingleAgentTurnEnvironmentServer(config=config, server_client=client), client


def _request() -> SingleAgentTurnRequest:
    return SingleAgentTurnRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(taskset="test:source", task_id="task"),
            task_input=SingleAgentTurnTaskInput(
                responses_create_params={"input": "task"},
                task_data={"instance_id": "task"},
            ),
        ),
    )


async def test_single_agent_turn_with_direct_resources_tools() -> None:
    environment_server, client = _environment_server(resources_tool_transports=["direct_http"])
    result = await environment_server.run_request(_request())

    assert result.result is not None
    assert result.result.reward == 1.0
    assert result.result.model_dump()["benchmark_field"] == "preserved"
    # The verify response is the result itself, so stored rollout records keep the flat verify-response shape.
    assert "verification" not in result.result.model_dump()
    assert result.result.model_dump()["response"]["id"] == "response"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    # /verify gets the Resources Server's own flat body, the one an Agent's /run sends, under the session cookie.
    verify_call = client.calls[4][2]
    verify_body = BaseVerifyRequest.model_validate(verify_call["json"])
    assert verify_call["json"]["instance_id"] == "task"
    assert verify_body.responses_create_params.input == "task"
    assert verify_body.response.id == "response"
    assert not {"episode_id", "task_id", "verification_input"} & verify_call["json"].keys()
    assert verify_call["cookies"] == {"session": "updated-cookie"}
    resources_seed_body = ResourcesSeedSessionRequest.model_validate(client.calls[0][2]["json"])
    create_body = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])
    agent_close_body = AgentCloseSessionRequest.model_validate(client.calls[3][2]["json"])
    resources_close_body = ResourcesCloseSessionRequest.model_validate(client.calls[5][2]["json"])
    assert resources_seed_body.resources_session_id == resources_close_body.resources_session_id
    assert create_body.agent_session_id == agent_close_body.agent_session_id
    assert create_body.task_id == TaskId(taskset="test:source", task_id="task")
    [tool_access] = create_body.tool_accesses
    assert tool_access.name == "resources.direct_http"
    assert str(tool_access.base_url) == "http://resources:8000/"
    assert tool_access.cookies == {"session": "cookie-value"}
    assert client.calls[1][2]["json"]["tool_accesses"][0]["kind"] == "direct_http"
    assert client.calls[2][2]["cookies"] == {"session": "cookie-value"}
    assert client.calls[3][2]["cookies"] == {"session": "cookie-value"}
    assert agent_close_body.episode_id == EpisodeId(rollout_id="rollout", attempt=2)
    assert client.calls[4][2]["cookies"] == {"session": "updated-cookie"}
    assert client.calls[5][2]["cookies"] == {"session": "updated-cookie"}
    assert resources_close_body.episode_id == EpisodeId(rollout_id="rollout", attempt=2)


async def test_single_agent_turn_translates_resources_mcp_metadata_to_canonical_tool_access() -> None:
    environment_server, client = _environment_server(resources_tool_transports=["direct_http", "mcp"])
    client.responses[0] = _Response(
        {
            "resources_session_id": "resources-session",
            "resources_tools": {
                "server_name": "resources",
                "headers": {"Authorization": "Bearer scoped"},
            },
        }
    )

    await environment_server.run_request(_request())

    create_body = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])
    direct_access, mcp_access = create_body.tool_accesses
    assert direct_access.name == "resources.direct_http"
    assert mcp_access.name == "resources"
    assert mcp_access.required is True
    assert mcp_access.connection.transport == "streamable_http"
    assert str(mcp_access.connection.url) == "http://resources:8000/mcp"
    assert mcp_access.connection.headers == {"Authorization": "Bearer scoped"}
    assert client.calls[1][2]["json"]["tool_accesses"][1]["kind"] == "mcp"
    assert client.calls[1][2]["json"]["tool_accesses"][1]["connection"]["transport"] == "streamable_http"


async def test_token_capture_keeps_prefixed_twin_route() -> None:
    environment_server, client = _environment_server(token_capture=True)

    await environment_server.run_request(_request())

    assert client.calls[2][1] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"


async def test_lost_resources_seed_response_closes_caller_assigned_session() -> None:
    environment_server, client = _environment_server()
    client.fail_path = "/seed_session"

    result = await environment_server.run_request(_request())

    assert result.failure is not None
    assert result.failure.stage == "seed"
    assert [path for _, path, _ in client.calls] == ["/seed_session", "/close_session"]
    seed_body = ResourcesSeedSessionRequest.model_validate(client.calls[0][2]["json"])
    close_body = ResourcesCloseSessionRequest.model_validate(client.calls[1][2]["json"])
    assert seed_body.resources_session_id == close_body.resources_session_id


async def test_lost_agent_seed_response_closes_both_caller_assigned_sessions() -> None:
    environment_server, client = _environment_server()
    client.fail_path = "/v1/agent_sessions"

    result = await environment_server.run_request(_request())

    assert result.failure is not None
    assert result.failure.stage == "agent"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    agent_seed_body = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])
    agent_close_body = AgentCloseSessionRequest.model_validate(client.calls[2][2]["json"])
    assert agent_seed_body.agent_session_id == agent_close_body.agent_session_id


async def test_lost_agent_close_response_is_a_retryable_failure_before_verification() -> None:
    environment_server, client = _environment_server()
    client.fail_path = "/v1/agent_sessions/close"

    result = await environment_server.run_request(_request())

    assert result.failure is not None
    assert result.failure.stage == "cleanup"
    assert result.failure.terminal is False
    assert "/verify" not in [path for _, path, _ in client.calls]


async def test_resources_close_failure_after_verification_keeps_result() -> None:
    environment_server, client = _environment_server()
    client.fail_path = "/close_session"

    result = await environment_server.run_request(_request())

    assert result.result is not None
    assert result.result.reward == 1.0
    assert [path for _, path, _ in client.calls].count("/close_session") == 2


async def test_slow_resources_close_after_verification_keeps_result() -> None:
    environment_server, client = _environment_server()
    environment_server.config.default_episode_timeout_seconds = 0.1
    client.slow_path = "/close_session"

    result = await environment_server.run_request(_request())

    assert result.failure is None
    assert result.result.reward == 1.0
    assert client.calls[-1][1] == "/close_session"


async def test_every_downstream_call_carries_the_attempt_rollout_id() -> None:
    environment_server, client = _environment_server()

    await environment_server.run_request(_request())

    assert [path for _, path, _ in client.calls][-2:] == ["/verify", "/close_session"]
    assert client.rollout_ids == ["rollout-a2"] * len(client.calls)


def test_dependency_failure_reasons_are_bounded() -> None:
    environment_server, _ = _environment_server()
    error = environment_server._failure(stage="agent", failure_reason="x" * 3000, terminal=False)
    assert len(error.failure.failure_reason) == 2000


def test_retry_requires_a_transient_dependency_error() -> None:
    assert _is_retryable_dependency_error(TimeoutError()) is True
    assert _is_retryable_dependency_error(ClientPayloadError("response body interrupted")) is True
    assert _is_retryable_dependency_error(ValueError("invalid response")) is False


async def test_verifier_only_episode_does_not_expose_resource_tools() -> None:
    environment, client = _environment_server()
    seed = orjson.loads(client.responses[0].body)
    seed["sandbox_access"] = {
        "connection": {"kind": "direct", "provider_config_ref": "runtime", "descriptor": {"sandbox_id": "task"}},
        "workdir": "/app",
    }
    client.responses[0] = _Response(seed)
    result = await environment.run_request(_request())

    assert result.result.reward == 1.0
    assert client.calls[1][2]["json"]["tool_accesses"] == []
    assert client.calls[1][2]["json"]["sandbox_access"] is not None


@pytest.mark.parametrize("reward", [0.0, 1.0])
@pytest.mark.parametrize("mask_sample", [False, True])
async def test_failed_agent_response_still_reaches_verification(reward: float, mask_sample: bool) -> None:
    environment, client = _environment_server()
    failed_response = _agent_response().model_dump(mode="json")
    failed_response.update(
        status="failed",
        error={"code": "server_error", "message": "Model generated invalid tool call: finish"},
        metadata={"partial": "true", "turns": "26"},
    )
    client.responses[2] = _Response(failed_response)
    verification = orjson.loads(client.responses[4].body)
    verification.update(response=failed_response, reward=reward, mask_sample=mask_sample)
    client.responses[4] = _Response(verification)

    result = await environment.run_request(_request())

    assert result.failure is None
    assert [path for _, path, _ in client.calls][-3:] == [
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    forwarded = NeMoGymResponse.model_validate(client.calls[4][2]["json"]["response"])
    assert forwarded == NeMoGymResponse.model_validate(failed_response)
    assert client.calls[4][2]["cookies"] == {"session": "updated-cookie"}
    assert result.result.response == forwarded
    assert result.result.reward == reward
    assert result.result.mask_sample is mask_sample
    assert result.result.response.status == "failed"


@pytest.mark.parametrize("mask_sample", [False, True])
@pytest.mark.parametrize("result_path", ["native", "flat-adapter"])
async def test_results_preserve_verification_and_observations(mask_sample: bool, result_path: str) -> None:
    environment, client = _environment_server()
    observations = {
        "source": "hermes",
        "records": [
            {
                "kind": "agent_invocation",
                "invocation_id": "root",
                "status": "completed",
                "model_calls": [{"model_call_id": "model-call-1"}],
            }
        ],
    }
    close_body = orjson.loads(client.responses[3].body)
    close_body["agent_observations"] = observations
    client.responses[3] = _Response(close_body)
    verification_body = orjson.loads(client.responses[4].body)
    verification_body.update(mask_sample=mask_sample, grader_output={"resolved": True})
    client.responses[4] = _Response(verification_body)

    result = await environment.run_request(_request())
    result = SingleAgentTurnResponse.model_validate_json(result.model_dump_json())
    assert result.result.mask_sample is mask_sample
    assert result.result.reward == 1.0
    assert result.result.model_extra["benchmark_field"] == "preserved"
    assert result.result.model_extra["grader_output"] == {"resolved": True}
    assert result.result.response == _agent_response()
    assert result.result.ng_agent_observations == AgentObservationBundle.model_validate(observations)
    if result_path != "flat-adapter":
        return
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment.config, server_client=client)
    legacy = adapter._legacy_result(result)
    assert legacy["mask_sample"] is mask_sample
    assert legacy["reward"] == 1.0
    assert legacy["benchmark_field"] == "preserved"
    assert legacy["grader_output"] == {"resolved": True}
    assert legacy["response"] == _agent_response().model_dump(mode="json")
    assert legacy["ng_agent_observations"] == AgentObservationBundle.model_validate(observations).model_dump(
        mode="json"
    )


@pytest.mark.parametrize("stage", ["agent", "verification"])
@pytest.mark.parametrize(
    "error",
    [
        ClientConnectionError("connection lost"),
        ValueError("invalid payload"),
        ClientResponseError(
            request_info=SimpleNamespace(real_url="http://agent/v1/responses"),
            history=(),
            status=500,
            message="Hermes agent failed: HTTP 429 Too Many Requests",
        ),
    ],
)
async def test_dependency_failure_closes_sessions(stage: str, error: Exception) -> None:
    environment, client = _environment_server()
    responses = client.responses
    if stage == "agent":
        client.responses = [*responses[:2], error, responses[3], responses[5]]
    else:
        client.responses = [*responses[:4], error, responses[5]]

    result = await environment.run_request(_request())

    assert result.result is None
    assert result.failure.stage == stage
    assert result.failure.failure_reason == str(error)
    assert result.failure.terminal is isinstance(error, ValueError)
    assert result.failure.partial_response == (_agent_response() if stage == "verification" else None)
    paths = [path for _, path, _ in client.calls]
    assert paths == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        *(["/verify"] if stage == "verification" else []),
        "/close_session",
    ]
    assert not client.responses


async def test_admission_timeout_is_retryable_without_starting_sessions() -> None:
    environment, client = _environment_server()
    environment = SingleAgentTurnEnvironmentServer(
        config=environment.config.model_copy(update={"max_concurrent_episodes": 1, "queue_timeout_seconds": 0.01}),
        server_client=client,
    )
    async with environment._admission:
        result = await environment.run_request(_request())

    assert result.result is None
    assert result.failure.failure_reason == "Episode admission timed out"
    assert result.failure.terminal is False
    assert client.calls == []


@pytest.mark.parametrize("cancel", [False, True], ids=["episode-timeout", "caller-cancellation"])
async def test_interrupted_activation_closes_agent_before_resources(
    monkeypatch: pytest.MonkeyPatch, *, cancel: bool
) -> None:
    environment, client = _environment_server()
    environment.config.default_episode_timeout_seconds = 0.05
    responses = client.responses
    client.responses = [*responses[:2], responses[3], responses[5]]
    activation_started = asyncio.Event()
    original_post = _Client.post

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path.endswith("/v1/responses"):
            self.calls.append((server_name, url_path, kwargs))
            activation_started.set()
            await asyncio.Event().wait()
            raise AssertionError("Activation must be interrupted")
        return await original_post(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    run_task = asyncio.create_task(environment.run_request(_request()))
    await asyncio.wait_for(activation_started.wait(), timeout=1)
    if cancel:
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
    else:
        result = await asyncio.wait_for(run_task, timeout=1)
        assert result.result is None
        assert result.failure.failure_reason == "Episode timed out"
        assert result.failure.terminal is False

    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    assert not client.responses
