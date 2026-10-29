# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import runpy
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import orjson
import pytest
from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
    SingleAgentTurnResult,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentOutcome,
    AgentOutcomePolicyConfig,
    AgentSeedSessionRequest,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import (
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseServerConfig,
    ResourcesServerRef,
)
from nemo_gym.core import MODEL_TRANSPORT_ERROR, EpisodeId, NeMoGymResponse, TaskId, current_rollout_id
from nemo_gym.server_runtime import (
    BaseVerifyRequest,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    SingleAgentTurnRequest,
    SingleAgentTurnResponse,
    is_retryable_dependency_error,
)
from nemo_gym.server_runtime.episode_types import MaterializedTask
from nemo_gym.server_runtime.server_utils import SimpleServer
from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnTaskInput
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord


class _Cookie:
    value = "cookie-value"


def test_module_exports_app_for_multi_worker_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_app = object()
    monkeypatch.setattr("nemo_gym.utils.server_process.is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))

    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "app.py"),
        run_name="environment_servers.single_agent_turn.worker_test",
    )

    assert namespace["app"] is worker_app


class _Response:
    ok = True
    cookies = {"session": _Cookie()}

    def __init__(self, body: dict, headers: dict | None = None) -> None:
        self.body = orjson.dumps(body)
        self.headers = headers or {}

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
        return type(response)(payload, response.headers)

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _environment_server(
    *,
    token_capture: bool = False,
    resources_tool_transports: list[Literal["direct_http", "mcp"]] | None = None,
    client_type: type[_Client] = _Client,
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
    client = client_type(
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


async def test_hung_agent_close_still_closes_resources_without_verifying() -> None:
    environment_server, client = _environment_server()
    environment_server.config.cleanup_timeout_seconds = 0.02
    client.slow_path = "/v1/agent_sessions/close"
    # Neither timed-out agent close consumes a response; Resources closes next.
    client.responses[3:] = [_Response({"resources_session_id": "resources-session"})]

    result = await environment_server.run_request(_request())

    assert result.failure is not None
    assert result.failure.stage == "cleanup"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/v1/agent_sessions/close",
        "/close_session",
    ]


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
    assert is_retryable_dependency_error(TimeoutError()) is True
    assert is_retryable_dependency_error(ClientPayloadError("response body interrupted")) is True
    assert is_retryable_dependency_error(ValueError("invalid response")) is False


@pytest.mark.parametrize(("status", "retryable"), [(422, False), (409, False), (429, True), (500, True)])
def test_an_agent_seed_refusal_is_final_but_a_server_error_is_retried(status: int, retryable: bool) -> None:
    # An agent refuses a seed it cannot serve with AgentSeedRejectedError, which answers 422.
    refusal = ClientResponseError(
        request_info=SimpleNamespace(real_url="http://agent/v1/agent_sessions"), history=(), status=status
    )

    assert is_retryable_dependency_error(refusal) is retryable


async def test_an_agent_seed_refusal_fails_the_episode_once_with_the_agents_reason() -> None:
    environment, client = _environment_server()
    # What raise_for_status raises for an agent's AgentSeedRejectedError.
    refusal = ClientResponseError(
        request_info=SimpleNamespace(real_url="http://agent:8001/v1/agent_sessions"),
        history=(),
        status=422,
        message="Unprocessable Content",
    )
    refusal.response_content = b'{"detail":"Simple Agent does not support required MCP tool access: resources"}'
    responses = client.responses
    client.responses = [responses[0], refusal, responses[3], responses[5]]

    result = await environment.run_request(_request())

    assert result.result is None
    assert result.failure.stage == "agent"
    assert result.failure.terminal is True
    assert "Simple Agent does not support required MCP tool access: resources" in result.failure.failure_reason
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/v1/agent_sessions/close",
        "/close_session",
    ]


async def test_a_resources_seed_refusal_fails_the_episode_once_with_the_servers_reason() -> None:
    environment, client = _environment_server()
    # What raise_for_status raises for a resources server's ResourcesSessionConflictError.
    refusal = ClientResponseError(
        request_info=SimpleNamespace(real_url="http://resources:8000/seed_session"),
        history=(),
        status=409,
        message="Conflict",
    )
    refusal.response_content = b'{"detail":"resources_session_id is already bound to another episode or task"}'
    responses = client.responses
    client.responses = [refusal, responses[5]]

    result = await environment.run_request(_request())

    assert result.result is None
    assert result.failure.stage == "seed"
    assert result.failure.terminal is True
    assert "already bound to another episode or task" in result.failure.failure_reason
    # The close registered before the seed still runs, so a seed whose reply was a refusal leaks nothing.
    assert [path for _, path, _ in client.calls] == ["/seed_session", "/close_session"]


@pytest.mark.parametrize(
    ("stage", "path", "status", "detail"),
    [
        ("agent", "/ng-rollout/rollout-a2/v1/responses", 422, "Agent turn refused: unsupported tool_choice"),
        ("cleanup", "/v1/agent_sessions/close", 409, "Agent close receipt has expired"),
        ("verification", "/verify", 422, "verify refused: task_data is missing expected_answer"),
    ],
)
async def test_every_dependency_refusal_carries_the_servers_reason(
    stage: str, path: str, status: int, detail: str
) -> None:
    environment, client = _environment_server()
    refusal = ClientResponseError(
        request_info=SimpleNamespace(real_url=f"http://dependency:8000{path}"), history=(), status=status
    )
    refusal.response_content = orjson.dumps({"detail": detail})
    responses = client.responses
    client.responses = {
        "agent": [responses[0], responses[1], refusal, responses[3], responses[5]],
        # The final unwind retries the refused agent close once before closing resources.
        "cleanup": [responses[0], responses[1], responses[2], refusal, responses[3], responses[5]],
        "verification": [responses[0], responses[1], responses[2], responses[3], refusal, responses[5]],
    }[stage]

    result = await environment.run_request(_request())

    assert result.result is None
    assert result.failure.stage == stage
    assert result.failure.terminal is True
    assert detail in result.failure.failure_reason


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


def _skipping_environment_server(
    *, environment: dict[str, Any], agent: dict[str, Any]
) -> tuple[SingleAgentTurnEnvironmentServer, _Client]:
    environment_server, client = _environment_server()
    client.global_config_dict.agent.responses_api_agents.test.update(agent)
    environment_server.config = SingleAgentTurnEnvironmentServerConfig.model_validate(
        {**environment_server.config.model_dump(), **environment}
    )
    # No /verify reply is consumed.
    del client.responses[4]
    return environment_server, client


@pytest.mark.parametrize(
    ("environment", "agent", "reward"),
    [
        ({"skip_verification": True, "skip_verification_reward": 0.5}, {}, 0.5),
        ({"skip_verification": True}, {}, 0.0),
        # Unset here, the bound agent's setting applies; Gym copies the global flag into it.
        ({}, {"skip_verification": True, "skip_verification_reward": -1.5}, -1.5),
        ({"skip_verification": True}, {"skip_verification_reward": 2.0}, 2.0),
        ({"skip_verification_reward": 3.0}, {"skip_verification": True, "skip_verification_reward": 0.25}, 3.0),
        # Strings, as a quoted override or an ${oc.env:...} interpolation leaves them, parse as the agent parses them.
        ({}, {"skip_verification": "true", "skip_verification_reward": "0.5"}, 0.5),
    ],
)
async def test_skip_verification_returns_the_fixed_reward_and_still_closes_both_sessions(
    environment: dict[str, Any], agent: dict[str, Any], reward: float
) -> None:
    environment_server, client = _skipping_environment_server(environment=environment, agent=agent)

    result = await environment_server.run_request(_request())

    assert result.failure is None
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    dumped = result.result.model_dump()
    assert dumped["reward"] == reward
    assert dumped["verification_skipped"] is True
    # The same flat shape a verified result has: the verify body plus the reward.
    assert dumped["instance_id"] == "task"
    assert dumped["response"]["id"] == "response"
    assert dumped["responses_create_params"]["input"] == "task"


async def test_the_environment_servers_own_setting_overrides_the_agents() -> None:
    environment_server, client = _environment_server()
    client.global_config_dict.agent.responses_api_agents.test.update({"skip_verification": True})
    environment_server.config = SingleAgentTurnEnvironmentServerConfig.model_validate(
        {**environment_server.config.model_dump(), "skip_verification": False}
    )

    result = await environment_server.run_request(_request())

    assert result.result.reward == 1.0
    assert "verification_skipped" not in result.result.model_dump()
    assert "/verify" in [path for _, path, _ in client.calls]


@pytest.mark.parametrize("agent_skip", [False, "false"])
async def test_an_agent_that_does_not_skip_keeps_verifying(agent_skip: bool | str) -> None:
    environment_server, client = _environment_server()
    client.global_config_dict.agent.responses_api_agents.test.update(
        {"skip_verification": agent_skip, "skip_verification_reward": 0.5}
    )

    result = await environment_server.run_request(_request())

    assert result.result.reward == 1.0
    assert "verification_skipped" not in result.result.model_dump()
    assert "/verify" in [path for _, path, _ in client.calls]


async def test_skip_verification_still_applies_the_agent_outcome_policy() -> None:
    environment_server, client = _skipping_environment_server(environment={"skip_verification": True}, agent={})
    failed = _agent_response().model_dump(mode="json")
    failed.update(
        status="failed",
        metadata={AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR).model_dump_json()},
    )
    client.responses[2] = _Response(failed)

    result = await environment_server.run_request(_request())

    # A retryable agent failure stays a retryable failure, not a fixed reward.
    assert result.result is None
    assert result.failure.stage == "agent"
    assert result.failure.terminal is False
    assert [path for _, path, _ in client.calls][-1] == "/close_session"


@pytest.mark.parametrize(
    ("environment", "agent"),
    [({"skip_verification": True}, {}), ({}, {"skip_verification": True})],
)
async def test_skip_verification_returns_empty_aggregate_metrics_without_calling_resources(
    environment: dict[str, Any], agent: dict[str, Any]
) -> None:
    environment_server, client = _skipping_environment_server(environment=environment, agent=agent)

    with pytest.warns(RuntimeWarning, match="skip_verification=True"):
        metrics = await environment_server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))

    assert metrics == AggregateMetrics()
    assert client.calls == []


def _unbound_config(environment_server: SingleAgentTurnEnvironmentServer) -> SingleAgentTurnEnvironmentServerConfig:
    return SingleAgentTurnEnvironmentServerConfig.model_validate(
        {**environment_server.config.model_dump(), "resources_server": None}
    )


async def test_an_unset_resources_server_is_the_agents_own() -> None:
    # A server that replaced a legacy_agent relay binds the resources server the agent's /run used.
    configured, client = _environment_server(resources_tool_transports=["direct_http"])
    client.global_config_dict.agent.responses_api_agents.test.update(
        {"resources_server": {"type": "resources_servers", "name": "agents_resources"}}
    )
    environment_server = SingleAgentTurnEnvironmentServer(config=_unbound_config(configured), server_client=client)

    result = await environment_server.run_request(_request())

    assert result.result.reward == 1.0
    resources_calls = [path for server, path, _ in client.calls if server == "agents_resources"]
    assert resources_calls == ["/seed_session", "/verify", "/close_session"]
    create_body = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"])
    assert [access.name for access in create_body.tool_accesses] == ["agents_resources.direct_http"]


def test_a_server_whose_agent_names_no_resources_server_does_not_start() -> None:
    configured, client = _environment_server()

    with pytest.raises(ValueError, match="sets no resources_server"):
        SingleAgentTurnEnvironmentServer(config=_unbound_config(configured), server_client=client)


@pytest.mark.parametrize("close_cookies", [{}, None])
async def test_agent_without_a_resources_cookie_jar_keeps_the_seed_cookie_for_verify(
    close_cookies: dict[str, str] | None,
) -> None:
    # An agent with no direct HTTP grant never sees the Resources cookie, so it closes with no jar or an empty jar.
    # Verify and close must still carry the seed's cookie, which is how a stateful Resources Server finds the episode.
    environment, client = _environment_server()
    close = {"agent_session_id": "agent-session"}
    if close_cookies is not None:
        close["resources_cookies"] = close_cookies
    client.responses[3] = _Response(close)

    result = await environment.run_request(_request())

    assert result.result.reward == 1.0
    verify_call, resources_close_call = client.calls[4], client.calls[5]
    assert (verify_call[1], resources_close_call[1]) == ("/verify", "/close_session")
    assert verify_call[2]["cookies"] == {"session": "cookie-value"}
    assert resources_close_call[2]["cookies"] == {"session": "cookie-value"}


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
@pytest.mark.parametrize("result_path", ["episode-request", "flat-adapter"])
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
    trajectory = {
        "task_id": "task",
        "rollout_id": "rollout",
        "turns": [
            {
                "invocation_id": "root",
                "task_id": "task",
                "rollout_id": "rollout",
                "turn_no": 1,
                "timestamp": 1.0,
                "step_count": 2,
                "model_calls": [{"model_call_id": "model-call-1"}],
            }
        ],
    }
    close_body = orjson.loads(client.responses[3].body)
    close_body["agent_observations"] = observations
    close_body["trajectory"] = trajectory
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
    # The harness's own turns from the close reach the result in the slot rollout collection reads.
    assert result.result.ng_trajectory == TrajectoryRecord.model_validate(trajectory)
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
    assert legacy["ng_trajectory"] == TrajectoryRecord.model_validate(trajectory).model_dump(mode="json")


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
    observations = {
        "source": "hermes",
        "records": [{"kind": "agent_invocation", "invocation_id": "root", "status": "failed", "model_calls": []}],
    }
    # One decision taken before the failure: the collector keeps it only if the failure row carries it.
    trajectory = {
        "task_id": "t",
        "rollout_id": "r",
        "turns": [
            {
                "invocation_id": "root",
                "task_id": "t",
                "rollout_id": "r",
                "turn_no": 1,
                "timestamp": 1.0,
                "question": "do it",
                "answer": [],
                "step_count": 2,
                "model_calls": [
                    {"model_ref": {"type": "responses_api_models", "name": "policy"}, "response_id": "resp_0"}
                ],
            }
        ],
    }
    close_body = orjson.loads(client.responses[3].body)
    close_body["agent_observations"] = observations
    close_body["trajectory"] = trajectory
    client.responses[3] = _Response(close_body)
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
    # The closed agent session's evidence survives the failure, including in the legacy flat reply: the
    # observations and the decisions taken before the failure, with their model-call references and step counts.
    assert result.failure.agent_observations == AgentObservationBundle.model_validate(observations)
    assert result.failure.trajectory == TrajectoryRecord.model_validate(trajectory)
    [turn] = result.failure.trajectory.turns
    assert (turn.step_count, turn.model_calls[0].response_id) == (2, "resp_0")
    assert result.model_dump(mode="json")["failure"]["trajectory"]["turns"][0]["step_count"] == 2
    legacy = SingleAgentTurnLegacyEnvironmentServer(config=environment.config, server_client=client)._legacy_result(
        result
    )
    assert legacy["ng_agent_observations"] == AgentObservationBundle.model_validate(observations).model_dump(
        mode="json"
    )
    assert legacy["ng_trajectory"] == TrajectoryRecord.model_validate(trajectory).model_dump(mode="json")
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


@pytest.mark.parametrize("reported, expected", [({"x-ng-checkpoint-verify": "replay"}, "replay"), ({}, "wait")])
async def test_verify_step_mode_comes_from_the_resources_seed_reply(reported: dict, expected: str) -> None:
    environment_server, client = _environment_server()
    client.responses[0] = _Response({"resources_session_id": "resources-session"}, reported)
    modes: list[str] = []

    def record_step(request: object, mode: str) -> contextlib.AbstractAsyncContextManager[None]:
        modes.append(mode)
        return contextlib.nullcontext()

    object.__setattr__(environment_server, "checkpoint_step", record_step)
    await environment_server.run_request(_request())

    # Seeding and the agent invocation replay; closing the agent waits; verification follows the seed reply.
    assert modes == ["replay", "replay", "wait", expected]


@pytest.mark.parametrize("restarting_seed", [None, 0, 1], ids=["neither", "resources", "agent"])
async def test_a_seed_reply_that_cannot_be_captured_makes_the_episode_a_restart(restarting_seed: int | None) -> None:
    environment_server, client = _environment_server()
    if restarting_seed is not None:
        # Response 0 answers the resources seed, response 1 the agent seed.
        reply = client.responses[restarting_seed]
        client.responses[restarting_seed] = _Response(orjson.loads(reply.body), {"x-ng-checkpoint-restart": "1"})
    restarts: list[object] = []

    async def record_restart(request: object) -> None:
        restarts.append(request)

    object.__setattr__(environment_server, "checkpoint_restart", record_restart)
    await environment_server.run_request(_request())

    assert len(restarts) == (0 if restarting_seed is None else 1)


@pytest.mark.parametrize("transports", [["direct_http"], ["mcp"], ["direct_http", "mcp"]])
async def test_an_episode_that_grants_mcp_access_is_a_restart(transports: list[Literal["direct_http", "mcp"]]) -> None:
    environment_server, client = _environment_server(resources_tool_transports=transports)
    client.responses[0] = _Response(
        {
            "resources_session_id": "resources-session",
            "resources_tools": {"server_name": "resources", "headers": {"X-NeMo-Gym-Session-Token": "token"}},
        }
    )
    restarts: list[list[str]] = []

    async def record_restart(request: object) -> None:
        restarts.append([path for _, path, _ in client.calls])

    object.__setattr__(environment_server, "checkpoint_restart", record_restart)
    await environment_server.run_request(_request())

    # Marked before the agent receives the token, so no checkpoint ever records the episode past its seed.
    assert restarts == ([["/seed_session"]] if "mcp" in transports else [])


async def test_an_agent_that_returns_during_a_checkpoint_parks_at_close_agent_and_stays_ready() -> None:
    from nemo_gym.server_runtime._checkpoint.control import CheckpointRequest, ParticipantControlPlane
    from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant

    environment_server, client = _environment_server()
    participant = EnvironmentParticipant(mode="exported")
    environment_server._checkpoint = participant
    controller = ParticipantControlPlane(participant, instance_name="environment", lease_grace_seconds=60)
    client.slow_path = environment_server._agent_responses_path(_request())
    running = asyncio.create_task(environment_server.run_request(_request()))
    # The agent call is under way before the checkpoint opens.
    async with asyncio.timeout(5):
        while not any(path == client.slow_path for _, path, _ in client.calls):
            await asyncio.sleep(0.01)
    checkpoint = {"checkpoint_id": "c1", "deadline_ts": time.time() + 5}
    prepared = await controller.prepare(CheckpointRequest(**checkpoint))

    def parked_at_close_agent() -> bool:
        records = participant.export_records(None)
        return len(records) == 1 and (records[0].boundary or {}).get("next") == "close_agent"

    # The agent returns while the checkpoint is open, and the episode parks at its next boundary.
    async with asyncio.timeout(5):
        while not parked_at_close_agent():
            await asyncio.sleep(0.01)
    report = participant.readiness()
    [record] = participant.export_records(None)
    await controller.resume(CheckpointRequest(**checkpoint))
    response = await running

    assert prepared["phase"] == "prepared"
    assert report.ready, report.blockers
    assert record.boundary["next"] == "close_agent"
    assert response.result.reward == 1.0


async def test_an_episode_restored_at_close_agent_closes_and_verifies_without_invoking_the_agent() -> None:
    from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant, EpisodeRecord, task_digest

    environment_server, client = _environment_server()
    participant = EnvironmentParticipant(mode="exported")
    environment_server._checkpoint = participant
    request = _request()
    handles = {"resources_session_id": "resources-session", "agent_session_id": "agent-session"}
    participant.restore_records(
        [
            EpisodeRecord(
                episode_id=request.episode_id.model_copy(update={"attempt": request.episode_id.attempt - 1}),
                task_digest=task_digest(request.task.model_dump(mode="json")),
                boundary={
                    "next": "close_agent",
                    "handles": handles,
                    "response": _agent_response().model_dump(mode="json"),
                },
            )
        ]
    )
    # Only the close, the verification, and the final resources close remain.
    del client.responses[:3]
    response = await environment_server.run_request(request)

    assert [path for _, path, _ in client.calls] == ["/v1/agent_sessions/close", "/verify", "/close_session"]
    assert response.result.reward == 1.0


async def test_an_episode_restored_at_verify_with_skip_on_closes_without_verifying() -> None:
    from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant, EpisodeRecord, task_digest

    environment_server, client = _skipping_environment_server(
        environment={"skip_verification": True, "skip_verification_reward": 0.5}, agent={}
    )
    participant = EnvironmentParticipant(mode="exported")
    environment_server._checkpoint = participant
    request = _request()
    participant.restore_records(
        [
            EpisodeRecord(
                episode_id=request.episode_id.model_copy(update={"attempt": request.episode_id.attempt - 1}),
                task_digest=task_digest(request.task.model_dump(mode="json")),
                boundary={
                    "next": "verify",
                    "handles": {"resources_session_id": "resources-session", "agent_session_id": "agent-session"},
                    "response": _agent_response().model_dump(mode="json"),
                },
            )
        ]
    )
    # Only the final resources close remains.
    del client.responses[:4]
    response = await environment_server.run_request(request)

    assert [path for _, path, _ in client.calls] == ["/close_session"]
    assert response.result.reward == 0.5
    assert response.result.model_dump()["verification_skipped"] is True


class _HoldingClient(_Client):
    """Holds every call until ``released`` is set, and records which calls started before then."""

    released: asyncio.Event = Field(default_factory=asyncio.Event)
    started: list[str] = []

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        self.started.append(url_path)
        await self.released.wait()
        return await super().post(server_name, url_path, **kwargs)


@pytest.mark.parametrize("stage", ["close_agent", "return"])
async def test_a_restored_replacement_admitted_during_a_second_checkpoint_parks_before_its_stage(
    stage: str, tmp_path: Path
) -> None:
    from environment_servers.single_agent_turn.app import SessionHandles
    from nemo_gym.server_runtime._checkpoint.control import (
        CheckpointRequest,
        CommitRequest,
        ParticipantControlPlane,
        RestoreRequest,
    )
    from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant, EpisodeRecord, task_digest
    from nemo_gym.server_runtime._checkpoint.store import write_participant_state

    environment_server, client = _environment_server(client_type=_HoldingClient)
    participant = EnvironmentParticipant(mode="exported")
    environment_server._checkpoint = participant
    controller = ParticipantControlPlane(participant, instance_name="environment", lease_grace_seconds=60)
    request = _request()
    first_attempt = request.episode_id.model_copy(update={"attempt": request.episode_id.attempt - 1})
    handles = SessionHandles(resources_session_id="resources-session", agent_session_id="agent-session").model_dump()
    response = _agent_response().model_dump(mode="json")
    result = SingleAgentTurnResult.model_validate(
        {"responses_create_params": {"input": "task"}, "response": response, "reward": 1.0}
    ).model_dump(mode="json")
    if stage == "close_agent":
        boundary = {"next": "close_agent", "handles": handles, "response": response}
        remaining = ["/v1/agent_sessions/close", "/verify", "/close_session"]
    else:
        boundary = {
            "next": "return",
            "handles": handles,
            "result": result,
            "observations": None,
            "trajectory": None,
        }
        remaining = ["/close_session"]
    record = EpisodeRecord(
        episode_id=first_attempt, task_digest=task_digest(request.task.model_dump(mode="json")), boundary=boundary
    )
    write_participant_state(
        tmp_path / "c0",
        kind="environment",
        instance="environment",
        checkpoint_id="c0",
        records=[record.to_json_record()],
    )
    deadline = time.time() + 30
    await controller.restore(
        RestoreRequest(
            checkpoint_id="r1", deadline_ts=deadline, checkpoint_dir=str(tmp_path / "c0"), episode_ids=[first_attempt]
        )
    )
    await controller.resume(CheckpointRequest(checkpoint_id="r1", deadline_ts=deadline))
    # A second checkpoint prepares before the replacement arrives.
    second = CheckpointRequest(checkpoint_id="c2", deadline_ts=deadline)
    prepared = await controller.prepare(second)
    del client.responses[: len(client.responses) - len(remaining)]
    running = asyncio.create_task(environment_server.run_request(request))

    def parked() -> bool:
        live = request.episode_id.capture_key in participant.steps.keys()
        return live and len(participant.export_records(None)) == 1

    async with asyncio.timeout(5):
        while not (parked() or client.started or running.done()):
            await asyncio.sleep(0.01)
    ready = participant.readiness().ready
    started_before_resume = list(client.started)
    replied_before_resume = running.done()

    assert prepared["phase"] == "prepared"
    # The replacement parks at its restored boundary before it closes anything or ends.
    assert (ready, started_before_resume, replied_before_resume) == (True, [], False)
    committed = await controller.commit(
        CommitRequest(**second.model_dump(), checkpoint_dir=str(tmp_path / "c2"), episode_ids=[request.episode_id])
    )
    [exported] = participant.export_records([request.episode_id])
    await controller.resume(second)
    client.released.set()
    reply = await running

    assert committed["episode_ids"] == [request.episode_id.capture_key]
    assert exported.boundary == boundary
    assert [path for _, path, _ in client.calls] == remaining
    assert reply.result.reward == 1.0


class _AgentGateClient(_Client):
    """Holds the agent call until ``released`` is set."""

    released: asyncio.Event = Field(default_factory=asyncio.Event)

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path.endswith("/v1/responses"):
            self.calls.append((server_name, url_path, kwargs))
            await self.released.wait()
            raise ClientConnectionError("agent connection lost")
        return await super().post(server_name, url_path, **kwargs)


async def test_an_agent_failure_during_a_checkpoint_closes_the_agent_only_after_resume() -> None:
    from nemo_gym.server_runtime._checkpoint.control import CheckpointRequest, ParticipantControlPlane
    from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant

    environment_server, client = _environment_server(client_type=_AgentGateClient)
    participant = EnvironmentParticipant(mode="exported")
    environment_server._checkpoint = participant
    controller = ParticipantControlPlane(participant, instance_name="environment", lease_grace_seconds=60)
    observations = {
        "source": "hermes",
        "records": [{"kind": "agent_invocation", "invocation_id": "root", "status": "failed", "model_calls": []}],
    }
    close_body = orjson.loads(client.responses[3].body)
    close_body["agent_observations"] = observations
    # The agent call takes no scripted reply; the close and the final resources close follow it.
    client.responses = [*client.responses[:2], _Response(close_body), client.responses[5]]
    running = asyncio.create_task(environment_server.run_request(_request()))
    agent_path = environment_server._agent_responses_path(_request())
    async with asyncio.timeout(5):
        while not any(path == agent_path for _, path, _ in client.calls):
            await asyncio.sleep(0.01)
    checkpoint = CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 30)
    prepared = await controller.prepare(checkpoint)
    client.released.set()
    # The failed activation leaves the episode with no boundary, parked until resume.
    async with asyncio.timeout(5):
        while [record.boundary for record in participant.export_records(None)] != [None]:
            await asyncio.sleep(0.01)
    ready = participant.readiness().ready
    paths_before_resume = [path for _, path, _ in client.calls]
    replied_before_resume = running.done()
    await controller.resume(checkpoint)
    result = await running

    assert prepared["phase"] == "prepared" and ready and not replied_before_resume
    assert "/v1/agent_sessions/close" not in paths_before_resume
    # After resume the agent is closed at once, and its evidence travels with the failure.
    assert [path for _, path, _ in client.calls][-2:] == ["/v1/agent_sessions/close", "/close_session"]
    assert result.failure.stage == "agent"
    assert result.failure.agent_observations == AgentObservationBundle.model_validate(observations)


async def test_boundary_payloads_are_not_built_without_checkpointing(monkeypatch: pytest.MonkeyPatch) -> None:
    environment_server, _ = _environment_server()
    dumps: list[str] = []
    original = SingleAgentTurnResult.model_dump

    def counting(self: SingleAgentTurnResult, *args: Any, **kwargs: Any) -> dict:
        dumps.append("result")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SingleAgentTurnResult, "model_dump", counting)
    await environment_server.run_request(_request())

    # The return boundary's payload is the only dump of the verification result, and nothing records it here.
    assert dumps == []


class _ForwardedResponse:
    """An agent server's reply, as the environment server's HTTP client presents it."""

    def __init__(self, reply: httpx.Response) -> None:
        self.ok = reply.is_success
        self.status = reply.status_code
        self.headers = reply.headers
        self.cookies = {name: SimpleNamespace(value=value) for name, value in reply.cookies.items()}
        self.body = reply.content

    async def read(self) -> bytes:
        return self.body


class _AgentForwardingClient(_Client):
    """Sends the environment server's agent calls to a real agent server; scripts every other call."""

    agent: Any = None

    async def post(self, server_name: str, url_path: str, **kwargs) -> Any:
        if server_name != "agent":
            return await super().post(server_name, url_path, **kwargs)
        self.calls.append((server_name, url_path, kwargs))
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            body = body.model_dump(mode="json", exclude_unset=True)
        headers = dict(kwargs.get("headers") or {})
        cookies = kwargs.get("cookies") or {}
        if cookies:
            headers["cookie"] = "; ".join(f"{name}={value}" for name, value in cookies.items())
        reply = await self.agent.post(url_path, json=body, headers=headers)
        # The environment server keeps the agent's cookies in its handles, as a checkpoint records them.
        self.agent.cookies.clear()
        return _ForwardedResponse(reply)


def _checkpointed_agent(model_calls: list[str]) -> tuple[Any, httpx.AsyncClient]:
    """A Simple Agent with checkpointing on, whose model replies at once and records each call."""
    from responses_api_agents.simple_agent.app import ModelServerRef, SimpleAgent, SimpleAgentConfig

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {
        "observability_enabled": True,
        "checkpoint": {"enabled": True, "control_auth_token": "t"},
    }

    async def model(*, url_path: str, **kwargs: Any) -> Any:
        model_calls.append(url_path)
        reply = MagicMock(status=200, cookies={}, ok=True)
        reply.read = AsyncMock(return_value=orjson.dumps(_agent_response().model_dump(mode="json")))
        return reply

    client.post = AsyncMock(side_effect=model)
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="agent",
            port=8001,
            entrypoint="app.py",
            name="agent",
            model_server=ModelServerRef(type="responses_api_models", name="model"),
        ),
        server_client=client,
    )
    return agent, httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent.setup_webserver()), base_url="http://agent"
    )


async def test_an_agent_reply_lost_to_a_restore_is_given_again_without_running_the_activation() -> None:
    from environment_servers.single_agent_turn.app import SessionHandles
    from nemo_gym.server_runtime import ResourcesSeedSessionResponse
    from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant, EpisodeRecord, task_digest

    request = _request()
    first_attempt = request.model_copy(update={"episode_id": request.episode_id.model_copy(update={"attempt": 1})})
    model_calls: list[str] = []
    handles = SessionHandles(resources_session_id="resources-session", agent_session_id="agent-session")

    # Before the crash, the agent finishes the activation and its checkpoint follows the reply,
    # while the environment server's checkpoint precedes the reply: its episode is still at invoke_agent.
    agent, agent_http = _checkpointed_agent(model_calls)
    environment, client = _environment_server(client_type=_AgentForwardingClient)
    client.agent = agent_http
    async with agent_http:
        await environment._seed_agent(
            first_attempt, handles, ResourcesSeedSessionResponse(resources_session_id="resources-session")
        )
        environment_boundary = {"next": "invoke_agent", "handles": handles.model_dump()}
        await environment._invoke_agent(first_attempt, handles)
        agent_records = await agent.checkpoint_participant.export(None)
    observations = agent._session_state("agent-session").observations

    # Both are restored, and the environment server invokes the activation again.
    restored_agent, restored_http = _checkpointed_agent(model_calls)
    await restored_agent.checkpoint_participant.install(agent_records, [first_attempt.episode_id])
    environment, client = _environment_server(client_type=_AgentForwardingClient)
    client.agent = restored_http
    # Only the verification and the resources close are scripted; the agent answers the rest.
    del client.responses[:4]
    participant = EnvironmentParticipant(mode="exported")
    environment._checkpoint = participant
    participant.restore_records(
        [
            EpisodeRecord(
                episode_id=first_attempt.episode_id,
                task_digest=task_digest(request.task.model_dump(mode="json")),
                boundary=environment_boundary,
            )
        ]
    )
    async with restored_http:
        response = await environment.run_request(request)

    assert [path for _, path, _ in client.calls] == [
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    # The agent gave the reply it already gave, without calling the model again.
    assert model_calls == ["/ng-rollout/rollout-a1/v1/responses"]
    assert response.result.reward == 1.0
    assert response.result.ng_agent_observations == observations


async def test_with_checkpointing_on_a_direct_http_grant_names_the_seeding_attempt() -> None:
    from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant

    environment_server, client = _environment_server(resources_tool_transports=["direct_http"])
    environment_server._checkpoint = EnvironmentParticipant(mode="exported")
    await environment_server.run_request(_request())

    [tool_access] = AgentSeedSessionRequest.model_validate(client.calls[1][2]["json"]).tool_accesses
    # The Resources Server refuses a tool call that names no attempt, so the grant carries the rollout prefix.
    assert str(tool_access.base_url) == "http://resources:8000/ng-rollout/rollout-a2"


async def test_verify_says_that_the_resources_session_is_closed_afterwards() -> None:
    environment_server, client = _environment_server()
    await environment_server.run_request(_request())

    [verify_call] = [call for _, path, call in client.calls if path == "/verify"]
    paths = [path for _, path, _ in client.calls]
    # So the Resources Server keeps the session tracked until the close, which then runs its close handler.
    assert verify_call["headers"] == {"x-ng-checkpoint-close-follows": "1"}
    assert paths.index("/close_session") > paths.index("/verify")


@pytest.mark.parametrize(
    "outcome, verifiable, retryable",
    [
        ({"failure_kind": "agent_invalid_output"}, True, False),
        ({"failure_kind": "agent_limit"}, True, False),
        ({"failure_kind": "model_request_error", "status_code": 400}, True, False),
        ({"failure_kind": "model_request_error", "status_code": 413}, True, False),
        ({"failure_kind": "model_request_error", "status_code": 422}, True, False),
        ({"failure_kind": "model_request_error", "status_code": 401}, False, False),
        ({"failure_kind": "model_request_error", "status_code": 403}, False, False),
        ({"failure_kind": "model_request_error", "status_code": 404}, False, False),
        ({"failure_kind": "model_request_error", "status_code": 429}, False, True),
        ({"failure_kind": "model_request_error", "status_code": 500}, False, True),
        ({"failure_kind": "model_request_error", "status_code": 503}, False, True),
        ({"failure_kind": "model_transport_error"}, False, True),
        ({"failure_kind": "unknown"}, False, True),
        ({"failure_kind": "model_request_error", "status_code": 402}, False, False),
        ({"failure_kind": "model_request_error", "status_code": 409}, False, False),
        ({"failure_kind": "agent_run_error"}, False, False),
    ],
)
async def test_classified_agent_failure_policy(outcome: dict, verifiable: bool, retryable: bool) -> None:
    await _assert_classified_agent_failure_policy(outcome, verifiable, retryable)


@pytest.mark.parametrize(
    "config, outcome, verifiable, retryable",
    [
        (
            {"by_failure_kind": {"model_request_error": "retry", "unknown": "retry"}},
            {"failure_kind": "model_request_error", "status_code": 401},
            False,
            True,
        ),
        (
            {"by_failure_kind": {"model_request_error": "retry", "unknown": "retry"}},
            {"failure_kind": "unknown"},
            False,
            True,
        ),
        (
            {"by_failure_kind": {"model_request_error": "retry"}},
            {"failure_kind": "model_request_error", "status_code": 400},
            True,
            False,
        ),
        ({"by_status_code": {400: "retry"}}, {"failure_kind": "model_request_error", "status_code": 400}, False, True),
        ({"by_status_code": {503: "fail"}}, {"failure_kind": "model_request_error", "status_code": 503}, False, False),
        ({"by_status_code": {404: "retry"}}, {"failure_kind": "model_request_error", "status_code": 404}, False, True),
        ({"by_status_code": {404: "retry"}}, {"failure_kind": "model_request_error", "status_code": 429}, False, True),
        ({"by_status_code": {404: "retry"}}, {"failure_kind": "model_request_error", "status_code": 503}, False, True),
        ({"by_status_code": {404: "retry"}}, {"failure_kind": "model_request_error", "status_code": 400}, True, False),
        (
            {"by_status_code": {"503": "fail"}},
            {"failure_kind": "model_request_error", "status_code": 503},
            False,
            False,
        ),
        ({"by_failure_kind": {"unknown": "retry"}}, {"failure_kind": "model_transport_error"}, False, True),
        ({"by_failure_kind": {"unknown": "retry"}}, {"failure_kind": "agent_invalid_output"}, True, False),
        ({"by_failure_kind": {"unknown": "verify"}}, {"failure_kind": "unknown"}, True, False),
        ({"by_failure_kind": {"unknown": "fail"}}, {"failure_kind": "unknown"}, False, False),
        ({"default_action": "retry"}, {"failure_kind": "agent_run_error"}, False, True),
        ({"by_failure_kind": {}, "by_status_code": {}}, {"failure_kind": "model_transport_error"}, False, True),
    ],
)
async def test_configured_agent_failure_policy(config: dict, outcome: dict, verifiable: bool, retryable: bool) -> None:
    await _assert_classified_agent_failure_policy(outcome, verifiable, retryable, config=config)


@pytest.mark.parametrize(
    "config",
    [
        {"by_failure_kind": {"typo": "retry"}},
        {"by_failure_kind": {"unknown": "typo"}},
        {"by_status_code": {399: "retry"}},
        {"by_status_code": {600: "retry"}},
        {"default_action": "typo"},
        {"typo": "retry"},
    ],
)
def test_invalid_agent_policy_config_rejected(config: dict) -> None:
    with pytest.raises(ValidationError):
        AgentOutcomePolicyConfig.model_validate(config)


async def _assert_classified_agent_failure_policy(
    outcome: dict, verifiable: bool, retryable: bool, *, config: dict | None = None
) -> None:
    from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentOutcome

    environment, client = _environment_server()
    if config is not None:
        environment.config = SingleAgentTurnEnvironmentServerConfig.model_validate(
            {**environment.config.model_dump(), "agent_outcome_policy": config}
        )
    response = _agent_response().model_dump(mode="json")
    response.update(
        status="failed",
        error={"code": "server_error", "message": "Native failure details"},
        metadata={AGENT_OUTCOME_METADATA_KEY: AgentOutcome.model_validate(outcome).model_dump_json()},
    )
    client.responses[2] = _Response(response)
    observations = {
        "source": "agent",
        "records": [{"kind": "agent_invocation", "invocation_id": "root", "status": "failed"}],
    }
    close = orjson.loads(client.responses[3].body)
    close["agent_observations"] = observations
    client.responses[3] = _Response(close)
    if not verifiable:
        del client.responses[4]

    result = await environment.run_request(_request())

    assert ("/verify" in [path for _, path, _ in client.calls]) is verifiable
    assert not client.responses
    if verifiable:
        assert result.failure is None
        assert result.result.reward == 1.0
    else:
        assert result.result is None
        assert result.failure.stage == "agent"
        assert result.failure.terminal is not retryable
        assert result.failure.failure_reason == "Native failure details"
        assert result.failure.partial_response.model_dump(mode="json") == response
        assert result.failure.agent_observations == AgentObservationBundle.model_validate(observations)
        legacy = SingleAgentTurnLegacyEnvironmentServer(
            config=environment.config, server_client=client
        )._legacy_result(result)
        assert legacy["_ng_failure_terminal"] is not retryable
        assert legacy["_ng_failure_partial_response"] == response
        assert legacy["ng_agent_observations"] == result.failure.agent_observations.model_dump(mode="json")
        assert "reward" not in legacy
    assert [path for _, path, _ in client.calls][-1] == "/close_session"


@pytest.mark.parametrize("status", ["completed", "incomplete", None])
async def test_nonfailed_outcomes_still_reach_verification(status: str | None) -> None:
    environment, client = _environment_server()
    response = _agent_response().model_copy(update={"status": status})
    client.responses[2] = _Response(response.model_dump(mode="json"))

    result = await environment.run_request(_request())

    assert result.result.reward == 1.0
    assert result.failure is None
    verify = next(kwargs for _, path, kwargs in client.calls if path == "/verify")
    assert verify["json"]["response"]["status"] == status


@pytest.mark.parametrize("action", ["fail", "retry"])
async def test_incomplete_context_limit_verifies_partial_work(action: str) -> None:
    from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentOutcome

    environment, client = _environment_server()
    environment.config.agent_outcome_policy = AgentOutcomePolicyConfig(default_action=action)
    response = (
        _agent_response()
        .model_copy(
            update={
                "status": "incomplete",
                "metadata": {
                    AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind="model_context_limit").model_dump_json()
                },
            }
        )
        .model_dump(mode="json")
    )
    client.responses[2] = _Response(response)

    result = await environment.run_request(_request())

    assert result.failure is None
    assert result.result.reward == 1.0
    verify = next(kwargs for _, path, kwargs in client.calls if path == "/verify")
    assert verify["json"]["response"] == response


@pytest.mark.parametrize("raw", ["not valid JSON", '{"failure_kind":"unregistered_error"}'])
async def test_invalid_outcome_is_retained_without_grading_or_retry(raw: str) -> None:
    from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY

    environment, client = _environment_server()
    response = _agent_response().model_dump(mode="json")
    response.update(status="failed", metadata={AGENT_OUTCOME_METADATA_KEY: raw})
    client.responses[2] = _Response(response)
    del client.responses[4]

    result = await environment.run_request(_request())

    assert result.result is None
    assert result.failure.terminal is True
    assert result.failure.partial_response.model_dump(mode="json") == response
    assert "/verify" not in [path for _, path, _ in client.calls]
    assert [path for _, path, _ in client.calls][-1] == "/close_session"


@pytest.mark.parametrize("outcome", ["completed", "judge_failure", "exception", "timeout"])
async def test_verifier_evidence_reaches_collector_through_result_or_failure(outcome, tmp_path):
    from fastapi.testclient import TestClient

    from nemo_gym.rollout_collection import _attach_trajectory_record, _episode_record
    from nemo_gym.server_runtime import BaseResourcesServerConfig, BaseVerifyResponse, SimpleResourcesServer
    from nemo_gym.verification import JudgeError

    class Resources(SimpleResourcesServer):
        async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
            if outcome == "judge_failure":
                raise JudgeError("judge unavailable")
            if outcome == "exception":
                raise RuntimeError("scoring broke")
            if outcome == "timeout":
                raise TimeoutError("grading deadline")
            return BaseVerifyResponse(**body.model_dump(), reward=0.0)

    environment, client = _environment_server()
    resources_client = MagicMock(spec=ServerClient)
    resources_client.global_config_dict = {
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path),
    }
    resources = Resources(
        config=BaseResourcesServerConfig(name="resources", host="resources", port=8000, entrypoint="app.py"),
        server_client=resources_client,
    )
    http_response = TestClient(resources.setup_webserver()).post(
        "/ng-rollout/rollout-a2/verify",
        json={"responses_create_params": {"input": "task"}, "response": _agent_response().model_dump(mode="json")},
    )

    class VerifyResponse(_Response):
        ok = http_response.is_success
        status = http_response.status_code
        content = SimpleNamespace(read=AsyncMock(return_value=http_response.content))

        def raise_for_status(self):
            if not self.ok:
                raise ClientResponseError(
                    request_info=SimpleNamespace(
                        url="http://resources/verify", real_url="http://resources/verify", method="POST", headers={}
                    ),
                    history=(),
                    status=self.status,
                    message="verification failed",
                )

    client.responses[4] = VerifyResponse(http_response.json())
    episode = await environment.run_request(_request())
    assert (episode.failure is not None) == (outcome in {"exception", "timeout"})
    if episode.result is not None:
        assert episode.result.reward == 0.0
        assert episode.result.mask_sample == (outcome == "judge_failure")
    else:
        assert episode.failure.stage == "verification"
        assert episode.failure.terminal is False  # HTTP 500 remains retryable.

    record = _episode_record(episode.model_dump(mode="json"))
    legacy_record = SingleAgentTurnLegacyEnvironmentServer(
        config=environment.config, server_client=client
    )._legacy_result(episode)
    assert legacy_record["ng_verifier_invocations"] == record["ng_verifier_invocations"]
    _attach_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, record)
    (invocation,) = record["ng_trajectory"]["verifier_invocations"]
    expected_status = {
        "completed": "completed",
        "judge_failure": "failed",
        "exception": "failed",
        "timeout": "incomplete",
    }
    assert invocation["status"] == expected_status[outcome]
    expected_kind = {
        "completed": None,
        "judge_failure": "judge_failed",
        "exception": "verifier_error",
        "timeout": "transport_timeout",
    }
    assert invocation["failure_kind"] == expected_kind[outcome]
    assert invocation["model_calls"] == []
    assert "ng_verifier_invocations" not in record  # Only the canonical trajectory is persisted.
    assert list(tmp_path.iterdir()) == []  # Resources never writes verifier sidecars.
