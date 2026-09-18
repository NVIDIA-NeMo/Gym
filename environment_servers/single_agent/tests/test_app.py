# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import orjson
from omegaconf import OmegaConf
from pydantic import ConfigDict

from environment_servers.single_agent.app import (
    SingleAgentEnvironmentServer,
    SingleAgentEnvironmentServerConfig,
    _is_retryable_dependency_error,
)
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from nemo_gym.single_agent_episode_types import SingleAgentEpisodeRequest, SingleAgentTaskInput


class _Cookie:
    value = "cookie-value"


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
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    responses: list[_Response]

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        self.calls.append((server_name, url_path, kwargs))
        return self.responses.pop(0)

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _environment_server(
    *,
    token_capture: bool = False,
    resources_tool_transports: list[Literal["direct_http", "mcp"]] | None = None,
) -> tuple[SingleAgentEnvironmentServer, _Client]:
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
    config = SingleAgentEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        cleanup_timeout_seconds=10,
        resources_tool_transports=resources_tool_transports or [],
    )
    return SingleAgentEnvironmentServer(config=config, server_client=client), client


def _request() -> SingleAgentEpisodeRequest:
    return SingleAgentEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(taskset="test:source", task_id="task"),
            task_input=SingleAgentTaskInput(
                responses_create_params={"input": "task"},
                task_data={"instance_id": "task"},
            ),
        ),
    )


async def test_single_agent_protocol_with_direct_resources_tools() -> None:
    environment_server, client = _environment_server(resources_tool_transports=["direct_http"])
    result = await environment_server.run_request(_request())

    assert result.result is not None
    assert result.result.verification.reward == 1.0
    assert result.result.verification.model_dump()["benchmark_field"] == "preserved"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    create_body = client.calls[1][2]["json"]
    assert create_body.task_id == TaskId(taskset="test:source", task_id="task")
    [tool_access] = create_body.tool_accesses
    assert tool_access.name == "resources.direct_http"
    assert str(tool_access.base_url) == "http://resources:8000/"
    assert tool_access.cookies == {"session": "cookie-value"}
    assert client.calls[2][2]["cookies"] == {"session": "cookie-value"}
    assert client.calls[3][2]["cookies"] == {"session": "cookie-value"}
    assert client.calls[3][2]["json"].episode_id == EpisodeId(rollout_id="rollout", attempt=2)
    assert client.calls[4][2]["cookies"] == {"session": "updated-cookie"}
    assert client.calls[5][2]["cookies"] == {"session": "updated-cookie"}
    assert client.calls[5][2]["json"].episode_id == EpisodeId(rollout_id="rollout", attempt=2)


async def test_single_agent_translates_resources_mcp_metadata_to_canonical_tool_access() -> None:
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

    direct_access, mcp_access = client.calls[1][2]["json"].tool_accesses
    assert direct_access.name == "resources.direct_http"
    assert mcp_access.name == "resources"
    assert mcp_access.required is True
    assert mcp_access.connection.transport == "streamable_http"
    assert str(mcp_access.connection.url) == "http://resources:8000/mcp"
    assert mcp_access.connection.headers == {"Authorization": "Bearer scoped"}


async def test_token_capture_keeps_prefixed_twin_route() -> None:
    environment_server, client = _environment_server(token_capture=True)

    await environment_server.run_request(_request())

    assert client.calls[2][1] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"


def test_dependency_failure_messages_are_bounded() -> None:
    environment_server, _ = _environment_server()
    error = environment_server._failure(stage="agent", message="x" * 3000, terminal=False)
    assert len(error.failure.message) == 2000


def test_retry_requires_a_transient_dependency_error() -> None:
    assert _is_retryable_dependency_error(TimeoutError()) is True
    assert _is_retryable_dependency_error(ValueError("invalid response")) is False
