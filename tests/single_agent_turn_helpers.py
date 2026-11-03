# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared in-process single-agent-turn setup for server and collection boundary tests."""

import asyncio
from typing import Literal

import orjson
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import AgentServerRef, BaseServerConfig, ResourcesServerRef
from nemo_gym.core.episode import EpisodeId, TaskId, current_rollout_id
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.server_runtime.episode_types import MaterializedTask
from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnRequest, SingleAgentTurnTaskInput


class _Cookie:
    value = "cookie-value"


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
