# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, call

import orjson
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseFunctionToolCall
from nemo_gym.server_utils import ServerClient
from responses_api_agents.cube_agent.app import CubeAgent, CubeAgentConfig


def test_cube_agent_lifecycle() -> None:
    config = CubeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        model_server=ModelServerRef(
            type="responses_api_models",
            name="my model name",
        ),
        resources_server=ResourcesServerRef(
            type="resources_servers",
            name="my resources name",
        ),
        max_steps=1,
    )
    server = CubeAgent(config=config, server_client=MagicMock(spec=ServerClient))
    app = server.setup_webserver()
    client = TestClient(app)

    mock_seed_session_data = {
        "env_id": str(uuid.uuid4()),
        "obs": [{"role": "user", "content": "world", "type": "message"}],
        "tools": [],
    }
    mock_response_data = {
        "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
        "created_at": 1753983920.0,
        "model": "dummy_model",
        "object": "response",
        "output": [
            NeMoGymResponseFunctionToolCall(
                call_id="abc123",
                name="get_weather",
                arguments=json.dumps({"city": "San Francisco"}),
            ).model_dump(),
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    mock_step_data = {
        "obs": [{"role": "user", "content": "hello", "type": "message"}],
        "reward": 0.0,
        "done": False,
    }
    dotjson_mock = AsyncMock()
    dotjson_mock.json = AsyncMock(side_effect=[mock_seed_session_data])
    dotjson_mock.read = AsyncMock(
        side_effect=[
            orjson.dumps(mock_response_data),
            orjson.dumps(mock_step_data),
        ]
    )
    dotjson_mock.status = 200
    dotjson_mock.raise_for_status = MagicMock()
    dotjson_mock.cookies = None
    server.server_client.post = AsyncMock(return_value=dotjson_mock)

    res = client.post(
        "/v1/responses",
        json={
            "task_idx": 0,
            "responses_create_params": {"input": [{"role": "user", "content": "hello", "type": "message"}]},
        },
    )
    assert res.status_code == 200

    calls = server.server_client.post.await_args_list
    assert len(calls) == 4

    assert calls[0] == call(server_name="my resources name", url_path="/seed_session", json={"task_idx": 0})

    assert calls[1][1]["server_name"] == "my model name"
    assert calls[1][1]["url_path"] == "/v1/responses"
    model_input = calls[1][1]["json"].input
    assert len(model_input) == 2
    assert model_input[0].content == "hello"
    assert model_input[1].content == "world"

    assert calls[2][1]["server_name"] == "my resources name"
    assert calls[2][1]["url_path"] == "/step"
    assert calls[2][1]["json"]["action"][0]["call_id"] == "abc123"
    assert calls[2][1]["json"]["env_id"] == mock_seed_session_data["env_id"]

    assert calls[3] == call(
        server_name="my resources name", url_path="/close", json={"env_id": mock_seed_session_data["env_id"]}
    )
