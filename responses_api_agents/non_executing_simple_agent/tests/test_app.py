# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from unittest.mock import AsyncMock, MagicMock, call

from fastapi import Request, Response

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.non_executing_simple_agent.app import (
    ModelServerRef,
    NonExecutingSimpleAgent,
    NonExecutingSimpleAgentConfig,
    NonExecutingSimpleAgentRunRequest,
    ResourcesServerRef,
)


def _mock_response(payload: dict, cookies: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.ok = True
    response.cookies = cookies or {}
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    return response


def _config() -> NonExecutingSimpleAgentConfig:
    return NonExecutingSimpleAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="non_executing_agent",
        resources_server=ResourcesServerRef(
            type="resources_servers",
            name="resource",
        ),
        model_server=ModelServerRef(
            type="responses_api_models",
            name="model",
        ),
    )


def _tool_call_response(arguments: str = '{"name":') -> dict:
    return {
        "id": "resp_123",
        "created_at": 1753983920.0,
        "model": "dummy_model",
        "object": "response",
        "output": [
            {
                "arguments": arguments,
                "call_id": "call_123",
                "name": "submit_answer",
                "type": "function_call",
                "id": "fc_123",
                "status": "completed",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "required",
        "tools": [],
    }


class TestApp:
    def test_sanity(self) -> None:
        NonExecutingSimpleAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    async def test_responses_returns_tool_calls_without_executing_or_parsing_arguments(self) -> None:
        server = NonExecutingSimpleAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        server.server_client.post.return_value = _mock_response(_tool_call_response(), cookies={"model_cookie": "1"})

        request = MagicMock(spec=Request)
        request.cookies = {"session_cookie": "1"}
        response = Response()

        result = await server.responses(
            request=request,
            response=response,
            body=NeMoGymResponseCreateParamsNonStreaming(input="hello"),
        )

        assert result.output[0].type == "function_call"
        assert result.output[0].arguments == '{"name":'
        set_cookie_headers = [value.decode() for key, value in response.raw_headers if key == b"set-cookie"]
        assert any(header.startswith("session_cookie=1;") for header in set_cookie_headers)
        assert any(header.startswith("model_cookie=1;") for header in set_cookie_headers)
        server.server_client.post.assert_called_once_with(
            server_name="model",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
            ),
        )

    async def test_run_seeds_and_verifies_model_response_without_tool_execution(self) -> None:
        server = NonExecutingSimpleAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input="hello")
        model_response = _tool_call_response(arguments='{"summary":"ok"}')
        verify_response = {
            "responses_create_params": responses_create_params.model_dump(),
            "response": model_response,
            "reward": 1.0,
        }
        server.server_client.post.side_effect = [
            _mock_response({}, cookies={"session": "seeded"}),
            _mock_response(model_response, cookies={"session": "seeded"}),
            _mock_response(verify_response),
        ]

        request = MagicMock(spec=Request)
        request.cookies = {}
        result = await server.run(
            request=request,
            body=NonExecutingSimpleAgentRunRequest(responses_create_params=responses_create_params),
        )

        assert result.reward == 1.0
        assert server.server_client.post.call_args_list[0] == call(
            server_name="resource",
            url_path="/seed_session",
            json={"responses_create_params": responses_create_params.model_dump()},
            cookies={},
        )
        assert server.server_client.post.call_args_list[1] == call(
            server_name="non_executing_agent",
            url_path="/v1/responses",
            json=responses_create_params,
            cookies={"session": "seeded"},
        )
        assert server.server_client.post.call_args_list[2].kwargs["server_name"] == "resource"
        assert server.server_client.post.call_args_list[2].kwargs["url_path"] == "/verify"
        assert server.server_client.post.call_args_list[2].kwargs["cookies"] == {"session": "seeded"}
        assert (
            server.server_client.post.call_args_list[2].kwargs["json"]["response"]["output"][0]["arguments"]
            == '{"summary":"ok"}'
        )

    async def test_aggregate_metrics_proxies_to_resource_server(self) -> None:
        server = NonExecutingSimpleAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        aggregate_metrics = {
            "group_level_metrics": [],
            "agent_metrics": {"mean/reward": 1.0},
            "key_metrics": {"mean/reward": 1.0},
        }
        server.server_client.post.return_value = _mock_response(aggregate_metrics)

        result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))

        assert result.agent_metrics == {"mean/reward": 1.0}
        server.server_client.post.assert_called_once_with(
            server_name="resource",
            url_path="/aggregate_metrics",
            json=AggregateMetricsRequest(verify_responses=[]),
        )
