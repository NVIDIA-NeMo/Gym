# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, call

from fastapi.testclient import TestClient

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.aviary_agent.app import AviaryAgent, AviaryAgentConfig, ModelServerRef, ResourcesServerRef


class TestApp:
    def test_lifecycle(self) -> None:
        config = AviaryAgentConfig(
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
        )
        server = AviaryAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_seed_session_data = {
            "env_id": str(uuid.uuid4()),
            "obs": [{"role": "user", "content": "world"}],
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
            "obs": [{"role": "user", "content": "hello"}],
            "reward": 0.0,
            "done": False,
        }
        mock_close_data = {"message": "Success", "success": True}

        dotjson_mock = MagicMock()
        dotjson_mock.json.side_effect = [mock_seed_session_data, mock_response_data, mock_step_data, mock_close_data]
        server.server_client.post = AsyncMock(return_value=dotjson_mock)

        # No model provided should use the one from the config
        res_no_model = client.post(
            "/v1/responses",
            json={
                "task_idx": 0,
                "responses_create_params": {"input": [{"role": "user", "content": "hello"}]},
                "max_steps": 1,
            },
        )
        assert res_no_model.status_code == 200
        server.server_client.post.assert_has_awaits(
            [
                call(
                    server_name="my resources name",
                    url_path="/seed_session",
                    json={"task_idx": 0},
                ),
                call(
                    server_name="my model name",
                    url_path="/v1/responses",
                    json=NeMoGymResponseCreateParamsNonStreaming.model_validate(
                        {"input": [{"role": "user", "content": "hello"}, {"role": "user", "content": "world"}]}
                    ),
                ),
                call(
                    server_name="my resources name",
                    url_path="/step",
                    json={
                        "action": [
                            {
                                "arguments": '{"city": "San Francisco"}',
                                "call_id": "abc123",
                                "name": "get_weather",
                                "type": "function_call",
                                "id": None,
                                "status": None,
                            }
                        ],
                        "env_id": mock_seed_session_data["env_id"],
                    },
                ),
                call(
                    server_name="my resources name",
                    url_path="/close",
                    json={"env_id": mock_seed_session_data["env_id"]},
                ),
            ],
            any_order=True,
        )
