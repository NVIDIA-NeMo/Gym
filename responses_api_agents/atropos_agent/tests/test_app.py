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
import uuid
from unittest.mock import AsyncMock, MagicMock, call

from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.atropos_agent.app import (
    AtroposAgent,
    AtroposAgentConfig,
    ModelServerRef,
    ResourcesServerRef,
)


class TestAtroposAgent:
    def test_lifecycle(self) -> None:
        config = AtroposAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my_model_name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="my_resources_name",
            ),
        )
        server = AtroposAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        env_id = str(uuid.uuid4())
        mock_seed_session_data = {
            "env_id": env_id,
            "obs": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            "system_prompt": "You are a helpful assistant.",
        }

        mock_model_response_data = {
            "id": "resp_123",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {"type": "message", "role": "assistant", "content": "The answer is \\boxed{4}"},
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }

        mock_step_data = {
            "obs": [],
            "reward": 1.0,
            "done": True,
            "info": {"correct": True},
        }

        mock_close_data = {"message": "Success", "success": True}

        dotjson_mock = MagicMock()
        dotjson_mock.json.side_effect = [
            mock_seed_session_data,
            mock_model_response_data,
            mock_step_data,
            mock_close_data,
        ]
        server.server_client.post = AsyncMock(return_value=dotjson_mock)

        res = client.post(
            "/v1/responses",
            json={
                "task_idx": 0,
                "responses_create_params": {"input": []},
                "max_steps": 1,
            },
        )

        assert res.status_code == 200
        response_data = res.json()
        assert "env_id" in response_data
        assert response_data["env_id"] == env_id

        server.server_client.post.assert_has_awaits(
            [
                call(
                    server_name="my_resources_name",
                    url_path="/seed_session",
                    json={"task_idx": 0},
                ),
                call(
                    server_name="my_model_name",
                    url_path="/v1/responses",
                    json=NeMoGymResponseCreateParamsNonStreaming.model_validate(
                        {
                            "input": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": "What is 2 + 2?"},
                            ]
                        }
                    ),
                ),
                call(
                    server_name="my_resources_name",
                    url_path="/step",
                    json={
                        "action": "The answer is \\boxed{4}",
                        "env_id": env_id,
                    },
                ),
                call(
                    server_name="my_resources_name",
                    url_path="/close",
                    json={"env_id": env_id},
                ),
            ],
            any_order=True,
        )
