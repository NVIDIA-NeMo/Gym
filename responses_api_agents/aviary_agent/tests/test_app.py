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
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.aviary_agent.app import (
    AviaryAgent,
    AviaryAgentConfig,
    AviaryAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
)


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
            max_steps=1,
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
        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [mock_seed_session_data, mock_response_data, mock_step_data]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        server.server_client.post = AsyncMock(return_value=dotjson_mock)

        # No model provided should use the one from the config
        res_no_model = client.post(
            "/v1/responses",
            json={
                "task_idx": 0,
                "responses_create_params": {"input": [{"role": "user", "content": "hello"}]},
            },
        )
        assert res_no_model.status_code == 200

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

    def test_sanity(self) -> None:
        config = AviaryAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        AviaryAgent(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_responses_done_if_no_tool_calls_true(self) -> None:
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
            done_if_no_tool_calls=True,
        )
        agent = AviaryAgent(config=config, server_client=MagicMock(spec=ServerClient))

        mock_seed_session_data = {
            "env_id": str(uuid.uuid4()),
            "obs": [{"role": "user", "content": "Initial observation"}],
            "tools": [],
        }
        mock_response_data = {
            "id": "resp_123",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_123",
                    "content": [{"annotations": [], "text": "I don't need to call any tools", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        mock_close_data = {"message": "Success", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [mock_seed_session_data, mock_response_data, mock_close_data]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = AviaryAgentRunRequest(
            task_idx=0, responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
        response = await agent.responses(request)

        assert response.env_id == mock_seed_session_data["env_id"]
        assert response.group_id == "0"

        agent.server_client.post.assert_has_awaits(
            [
                call(server_name="my resources name", url_path="/seed_session", json={"task_idx": 0}),
                call(
                    server_name="my model name",
                    url_path="/v1/responses",
                    json=NeMoGymResponseCreateParamsNonStreaming.model_validate(
                        {"input": [{"role": "user", "content": "Initial observation"}], "tools": []}
                    ),
                    cookies=None,
                ),
                call(
                    server_name="my resources name",
                    url_path="/close",
                    json={"env_id": mock_seed_session_data["env_id"]},
                ),
            ]
        )

    async def test_responses_multi_step(self) -> None:
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
        agent = AviaryAgent(config=config, server_client=MagicMock(spec=ServerClient))

        env_id = str(uuid.uuid4())
        mock_seed_session_data = {"env_id": env_id, "obs": [{"role": "user", "content": "Step 0"}], "tools": []}

        mock_response_1 = {
            "id": "resp_1",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                NeMoGymResponseFunctionToolCall(
                    call_id="call_1", name="tool_1", arguments=json.dumps({"arg": "val1"})
                ).model_dump()
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        mock_step_1 = {
            "obs": [{"type": "function_call_output", "call_id": "call_1", "output": "Result 1"}],
            "reward": 0.0,
            "done": False,
        }

        mock_response_2 = {
            "id": "resp_2",
            "created_at": 1753983921.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                NeMoGymResponseFunctionToolCall(
                    call_id="call_2", name="tool_2", arguments=json.dumps({"arg": "val2"})
                ).model_dump()
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        mock_step_2 = {
            "obs": [{"type": "function_call_output", "call_id": "call_2", "output": "Result 2"}],
            "reward": 1.0,
            "done": True,
        }

        mock_close_data = {"message": "Success", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            mock_seed_session_data,
            mock_response_1,
            mock_step_1,
            mock_response_2,
            mock_step_2,
            mock_close_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = MagicMock()
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = AviaryAgentRunRequest(
            task_idx=42, responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
        response = await agent.responses(request)

        assert response.env_id == env_id
        assert response.group_id == "42"
        assert len(response.output) == 2

        calls = agent.server_client.post.await_args_list
        assert len(calls) == 6
        assert calls[0] == call(server_name="my resources name", url_path="/seed_session", json={"task_idx": 42})
        assert calls[1][1]["server_name"] == "my model name"
        assert calls[1][1]["url_path"] == "/v1/responses"
        assert calls[2][1]["server_name"] == "my resources name"
        assert calls[2][1]["url_path"] == "/step"
        assert calls[2][1]["json"]["action"][0]["call_id"] == "call_1"
        assert calls[3][1]["server_name"] == "my model name"
        assert calls[3][1]["url_path"] == "/v1/responses"
        assert calls[4][1]["server_name"] == "my resources name"
        assert calls[4][1]["url_path"] == "/step"
        assert calls[4][1]["json"]["action"][0]["call_id"] == "call_2"
        assert calls[5] == call(server_name="my resources name", url_path="/close", json={"env_id": env_id})

    async def test_responses_return_transitions_false(self) -> None:
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
            return_transitions=False,
        )
        agent = AviaryAgent(config=config, server_client=MagicMock(spec=ServerClient))

        env_id = str(uuid.uuid4())
        mock_seed_session_data = {"env_id": env_id, "obs": [{"role": "user", "content": "Step 0"}], "tools": []}

        mock_response_1 = {
            "id": "resp_1",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                NeMoGymResponseFunctionToolCall(
                    call_id="call_1", name="tool_1", arguments=json.dumps({"arg": "val1"})
                ).model_dump()
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        mock_step_1 = {
            "obs": [{"type": "function_call_output", "call_id": "call_1", "output": "Result 1"}],
            "reward": 0.0,
            "done": False,
        }

        mock_response_2 = {
            "id": "resp_2",
            "created_at": 1753983921.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                NeMoGymResponseFunctionToolCall(
                    call_id="call_2", name="tool_2", arguments=json.dumps({"arg": "val2"})
                ).model_dump()
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        mock_step_2 = {
            "obs": [{"type": "function_call_output", "call_id": "call_2", "output": "Result 2"}],
            "reward": 1.0,
            "done": True,
        }

        mock_close_data = {"message": "Success", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            mock_seed_session_data,
            mock_response_1,
            mock_step_1,
            mock_response_2,
            mock_step_2,
            mock_close_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = MagicMock()
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = AviaryAgentRunRequest(
            task_idx=42, responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
        response = await agent.responses(request)

        assert response.env_id == env_id
        assert response.group_id == "42"
        assert response.contains_transitions is False
        assert len(response.output) == 4
        assert response.output[0].type == "function_call"
        assert response.output[0].call_id == "call_1"
        assert response.output[1].type == "function_call_output"
        assert response.output[1].call_id == "call_1"
        assert response.output[2].type == "function_call"
        assert response.output[2].call_id == "call_2"
        assert response.output[3].type == "function_call_output"
        assert response.output[3].call_id == "call_2"

    async def test_responses_collapse_old_env_states(self) -> None:
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
            collapse_old_env_states=True,
            old_env_state_message="[Hidden env state]",
        )
        agent = AviaryAgent(config=config, server_client=MagicMock(spec=ServerClient))

        env_id = str(uuid.uuid4())
        mock_seed_session_data = {
            "env_id": env_id,
            "obs": [{"role": "user", "content": "Initial env state", "is_env_state": True}],
            "tools": [],
        }

        mock_response_1 = {
            "id": "resp_1",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                NeMoGymResponseFunctionToolCall(
                    call_id="call_1", name="tool_1", arguments=json.dumps({"arg": "val"})
                ).model_dump()
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        mock_step_1 = {
            "obs": [{"role": "user", "content": "New env state after step 1", "is_env_state": True}],
            "reward": 0.0,
            "done": False,
        }

        mock_response_2 = {
            "id": "resp_2",
            "created_at": 1753983921.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                NeMoGymResponseFunctionToolCall(
                    call_id="call_2", name="tool_2", arguments=json.dumps({"arg": "val2"})
                ).model_dump()
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        mock_step_2 = {"obs": [{"role": "user", "content": "Final observation"}], "reward": 1.0, "done": True}

        mock_close_data = {"message": "Success", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            mock_seed_session_data,
            mock_response_1,
            mock_step_1,
            mock_response_2,
            mock_step_2,
            mock_close_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = AviaryAgentRunRequest(
            task_idx=0, responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
        await agent.responses(request)

        calls = agent.server_client.post.await_args_list
        second_model_call = calls[3]
        second_model_input = second_model_call[1]["json"].input

        has_hidden_message = any(
            isinstance(msg, NeMoGymEasyInputMessage) and msg.content == "[Hidden env state]"
            for msg in second_model_input
        )
        assert has_hidden_message, "Old env state should be collapsed to hidden message"

        old_env_state_present = any(
            isinstance(msg, dict) and msg.get("content") == "Initial env state" for msg in second_model_input
        )
        assert not old_env_state_present, "Old env state content should not be present"

    async def test_run_workflow(self) -> None:
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
        agent = AviaryAgent(config=config, server_client=MagicMock(spec=ServerClient))

        env_id = str(uuid.uuid4())
        mock_seed_session_data = {"env_id": env_id, "obs": [{"role": "user", "content": "obs"}], "tools": []}

        mock_response_data = {
            "id": "resp_1",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                NeMoGymResponseFunctionToolCall(
                    call_id="call_1", name="tool_1", arguments=json.dumps({"arg": "val"})
                ).model_dump()
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        mock_step_data = {
            "obs": [{"type": "function_call_output", "call_id": "call_1", "output": "result"}],
            "reward": 1.0,
            "done": True,
        }
        mock_verify_data = {
            "reward": 1.0,
            "success": True,
            "task_idx": 0,
            "responses_create_params": {"input": []},
            "response": {
                "id": "resp_1",
                "created_at": 1753983920.0,
                "model": "dummy_model",
                "object": "response",
                "env_id": env_id,
                "group_id": "0",
                "contains_transitions": True,
                "output": [[{"role": "user", "content": "obs"}]],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            mock_seed_session_data,
            mock_response_data,
            mock_step_data,
            mock_verify_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = AviaryAgentRunRequest(
            task_idx=0, responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[])
        )
        verify_response = await agent.run(request)

        assert verify_response.reward == 1.0
        assert verify_response.success is True

        agent.server_client.post.assert_has_awaits(
            [
                call(server_name="my resources name", url_path="/seed_session", json={"task_idx": 0}),
                call(
                    server_name="my model name",
                    url_path="/v1/responses",
                    json=NeMoGymResponseCreateParamsNonStreaming.model_validate(
                        {"input": [{"role": "user", "content": "obs"}], "tools": []}
                    ),
                    cookies=None,
                ),
                call(
                    server_name="my resources name",
                    url_path="/step",
                    json={
                        "action": [
                            {
                                "call_id": "call_1",
                                "name": "tool_1",
                                "arguments": '{"arg": "val"}',
                                "type": "function_call",
                                "id": None,
                                "status": None,
                            }
                        ],
                        "env_id": env_id,
                    },
                ),
                call(server_name="my resources name", url_path="/close", json={"env_id": env_id}),
            ]
        )
