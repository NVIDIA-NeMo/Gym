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
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.mini_swe_agent.app import (
    MiniSWEAgent,
    MiniSWEAgentConfig,
    MiniSWEAgentRunRequest,
    MiniSWEAgentVerifyResponse,
)


DEFAULT_RUN_SWEGYM_RESULT = {
    "test_instance_123": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix this bug."},
            {"role": "assistant", "content": "I'll help you fix the bug."},
            {"role": "user", "content": "Thank you!"},
        ],
        "responses": [
            {
                "choices": [],
                "provider_specific_fields": {
                    "prompt_token_ids": [],
                    "generation_token_ids": [],
                    "generation_log_probs": [],
                },
            }
        ],
        "eval_report": {
            "eval_report": {
                "test_instance_123": {
                    "resolved": True,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["test1"], "failure": []},
                        "PASS_TO_PASS": {"success": ["test2"], "failure": []},
                    },
                }
            }
        },
    }
}

DEFAULT_CONFIG_YAML = """
model:
  model_kwargs:
    temperature: 0.5
    top_p: 0.8
"""

DEFAULT_CHAT_COMPLETION = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "test_model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}


def create_test_config(
    host: str = "0.0.0.0",
    port: int = 8080,
    model_name: str = "test_model",
    env: str = "singularity",
    cache_dir_template: str = "/tmp/cache/gym.sif",
) -> MiniSWEAgentConfig:
    return MiniSWEAgentConfig(
        name="mini_swe_agent",
        host=host,
        port=port,
        entrypoint="",
        resources_server=ResourcesServerRef(
            type="resources_servers",
            name="mini_swe_resource",
        ),
        model_server=ModelServerRef(
            type="responses_api_models",
            name=model_name,
        ),
        env=env,
        concurrency=1,
        cache_dir_template=cache_dir_template,
    )


def setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict):
    mock_server_client_instance = MagicMock()
    mock_server_client_instance.global_config_dict = {"policy_model_name": "test_model"}
    mock_load_from_global_config.return_value = mock_server_client_instance

    mock_get_first_server_config_dict.return_value = {
        "host": "0.0.0.0",
        "port": 8080,
    }


def setup_config_path_mock(mock_get_config_path, config_yaml: str = DEFAULT_CONFIG_YAML):
    mock_config_path = MagicMock()
    mock_config_path.read_text.return_value = config_yaml
    mock_get_config_path.return_value = mock_config_path


def setup_run_swegym_mock(
    mock_to_thread,
    mock_runner_ray_remote,
    run_swegym_result: Dict[str, Any] = None,
):
    """Setup mock for Ray-based run_swegym execution"""
    if run_swegym_result is None:
        run_swegym_result = DEFAULT_RUN_SWEGYM_RESULT

    # Mock the Ray remote function to return a future-like object
    mock_future = MagicMock()
    mock_runner_ray_remote.remote.return_value = mock_future

    # Mock asyncio.to_thread (which calls ray.get) to return the result
    mock_to_thread.return_value = run_swegym_result


def create_run_request(
    instance_id: str = "test_instance_123",
    temperature: float = 0.5,
    top_p: float = 0.8,
    subset: str = "gym",
    split: str = "train",
    input_data: list = None,
) -> MiniSWEAgentRunRequest:
    """Create a test run request with default values."""
    if input_data is None:
        input_data = []

    return MiniSWEAgentRunRequest(
        instance_id=instance_id,
        subset=subset,
        split=split,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            temperature=temperature, top_p=top_p, input=input_data
        ),
    )


def create_chat_completion_request(
    model: str = "test_model",
    messages: list = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> NeMoGymChatCompletionCreateParamsNonStreaming:
    if messages is None:
        messages = [{"role": "user", "content": "Hello!"}]

    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return NeMoGymChatCompletionCreateParamsNonStreaming(**kwargs)


def assert_run_response(
    response: MiniSWEAgentVerifyResponse,
    expected_reward: float = 1.0,
    expected_temperature: float = 0.5,
    expected_top_p: float = 0.8,
    expected_input_length: int = 2,
):
    assert isinstance(response, MiniSWEAgentVerifyResponse)
    assert response.reward == expected_reward
    assert response.responses_create_params.temperature == expected_temperature
    assert response.responses_create_params.top_p == expected_top_p
    assert len(response.responses_create_params.input) == expected_input_length

    if expected_input_length >= 2:
        assert response.responses_create_params.input[0]["role"] == "system"
        assert response.responses_create_params.input[1]["role"] == "user"


def assert_run_swegym_called(
    mock_to_thread,
    subset: str = "gym",
    split: str = "train",
    instance_id: str = "test_instance_123",
):
    mock_to_thread.assert_called_once()
    call_args = mock_to_thread.call_args
    args = call_args[0]
    assert len(args) >= 1


class TestApp:
    def test_sanity(self) -> None:
        config = create_test_config(model_name="", cache_dir_template="/")
        MiniSWEAgent(config=config, server_client=MagicMock(spec=ServerClient))

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_successful_execution(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        """Test successful execution of the run method with mocked run_swegym."""

        config = create_test_config()
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)
        setup_run_swegym_mock(mock_to_thread, mock_runner_ray_remote)

        run_request = create_run_request()

        response = await server.run(run_request)

        assert_run_response(response)

        assert_run_swegym_called(mock_to_thread)

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_failed_execution(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        """Test run method when run_swegym fails."""

        config = create_test_config(env="docker")
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)

        # Mock Ray remote function
        mock_future = MagicMock()
        mock_runner_ray_remote.remote.return_value = mock_future

        # Mock asyncio.to_thread (ray.get) to raise an exception
        mock_to_thread.side_effect = Exception("run_swegym failed")

        run_request = create_run_request(instance_id="test_instance_456", temperature=0.3, top_p=0.95)

        response = await server.run(run_request)

        assert_run_response(
            response,
            expected_reward=0.0,
            expected_temperature=0.3,
            expected_top_p=0.95,
            expected_input_length=0,
        )

        assert_run_swegym_called(mock_to_thread, instance_id="test_instance_456")

    @patch("responses_api_agents.mini_swe_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.mini_swe_agent.app.get_first_server_config_dict")
    @patch("responses_api_agents.mini_swe_agent.app.get_config_path")
    @patch("responses_api_agents.mini_swe_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_swegym_not_found(
        self,
        mock_to_thread,
        mock_runner_ray_remote,
        mock_get_config_path,
        mock_get_first_server_config_dict,
        mock_load_from_global_config,
    ) -> None:
        config = create_test_config(env="docker")
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        setup_server_client_mocks(mock_load_from_global_config, mock_get_first_server_config_dict)
        setup_config_path_mock(mock_get_config_path)

        # Mock Ray remote function
        mock_future = MagicMock()
        mock_runner_ray_remote.remote.return_value = mock_future

        # Mock asyncio.to_thread (ray.get) to raise FileNotFoundError
        mock_to_thread.side_effect = FileNotFoundError("run_swegym not found")

        run_request = create_run_request(instance_id="test_instance_789", temperature=0.2, top_p=1.0)

        response = await server.run(run_request)

        assert_run_response(
            response,
            expected_reward=0.0,
            expected_temperature=0.2,
            expected_top_p=1.0,
            expected_input_length=0,
        )

        assert_run_swegym_called(mock_to_thread, instance_id="test_instance_789")

    async def test_responses_not_implemented(self) -> None:
        config = create_test_config(env="docker")
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        request_body = NeMoGymResponseCreateParamsNonStreaming(temperature=0.7, top_p=0.9, input=[])

        with pytest.raises(NotImplementedError):
            await server.responses(request_body)

    def test_endpoints_registration(self) -> None:
        config = create_test_config(env="docker")
        mock_server_client = MagicMock(spec=ServerClient)
        server = MiniSWEAgent(config=config, server_client=mock_server_client)

        app = server.setup_webserver()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/v1/responses", json={"temperature": 0.7, "top_p": 0.9, "input": []})
        assert response.status_code == 500

        run_response = client.post("/run", json={})
        assert run_response.status_code != 404
