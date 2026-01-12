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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.swe_agents.app import (
    ModelServerRef,
    SWEBenchRunRequest,
    SWEBenchVerifyResponse,
    SWEBenchWrapper,
    SWEBenchWrapperConfig,
)


class TestSWEBenchWrapperConfig:
    def test_configuration(self) -> None:
        """Test configuration options."""
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_swe_agent",
            entrypoint="responses_api_agents/swe_agents",
            agent_framework="swe_agent",
            agent_config="custom/config",
            agent_tools_file="tools.json",
            agent_max_turns=50,
            container_formatter=["docker://custom/{instance_id}"],
            swebench_tests_timeout=900,
            model_server=ModelServerRef(
                type="responses_api_models",
                name="test_model",
            ),
        )
        assert config.agent_framework == "swe_agent"
        assert config.agent_config == "custom/config"
        assert config.agent_tools_file == "tools.json"
        assert config.agent_max_turns == 50
        assert config.container_formatter == ["docker://custom/{instance_id}"]
        assert config.swebench_tests_timeout == 900

    def test_configuration_multiple_container_formatters(self) -> None:
        """Test configuration with multiple container_formatter paths."""
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_swe_agent",
            entrypoint="responses_api_agents/swe_agents",
            agent_framework="swe_agent",
            container_formatter=[
                "docker://single/{instance_id}",
                "docker://second/{instance_id}",
            ],
            swebench_tests_timeout=900,
            model_server=ModelServerRef(
                type="responses_api_models",
                name="test_model",
            ),
        )
        assert len(config.container_formatter) == 2
        assert config.container_formatter[0] == "docker://single/{instance_id}"
        assert config.container_formatter[1] == "docker://second/{instance_id}"

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            model_server=ModelServerRef(type="responses_api_models", name="test"),
        )
        assert config.agent_framework == "swe_agent"
        assert config.agent_config is None
        assert config.agent_tools_file is None
        assert config.agent_max_turns == 100
        assert config.swebench_tests_timeout == 1800
        assert config.concurrency == 256
        assert config.openhands_setup_dir is None
        assert config.swebench_setup_dir is None
        assert config.r2e_gym_setup_dir is None


class TestSWEBenchWrapper:
    """Tests for SWEBenchWrapper class."""

    def _create_config(self, agent_framework: str = "swe_agent") -> SWEBenchWrapperConfig:
        """Helper to create a test config."""
        return SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_swe_agent",
            entrypoint="responses_api_agents/swe_agents",
            agent_framework=agent_framework,
            container_formatter=["docker://custom/{instance_id}"],
            swebench_tests_timeout=900,
            model_server=ModelServerRef(type="responses_api_models", name="test_model"),
            concurrency=1,
        )

    @patch("responses_api_agents.swe_agents.app.setup_openhands_environment")
    @patch("responses_api_agents.swe_agents.app.setup_swebench_environment")
    @patch("responses_api_agents.swe_agents.app.setup_r2e_gym_environment")
    def test_model_post_init_swe_agent(self, mock_r2e, mock_swebench, mock_openhands) -> None:
        """Test model_post_init for swe_agent framework."""
        mock_swebench.return_value = Path("/mock/swebench")
        mock_r2e.return_value = Path("/mock/r2e")

        config = self._create_config("swe_agent")
        wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))

        mock_openhands.assert_not_called()
        mock_swebench.assert_called_once()
        mock_r2e.assert_called_once()
        assert wrapper.config.swebench_setup_dir == Path("/mock/swebench")
        assert wrapper.config.r2e_gym_setup_dir == Path("/mock/r2e")
        assert wrapper.config.run_session_id is not None
        assert wrapper.sem is not None

    @patch("responses_api_agents.swe_agents.app.setup_openhands_environment")
    @patch("responses_api_agents.swe_agents.app.setup_swebench_environment")
    @patch("responses_api_agents.swe_agents.app.setup_r2e_gym_environment")
    def test_model_post_init_openhands(self, mock_r2e, mock_swebench, mock_openhands) -> None:
        """Test model_post_init for openhands framework."""
        mock_openhands.return_value = Path("/mock/openhands")
        mock_swebench.return_value = Path("/mock/swebench")
        mock_r2e.return_value = Path("/mock/r2e")

        config = self._create_config("openhands")
        wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))

        mock_openhands.assert_called_once()
        mock_swebench.assert_called_once()
        mock_r2e.assert_called_once()
        assert wrapper.config.openhands_setup_dir == Path("/mock/openhands")
        assert wrapper.config.swebench_setup_dir == Path("/mock/swebench")
        assert wrapper.config.r2e_gym_setup_dir == Path("/mock/r2e")
        assert wrapper.config.run_session_id is not None

    @pytest.mark.asyncio
    @patch("responses_api_agents.swe_agents.app.setup_openhands_environment")
    @patch("responses_api_agents.swe_agents.app.setup_swebench_environment")
    @patch("responses_api_agents.swe_agents.app.setup_r2e_gym_environment")
    @patch("responses_api_agents.swe_agents.app.runner_ray_remote")
    @patch("responses_api_agents.swe_agents.app.extract_problem_info")
    @patch("responses_api_agents.swe_agents.app.get_model_endpoint")
    @patch("responses_api_agents.swe_agents.app.convert_tools_to_function_format")
    @patch("responses_api_agents.swe_agents.app.convert_trajectory_to_output_items")
    async def test_responses_success_with_trajectory(
        self,
        mock_convert_traj,
        mock_convert_tools,
        mock_get_endpoint,
        mock_extract_info,
        mock_ray_remote,
        mock_r2e,
        mock_swebench,
        mock_openhands,
    ) -> None:
        """Test responses method with successful trajectory."""
        mock_swebench.return_value = Path("/mock/swebench")
        mock_r2e.return_value = Path("/mock/r2e")

        mock_extract_info.return_value = {
            "problem_statement": "Fix bug",
            "instance_id": "test-instance",
            "base_commit": "abc123",
            "dataset_name": "SWE-bench",
            "split": "test",
            "instance_dict": "{}",
            "container_formatter": ["docker://test/{instance_id}"],
        }
        mock_get_endpoint.return_value = "http://localhost:8000/v1"

        ray_result = {
            "trajectory": [{"role": "assistant", "content": "Fixed the bug"}],
            "tools": [{"type": "function", "function": {"name": "test_tool"}}],
            "resolved": True,
            "patch_exists": True,
            "patch_successfully_applied": True,
            "swe-bench-metrics": {"resolved": True},
            "instance_id": "test-instance",
        }

        async def make_coroutine(*args, **kwargs):
            return ray_result

        mock_ray_remote.remote.side_effect = make_coroutine

        mock_convert_tools.return_value = [{"type": "function", "name": "test_tool"}]
        mock_convert_traj.return_value = [
            NeMoGymResponseOutputMessage(
                id="msg-1",
                content=[NeMoGymResponseOutputText(type="output_text", text="Fixed the bug", annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

        config = self._create_config("swe_agent")
        wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))

        body = NeMoGymResponseCreateParamsNonStreaming(
            model="test-model",
            input=[],
            metadata={
                "problem_statement": "Fix bug",
                "instance_id": "test-instance",
                "base_commit": "abc123",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": "{}",
            },
        )

        response = await wrapper.responses(body)

        assert response.id.startswith("swebench-")
        assert response.object == "response"
        assert len(response.output) > 0
        assert response.metadata["resolved"] == "True"
        mock_ray_remote.remote.assert_called_once()

    @pytest.mark.asyncio
    @patch("responses_api_agents.swe_agents.app.setup_openhands_environment")
    @patch("responses_api_agents.swe_agents.app.setup_swebench_environment")
    @patch("responses_api_agents.swe_agents.app.setup_r2e_gym_environment")
    @patch("responses_api_agents.swe_agents.app.runner_ray_remote")
    @patch("responses_api_agents.swe_agents.app.extract_problem_info")
    @patch("responses_api_agents.swe_agents.app.get_model_endpoint")
    async def test_responses_success_no_trajectory(
        self, mock_get_endpoint, mock_extract_info, mock_ray_remote, mock_r2e, mock_swebench, mock_openhands
    ) -> None:
        """Test responses method with no trajectory (creates summary message)."""
        mock_swebench.return_value = Path("/mock/swebench")
        mock_r2e.return_value = Path("/mock/r2e")

        mock_extract_info.return_value = {
            "problem_statement": "Fix bug",
            "instance_id": "test-instance",
            "base_commit": "abc123",
            "dataset_name": "SWE-bench",
            "split": "test",
            "instance_dict": "{}",
            "container_formatter": ["docker://test/{instance_id}"],
        }
        mock_get_endpoint.return_value = "http://localhost:8000/v1"

        ray_result = {
            "trajectory": [],
            "tools": [],
            "resolved": False,
            "instance_id": "test-instance",
        }

        async def make_coroutine(*args, **kwargs):
            return ray_result

        mock_ray_remote.remote.side_effect = make_coroutine

        config = self._create_config("swe_agent")
        wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))

        body = NeMoGymResponseCreateParamsNonStreaming(
            model="test-model",
            input=[],
            metadata={
                "problem_statement": "Fix bug",
                "instance_id": "test-instance",
                "base_commit": "abc123",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": "{}",
            },
        )

        response = await wrapper.responses(body)

        assert response.id.startswith("swebench-")
        assert len(response.output) == 1
        assert response.output[0].role == "assistant"

    @pytest.mark.asyncio
    @patch("responses_api_agents.swe_agents.app.setup_openhands_environment")
    @patch("responses_api_agents.swe_agents.app.setup_swebench_environment")
    @patch("responses_api_agents.swe_agents.app.setup_r2e_gym_environment")
    @patch("responses_api_agents.swe_agents.app.runner_ray_remote")
    @patch("responses_api_agents.swe_agents.app.extract_problem_info")
    @patch("responses_api_agents.swe_agents.app.get_model_endpoint")
    async def test_responses_exception_handling(
        self, mock_get_endpoint, mock_extract_info, mock_ray_remote, mock_r2e, mock_swebench, mock_openhands
    ) -> None:
        """Test responses method handles exceptions gracefully."""
        mock_swebench.return_value = Path("/mock/swebench")
        mock_r2e.return_value = Path("/mock/r2e")

        mock_extract_info.return_value = {
            "problem_statement": "Fix bug",
            "instance_id": "test-instance",
            "base_commit": "abc123",
            "dataset_name": "SWE-bench",
            "split": "test",
            "instance_dict": "{}",
            "container_formatter": ["docker://test/{instance_id}"],
        }
        mock_get_endpoint.return_value = "http://localhost:8000/v1"

        mock_ray_remote.remote.side_effect = RuntimeError("Container failed")

        config = self._create_config("swe_agent")
        wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))

        body = NeMoGymResponseCreateParamsNonStreaming(
            model="test-model",
            input=[],
            metadata={
                "problem_statement": "Fix bug",
                "instance_id": "test-instance",
                "base_commit": "abc123",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": "{}",
            },
        )

        response = await wrapper.responses(body)

        assert "error" in response.id
        assert response.metadata["error"] == "Container failed"
        assert response.tool_choice == "none"
        assert response.tools == []

    @pytest.mark.asyncio
    @patch("responses_api_agents.swe_agents.app.setup_openhands_environment")
    @patch("responses_api_agents.swe_agents.app.setup_swebench_environment")
    @patch("responses_api_agents.swe_agents.app.setup_r2e_gym_environment")
    @patch("responses_api_agents.swe_agents.app.extract_input_messages_from_trajectory")
    async def test_run_resolved(self, mock_extract_input, mock_r2e, mock_swebench, mock_openhands) -> None:
        """Test run method with resolved problem."""
        import json

        mock_swebench.return_value = Path("/mock/swebench")
        mock_r2e.return_value = Path("/mock/r2e")

        output_msg = NeMoGymResponseOutputMessage(
            id="msg-output",
            content=[NeMoGymResponseOutputText(type="output_text", text="Fixed", annotations=[])],
            role="assistant",
            status="completed",
            type="message",
        )
        mock_extract_input.return_value = (
            [{"role": "user", "content": "Fix this"}],
            [output_msg],
        )

        config = self._create_config("swe_agent")
        wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))

        mock_response = NeMoGymResponse(
            id="swebench-test-instance",
            created_at=1234567890,
            model="test-model",
            object="response",
            output=[output_msg],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            metadata={
                "resolved": "True",
                "patch_exists": "True",
                "patch_successfully_applied": "True",
                "instance_id": "test-instance",
                "swe-bench-metrics": json.dumps({"resolved": True}),
            },
        )

        with patch.object(SWEBenchWrapper, "responses", new_callable=AsyncMock) as mock_responses:
            mock_responses.return_value = mock_response

            body = SWEBenchRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    model="test-model",
                    input=[],
                    tool_choice="auto",
                    metadata={
                        "problem_statement": "Fix bug",
                        "instance_id": "test-instance",
                        "base_commit": "abc123",
                        "dataset_name": "SWE-bench",
                        "split": "test",
                        "instance_dict": "{}",
                    },
                )
            )

            result = await wrapper.run(body)

            assert isinstance(result, SWEBenchVerifyResponse)
            assert result.reward == 1.0
            assert result.resolved == 1.0
            assert result.patch_exists == 1.0
            assert result.patch_successfully_applied == 1.0
            mock_responses.assert_called_once()

    @pytest.mark.asyncio
    @patch("responses_api_agents.swe_agents.app.setup_openhands_environment")
    @patch("responses_api_agents.swe_agents.app.setup_swebench_environment")
    @patch("responses_api_agents.swe_agents.app.setup_r2e_gym_environment")
    @patch("responses_api_agents.swe_agents.app.extract_input_messages_from_trajectory")
    async def test_run_not_resolved(self, mock_extract_input, mock_r2e, mock_swebench, mock_openhands) -> None:
        """Test run method with unresolved problem."""
        mock_swebench.return_value = Path("/mock/swebench")
        mock_r2e.return_value = Path("/mock/r2e")

        output_msg = NeMoGymResponseOutputMessage(
            id="msg-output",
            content=[NeMoGymResponseOutputText(type="output_text", text="Failed", annotations=[])],
            role="assistant",
            status="completed",
            type="message",
        )
        mock_extract_input.return_value = (
            [{"role": "user", "content": "Fix this"}],
            [output_msg],
        )

        config = self._create_config("swe_agent")
        wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))

        mock_response = NeMoGymResponse(
            id="swebench-test-instance",
            created_at=1234567890,
            model="test-model",
            object="response",
            output=[output_msg],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            metadata={
                "resolved": "False",
                "patch_exists": "True",
                "patch_successfully_applied": "False",
                "instance_id": "test-instance",
            },
        )

        with patch.object(SWEBenchWrapper, "responses", new_callable=AsyncMock) as mock_responses:
            mock_responses.return_value = mock_response

            body = SWEBenchRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    model="test-model",
                    input=[],
                    tool_choice="auto",
                    metadata={
                        "problem_statement": "Fix bug",
                        "instance_id": "test-instance",
                        "base_commit": "abc123",
                        "dataset_name": "SWE-bench",
                        "split": "test",
                        "instance_dict": "{}",
                    },
                )
            )

            result = await wrapper.run(body)

            assert result.reward == 0.0
            assert result.resolved == 0.0
            assert result.patch_exists == 1.0
            assert result.patch_successfully_applied == 0.0


class TestRunnerRayRemote:
    """Tests for the runner_ray_remote function."""

    def test_runner_ray_remote_exists(self) -> None:
        """Test that runner_ray_remote is a ray remote function."""
        from responses_api_agents.swe_agents.app import runner_ray_remote

        assert hasattr(runner_ray_remote, "remote")


class TestRequestResponseModels:
    """Tests for request/response model classes."""

    def test_swebench_run_request_extra_fields(self) -> None:
        """Test SWEBenchRunRequest allows extra fields."""
        assert SWEBenchRunRequest.model_config.get("extra") == "allow"

    def test_swebench_verify_response_fields(self) -> None:
        """Test SWEBenchVerifyResponse has correct fields."""
        assert hasattr(SWEBenchVerifyResponse, "model_fields")
        fields = SWEBenchVerifyResponse.model_fields
        assert "swebench_metrics" in fields
        assert "resolved" in fields
        assert "patch_exists" in fields
        assert "patch_successfully_applied" in fields
