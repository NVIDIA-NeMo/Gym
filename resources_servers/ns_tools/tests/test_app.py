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
from unittest.mock import AsyncMock, MagicMock

from app import (
    NSToolsConfig,
    NSToolsResourcesServer,
    NSToolsVerifyRequest,
)

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


def create_mock_request(session_id: str = "test-session-id") -> MagicMock:
    """Create a mock FastAPI Request with session."""
    mock_request = MagicMock()
    mock_request.session = {SESSION_ID_KEY: session_id}
    return mock_request


class TestApp:
    def test_sanity(self) -> None:
        """Test that the server can be instantiated with minimal config."""
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
        )
        NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_config_with_verifiers(self) -> None:
        """Test configuration with verifiers."""
        verifiers = {
            "math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge"),
        }
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            verifiers=verifiers,
            default_verifier="math_with_judge",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        assert len(server.config.verifiers) == 1
        assert "math_with_judge" in server.config.verifiers
        assert server.config.default_verifier == "math_with_judge"

    async def test_verify_delegates_to_math_with_judge(self) -> None:
        """Test that verification is delegated to math_with_judge verifier."""
        verifiers = {
            "math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge"),
        }
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            verifiers=verifiers,
            default_verifier="math_with_judge",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # Mock the server_client.post to return a successful verification
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"reward": 1.0, "extracted_answer": "4"})
        server.server_client.post = AsyncMock(return_value=mock_response)

        # Build a NeMoGymResponse with a valid output
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [
                        {
                            "annotations": [],
                            "text": "The answer is \\boxed{4}.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request = NSToolsVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "system", "content": "You are a helpful math assistant."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
            },
            response=response,
            question="What is 2 + 2?",
            expected_answer="4",
        )

        mock_request = create_mock_request()
        result = await server.verify(mock_request, verify_request)

        assert result.reward == 1.0
        assert result.delegated_response is not None
        assert result.delegated_response["reward"] == 1.0
        # Verify timing metrics are included (0 since no tool calls in this test)
        assert result.total_tool_execution_time_seconds == 0.0
        assert result.num_tool_calls == 0
        assert result.avg_tool_call_time_seconds == 0.0
        assert result.tool_timeout_count == 0
        assert result.tool_request_timeout_count == 0

        # Verify the server_client.post was called with correct args
        server.server_client.post.assert_called_once()
        call_args = server.server_client.post.call_args
        assert call_args.kwargs["server_name"] == "math_with_judge"
        assert call_args.kwargs["url_path"] == "/verify"

    async def test_verify_uses_default_verifier(self) -> None:
        """Test that default verifier is used when verifier_type not specified."""
        verifiers = {
            "math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge"),
        }
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            verifiers=verifiers,
            default_verifier="math_with_judge",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"reward": 0.0})
        server.server_client.post = AsyncMock(return_value=mock_response)

        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [
                        {
                            "annotations": [],
                            "text": "The answer is \\boxed{5}.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        # No verifier_type specified - should use default
        verify_request = NSToolsVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "What is 2 + 2?"}],
            },
            response=response,
            question="What is 2 + 2?",
            expected_answer="4",
        )

        mock_request = create_mock_request()
        result = await server.verify(mock_request, verify_request)

        assert result.reward == 0.0
        call_args = server.server_client.post.call_args
        assert call_args.kwargs["server_name"] == "math_with_judge"

    async def test_verify_passes_through_fields(self) -> None:
        """Test that all sample fields are passed through to the delegated verifier."""
        verifiers = {
            "math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge"),
        }
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            verifiers=verifiers,
            default_verifier="math_with_judge",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=mock_response)

        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "\\boxed{4}", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request = NSToolsVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "What is 2 + 2?"}],
            },
            response=response,
            question="What is 2 + 2?",
            expected_answer="4",
        )

        mock_request = create_mock_request()
        await server.verify(mock_request, verify_request)

        call_args = server.server_client.post.call_args
        json_data = call_args.kwargs["json"]

        # Verify fields are passed through
        assert "question" in json_data
        assert "expected_answer" in json_data
        assert "responses_create_params" in json_data
        assert "response" in json_data

    def test_aggregate_timing_metrics_basic(self) -> None:
        """Test _aggregate_timing_metrics with normal tool calls (no mocking needed)."""
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # Set up timing data directly
        session_id = "test-session-with-tools"
        server._timing_by_session[session_id] = [
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": 0.5,
                "is_internal_timeout": False,
                "is_request_timeout": False,
            },
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": 0.3,
                "is_internal_timeout": False,
                "is_request_timeout": False,
            },
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": 0.2,
                "is_internal_timeout": False,
                "is_request_timeout": False,
            },
        ]

        metrics = server._aggregate_timing_metrics(session_id)

        # Verify aggregation
        assert metrics["total_tool_execution_time_seconds"] == 1.0  # 0.5 + 0.3 + 0.2
        assert metrics["num_tool_calls"] == 3
        assert abs(metrics["avg_tool_call_time_seconds"] - (1.0 / 3)) < 0.001
        assert metrics["tool_timeout_count"] == 0
        assert metrics["tool_request_timeout_count"] == 0

        # Verify timing data is cleaned up
        assert session_id not in server._timing_by_session

    def test_aggregate_timing_metrics_internal_timeouts(self) -> None:
        """Test _aggregate_timing_metrics counts internal sandbox timeouts (no mocking needed)."""
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        session_id = "test-session-with-timeouts"
        server._timing_by_session[session_id] = [
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": 0.5,
                "is_internal_timeout": False,
                "is_request_timeout": False,
            },
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": 10.0,
                "is_internal_timeout": True,
                "is_request_timeout": False,
            },
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": 10.0,
                "is_internal_timeout": True,
                "is_request_timeout": False,
            },
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": 0.3,
                "is_internal_timeout": False,
                "is_request_timeout": False,
            },
        ]

        metrics = server._aggregate_timing_metrics(session_id)

        assert metrics["num_tool_calls"] == 4
        assert metrics["tool_timeout_count"] == 2
        assert metrics["tool_request_timeout_count"] == 0

    def test_aggregate_timing_metrics_request_timeouts(self) -> None:
        """Test _aggregate_timing_metrics counts HTTP request timeouts (no mocking needed)."""
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        session_id = "test-session-with-request-timeout"
        server._timing_by_session[session_id] = [
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": 0.5,
                "is_internal_timeout": False,
                "is_request_timeout": False,
            },
            {
                "tool_name": "stateful_python_code_exec",
                "execution_time_seconds": 30.0,
                "is_internal_timeout": False,
                "is_request_timeout": True,
            },
        ]

        metrics = server._aggregate_timing_metrics(session_id)

        assert metrics["num_tool_calls"] == 2
        assert metrics["tool_timeout_count"] == 0
        assert metrics["tool_request_timeout_count"] == 1

    def test_aggregate_timing_metrics_zero_calls(self) -> None:
        """Test _aggregate_timing_metrics handles zero tool calls without div-by-zero (no mocking needed)."""
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # Empty timing list
        session_id = "test-session-no-tools"
        server._timing_by_session[session_id] = []

        metrics = server._aggregate_timing_metrics(session_id)

        assert metrics["num_tool_calls"] == 0
        assert metrics["total_tool_execution_time_seconds"] == 0.0
        assert metrics["avg_tool_call_time_seconds"] == 0.0
        assert metrics["tool_timeout_count"] == 0
        assert metrics["tool_request_timeout_count"] == 0

    def test_aggregate_timing_metrics_unknown_session(self) -> None:
        """Test _aggregate_timing_metrics handles unknown session gracefully (no mocking needed)."""
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # Don't add any timing data for this session
        metrics = server._aggregate_timing_metrics("nonexistent-session")

        assert metrics["num_tool_calls"] == 0
        assert metrics["total_tool_execution_time_seconds"] == 0.0
        assert metrics["avg_tool_call_time_seconds"] == 0.0

    def test_aggregate_timing_metrics_none_session(self) -> None:
        """Test _aggregate_timing_metrics handles None session gracefully (no mocking needed)."""
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        metrics = server._aggregate_timing_metrics(None)

        assert metrics["num_tool_calls"] == 0
        assert metrics["total_tool_execution_time_seconds"] == 0.0
        assert metrics["avg_tool_call_time_seconds"] == 0.0
