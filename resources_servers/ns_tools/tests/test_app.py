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
from nemo_gym.server_utils import ServerClient


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

        result = await server.verify(None, verify_request)

        assert result.reward == 1.0
        assert result.delegated_response is not None
        assert result.delegated_response["reward"] == 1.0

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

        result = await server.verify(None, verify_request)

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

        await server.verify(None, verify_request)

        call_args = server.server_client.post.call_args
        json_data = call_args.kwargs["json"]

        # Verify fields are passed through
        assert "question" in json_data
        assert "expected_answer" in json_data
        assert "responses_create_params" in json_data
        assert "response" in json_data
