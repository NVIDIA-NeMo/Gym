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
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import (
    ExecuteCalculationRequest,
    FetchDataRequest,
    MyCustomToolsConfig,
    MyCustomToolsResourcesServer,
    MyCustomToolsVerifyRequest,
    SearchDatabaseRequest,
)

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def create_server(available_verifiers=None, default_verifier="xlam_fc"):
    """Helper to create a server instance with mocked dependencies."""
    if available_verifiers is None:
        available_verifiers = {
            "xlam_fc": ResourcesServerRef(type="resources_servers", name="xlam_fc_verifier"),
            "mcqa": ResourcesServerRef(type="resources_servers", name="mcqa_verifier"),
        }

    config = MyCustomToolsConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="my_custom_tools",
        available_verifiers=available_verifiers,
        default_verifier=default_verifier,
    )
    return MyCustomToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestToolEndpoints:
    """Tests for the custom tool endpoints."""

    async def test_search_database(self):
        """Test the search_database tool."""
        server = create_server()
        request = SearchDatabaseRequest(query="test query", database="test_db")
        result = await server.search_database(request)

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0]["database"] == "test_db"
        assert "test query" in result.results[0]["name"]

    async def test_search_database_default_db(self):
        """Test search_database with default database."""
        server = create_server()
        request = SearchDatabaseRequest(query="another query")
        result = await server.search_database(request)

        assert result.results[0]["database"] == "default"

    async def test_execute_calculation_simple(self):
        """Test execute_calculation with simple expression."""
        server = create_server()
        request = ExecuteCalculationRequest(expression="2 + 2", precision=2)
        result = await server.execute_calculation(request)

        assert result.result == 4.0
        assert result.formatted == "4.00"

    async def test_execute_calculation_with_precision(self):
        """Test execute_calculation with custom precision."""
        server = create_server()
        request = ExecuteCalculationRequest(expression="100 * 1.15", precision=2)
        result = await server.execute_calculation(request)

        assert result.result == 115.0
        assert result.formatted == "115.00"

    async def test_execute_calculation_complex(self):
        """Test execute_calculation with more complex expression."""
        server = create_server()
        request = ExecuteCalculationRequest(expression="(10 + 5) * 2 / 3", precision=4)
        result = await server.execute_calculation(request)

        assert result.result == pytest.approx(10.0, rel=1e-4)

    async def test_execute_calculation_invalid(self):
        """Test execute_calculation with invalid expression returns 0."""
        server = create_server()
        request = ExecuteCalculationRequest(expression="invalid_expression", precision=2)
        result = await server.execute_calculation(request)

        assert result.result == 0.0

    async def test_fetch_data(self):
        """Test the fetch_data tool."""
        server = create_server()
        request = FetchDataRequest(source_id="customers", fields=["name", "email"])
        result = await server.fetch_data(request)

        assert result.source == "customers"
        assert result.data["source_id"] == "customers"
        assert result.data["fields"] == ["name", "email"]
        assert result.data["fetched"] is True

    async def test_fetch_data_no_fields(self):
        """Test fetch_data with no fields specified."""
        server = create_server()
        request = FetchDataRequest(source_id="products")
        result = await server.fetch_data(request)

        assert result.source == "products"
        assert result.data["fields"] == []


class TestDynamicVerification:
    """Tests for the dynamic verification delegation."""

    def _create_mock_response(self, reward: float, extra_fields: dict = None):
        """Create a mock HTTP response for the server_client."""
        response_data = {"reward": reward, **(extra_fields or {})}

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=response_data)
        return mock_response

    def _create_verify_request(self, verifier_type: str = None, **kwargs):
        """Create a verify request with a mock model response."""
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "search_database",
                    "arguments": json.dumps({"query": "test", "database": "default"}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        return MyCustomToolsVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Test query"}],
            },
            response=response,
            verifier_type=verifier_type,
            **kwargs,
        )

    async def test_verify_delegates_to_xlam_fc(self):
        """Test that verification is delegated to xlam_fc verifier."""
        server = create_server()

        # Mock the server_client.post to return a successful verification
        mock_response = self._create_mock_response(
            reward=1.0, extra_fields={"num_correct": 1, "num_expected": 1}
        )
        server.server_client.post = AsyncMock(return_value=mock_response)

        request = self._create_verify_request(
            verifier_type="xlam_fc",
            expected_answers=[{"name": "search_database", "arguments": {"query": "test"}}],
        )

        result = await server.verify(request)

        assert result.reward == 1.0
        assert result.verifier_used == "xlam_fc"
        assert result.delegated_response is not None
        assert result.delegated_response["reward"] == 1.0

        # Verify the server_client.post was called with correct args
        server.server_client.post.assert_called_once()
        call_args = server.server_client.post.call_args
        assert call_args.kwargs["server_name"] == "xlam_fc_verifier"
        assert call_args.kwargs["url_path"] == "/verify"

    async def test_verify_delegates_to_mcqa(self):
        """Test that verification is delegated to mcqa verifier."""
        server = create_server()

        mock_response = self._create_mock_response(
            reward=1.0, extra_fields={"expected_answer": "B", "extracted_answer": "B"}
        )
        server.server_client.post = AsyncMock(return_value=mock_response)

        request = self._create_verify_request(
            verifier_type="mcqa",
            expected_answer="B",
            options=[{"A": "option1"}, {"B": "option2"}],
        )

        result = await server.verify(request)

        assert result.reward == 1.0
        assert result.verifier_used == "mcqa"

        call_args = server.server_client.post.call_args
        assert call_args.kwargs["server_name"] == "mcqa_verifier"

    async def test_verify_uses_default_verifier(self):
        """Test that default verifier is used when verifier_type not specified."""
        server = create_server(default_verifier="xlam_fc")

        mock_response = self._create_mock_response(reward=0.5)
        server.server_client.post = AsyncMock(return_value=mock_response)

        # No verifier_type specified
        request = self._create_verify_request(verifier_type=None)

        result = await server.verify(request)

        assert result.verifier_used == "xlam_fc"
        call_args = server.server_client.post.call_args
        assert call_args.kwargs["server_name"] == "xlam_fc_verifier"

    async def test_verify_unknown_verifier(self):
        """Test handling of unknown verifier type."""
        server = create_server()

        request = self._create_verify_request(verifier_type="unknown_verifier")

        result = await server.verify(request)

        assert result.reward == 0.0
        assert "unknown:unknown_verifier" in result.verifier_used
        assert "error" in result.delegated_response

    async def test_verify_passes_through_all_fields(self):
        """Test that all sample fields are passed through to the delegated verifier."""
        server = create_server()

        mock_response = self._create_mock_response(reward=1.0)
        server.server_client.post = AsyncMock(return_value=mock_response)

        # Include various fields that should pass through
        request = self._create_verify_request(
            verifier_type="xlam_fc",
            expected_answers=[{"name": "search_database", "arguments": {"query": "test"}}],
            expected_answer="also_included",  # Extra field
            options=[{"A": "opt1"}],  # Extra field
        )

        await server.verify(request)

        call_args = server.server_client.post.call_args
        json_data = call_args.kwargs["json"]

        # Verify fields are passed through
        assert "expected_answers" in json_data
        assert "expected_answer" in json_data
        assert "options" in json_data
        assert "responses_create_params" in json_data
        assert "response" in json_data


class TestServerSetup:
    """Tests for server setup and configuration."""

    def test_sanity(self):
        """Test that the server can be instantiated."""
        server = create_server()
        assert server is not None
        assert server.config.name == "my_custom_tools"

    def test_setup_webserver_registers_endpoints(self):
        """Test that setup_webserver registers the tool endpoints."""
        server = create_server()
        app = server.setup_webserver()

        # Check that routes are registered
        routes = [route.path for route in app.routes]
        assert "/search_database" in routes
        assert "/execute_calculation" in routes
        assert "/fetch_data" in routes
        assert "/verify" in routes
        assert "/seed_session" in routes

    def test_config_with_multiple_verifiers(self):
        """Test configuration with multiple available verifiers."""
        verifiers = {
            "xlam_fc": ResourcesServerRef(type="resources_servers", name="verifier1"),
            "mcqa": ResourcesServerRef(type="resources_servers", name="verifier2"),
            "custom": ResourcesServerRef(type="resources_servers", name="verifier3"),
        }
        server = create_server(available_verifiers=verifiers)

        assert len(server.config.available_verifiers) == 3
        assert "xlam_fc" in server.config.available_verifiers
        assert "mcqa" in server.config.available_verifiers
        assert "custom" in server.config.available_verifiers


