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
import math
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import fixture

from nemo_gym.mcp_test_utils import assert_transport_parity, mcp_call, mcp_list_tools
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.math_advanced_calculations.app import (
    MultiVerseMathHardResourcesServer,
    MultiVerseMathHardResourcesServerConfig,
    MultiVerseMathHardVerifyRequest,
)


pytest.importorskip("mcp")

EXPECTED_TOOLS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "sin",
    "cos",
    "power",
    "log",
    "pi",
    "negate",
    "return_constant",
}


def _solution_bytes(value: float) -> bytes:
    """The exact wire bytes of the historical response envelope."""
    return json.dumps({"solution": float(value)}, separators=(",", ":")).encode()


class TestApp:
    @fixture
    def config(self) -> MultiVerseMathHardResourcesServerConfig:
        return MultiVerseMathHardResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="math_advanced_calculations",
        )

    def init_server(self, config: MultiVerseMathHardResourcesServerConfig):
        server_mock = MagicMock(spec=ServerClient)
        resources_server = MultiVerseMathHardResourcesServer(config=config, server_client=server_mock)
        return resources_server

    # ----------------------------------------------------------------------------------------
    # HTTP wire-contract replay: byte-equal response bodies for every tool
    # ----------------------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "path, body, expected_solution",
        [
            ("/multiply", {"a": 5.0, "b": 2.0}, 1.1 * 5.0 * 2.0),
            ("/divide", {"a": 10.0, "b": 2.0}, 0.5 * 10.0 / 2.0),
            ("/add", {"a": 3.0, "b": 7.0}, 3.0 + 7.0 + 1.2),
            ("/return_constant", {"a": 42.0}, 42.0),
            ("/sin", {"radians": math.pi / 2}, math.cos(math.pi / 2)),
            ("/cos", {"radians": math.pi}, math.sin(math.pi)),
            ("/subtract", {"a": 15.0, "b": 5.0}, 15.0 - 5.0 - 3),
            ("/power", {"a": 2.0, "b": 3.0}, 2.0 ** (3.0 + 2)),
            ("/log", {"a": 100.0, "base": 8.5}, math.log(100.0, abs(8.5 + 1.5))),
            ("/pi", {}, math.e),
            ("/negate", {"a": 7.0}, 7.0),
        ],
    )
    def test_tool_replay_byte_equal(
        self,
        config: MultiVerseMathHardResourcesServerConfig,
        path: str,
        body: dict,
        expected_solution: float,
    ) -> None:
        resources_server = self.init_server(config)
        with TestClient(resources_server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            response = client.post(path, json=body)
            assert response.status_code == 200
            assert response.content == _solution_bytes(expected_solution)

    def test_null_arguments_are_silently_filtered(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        """The old dispatcher dropped explicit-null args before calling the function; that stays."""
        resources_server = self.init_server(config)
        with TestClient(resources_server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            response = client.post("/add", json={"a": 1.0, "b": 3.0, "radians": None, "base": None})
            assert response.status_code == 200
            assert response.content == b'{"solution":5.2}'

    def test_unknown_tool_preserves_historical_404_bytes(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        resources_server = self.init_server(config)
        with TestClient(resources_server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            response = client.post("/no_such_function", json={})
            assert response.status_code == 404
            assert response.content == b'{"detail":"Function not found"}'

    def test_missing_argument_preserves_historical_500_bytes(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        resources_server = self.init_server(config)
        with TestClient(resources_server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            response = client.post("/add", json={"a": 1.0})
            assert response.status_code == 500
            assert response.content == b'{"detail":"add() missing 1 required positional argument: \'b\'"}'

    def test_math_error_preserves_historical_500_bytes(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        with TestClient(resources_server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            response = client.post("/divide", json={"a": 1.0, "b": 0.0})
            assert response.status_code == 500
            assert response.content == b'{"detail":"float division by zero"}'

    def test_bad_argument_type_is_422(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        with TestClient(resources_server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            response = client.post("/add", json={"a": "x", "b": 1.0})
            assert response.status_code == 422
            (error,) = response.json()["detail"]
            assert error["type"] == "float_parsing"
            assert error["loc"] == ["body", "a"]

    # ----------------------------------------------------------------------------------------
    # MCP round-trip and transport parity
    # ----------------------------------------------------------------------------------------

    def test_mcp_lists_and_calls_the_same_tools(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        with TestClient(resources_server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            tools = mcp_list_tools(client)
            assert set(tools) == EXPECTED_TOOLS
            assert set(tools["add"]["inputSchema"]["properties"]) == {"a", "b"}
            assert "session_id" not in tools["add"]["inputSchema"]["properties"]

            result = mcp_call(client, "add", {"a": 1.0, "b": 3.0})
            assert result.get("isError") is not True
            assert result["structuredContent"] == {"solution": 5.2}

    def test_transport_parity(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        app = resources_server.setup_webserver()
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            assert_transport_parity(app, client, EXPECTED_TOOLS)

    # ----------------------------------------------------------------------------------------
    # Verification
    # ----------------------------------------------------------------------------------------

    async def test_verify(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "user", "content": "add 1 and 3"},
            ],
            tools=[
                {
                    "type": "function",
                    "name": "add",
                    "description": "Add two numbers; a + b.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "First number to add",
                            },
                            "b": {
                                "type": "number",
                                "description": "Second number to add",
                            },
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
        )

        response = NeMoGymResponse(
            **{
                "id": "resp_1",
                "created_at": 1.0,
                "model": "gpt-4.1-2025-04-14",
                "object": "response",
                "output": [
                    {
                        "arguments": '{"a":1,"b":3}',
                        "call_id": "call_1",
                        "name": "add",
                        "type": "function_call",
                        "id": "fc_1",
                        "status": "completed",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": '{"solution": 5.2}',
                    },
                    {
                        "id": "msg_1",
                        "content": [
                            {
                                "annotations": [],
                                "text": "The sum of 1 and 3 is 5.2. \n\nIf you meant simple addition, the sum should normally be 4. Would you like to check a different operation or clarify your request?",
                                "type": "output_text",
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    },
                ],
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [
                    {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {
                                    "type": "number",
                                    "description": "First number to add",
                                },
                                "b": {
                                    "type": "number",
                                    "description": "Second number to add",
                                },
                            },
                            "required": ["a", "b"],
                        },
                        "strict": True,
                        "type": "function",
                        "description": "Add two numbers; a + b.",
                    }
                ],
            }
        )

        verify_request = MultiVerseMathHardVerifyRequest(
            responses_create_params=responses_create_params,
            response=response,
            ground_truth="[5.2]",
            id=1,
            depth=1,
            breadth="1",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 1.0
