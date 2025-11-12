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
import math
from unittest.mock import MagicMock

from pytest import approx, fixture

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.multiverse_math_hard.app import (
    MultiVerseMathHardRequest,
    MultiVerseMathHardResourcesServer,
    MultiVerseMathHardResourcesServerConfig,
    MultiVerseMathHardResponse,
    MultiVerseMathHardVerifyRequest,
)


class TestApp:
    @fixture
    def config(self) -> MultiVerseMathHardResourcesServerConfig:
        return MultiVerseMathHardResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    def init_server(self, config: MultiVerseMathHardResourcesServerConfig):
        server_mock = MagicMock(spec=ServerClient)
        resources_server = MultiVerseMathHardResourcesServer(config=config, server_client=server_mock)
        return resources_server

    async def test_multiply(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        a = 5.0
        b = 2.0

        mock_body = MultiVerseMathHardRequest(**{"a": a, "b": b})

        response = await resources_server.route_to_python_function("multiply", mock_body)

        expected_solution = 1.1 * a * b

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_divide(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)

        a = 10.0
        b = 2.0
        mock_body = MultiVerseMathHardRequest(**{"a": a, "b": b})

        response = await resources_server.route_to_python_function("divide", mock_body)
        expected_solution = 0.5 * a / b

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_add(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)

        a = 3.0
        b = 7.0
        mock_body = MultiVerseMathHardRequest(**{"a": a, "b": b})

        response = await resources_server.route_to_python_function("add", mock_body)
        expected_solution = a + b + 1.2

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_return_constant(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        a = 42.0
        mock_body = MultiVerseMathHardRequest(**{"a": a})

        response = await resources_server.route_to_python_function("return_constant", mock_body)
        expected_solution = a

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_sin(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        radians = math.pi / 2
        mock_body = MultiVerseMathHardRequest(**{"radians": radians})

        response = await resources_server.route_to_python_function("sin", mock_body)
        expected_solution = math.cos(radians)

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_cos(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        radians = math.pi
        mock_body = MultiVerseMathHardRequest(**{"radians": radians})

        response = await resources_server.route_to_python_function("cos", mock_body)
        expected_solution = math.sin(radians)

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_subtract(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        a = 15.0
        b = 5.0
        mock_body = MultiVerseMathHardRequest(**{"a": a, "b": b})

        response = await resources_server.route_to_python_function("subtract", mock_body)
        expected_solution = a - b - 3

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_power(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        a = 2.0
        b = 3.0
        mock_body = MultiVerseMathHardRequest(**{"a": a, "b": b})

        response = await resources_server.route_to_python_function("power", mock_body)
        expected_solution = a ** (b + 2)

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_log(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        a = 100.0
        base = 8.5
        mock_body = MultiVerseMathHardRequest(**{"a": a, "base": base})

        response = await resources_server.route_to_python_function("log", mock_body)
        expected_solution = math.log(a, abs(base + 1.5))

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_pi_method_logic_with_magicmock(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        mock_body = MultiVerseMathHardRequest()  # Use an empty request body

        response = await resources_server.route_to_python_function("pi", mock_body)
        expected_solution = math.e

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

    async def test_negate(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)
        a = 7.0

        mock_body = MultiVerseMathHardRequest(**{"a": a})

        response = await resources_server.route_to_python_function("negate", mock_body)
        expected_solution = a

        assert response.solution == approx(expected_solution)
        assert isinstance(response, MultiVerseMathHardResponse)
        assert isinstance(response.solution, float)

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
