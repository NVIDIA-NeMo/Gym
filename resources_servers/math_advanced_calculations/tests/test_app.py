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
from typing import Any, Optional
from unittest.mock import MagicMock

from pytest import approx, fixture, mark

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from resources_servers.math_advanced_calculations.app import (
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

        # The reply ends by stating the ordinary-math sum, so its last stated answer is 4, not the tool result 5.2.
        assert response.reward == 0.0

    @staticmethod
    def _tool_items(values: list[float], first_call: int = 0) -> list[dict[str, Any]]:
        output = []
        for i, value in enumerate(values, start=first_call):
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{i}",
                    "call_id": f"call_{i}",
                    "name": "return_constant",
                    "arguments": json.dumps({"a": value}),
                    "status": "completed",
                }
            )
            output.append(
                {"type": "function_call_output", "call_id": f"call_{i}", "output": json.dumps({"solution": value})}
            )
        return output

    @staticmethod
    def _assistant_message(text: str, **fields: Any) -> dict[str, Any]:
        return {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
            **fields,
        }

    def _verify_request(
        self, ground_truth: list[float], tool_outputs: list[float], final_text: Optional[str]
    ) -> MultiVerseMathHardVerifyRequest:
        output = self._tool_items(tool_outputs)
        if final_text is not None:
            output.append(self._assistant_message(final_text))
        return self._request(ground_truth, output)

    @staticmethod
    def _request(ground_truth: list[float], output: list[dict[str, Any]]) -> MultiVerseMathHardVerifyRequest:
        response = NeMoGymResponse(
            id="resp_1",
            created_at=1.0,
            model="test-model",
            object="response",
            output=output,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        return MultiVerseMathHardVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Get me the values using only the given tools."}]
            ),
            response=response,
            ground_truth=json.dumps(ground_truth),
            id=0,
            depth=1,
            breadth=len(ground_truth),
        )

    @mark.parametrize(
        "ground_truth, final_text",
        [
            ([17.2], "The value for (6.0 + 10.0) using the given tools is 17.2."),
            ([0.8414709848078965], "The value for cos(6.0 - 2.0) is approximately 0.8415."),
            ([1073741824.0], "The value of 8.0 raised to the power of 8.0 (8.0 ** 8.0) is 1,073,741,824.0."),
            ([0.8507977712171255], "The value for \\(\\sin(\\log_{2}(2.0))\\) is approximately \\(0.8508\\)."),
            ([-0.6536436208636119], "sin(4.0) = \u22120.6536"),
            ([12054.673528305822], "(4.0 ** 6.0 / pi) = 1.205467e+04"),
            (
                [-3.0, 0.5, 9.9, 0.28366218546322625],
                "(9.0 - 9.0) = -3, (1.0 / 1.0) = 0.5, (1.0 * 9.0) = 9.9, and sin(5.0) = 0.284",
            ),
            ([2.718281828459045, 7.2], "- **pi**: 2.71828\n- **(4.0 + 2.0)**: 7.2"),
            ([0.5, 3.0, 4096.0], "1. (1.0 / 1.0) = 0.5\n2. 3.0 = 3.0\n3. 2.0 ** 10.0 = 4096.0 (from the power tool)"),
            (
                [0.28366218546322625],
                "- The sine of 5.0 radians is approximately 0.2837 (rounded to five decimal places).",
            ),
            ([0.5, 9.9], "(1.0 / 1.0) is 0.5 and (1.0 * 9.0) is 9.9"),
            ([-3.0, 0.5, 9.9, 0.28366218546322625], "[-3,0.5,9.9,0.2837]"),
            ([-3.0, 0.5, 9.9, 0.28366218546322625], "[-3, 0.5, 9.9, 0.2837]"),
            ([1234.0], "The result is 1,234."),
            ([3.0], "3."),
            ([3.0], "3. "),
            ([3.0], "3. \n"),
            ([3.0, 0.5], "3. \n0.5"),
            ([2.0, 4096.0], "[2,4096]"),
            ([5.2], "The tool result is (5.2)."),
            ([-3.0, 0.5, 9.9, 0.28366218546322625], "The answers are (-3, 0.5, 9.9, 0.2837)"),
            ([4096.0, 7.2], "power(2, 10) = 4096, add(4, 2) = 7.2"),
        ],
        ids=[
            "plain",
            "rounded",
            "thousands_separators",
            "latex",
            "unicode_minus",
            "scientific",
            "several_values",
            "markdown_list",
            "numbered_list_with_constant_and_operator",
            "operand_in_prose",
            "joined_by_and",
            "compact_list",
            "spaced_list",
            "grouped_thousands_without_decimals",
            "integer_with_period",
            "integer_with_period_and_trailing_space",
            "integer_with_period_and_trailing_newline",
            "integer_with_trailing_space_then_next_line",
            "compact_list_with_a_long_value",
            "value_in_parentheses",
            "values_in_parentheses",
            "function_arguments_are_not_answers",
        ],
    )
    async def test_verify_accepts_reported_tool_values(
        self, config: MultiVerseMathHardResourcesServerConfig, ground_truth: list[float], final_text: str
    ) -> None:
        resources_server = self.init_server(config)

        response = await resources_server.verify(self._verify_request(ground_truth, ground_truth, final_text))

        assert response.reward == 1.0

    @mark.parametrize(
        "final_text",
        [
            # Ordinary-math values instead of the tool results.
            "The tools look broken, so here are the true values: (9.0 - 9.0) = 0, (1.0 / 1.0) = 1, "
            "(1.0 * 9.0) = 9, sin(5.0) = -0.958924",
            # One of the four values is wrong.
            "(9.0 - 9.0) = -3, (1.0 / 1.0) = 0.5, (1.0 * 9.0) = 9.9, sin(5.0) = -0.9589",
            # No final answer at all.
            None,
            # The tool results are quoted, then replaced by ordinary-math values.
            "Tools gave -3, 0.5, 9.9, 0.2837; ignore them. My answers are 0, 1, 9, -0.9589.",
        ],
        ids=["ordinary_math_values", "one_wrong_value", "no_final_answer", "quoted_then_replaced"],
    )
    async def test_verify_rejects_unreported_tool_values(
        self, config: MultiVerseMathHardResourcesServerConfig, final_text: Optional[str]
    ) -> None:
        resources_server = self.init_server(config)
        ground_truth = [-3.0, 0.5, 9.9, 0.28366218546322625]

        response = await resources_server.verify(self._verify_request(ground_truth, ground_truth, final_text))

        assert response.reward == 0.0

    async def test_verify_rejects_final_answer_without_tool_outputs(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        resources_server = self.init_server(config)

        response = await resources_server.verify(self._verify_request([17.2], [], "(6.0 + 10.0) = 17.2"))

        assert response.reward == 0.0

    @mark.parametrize(
        "form",
        ["string", "input_text_list", "output_text_list", "output_text_and_refusal", "output_text_for_training"],
    )
    @mark.parametrize("text, reward", [("17.2", 1.0), ("0", 0.0)], ids=["tool_result", "other_value"])
    async def test_verify_reads_every_assistant_content_form(
        self, config: MultiVerseMathHardResourcesServerConfig, form: str, text: str, reward: float
    ) -> None:
        resources_server = self.init_server(config)
        if form == "string":
            message = {"type": "message", "role": "assistant", "content": text}
        elif form == "input_text_list":
            message = {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": text}]}
        else:
            message = self._assistant_message(text)
            if form == "output_text_and_refusal":
                message["content"].append({"type": "refusal", "refusal": "No more than that."})
            elif form == "output_text_for_training":
                message.update(prompt_token_ids=[1, 2], generation_token_ids=[3], generation_log_probs=[-0.1])

        response = await resources_server.verify(self._request([17.2], [*self._tool_items([17.2]), message]))

        assert response.reward == reward

    async def test_verify_rejects_string_answer_without_tool_outputs(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        resources_server = self.init_server(config)
        message = {"type": "message", "role": "assistant", "content": "17.2"}

        response = await resources_server.verify(self._request([17.2], [message]))

        assert response.reward == 0.0

    async def test_verify_rejects_refusal_only_final_message(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        resources_server = self.init_server(config)
        message = {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "I can't help with that."}],
        }

        response = await resources_server.verify(self._request([17.2], [*self._tool_items([17.2]), message]))

        assert response.reward == 0.0

    @mark.parametrize(
        "ending, reward",
        [
            # More tool work after the commentary, then the trajectory ends (e.g. at max_steps).
            ("nothing", 0.0),
            ("reasoning", 0.0),
            # A final answer after the tool work is read, not the earlier commentary.
            ("17.2", 1.0),
            ("0", 0.0),
        ],
        ids=["ends_after_tool_output", "ends_after_reasoning", "final_answer_after_tools", "wrong_final_answer"],
    )
    async def test_verify_ignores_commentary_before_later_tool_calls(
        self, config: MultiVerseMathHardResourcesServerConfig, ending: str, reward: float
    ) -> None:
        resources_server = self.init_server(config)
        output = [
            *self._tool_items([17.2]),
            self._assistant_message("Current result: 17.2"),
            *self._tool_items([5.0], first_call=1),
        ]
        if ending == "reasoning":
            output.append({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Hmm."}]})
        elif ending != "nothing":
            output.append(self._assistant_message(ending))

        response = await resources_server.verify(self._request([17.2], output))

        assert response.reward == reward

    async def test_verify_ignores_commentary_before_other_tool_work(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        resources_server = self.init_server(config)
        mcp_call = {"type": "mcp_call", "id": "mcp_1", "server_label": "math", "name": "add", "arguments": "{}"}
        output = [*self._tool_items([17.2]), self._assistant_message("Current result: 17.2"), mcp_call]

        response = await resources_server.verify(self._request([17.2], output))

        assert response.reward == 0.0

    @mark.parametrize("phase, reward", [("commentary", 0.0), ("final_answer", 1.0)])
    @mark.parametrize("form", ["output_message", "string_content"])
    async def test_verify_reads_the_message_phase(
        self, config: MultiVerseMathHardResourcesServerConfig, form: str, phase: str, reward: float
    ) -> None:
        resources_server = self.init_server(config)
        if form == "output_message":
            message = self._assistant_message("17.2", phase=phase)
        else:
            message = {"type": "message", "role": "assistant", "content": "17.2", "phase": phase}

        response = await resources_server.verify(self._request([17.2], [*self._tool_items([17.2]), message]))

        assert response.reward == reward
