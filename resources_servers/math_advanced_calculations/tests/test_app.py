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

from pytest import approx, fixture, raises

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.math_advanced_calculations.app import (
    GroundTruthExpressionError,
    MultiVerseMathHardRequest,
    MultiVerseMathHardResourcesServer,
    MultiVerseMathHardResourcesServerConfig,
    MultiVerseMathHardResponse,
    MultiVerseMathHardVerifyRequest,
    _eval_ground_truth_expressions,
    _safe_eval_expression,
)


def _make_verify_request(predicted_solutions: list, ground_truth_expression: str) -> MultiVerseMathHardVerifyRequest:
    """Build a MultiVerseMathHardVerifyRequest whose response.output has one
    function_call_output per predicted solution, mirroring test_verify's existing shape.
    """
    output = []
    for i, solution in enumerate(predicted_solutions):
        output.append(
            {
                "type": "function_call_output",
                "call_id": f"call_{i}",
                "output": json.dumps({"solution": solution}),
            }
        )

    response = NeMoGymResponse(
        **{
            "id": "resp_1",
            "created_at": 1.0,
            "model": "gpt-4.1-2025-04-14",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "temperature": 1.0,
            "tool_choice": "auto",
            "tools": [],
        }
    )

    return MultiVerseMathHardVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "solve"}],
        ),
        response=response,
        ground_truth_expression=ground_truth_expression,
        id=1,
        depth=1,
        breadth="1",
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
            ground_truth_expression="add(a=1.0, b=3.0)",
            id=1,
            depth=1,
            breadth="1",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 1.0

    def test_safe_eval_expression_matches_true_full_precision(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        """Real example from this session's investigation: the upstream dataset's own
        rounded-chain trace for this expression computes sin(radians=1.14) [using the
        ROUNDED log() output], landing on 0.41759 -- but a model chaining real,
        full-precision tool outputs gets a different true value. This is exactly the value
        the fresh evaluator must reproduce.
        """
        function_map = self.init_server(config)._function_map
        value = _safe_eval_expression("sin(radians=log(a=7.0, base=4.0))", function_map)
        assert value == approx(0.4162629786679435)

    def test_safe_eval_expression_nested_nontrivial(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        function_map = self.init_server(config)._function_map
        # pi() = math.e; power(a, b) = a ** (b + 2)
        value = _safe_eval_expression("power(a=pi(), b=2.0)", function_map)
        assert value == approx(math.e ** (2.0 + 2))

    def test_eval_ground_truth_expressions_splits_on_semicolon(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        function_map = self.init_server(config)._function_map
        values = _eval_ground_truth_expressions("add(a=1.0, b=3.0); multiply(a=5.0, b=2.0)", function_map)
        assert values == approx([1.0 + 3.0 + 1.2, 1.1 * 5.0 * 2.0])

    def test_safe_eval_expression_rejects_disallowed_function(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        function_map = self.init_server(config)._function_map
        with raises(GroundTruthExpressionError):
            _safe_eval_expression("__import__('os').system('ls')", function_map)

    def test_safe_eval_expression_rejects_attribute_access(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        function_map = self.init_server(config)._function_map
        with raises(GroundTruthExpressionError):
            _safe_eval_expression("os.system('ls')", function_map)

    def test_safe_eval_expression_rejects_unknown_name(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        function_map = self.init_server(config)._function_map
        with raises(GroundTruthExpressionError):
            _safe_eval_expression("eval(a=1.0)", function_map)

    def test_safe_eval_expression_rejects_non_numeric_literal(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        function_map = self.init_server(config)._function_map
        with raises(GroundTruthExpressionError):
            _safe_eval_expression("add(a='1.0', b=3.0)", function_map)

    def test_safe_eval_expression_rejects_positional_args(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        function_map = self.init_server(config)._function_map
        with raises(GroundTruthExpressionError):
            _safe_eval_expression("add(1.0, 3.0)", function_map)

    def test_safe_eval_expression_rejects_operators(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        function_map = self.init_server(config)._function_map
        with raises(GroundTruthExpressionError):
            _safe_eval_expression("add(a=1.0, b=3.0) + 1.0", function_map)

    def test_safe_eval_expression_rejects_malformed_syntax(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        function_map = self.init_server(config)._function_map
        with raises(GroundTruthExpressionError):
            _safe_eval_expression("add(a=1.0, b=", function_map)

    async def test_verify_full_precision_compounding_case_now_passes(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        """The exact depth-2 case hand-verified this session: the upstream dataset's own
        rounded-chain ground truth for this expression is 0.41759 (computed via
        sin(radians=1.14), reusing the ROUNDED log() output) -- a value no tolerance could
        safely accept. A model chaining real, full-precision tool outputs correctly
        produces the TRUE value instead, which is what the fresh-expression evaluator now
        checks against, so this must score reward=1.0 (previously impossible to make pass
        safely under the old rounded-ground-truth design).
        """
        resources_server = self.init_server(config)

        log_value = math.log(7.0, abs(4.0 + 1.5))  # true log(a=7.0, base=4.0) = 1.1414649225389244
        true_final = math.cos(log_value)  # sin() is swapped to cos() per the altered tools

        verify_request = _make_verify_request(
            predicted_solutions=[log_value, true_final],
            ground_truth_expression="sin(radians=log(a=7.0, base=4.0))",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 1.0

    async def test_verify_tolerates_float_noise(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        """A predicted value differing only by float64-level noise (not a real answer
        difference) must still pass -- this is exactly what the tight math.isclose
        tolerance exists for.
        """
        resources_server = self.init_server(config)

        true_value = 1.1 * 5.0 * 7.0  # multiply(a=5.0, b=7.0) = 38.5
        noisy_value = true_value + 1e-11  # far below rel_tol=abs_tol=1e-9

        verify_request = _make_verify_request(
            predicted_solutions=[noisy_value],
            ground_truth_expression="multiply(a=5.0, b=7.0)",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 1.0

    async def test_verify_rejects_standard_math_substitution(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        """The most realistic failure mode for this benchmark: a model reverting to
        standard (non-altered) math for one operation. add(a,b) = a+b+1.2 in this
        benchmark; a model using plain a+b instead must still score 0.
        """
        resources_server = self.init_server(config)

        standard_math_answer = 5.0 + 7.0  # missing the +1.2 alteration -- a genuine wrong answer

        verify_request = _make_verify_request(
            predicted_solutions=[standard_math_answer],
            ground_truth_expression="add(a=5.0, b=7.0)",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 0.0

    async def test_verify_rejects_sign_flip(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        resources_server = self.init_server(config)

        true_value = 1.1 * 5.0 * 7.0  # multiply(a=5.0, b=7.0) = 38.5

        verify_request = _make_verify_request(
            predicted_solutions=[-true_value],
            ground_truth_expression="multiply(a=5.0, b=7.0)",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 0.0

    async def test_verify_requires_all_values_to_match(self, config: MultiVerseMathHardResourcesServerConfig) -> None:
        """Multi-answer (";"-joined) rows must still require every sub-expression's value
        to be matched (AND semantics) -- one correct, one wrong, must score 0 overall.
        """
        resources_server = self.init_server(config)

        correct = 1.0 + 3.0 + 1.2  # add(a=1.0, b=3.0)
        wrong = 999.0

        verify_request = _make_verify_request(
            predicted_solutions=[correct, wrong],
            ground_truth_expression="add(a=1.0, b=3.0); multiply(a=5.0, b=2.0)",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 0.0

    async def test_verify_scores_zero_on_malformed_ground_truth_expression(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        """[CUSTOM-MATHCALC-GT-ERRORHANDLING 2026-11-09] A malformed/disallowed
        ground_truth_expression must no longer crash verify() -- a single bad row
        previously took down an entire training run (job 63861875). It must be
        logged and scored reward=0 instead, same as route_to_python_function's own
        client-input-error handling.
        """
        resources_server = self.init_server(config)

        verify_request = _make_verify_request(
            predicted_solutions=[1.0],
            ground_truth_expression="__import__('os').system('ls')",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 0.0

    async def test_verify_scores_zero_when_altered_power_goes_complex(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        """[CUSTOM-MATHCALC-GT-ERRORHANDLING 2026-11-09] Real bug class found in ~0.9% of
        sampled rows: power(a, b) = a ** (b + 2) goes complex for a negative base with a
        non-integer exponent, so float(result) raises TypeError. This must score reward=0,
        not propagate the TypeError out of verify(). subtract(a=1.0, b=1.0) = 1-1-3 = -3.0
        supplies the negative base -- a bare negative literal isn't expressible in the
        restricted AST grammar (a unary minus parses as a disallowed UnaryOp node).
        """
        resources_server = self.init_server(config)

        verify_request = _make_verify_request(
            predicted_solutions=[1.0],
            ground_truth_expression="power(a=subtract(a=1.0, b=1.0), b=0.5)",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 0.0

    async def test_verify_scores_zero_when_altered_log_hits_domain_error(
        self, config: MultiVerseMathHardResourcesServerConfig
    ) -> None:
        """[CUSTOM-MATHCALC-GT-ERRORHANDLING 2026-11-09] Real bug class found in ~0.9% of
        sampled rows: log(a, base) = math.log(a, abs(base + 1.5)) raises ValueError ("math
        domain error") for a <= 0. This must score reward=0, not propagate the ValueError
        out of verify(). subtract(a=1.0, b=1.0) = 1-1-3 = -3.0 supplies the negative base --
        see the matching comment on the power() test above for why a literal can't be used.
        """
        resources_server = self.init_server(config)

        verify_request = _make_verify_request(
            predicted_solutions=[1.0],
            ground_truth_expression="log(a=subtract(a=1.0, b=1.0), base=4.0)",
        )

        response = await resources_server.verify(verify_request)

        assert response.reward == 0.0
