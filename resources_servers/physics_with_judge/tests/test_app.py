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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
from pytest import approx, fixture, raises

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.physics_with_judge.app import (
    PhysicsJudgeResourcesServer,
    PhysicsJudgeResourcesServerConfig,
    PhysicsJudgeVerifyRequest,
)


class TestApp:
    # ------------------------------------------------------------------
    # Fixtures / helpers
    # ------------------------------------------------------------------

    @fixture
    def config(self) -> PhysicsJudgeResourcesServerConfig:
        return PhysicsJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="physics_judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    def _make_server(self, config: PhysicsJudgeResourcesServerConfig) -> PhysicsJudgeResourcesServer:
        return PhysicsJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _make_response(self, id: str, output_item: NeMoGymResponseOutputItem) -> dict[str, Any]:
        return NeMoGymResponse(
            id=id,
            created_at=1234.5,
            model="test_model",
            object="response",
            output=[output_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()

    def _make_text_message(self, text: str) -> NeMoGymResponseOutputMessage:
        return NeMoGymResponseOutputMessage(
            id=f"msg_{text[:16]}",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )

    def _setup_judge_mock(self, server: PhysicsJudgeResourcesServer, *response_jsons: str) -> AsyncMock:
        """Wire server_client.post to return *response_jsons* in order."""
        response_mock = AsyncMock()
        response_mock.side_effect = list(response_jsons)
        post_mock = MagicMock()
        post_mock.read = response_mock
        server.server_client.post = AsyncMock(return_value=post_mock)
        return response_mock

    # ------------------------------------------------------------------
    # _extract_boxed_content
    # ------------------------------------------------------------------

    def test_extract_boxed_simple(self) -> None:
        assert PhysicsJudgeResourcesServer._extract_boxed_content(r"The answer is \boxed{1.3 kW}") == "1.3 kW"

    def test_extract_boxed_nested_braces(self) -> None:
        assert (
            PhysicsJudgeResourcesServer._extract_boxed_content(r"\boxed{1.3 \times 10^{5} \text{ km/s}}")
            == r"1.3 \times 10^{5} \text{ km/s}"
        )

    def test_extract_boxed_last_occurrence(self) -> None:
        # Should return the LAST \boxed{} when there are multiple
        text = r"First \boxed{wrong} then \boxed{correct}"
        assert PhysicsJudgeResourcesServer._extract_boxed_content(text) == "correct"

    def test_extract_boxed_none_when_absent(self) -> None:
        assert PhysicsJudgeResourcesServer._extract_boxed_content("no boxed here") is None

    def test_extract_boxed_unmatched_brace(self) -> None:
        assert PhysicsJudgeResourcesServer._extract_boxed_content(r"\boxed{unclosed") is None

    # ------------------------------------------------------------------
    # _preprocess_latex
    # ------------------------------------------------------------------

    def test_preprocess_plain(self) -> None:
        assert PhysicsJudgeResourcesServer._preprocess_latex("1.3 kW") == "1.3 kW"

    def test_preprocess_text_unit(self) -> None:
        assert PhysicsJudgeResourcesServer._preprocess_latex(r"1.3 \text{ kW}") == "1.3 kW"

    def test_preprocess_mathrm_unit(self) -> None:
        assert PhysicsJudgeResourcesServer._preprocess_latex(r"9.81 \mathrm{m/s^2}") == "9.81 m/s**2"

    def test_preprocess_scientific_notation_braced(self) -> None:
        result = PhysicsJudgeResourcesServer._preprocess_latex(r"1.3 \times 10^{5}")
        assert result == "1.3e5"

    def test_preprocess_scientific_notation_bare(self) -> None:
        result = PhysicsJudgeResourcesServer._preprocess_latex(r"3.0 \times 10^8 \text{ m/s}")
        assert result == "3.0e8 m/s"

    def test_preprocess_exponent_braced(self) -> None:
        result = PhysicsJudgeResourcesServer._preprocess_latex(r"9.81 \text{ m/s}^{2}")
        assert result == "9.81 m/s**2"

    def test_preprocess_omega(self) -> None:
        result = PhysicsJudgeResourcesServer._preprocess_latex(r"100 \Omega")
        assert result == "100 ohm"

    def test_preprocess_simple_frac(self) -> None:
        result = PhysicsJudgeResourcesServer._preprocess_latex(r"\frac{1}{2} \text{ J}")
        assert result == "0.5 J"

    def test_preprocess_celsius(self) -> None:
        result = PhysicsJudgeResourcesServer._preprocess_latex("100 °C")
        assert result == "100 degC"

    # ------------------------------------------------------------------
    # _verify_answer_with_library
    # ------------------------------------------------------------------

    def test_library_unit_conversion_watts_kilowatts(self, config) -> None:
        server = self._make_server(config)
        reward, extracted = server._verify_answer_with_library(
            r"1.3 \text{ kW}",
            r"The resistor dissipates \boxed{1300 \text{ W}}.",
        )
        assert reward == approx(1.0)
        assert extracted == r"1300 \text{ W}"

    def test_library_unit_conversion_reverse(self, config) -> None:
        server = self._make_server(config)
        reward, _ = server._verify_answer_with_library(
            "1300 W",
            r"Answer: \boxed{1.3 \text{ kW}}",
        )
        assert reward == approx(1.0)

    def test_library_scientific_notation(self, config) -> None:
        server = self._make_server(config)
        reward, _ = server._verify_answer_with_library(
            r"1.5 \times 10^{5} \text{ km/s}",
            r"\boxed{1.5 \times 10^{5} \text{ km/s}}",
        )
        assert reward == approx(1.0)

    def test_library_scientific_notation_equivalent(self, config) -> None:
        # 1.5e5 km/s and 150000 km/s are the same
        server = self._make_server(config)
        reward, _ = server._verify_answer_with_library(
            r"1.5 \times 10^{5} \text{ km/s}",
            r"\boxed{150000 \text{ km/s}}",
        )
        assert reward == approx(1.0)

    def test_library_speed_conversion(self, config) -> None:
        server = self._make_server(config)
        reward, _ = server._verify_answer_with_library(
            "108 km/h",
            r"\boxed{30 \text{ m/s}}",
        )
        assert reward == approx(1.0)

    def test_library_dimensionless(self, config) -> None:
        server = self._make_server(config)
        reward, _ = server._verify_answer_with_library("0.5", r"\boxed{0.5}")
        assert reward == approx(1.0)

    def test_library_wrong_value(self, config) -> None:
        server = self._make_server(config)
        reward, _ = server._verify_answer_with_library("100 J", r"\boxed{50 \text{ J}}")
        assert reward == approx(0.0)

    def test_library_wrong_units_dimensionality(self, config) -> None:
        # Watts vs Joules are dimensionally incompatible
        server = self._make_server(config)
        reward, _ = server._verify_answer_with_library("100 W", r"\boxed{100 \text{ J}}")
        assert reward == approx(0.0)

    def test_library_no_boxed(self, config) -> None:
        server = self._make_server(config)
        reward, extracted = server._verify_answer_with_library("100 J", "The kinetic energy is 100 J")
        assert reward == approx(0.0)
        assert extracted is None

    def test_library_unparseable_expected(self, config) -> None:
        server = self._make_server(config)
        # Symbolic expressions that pint cannot parse → fall back gracefully
        reward, extracted = server._verify_answer_with_library(
            r"\frac{mv^2}{2}",
            r"\boxed{0.5 \text{ J}}",
        )
        assert reward == approx(0.0)
        assert extracted is not None  # extraction still works

    def test_library_rtol_within(self, config) -> None:
        # |100.05 - 100.0| / 100.0 = 0.0005 <= rtol=0.001 → accepted
        server = self._make_server(config)
        reward, _ = server._verify_answer_with_library("100.0 J", r"\boxed{100.05 \text{ J}}")
        assert reward == approx(1.0)

    def test_library_rtol_exceeds(self, config) -> None:
        # Value outside rtol should be rejected
        server = self._make_server(config)
        reward, _ = server._verify_answer_with_library("100.0 J", r"\boxed{100.5 \text{ J}}")
        assert reward == approx(0.0)  # |100.5-100|/100 = 0.005 > 0.001

    # ------------------------------------------------------------------
    # _verify_answer (two-stage pipeline)
    # ------------------------------------------------------------------

    async def test_verify_answer_library_match_skips_judge(self, config) -> None:
        """When the library is confident, judge must NOT be called."""
        server = self._make_server(config)
        server.server_client.post = AsyncMock()  # should never be called

        reward, extracted, lib_reward, judge_evals = await server._verify_answer(
            "A resistor dissipates 1300 W. Express in kW.",
            "1.3 kW",
            r"The answer is \boxed{1300 \text{ W}}.",
        )
        assert reward == approx(1.0)
        assert lib_reward == approx(1.0)
        assert judge_evals is None
        server.server_client.post.assert_not_called()

    async def test_verify_answer_judge_called_on_library_miss(self, config) -> None:
        """When the library fails, the judge is invoked (should_use_judge=True)."""
        config = config.model_copy(deep=True)
        config.should_use_judge = True

        server = self._make_server(config)
        not_equal_item = self._make_text_message(f"{PhysicsJudgeResourcesServer.JUDGE_NOT_EQUAL_LABEL} verdict")
        self._setup_judge_mock(server, json.dumps(self._make_response("id1", not_equal_item)))

        reward, _, lib_reward, judge_evals = await server._verify_answer(
            "question", "expected", r"\boxed{wrong_answer}"
        )
        assert reward == approx(0.0)
        assert lib_reward == approx(0.0)
        assert judge_evals is not None
        server.server_client.post.assert_called_once()

    async def test_verify_answer_judge_disabled(self, config) -> None:
        """With should_use_judge=False, judge is never called even on library miss."""
        config = config.model_copy(deep=True)
        config.should_use_judge = False

        server = self._make_server(config)
        server.server_client.post = AsyncMock()

        reward, _, lib_reward, judge_evals = await server._verify_answer("question", "expected", "no boxed")
        assert reward == approx(0.0)
        assert lib_reward == approx(0.0)
        assert judge_evals is None
        server.server_client.post.assert_not_called()

    # ------------------------------------------------------------------
    # _verify_answer_with_judge (bidirectional judge logic)
    # ------------------------------------------------------------------

    async def test_judge_first_order_not_equal(self, config) -> None:
        server = self._make_server(config)
        not_equal_item = self._make_text_message(f"verdict: {PhysicsJudgeResourcesServer.JUDGE_NOT_EQUAL_LABEL}")
        self._setup_judge_mock(server, json.dumps(self._make_response("id1", not_equal_item)))

        reward, evals = await server._verify_answer_with_judge("q", "expected", "generated")
        assert reward == approx(0.0)
        assert len(evals) == 1
        server.server_client.post.assert_called_once()

    async def test_judge_both_orders_equal(self, config) -> None:
        server = self._make_server(config)
        equal_item_1 = self._make_text_message(f"{PhysicsJudgeResourcesServer.JUDGE_EQUAL_LABEL}")
        equal_item_2 = self._make_text_message(f"I conclude {PhysicsJudgeResourcesServer.JUDGE_EQUAL_LABEL}")
        response_mock = AsyncMock()
        response_mock.side_effect = [
            json.dumps(self._make_response("id1", equal_item_1)),
            json.dumps(self._make_response("id2", equal_item_2)),
        ]
        post_mock = MagicMock()
        post_mock.read = response_mock
        server.server_client.post = AsyncMock(return_value=post_mock)

        reward, evals = await server._verify_answer_with_judge("q", "expected", "generated")
        assert reward == approx(1.0)
        assert len(evals) == 2

    async def test_judge_second_order_not_equal(self, config) -> None:
        server = self._make_server(config)
        equal_item = self._make_text_message(f"{PhysicsJudgeResourcesServer.JUDGE_EQUAL_LABEL}")
        not_equal_item = self._make_text_message(f"{PhysicsJudgeResourcesServer.JUDGE_NOT_EQUAL_LABEL}")
        response_mock = AsyncMock()
        response_mock.side_effect = [
            json.dumps(self._make_response("id1", equal_item)),
            json.dumps(self._make_response("id2", not_equal_item)),
        ]
        post_mock = MagicMock()
        post_mock.read = response_mock
        server.server_client.post = AsyncMock(return_value=post_mock)

        reward, evals = await server._verify_answer_with_judge("q", "expected", "generated")
        assert reward == approx(0.0)
        assert len(evals) == 2

    # ------------------------------------------------------------------
    # _generate_judge_evaluation (label parsing)
    # ------------------------------------------------------------------

    async def test_generate_judge_evaluation_invalid_response(self, config) -> None:
        server = self._make_server(config)
        response_mock = AsyncMock(return_value=json.dumps({}))
        post_mock = MagicMock()
        post_mock.read = response_mock
        server.server_client.post = AsyncMock(return_value=post_mock)

        with raises(ValidationError):
            await server._generate_judge_evaluation("q", "a", "b")

    async def test_generate_judge_evaluation_reasoning_item(self, config) -> None:
        server = self._make_server(config)
        reasoning_item = NeMoGymResponseReasoningItem(id="r1", summary=[], type="reasoning")
        response_mock = AsyncMock(return_value=json.dumps(self._make_response("id1", reasoning_item)))
        post_mock = MagicMock()
        post_mock.read = response_mock
        server.server_client.post = AsyncMock(return_value=post_mock)

        equal, _ = await server._generate_judge_evaluation("q", "a", "b")
        assert equal is False

    async def test_generate_judge_evaluation_refusal(self, config) -> None:
        server = self._make_server(config)
        refusal_item = NeMoGymResponseOutputMessage(
            id="ref1",
            content=[NeMoGymResponseOutputRefusal(refusal="I refuse", type="refusal")],
            role="assistant",
            status="completed",
            type="message",
        )
        response_mock = AsyncMock(return_value=json.dumps(self._make_response("id1", refusal_item)))
        post_mock = MagicMock()
        post_mock.read = response_mock
        server.server_client.post = AsyncMock(return_value=post_mock)

        equal, _ = await server._generate_judge_evaluation("q", "a", "b")
        assert equal is False

    async def test_generate_judge_evaluation_equal_label_first(self, config) -> None:
        server = self._make_server(config)
        text = (
            f"{PhysicsJudgeResourcesServer.JUDGE_EQUAL_LABEL} not {PhysicsJudgeResourcesServer.JUDGE_NOT_EQUAL_LABEL}"
        )
        item = self._make_text_message(text)
        response_mock = AsyncMock(return_value=json.dumps(self._make_response("id1", item)))
        post_mock = MagicMock()
        post_mock.read = response_mock
        server.server_client.post = AsyncMock(return_value=post_mock)

        equal, _ = await server._generate_judge_evaluation("q", "a", "b")
        assert equal is True

    async def test_generate_judge_evaluation_not_equal_label_first(self, config) -> None:
        server = self._make_server(config)
        text = f"{PhysicsJudgeResourcesServer.JUDGE_NOT_EQUAL_LABEL} {PhysicsJudgeResourcesServer.JUDGE_EQUAL_LABEL}"
        item = self._make_text_message(text)
        response_mock = AsyncMock(return_value=json.dumps(self._make_response("id1", item)))
        post_mock = MagicMock()
        post_mock.read = response_mock
        server.server_client.post = AsyncMock(return_value=post_mock)

        equal, _ = await server._generate_judge_evaluation("q", "a", "b")
        assert equal is False

    # ------------------------------------------------------------------
    # Full verify() endpoint
    # ------------------------------------------------------------------

    async def test_verify_correct_unit_conversion(self, config) -> None:
        server = self._make_server(config)
        response = NeMoGymResponse(
            id="resp1",
            created_at=0.0,
            model="test",
            object="response",
            output=[self._make_text_message(r"The power is \boxed{1300 \text{ W}}.")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        body = PhysicsJudgeVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Express 1.3 kW in W."}]},
            response=response,
            question="Express 1.3 kW in W.",
            expected_answer=r"1.3 \text{ kW}",
        )
        result = await server.verify(body)
        assert result.reward == approx(1.0)
        assert result.library_reward == approx(1.0)
        assert result.judge_evaluations is None

    async def test_verify_wrong_answer(self, config) -> None:
        config = config.model_copy(deep=True)
        config.should_use_judge = False
        server = self._make_server(config)
        response = NeMoGymResponse(
            id="resp1",
            created_at=0.0,
            model="test",
            object="response",
            output=[self._make_text_message(r"The answer is \boxed{50 \text{ J}}.")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        body = PhysicsJudgeVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "What is KE of 2 kg at 10 m/s?"}]},
            response=response,
            question="What is KE of 2 kg at 10 m/s?",
            expected_answer="100 J",
        )
        result = await server.verify(body)
        assert result.reward == approx(0.0)
        assert result.library_reward == approx(0.0)

    async def test_verify_response_fields_preserved(self, config) -> None:
        """All request fields must appear in the response."""
        server = self._make_server(config)
        response = NeMoGymResponse(
            id="resp1",
            created_at=0.0,
            model="test",
            object="response",
            output=[self._make_text_message(r"\boxed{49 \text{ N}}")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        body = PhysicsJudgeVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Weight of 5 kg?"}]},
            response=response,
            question="Weight of 5 kg?",
            expected_answer="49 N",
        )
        result = await server.verify(body)
        assert sorted(result.model_dump().keys()) == [
            "expected_answer",
            "extracted_answer",
            "judge_evaluations",
            "library_reward",
            "response",
            "responses_create_params",
            "reward",
        ]

    async def test_verify_multi_output_items(self, config) -> None:
        """Verify concatenates text across multiple output messages."""
        server = self._make_server(config)
        response = NeMoGymResponse(
            id="resp1",
            created_at=0.0,
            model="test",
            object="response",
            output=[
                NeMoGymResponseReasoningItem(id="r1", summary=[], type="reasoning"),
                self._make_text_message("Intermediate step..."),
                self._make_text_message(r"Final: \boxed{108 \text{ km/h}}"),
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        body = PhysicsJudgeVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Convert 30 m/s to km/h."}]},
            response=response,
            question="Convert 30 m/s to km/h.",
            expected_answer="108 km/h",
        )
        result = await server.verify(body)
        assert result.reward == approx(1.0)
