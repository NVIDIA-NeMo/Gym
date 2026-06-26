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
from unittest.mock import MagicMock

from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.equivalence_rule.app import (
    EquivalenceRuleResourcesServer,
    EquivalenceRuleResourcesServerConfig,
    EquivalenceRuleVerifyRequest,
    GradingRule,
    _normalize,
)


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="test",
        created_at=0.0,
        model="test_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _make_server(grading_rule: GradingRule = GradingRule.EXACT) -> EquivalenceRuleResourcesServer:
    config = EquivalenceRuleResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        grading_rule=grading_rule,
    )
    return EquivalenceRuleResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestNormalize:
    def test_strips_whitespace(self) -> None:
        assert _normalize("  hello  ") == "hello"

    def test_lowercases(self) -> None:
        assert _normalize("Hello World") == "hello world"

    def test_collapses_internal_whitespace(self) -> None:
        assert _normalize("foo   bar") == "foo bar"

    def test_combined(self) -> None:
        assert _normalize("  Hello   WORLD  ") == "hello world"


class TestGradeExact:
    def test_exact_match(self) -> None:
        assert EquivalenceRuleResourcesServer._grade_exact("Paris", "Paris") == approx(1.0)

    def test_case_insensitive(self) -> None:
        assert EquivalenceRuleResourcesServer._grade_exact("paris", "Paris") == approx(1.0)

    def test_whitespace_normalized(self) -> None:
        assert EquivalenceRuleResourcesServer._grade_exact("  Paris  ", "Paris") == approx(1.0)

    def test_mismatch(self) -> None:
        assert EquivalenceRuleResourcesServer._grade_exact("London", "Paris") == approx(0.0)

    def test_empty_strings(self) -> None:
        assert EquivalenceRuleResourcesServer._grade_exact("", "") == approx(1.0)

    def test_partial_match_is_zero(self) -> None:
        assert EquivalenceRuleResourcesServer._grade_exact("Par", "Paris") == approx(0.0)


class TestGradeSeqMatch:
    def test_identical(self) -> None:
        assert EquivalenceRuleResourcesServer._grade_seq_match("hello", "hello") == approx(1.0)

    def test_completely_different(self) -> None:
        score = EquivalenceRuleResourcesServer._grade_seq_match("abc", "xyz")
        assert 0.0 <= score < 0.5

    def test_partial_overlap(self) -> None:
        score = EquivalenceRuleResourcesServer._grade_seq_match("hello world", "hello")
        assert 0.0 < score < 1.0

    def test_case_insensitive(self) -> None:
        assert EquivalenceRuleResourcesServer._grade_seq_match("Hello", "hello") == approx(1.0)

    def test_returns_float_in_range(self) -> None:
        score = EquivalenceRuleResourcesServer._grade_seq_match("foo bar", "foo baz")
        assert 0.0 <= score <= 1.0


class TestGradeWeightedSeqMatch:
    def test_identical(self) -> None:
        assert EquivalenceRuleResourcesServer._grade_weighted_seq_match("hello", "hello") == approx(1.0)

    def test_returns_float_in_range(self) -> None:
        score = EquivalenceRuleResourcesServer._grade_weighted_seq_match("some text", "other text")
        assert 0.0 <= score <= 1.0

    def test_prefix_bonus_helps(self) -> None:
        # Two responses with same similarity to answer but one shares prefix — weighted should be >= seq_match
        answer = "abcdefghij tail"
        response = "abcdefghij diff"
        weighted = EquivalenceRuleResourcesServer._grade_weighted_seq_match(response, answer)
        seq = EquivalenceRuleResourcesServer._grade_seq_match(response, answer)
        assert weighted >= seq - 1e-9

    def test_completely_different(self) -> None:
        score = EquivalenceRuleResourcesServer._grade_weighted_seq_match("xyz", "abc")
        assert 0.0 <= score < 0.5


class TestGradeDispatcher:
    def test_dispatches_exact(self) -> None:
        assert EquivalenceRuleResourcesServer._grade("Paris", "Paris", GradingRule.EXACT) == approx(1.0)

    def test_dispatches_seq_match(self) -> None:
        score = EquivalenceRuleResourcesServer._grade("hello", "hello", GradingRule.SEQ_MATCH)
        assert score == approx(1.0)

    def test_dispatches_weighted_seq_match(self) -> None:
        score = EquivalenceRuleResourcesServer._grade("hello", "hello", GradingRule.WEIGHTED_SEQ_MATCH)
        assert score == approx(1.0)


class TestVerify:
    async def test_exact_match_reward_1(self) -> None:
        server = _make_server(GradingRule.EXACT)
        request = EquivalenceRuleVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("Paris"),
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.equivalence_score == approx(1.0)

    async def test_exact_mismatch_reward_0(self) -> None:
        server = _make_server(GradingRule.EXACT)
        request = EquivalenceRuleVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("London"),
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.equivalence_score == approx(0.0)

    async def test_exact_case_insensitive(self) -> None:
        server = _make_server(GradingRule.EXACT)
        request = EquivalenceRuleVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("paris"),
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)

    async def test_seq_match_partial_reward(self) -> None:
        server = _make_server(GradingRule.SEQ_MATCH)
        request = EquivalenceRuleVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("hello world"),
            expected_answer="hello",
        )
        result = await server.verify(request)
        assert 0.0 < result.reward < 1.0

    async def test_weighted_seq_match_identical(self) -> None:
        server = _make_server(GradingRule.WEIGHTED_SEQ_MATCH)
        request = EquivalenceRuleVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("the answer"),
            expected_answer="the answer",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)

    async def test_response_fields_present(self) -> None:
        server = _make_server()
        request = EquivalenceRuleVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("test"),
            expected_answer="test",
        )
        result = await server.verify(request)
        dump = result.model_dump()
        assert "reward" in dump
        assert "equivalence_score" in dump

    async def test_empty_response(self) -> None:
        server = _make_server(GradingRule.EXACT)
        empty_response = NeMoGymResponse(
            id="test",
            created_at=0.0,
            model="test_model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        request = EquivalenceRuleVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=empty_response,
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.reward == approx(0.0)


class TestServerInstantiation:
    def test_default_grading_rule(self) -> None:
        server = _make_server()
        assert server.config.grading_rule == GradingRule.EXACT

    def test_custom_grading_rule(self) -> None:
        server = _make_server(GradingRule.SEQ_MATCH)
        assert server.config.grading_rule == GradingRule.SEQ_MATCH
