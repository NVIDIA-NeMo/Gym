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
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.lc_niah.app import (
    LCNIAHResourcesServer,
    LCNIAHResourcesServerConfig,
    LCNIAHVerifyRequest,
    OverlapGradingRule,
    OverlapMetricRule,
    _extract_answer_text,
    _extract_input_text,
    _extract_reasoning_text,
    _normalize,
)


def _make_response(answer: Optional[str] = None, reasoning: Optional[str] = None) -> NeMoGymResponse:
    output: List = []
    if reasoning is not None:
        output.append(
            NeMoGymResponseReasoningItem(
                id="rs",
                summary=[NeMoGymSummary(text=reasoning, type="summary_text")],
                type="reasoning",
            )
        )
    if answer is not None:
        output.append(
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=answer, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        )
    return NeMoGymResponse(
        id="test",
        created_at=0.0,
        model="test_model",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _make_request(
    *,
    expected_answer: str,
    answer: Optional[str] = None,
    reasoning: Optional[str] = None,
    input_text: str = "",
) -> LCNIAHVerifyRequest:
    return LCNIAHVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"content": input_text, "role": "user", "type": "message"}]
        ),
        response=_make_response(answer=answer, reasoning=reasoning),
        expected_answer=expected_answer,
    )


def _make_server(
    overlap_metric_rule: OverlapMetricRule = OverlapMetricRule.LCS,
    overlap_grading_rule: OverlapGradingRule = OverlapGradingRule.MULTIPLY,
) -> LCNIAHResourcesServer:
    config = LCNIAHResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        overlap_metric_rule=overlap_metric_rule,
        overlap_grading_rule=overlap_grading_rule,
    )
    return LCNIAHResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestNormalize:
    def test_strips_and_lowercases(self) -> None:
        assert _normalize("  Hello   WORLD  ") == "hello world"


class TestExtractors:
    def test_extract_answer_text(self) -> None:
        assert _extract_answer_text(_make_response(answer="Paris")) == "Paris"

    def test_extract_answer_text_empty_when_no_message(self) -> None:
        assert _extract_answer_text(_make_response(reasoning="thinking")) == ""

    def test_extract_reasoning_text(self) -> None:
        assert _extract_reasoning_text(_make_response(reasoning="step one")) == "step one"

    def test_extract_reasoning_text_empty_when_none(self) -> None:
        assert _extract_reasoning_text(_make_response(answer="Paris")) == ""

    def test_extract_input_text_from_string(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(input="hello there")
        assert _extract_input_text(params) == "hello there"

    def test_extract_input_text_from_message_list(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"content": "a question", "role": "user", "type": "message"}]
        )
        assert _extract_input_text(params) == "a question"

    def test_extract_input_text_from_content_blocks(self) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "content": [{"text": "block text", "type": "input_text"}],
                    "role": "user",
                    "type": "message",
                }
            ]
        )
        assert _extract_input_text(params) == "block text"


class TestGradeAnswer:
    """The answer grader parses a 'Final Answer: [..]' node list and scores F1 vs a JSON list."""

    def test_perfect_f1(self) -> None:
        score = LCNIAHResourcesServer._grade_answer("Final Answer: [a, b]", '["a", "b"]')
        assert score == approx(1.0)

    def test_no_overlap_f1_zero(self) -> None:
        score = LCNIAHResourcesServer._grade_answer("Final Answer: [c]", '["a", "b"]')
        assert score == approx(0.0)

    def test_partial_f1(self) -> None:
        # predicted {a, c}, expected {a, b}: precision 0.5, recall 0.5 -> f1 0.5
        score = LCNIAHResourcesServer._grade_answer("Final Answer: [a, c]", '["a", "b"]')
        assert score == approx(0.5)

    def test_parse_failure_is_zero(self) -> None:
        assert LCNIAHResourcesServer._grade_answer("no final answer here", '["a"]') == approx(0.0)

    def test_both_empty_is_one(self) -> None:
        assert LCNIAHResourcesServer._grade_answer("Final Answer: []", "[]") == approx(1.0)

    def test_non_json_expected_is_zero(self) -> None:
        # Unparseable expected_answer -> empty expected set; a non-empty prediction scores 0.
        assert LCNIAHResourcesServer._grade_answer("Final Answer: [a]", "not-json") == approx(0.0)

    def test_empty_response_is_zero(self) -> None:
        # No lines at all -> parse failure -> 0.
        assert LCNIAHResourcesServer._grade_answer("", '["a"]') == approx(0.0)


class TestOverlapSeqMatch:
    def test_no_reasoning_is_zero(self) -> None:
        assert LCNIAHResourcesServer._overlap_seq_match("", "some long input") == approx(0.0)

    def test_identical_is_one(self) -> None:
        assert LCNIAHResourcesServer._overlap_seq_match("copy me", "copy me") == approx(1.0)

    def test_unrelated_is_low(self) -> None:
        assert LCNIAHResourcesServer._overlap_seq_match("xyz", "completely different prompt") < 0.5

    def test_returns_float_in_range(self) -> None:
        score = LCNIAHResourcesServer._overlap_seq_match("partial overlap text", "some overlap here")
        assert 0.0 <= score <= 1.0


class TestOverlapNgram:
    def test_short_reasoning_is_zero(self) -> None:
        # Fewer than n words => no n-grams exist.
        assert LCNIAHResourcesServer._overlap_ngram("only a few words here", "only a few words here", n=16) == approx(
            0.0
        )

    def test_fully_copied_is_one(self) -> None:
        text = " ".join(f"word{i}" for i in range(20))  # 20 words -> 16-grams exist
        assert LCNIAHResourcesServer._overlap_ngram(text, text, n=16) == approx(1.0)

    def test_disjoint_is_zero(self) -> None:
        reasoning = " ".join(f"r{i}" for i in range(20))
        input_text = " ".join(f"i{i}" for i in range(20))
        assert LCNIAHResourcesServer._overlap_ngram(reasoning, input_text, n=16) == approx(0.0)

    def test_partial_overlap_in_range(self) -> None:
        shared = " ".join(f"s{i}" for i in range(20))
        reasoning = shared + " " + " ".join(f"extra{i}" for i in range(20))
        score = LCNIAHResourcesServer._overlap_ngram(reasoning, shared, n=16)
        assert 0.0 < score < 1.0


class TestOverlapLcs:
    def test_no_reasoning_is_zero(self) -> None:
        assert LCNIAHResourcesServer._overlap_lcs("", "anything") == approx(0.0)

    def test_full_copy_is_one(self) -> None:
        # Entire reasoning appears verbatim inside the input.
        assert LCNIAHResourcesServer._overlap_lcs("abcdefgh", "xx abcdefgh yy") == approx(1.0)

    def test_half_copied(self) -> None:
        # Longest shared block is 'abcd' (4 chars) out of 8 -> 0.5
        assert LCNIAHResourcesServer._overlap_lcs("abcdwxyz", "qq abcd pp") == approx(0.5)

    def test_returns_float_in_range(self) -> None:
        score = LCNIAHResourcesServer._overlap_lcs("some reasoning", "some other input text")
        assert 0.0 <= score <= 1.0


class TestVerify:
    async def test_base_rule_ignores_overlap(self) -> None:
        # base grading: reward == answer_score, overlap reported but ignored.
        server = _make_server(overlap_grading_rule=OverlapGradingRule.BASE)
        prompt = "The graph has edges from node a to node b and several other distinct nodes scattered around."
        request = _make_request(
            expected_answer='["a", "b"]',
            answer="Final Answer: [a, b]",
            reasoning=prompt,  # reasoning copies the prompt verbatim
            input_text=prompt,
        )
        result = await server.verify(request)
        assert result.answer_score == approx(1.0)
        assert result.reasoning_overlap == approx(1.0)  # still reported for inspection
        assert result.reward == approx(1.0)  # base rule ignores the overlap

    async def test_wrong_answer_gives_zero_reward(self) -> None:
        # Default grading (multiply): a wrong answer scores 0 regardless of overlap.
        server = _make_server()
        request = _make_request(
            expected_answer='["a", "b"]',
            answer="Final Answer: [c]",
            reasoning="anything",
            input_text="graph prompt",
        )
        result = await server.verify(request)
        assert result.answer_score == approx(0.0)
        assert result.reward == approx(0.0)

    async def test_multiply_rule_gates_reward_by_overlap(self) -> None:
        server = _make_server(overlap_grading_rule=OverlapGradingRule.MULTIPLY)
        prompt = "The graph has edges from node a to node b and several other distinct nodes scattered around."
        request = _make_request(
            expected_answer='["a", "b"]',
            answer="Final Answer: [a, b]",
            reasoning=prompt,  # verbatim copy -> overlap 1.0 -> reward 0.0
            input_text=prompt,
        )
        result = await server.verify(request)
        assert result.answer_score == approx(1.0)
        assert result.reasoning_overlap == approx(1.0)
        assert result.reward == approx(0.0)

    async def test_multiply_rule_correct_answer_low_overlap(self) -> None:
        server = _make_server(overlap_grading_rule=OverlapGradingRule.MULTIPLY)
        request = _make_request(
            expected_answer='["a", "b"]',
            answer="Final Answer: [a, b]",
            reasoning="I traced the graph briefly and picked the endpoints.",
            input_text="Some long haystack prompt describing a graph with many distinct edges.",
        )
        result = await server.verify(request)
        assert result.answer_score == approx(1.0)
        assert result.reward == approx(1.0 - result.reasoning_overlap)

    async def test_minus_rule_subtracts_overlap(self) -> None:
        server = _make_server(overlap_grading_rule=OverlapGradingRule.MINUS)
        request = _make_request(
            expected_answer='["a"]',
            answer="Final Answer: [a]",
            reasoning="a partially overlapping reasoning trace",
            input_text="a partially overlapping prompt with extra words",
        )
        result = await server.verify(request)
        assert result.reward == approx(result.answer_score - result.reasoning_overlap)

    async def test_no_reasoning_no_penalty(self) -> None:
        server = _make_server(overlap_grading_rule=OverlapGradingRule.MULTIPLY)
        request = _make_request(
            expected_answer='["a"]',
            answer="Final Answer: [a]",
            reasoning=None,
            input_text="graph prompt",
        )
        result = await server.verify(request)
        assert result.overlap_seq_match == approx(0.0)
        assert result.overlap_ngram16 == approx(0.0)
        assert result.overlap_lcs == approx(0.0)
        assert result.reasoning_overlap == approx(0.0)
        assert result.reward == approx(1.0)

    @pytest.mark.parametrize(
        "metric_rule, signal_field",
        [
            (OverlapMetricRule.SEQ_MATCH, "overlap_seq_match"),
            (OverlapMetricRule.NGRAM16, "overlap_ngram16"),
            (OverlapMetricRule.LCS, "overlap_lcs"),
        ],
    )
    async def test_metric_rule_selects_signal(self, metric_rule: OverlapMetricRule, signal_field: str) -> None:
        server = _make_server(metric_rule)
        request = _make_request(
            expected_answer='["a"]',
            answer="Final Answer: [a]",
            reasoning="a partially overlapping reasoning trace with several words",
            input_text="a partially overlapping prompt with several extra words",
        )
        result = await server.verify(request)
        assert result.reasoning_overlap == approx(getattr(result, signal_field))

    async def test_invalid_metric_rule_raises(self) -> None:
        server = _make_server()
        server.config.overlap_metric_rule = "bogus"
        request = _make_request(expected_answer='["a"]', answer="Final Answer: [a]", reasoning="r", input_text="i")
        with pytest.raises(ValueError, match="Invalid overlap metric rule"):
            await server.verify(request)

    async def test_invalid_grading_rule_raises(self) -> None:
        server = _make_server()
        server.config.overlap_grading_rule = "bogus"
        request = _make_request(expected_answer='["a"]', answer="Final Answer: [a]", reasoning="r", input_text="i")
        with pytest.raises(ValueError, match="Invalid overlap grading rule"):
            await server.verify(request)

    async def test_response_fields_present(self) -> None:
        server = _make_server()
        request = _make_request(expected_answer='["x"]', answer="Final Answer: [x]", reasoning="r", input_text="i")
        result = await server.verify(request)
        dump = result.model_dump()
        for field in (
            "reward",
            "answer_score",
            "overlap_seq_match",
            "overlap_ngram16",
            "overlap_lcs",
            "reasoning_overlap",
        ):
            assert field in dump


class TestServerInstantiation:
    def test_default_name(self) -> None:
        config = LCNIAHResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="")
        assert config.name == "lc_niah"
