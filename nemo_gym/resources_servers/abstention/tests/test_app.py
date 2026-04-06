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

from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.resources_servers.abstention.app import (
    AbstentionConfig,
    AbstentionServer,
    AbstentionVerifyRequest,
    _strip_thinking_traces,
    extract_boxed_answer,
    extract_text_from_response,
    normalize_answer,
    parse_judge_grade,
)
from nemo_gym.server_utils import ServerClient


class TestExtractBoxedAnswer:
    def test_simple_boxed(self) -> None:
        assert extract_boxed_answer("The answer is \\boxed{42}") == "42"

    def test_no_boxed(self) -> None:
        assert extract_boxed_answer("No boxed answer here") is None

    def test_nested_braces(self) -> None:
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_last_boxed_wins(self) -> None:
        assert extract_boxed_answer("\\boxed{wrong} then \\boxed{right}") == "right"

    def test_idk_token(self) -> None:
        assert extract_boxed_answer("\\boxed{[IDK]}") == "[IDK]"

    def test_empty_boxed(self) -> None:
        assert extract_boxed_answer("\\boxed{}") == ""

    def test_unclosed_boxed(self) -> None:
        assert extract_boxed_answer("\\boxed{unclosed") is None


class TestNormalizeAnswer:
    def test_lowercases(self) -> None:
        assert normalize_answer("Hello World") == "hello world"

    def test_strips_articles(self) -> None:
        assert normalize_answer("The quick brown fox") == "quick brown fox"

    def test_strips_punctuation(self) -> None:
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    def test_idk_normalization(self) -> None:
        assert normalize_answer("[IDK]") == "idk"


class TestStripThinkingTraces:
    def test_strips_think_tags(self) -> None:
        text = "<think>some reasoning</think>The answer is 42"
        assert _strip_thinking_traces(text) == "The answer is 42"

    def test_strips_thinking_tags(self) -> None:
        text = "<thinking>deep thought</thinking>Result: yes"
        assert _strip_thinking_traces(text) == "Result: yes"

    def test_strips_unpaired_closing_think(self) -> None:
        text = "reasoning here</think>The actual answer"
        assert _strip_thinking_traces(text) == "The actual answer"

    def test_strips_unpaired_closing_thinking(self) -> None:
        text = "reasoning here</thinking>The actual answer"
        assert _strip_thinking_traces(text) == "The actual answer"

    def test_no_tags(self) -> None:
        assert _strip_thinking_traces("plain text") == "plain text"

    def test_multiline_think(self) -> None:
        text = "<think>\nline1\nline2\n</think>\nAnswer"
        assert _strip_thinking_traces(text) == "Answer"


class TestParseJudgeGrade:
    def test_single_letter_a(self) -> None:
        assert parse_judge_grade("A") == "A"

    def test_single_letter_b(self) -> None:
        assert parse_judge_grade("B") == "B"

    def test_single_letter_c(self) -> None:
        assert parse_judge_grade("C") == "C"

    def test_letter_on_last_line(self) -> None:
        assert parse_judge_grade("Some reasoning\nA") == "A"

    def test_letter_in_text(self) -> None:
        assert parse_judge_grade("The grade is A based on analysis") == "A"

    def test_fallback_to_b(self) -> None:
        assert parse_judge_grade("no clear grade here") == "B"

    def test_whitespace_handling(self) -> None:
        assert parse_judge_grade("  C  ") == "C"

    def test_c_in_multiline(self) -> None:
        assert parse_judge_grade("reasoning\nmore reasoning\nC") == "C"


class TestExtractTextFromResponse:
    def _make_response(self, text: str) -> NeMoGymResponse:
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

    def test_extracts_text(self) -> None:
        response = self._make_response("Hello world")
        assert extract_text_from_response(response) == "Hello world"

    def test_strips_thinking(self) -> None:
        response = self._make_response("<think>reasoning</think>Answer")
        assert extract_text_from_response(response) == "Answer"

    def test_empty_output(self) -> None:
        response = NeMoGymResponse(
            id="test",
            created_at=0.0,
            model="test_model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        assert extract_text_from_response(response) == ""


class TestAbstentionServer:
    @fixture
    def config(self) -> AbstentionConfig:
        return AbstentionConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(
                type="responses_api_models",
                name="judge_model",
            ),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            abstention_reward=0.5,
            abstention_token="[IDK]",
            correct_reward=1.0,
            incorrect_reward=0.0,
        )

    def _make_model_response(self, text: str) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="policy_resp",
            created_at=0.0,
            model="policy_model",
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

    def _make_judge_response(self, grade: str) -> dict:
        return NeMoGymResponse(
            id="judge_resp",
            created_at=0.0,
            model="judge_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="judge_msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=grade, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()

    async def test_verify_abstention_via_boxed_idk(self, config: AbstentionConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = AbstentionServer(config=config, server_client=server_mock)

        model_response = self._make_model_response("I'm not sure. \\boxed{[IDK]}")
        request = AbstentionVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What is the capital of Atlantis?",
            answer="Unknown",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.5)
        assert result.verdict == "abstain"
        assert result.is_abstain == approx(1.0)
        assert result.is_correct == approx(0.0)
        assert result.is_incorrect == approx(0.0)
        assert result.judge_output is None
        assert result.extracted_answer == "[IDK]"

    async def test_verify_correct_answer(self, config: AbstentionConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = AbstentionServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("The answer is \\boxed{Yes}")
        request = AbstentionVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Were Scott Derrickson and Ed Wood of the same nationality?",
            answer="Yes",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.verdict == "correct"
        assert result.is_correct == approx(1.0)
        assert result.is_abstain == approx(0.0)
        assert result.is_incorrect == approx(0.0)
        assert result.omniscience_index == approx(1.0)
        assert result.extracted_answer == "Yes"

    async def test_verify_incorrect_answer(self, config: AbstentionConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = AbstentionServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("B"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("The answer is \\boxed{No}")
        request = AbstentionVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Were Scott Derrickson and Ed Wood of the same nationality?",
            answer="Yes",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "incorrect"
        assert result.is_correct == approx(0.0)
        assert result.is_incorrect == approx(1.0)
        assert result.omniscience_index == approx(-1.0)

    async def test_verify_judge_not_attempted(self, config: AbstentionConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = AbstentionServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("C"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("I don't have enough information to answer this.")
        request = AbstentionVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Some obscure question?",
            answer="Some answer",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.5)
        assert result.verdict == "abstain"
        assert result.is_abstain == approx(1.0)

    async def test_verify_with_thinking_traces(self, config: AbstentionConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = AbstentionServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response(
            "<think>Let me reason about this...</think>The answer is \\boxed{Paris}"
        )
        request = AbstentionVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What is the capital of France?",
            answer="Paris",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.verdict == "correct"
        assert result.extracted_answer == "Paris"

    async def test_verify_response_fields(self, config: AbstentionConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = AbstentionServer(config=config, server_client=server_mock)

        model_response = self._make_model_response("\\boxed{[IDK]}")
        request = AbstentionVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Test question?",
            answer="Test answer",
        )

        result = await server.verify(request)
        dump = result.model_dump()
        assert "reward" in dump
        assert "extracted_answer" in dump
        assert "ground_truth" in dump
        assert "verdict" in dump
        assert "judge_output" in dump
        assert "is_correct" in dump
        assert "is_abstain" in dump
        assert "is_incorrect" in dump
        assert "omniscience_index" in dump
        assert result.ground_truth == "Test answer"

    async def test_verify_no_boxed_uses_full_text(self, config: AbstentionConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = AbstentionServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("The capital of France is Paris")
        request = AbstentionVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What is the capital of France?",
            answer="Paris",
        )

        result = await server.verify(request)
        assert result.extracted_answer == "The capital of France is Paris"
        assert result.reward == approx(1.0)
