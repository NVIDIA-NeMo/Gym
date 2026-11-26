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

import orjson
import pytest
from pytest import approx, fixture

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_runtime import BaseRunRequest
from nemo_gym.verification import JudgeError
from resources_servers.omniscience.app import (
    OmniscienceConfig,
    OmniscienceRunRequest,
    OmniscienceServer,
    OmniscienceVerifyRequest,
    _strip_thinking_traces,
    extract_text_from_response,
    parse_judge_grade,
)
from resources_servers.omniscience.task_data import TaskData


class TestStripThinkingTraces:
    def test_strips_think_tags(self) -> None:
        assert _strip_thinking_traces("<think>some reasoning</think>The answer is 42") == "The answer is 42"

    def test_strips_thinking_tags(self) -> None:
        assert _strip_thinking_traces("<thinking>deep thought</thinking>Result: yes") == "Result: yes"

    def test_strips_unpaired_closing_think(self) -> None:
        assert _strip_thinking_traces("reasoning here</think>The actual answer") == "The actual answer"

    def test_strips_unpaired_closing_thinking(self) -> None:
        assert _strip_thinking_traces("reasoning here</thinking>The actual answer") == "The actual answer"

    def test_no_tags(self) -> None:
        assert _strip_thinking_traces("plain text") == "plain text"

    def test_multiline_think(self) -> None:
        assert _strip_thinking_traces("<think>\nline1\nline2\n</think>\nAnswer") == "Answer"


class TestParseJudgeGrade:
    def test_single_letter_a(self) -> None:
        assert parse_judge_grade("A") == "A"

    def test_single_letter_b(self) -> None:
        assert parse_judge_grade("B") == "B"

    def test_single_letter_c(self) -> None:
        assert parse_judge_grade("C") == "C"

    def test_single_letter_d(self) -> None:
        assert parse_judge_grade("D") == "D"

    def test_letter_on_last_line(self) -> None:
        assert parse_judge_grade("Some reasoning\nA") == "A"

    def test_labelled_letter_in_text(self) -> None:
        assert parse_judge_grade("The grade is A based on analysis") == "A"

    def test_no_grade_returns_none(self) -> None:
        assert parse_judge_grade("no clear grade here") is None

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("NOT_ATTEMPTED", "D"),
            ("CORRECT", "A"),
            ("INCORRECT", "B"),
            ("PARTIAL_ANSWER", "C"),
            ("PARTIAL", "C"),
            ("The grade is NOT_ATTEMPTED because the model refuses.", "D"),
            ("C: PARTIAL_ANSWER", "C"),
            ("Grade: **B**", "B"),
            ("reasoning\n**A**", "A"),
            ("reasoning\nD.", "D"),
        ],
    )
    def test_unambiguous_grade_tokens(self, text: str, expected: str) -> None:
        assert parse_judge_grade(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "A reasonable attempt, but it is not clear",
            "It is CORRECT in part but also INCORRECT in part",
            "PARTIALLY_CORRECT",
            "Not attempted at all",
            "",
        ],
    )
    def test_ambiguous_text_has_no_grade(self, text: str) -> None:
        assert parse_judge_grade(text) is None

    def test_whitespace_handling(self) -> None:
        assert parse_judge_grade("  D  ") == "D"

    def test_d_in_multiline(self) -> None:
        assert parse_judge_grade("reasoning\nmore reasoning\nD") == "D"


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


class TestOmniscienceServer:
    @fixture
    def config(self) -> OmniscienceConfig:
        return OmniscienceConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(
                type="responses_api_models",
                name="judge_model",
            ),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
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

    async def test_verify_correct(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("1989")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="In what year did the Berlin Wall fall?",
            expected_answer="1989",
            domain="Humanities and Social Sciences",
            topic="History",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.verdict == "correct"
        assert result.is_correct == approx(1.0)
        assert result.is_incorrect == approx(0.0)
        assert result.is_partial == approx(0.0)
        assert result.is_not_attempted == approx(0.0)
        assert result.omniscience_index == approx(1.0)
        assert result.is_hallucination == approx(0.0)

    async def test_verify_incorrect(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("B"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("1991")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="In what year did the Berlin Wall fall?",
            expected_answer="1989",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "incorrect"
        assert result.is_correct == approx(0.0)
        assert result.is_incorrect == approx(1.0)
        assert result.omniscience_index == approx(-1.0)
        assert result.is_hallucination == approx(1.0)

    async def test_verify_partial(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("C"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("About 28 million")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="In millions of dollars, how much profit did Company X make in 2024?",
            expected_answer="28",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "partial"
        assert result.is_partial == approx(1.0)
        assert result.is_correct == approx(0.0)
        assert result.is_incorrect == approx(0.0)
        assert result.omniscience_index == approx(0.0)
        assert result.is_hallucination == approx(0.0)

    async def test_verify_not_attempted(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("D"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("I don't have enough information to answer this question.")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Some obscure question?",
            expected_answer="Some answer",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "not_attempted"
        assert result.is_not_attempted == approx(1.0)
        assert result.is_correct == approx(0.0)
        assert result.is_incorrect == approx(0.0)
        assert result.omniscience_index == approx(0.0)
        assert result.is_hallucination == approx(0.0)

    async def test_verify_with_thinking_traces(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("<think>Let me recall...</think>Pancreas")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What organ in the human body produces insulin?",
            expected_answer="Pancreas",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.verdict == "correct"
        assert result.extracted_answer == "Pancreas"

    async def test_verify_response_fields(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("Test answer")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Test question?",
            expected_answer="Test answer",
        )

        result = await server.verify(request)
        dump = result.model_dump()
        assert "reward" in dump
        assert "extracted_answer" in dump
        assert "expected_answer" in dump
        assert "verdict" in dump
        assert "judge_output" in dump
        assert "is_correct" in dump
        assert "is_incorrect" in dump
        assert "is_partial" in dump
        assert "is_not_attempted" in dump
        assert "omniscience_index" in dump
        assert "is_hallucination" in dump
        assert result.expected_answer == "Test answer"

    async def test_verify_empty_response(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("D"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        # Empty model output
        empty_response = NeMoGymResponse(
            id="empty",
            created_at=0.0,
            model="policy_model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=empty_response,
            question="Test?",
            expected_answer="Answer",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.extracted_answer == ""

    async def test_judge_prompt_contains_question_and_answer(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("Paris")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What is the capital of France?",
            expected_answer="Paris",
        )

        await server.verify(request)

        # Verify the judge was called with the right prompt content
        call_kwargs = server_mock.post.call_args
        json_payload = call_kwargs.kwargs["json"]
        judge_input = json_payload.input[0].content
        assert "What is the capital of France?" in judge_input
        assert "Paris" in judge_input

    async def test_verify_no_think_tag_preserves_plain_answer(self, config: OmniscienceConfig) -> None:
        """Plain answers from non-reasoning models should be judged as-is."""
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("Paris")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What is the capital of France?",
            expected_answer="Paris",
        )

        result = await server.verify(request)
        assert result.extracted_answer == "Paris"
        assert result.verdict == "correct"

    async def test_verify_unclosed_think_tag_produces_empty(self, config: OmniscienceConfig) -> None:
        """An unclosed reasoning block indicates a response truncated before its answer."""
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("D"))
        response_mock.read = AsyncMock(return_value=orjson.dumps(response_mock.json.return_value))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("<think>Let me think about this...")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Some question?",
            expected_answer="Some answer",
        )

        result = await server.verify(request)
        assert result.extracted_answer == ""
        assert result.verdict == "not_attempted"

    async def test_verify_without_a_grade_raises_judge_error(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)
        self._mock_judge_reply(server_mock, self._make_judge_response("The predicted answer matches the gold target."))

        with pytest.raises(JudgeError, match="matches the gold target"):
            await server.verify(self._make_request("1989"))

    async def test_chat_completions_judge_strips_think_blocks(self, config: OmniscienceConfig) -> None:
        config.use_chat_completions_for_judge = True
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)
        self._mock_judge_reply(
            server_mock,
            {
                "id": "judge_chat",
                "object": "chat.completion",
                "created": 0,
                "model": "judge_model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "<think>Same year as the gold target.</think>\nA"},
                    }
                ],
            },
        )

        result = await server.verify(self._make_request("1989"))

        assert result.verdict == "correct"
        assert result.judge_output == "A"

    def _mock_judge_reply(self, server_mock: MagicMock, payload: dict) -> None:
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=payload)
        response_mock.read = AsyncMock(return_value=orjson.dumps(payload))
        server_mock.post = AsyncMock(return_value=response_mock)

    def _make_request(self, policy_text: str) -> OmniscienceVerifyRequest:
        return OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._make_model_response(policy_text),
            question="In what year did the Berlin Wall fall?",
            expected_answer="1989",
        )


def test_run_request_fields_agree_with_task_data() -> None:
    provenance_only = {
        "aa_omniscience_version",
        "aa_omniscience_dataset_revision",
    }
    wire_fields = set(OmniscienceRunRequest.model_fields) - set(BaseRunRequest.model_fields)
    assert wire_fields == set(TaskData.model_fields) - provenance_only
    assert OmniscienceRunRequest.model_config.get("extra") == TaskData.model_config.get("extra") == "allow"
    for name in wire_fields:
        wire_field = OmniscienceRunRequest.model_fields[name]
        schema_field = TaskData.model_fields[name]
        assert wire_field.annotation == schema_field.annotation, name
        assert wire_field.is_required() == schema_field.is_required(), name
        assert wire_field.default == schema_field.default, name


class TestOmniscienceScoreFn:
    def test_correct(self) -> None:
        assert OmniscienceServer._omni_score_fn({"verdict": "correct"}) == {
            "judge_correct": 1.0,
            "judge_incorrect": 0.0,
            "judge_partially_correct": 0.0,
            "judge_abstained": 0.0,
        }

    def test_incorrect(self) -> None:
        scores = OmniscienceServer._omni_score_fn({"verdict": "incorrect"})
        assert scores["judge_correct"] == 0.0
        assert scores["judge_incorrect"] == -1.0

    def test_partial(self) -> None:
        scores = OmniscienceServer._omni_score_fn({"verdict": "partial"})
        assert scores["judge_partially_correct"] == 1.0
        assert scores["judge_correct"] == 0.0

    def test_not_attempted(self) -> None:
        scores = OmniscienceServer._omni_score_fn({"verdict": "not_attempted"})
        assert scores["judge_abstained"] == 1.0
        assert scores["judge_correct"] == 0.0


class TestExtractTextNoStrip:
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

    def test_strip_false_preserves_reasoning(self) -> None:
        response = self._make_response("reasoning here</think>answer")
        assert extract_text_from_response(response, strip_thinking=False) == "reasoning here</think>answer"

    def test_strip_true_removes_reasoning(self) -> None:
        response = self._make_response("reasoning here</think>answer")
        assert extract_text_from_response(response, strip_thinking=True) == "answer"


@pytest.mark.parametrize(
    "text, expected, parsed",
    [
        ("A", "A", True),
        ("A\n", "A", True),
        ("B", "B", True),
        ("B\nA", "A", True),
        ("explanation\nD", "D", True),
        ("reasoning\n\u00a0A", "A", True),
        ("reasoning\n\u00a0B", "B", True),
        ("reasoning\n\u2003D", "D", True),
        ("The grade is A based on analysis", "A", True),
        ("The grade is A based on option B", "A", True),
        ("The grade is A based on answer B", "A", True),
        ("Grade: **B**", "B", True),
        ("ANSWER: B", "B", True),
        ("Answer is D", "D", True),
        ("Verdict: D", "D", True),
        ("Reasoning...\nFinal grade: C", "C", True),
        ("NOT_ATTEMPTED", "D", True),
        ("CORRECT", "A", True),
        ("INCORRECT", "B", True),
        ("C: PARTIAL_ANSWER", "C", True),
        ("Grade: INCORRECT", "B", True),
        ("Grade: B: INCORRECT", "B", True),
        ("Grade: C: PARTIAL_ANSWER", "C", True),
        ("Grade: B; Verdict: B", "B", True),
        ("NOT CORRECT", None, False),
        ("It is not CORRECT", None, False),
        ("The predicted answer is NOT CORRECT.", None, False),
        ("PARTIALLY CORRECT", None, False),
        ("Grade: NOT CORRECT", None, False),
        ("Grade: PARTIALLY CORRECT", None, False),
        ("<think>Is it CORRECT? No.</think>B", None, False),
        ("<think>\nCORRECT\n</think>B", None, False),
        ("A\n\nExplanation: the answer is B-level", None, False),
        ("The gold answer is C and the predicted answer is C++, which is different.", None, False),
        ("<your_message>A</your_message>", None, False),
        ("b", None, False),
        ("Predicted answer: Berlin", None, False),
        ("The predicted answer is a city in France.", None, False),
        ("Answer: Berlin", None, False),
        ("Grade: Athens", None, False),
        ("Answer: C++", None, False),
        ("Grade: A+", None, False),
        ("Grade: A/B-level", None, False),
        ("Grade: A/B+", None, False),
        ("Grade: A/E", None, False),
        ("Grade:\nA reasonable attempt", None, False),
        ("Reasoning; Grade: A", None, False),
        ("C: CORRECT", None, False),
        ("Grade: A: INCORRECT", None, False),
        ("Grade: A/B", None, False),
        ("Grade: A & B", None, False),
        ("Grade: A | C", None, False),
        ("Grade: CORRECT or INCORRECT", None, False),
        ("The grade is A or B", None, False),
        ("Grade: A/B; Grade: A", None, False),
        ("Grade: A; Grade: B", None, False),
        ("Grade: A; Grade: B+", None, False),
        ("Grade: **A**/**B**", None, False),
        ("Grade: **A** and **B**", None, False),
        ("Grade: **A**, **B**", None, False),
        ("The answer A is not correct.", None, False),
        ("PARTIALLY_CORRECT", None, False),
        ("no clear grade", None, False),
        ("It is CORRECT in part but also INCORRECT in part", None, False),
        # A verdict followed by a short reason or closing punctuation still counts.
        ("CORRECT because the predicted answer matches the gold target.", "A", True),
        ("Grade: B (the predicted year differs)", "B", True),
        ("Answer: D - the model declined to answer", "D", True),
        ("Final grade: C since only part of the answer is given", "C", True),
        ("Verdict: **A**.", "A", True),
        # Prose that only starts like a verdict, from the #5783 review.
        ("Answer: A different year (1990) was given, so this is incorrect.", None, False),
        ("CORRECT answer would be 1989, but the model said 1990.", None, False),
        ("CORRECT? No, the answers differ.", None, False),
        ("Answer: C#", None, False),
        ("<think>Let me consider the options:\nA", None, False),
        ("<THINK>reasoning</THINK>\nA", None, False),
        ("Grade A would require an exact match, so this is B.", None, False),
        ("Grade A", None, False),
        ("A: CORRECT\nB: INCORRECT\nC:", None, False),
        ("B:", None, False),
        ("B\n\nThe answer is CORRECT to one significant figure, but the gold target requires three.", None, False),
        ("Grade: A，B", None, False),
    ],
)
def test_only_explicit_judge_verdict_marks_grading(text: str, expected: str | None, parsed: bool) -> None:
    from nemo_gym.trajectory import ModelCallRef
    from nemo_gym.verification import JudgeGrading
    from nemo_gym.verification.judge import _judge_grading

    grading = JudgeGrading(dependencies=[ModelCallRef(model_call_id="judge-response")])
    token = _judge_grading.set(grading)
    try:
        assert parse_judge_grade(text) == expected
        assert grading.parsed_dependencies == ({0} if parsed else set())
    finally:
        _judge_grading.reset(token)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Grade: B; Verdict: B would be too harsh", None),
        ("Grade: B; Verdict: B; Grade: B is too harsh", None),
        ("Grade: B; Verdict: B would be too harsh\nGrade: A", None),
        ("CORRECT (A) requires a full match.\nGrade: B. The predicted answer differs.", None),
        ("Grade: B — the year differs.\nCORRECT (gold): 1989", None),
        ("CORRECT answer would be 1989, but the model said 1990.\nGrade: B", None),
        ("INCORRECT. The predicted answer names another county.\n\nGrade: B", "B"),
        ("Grade: A Because the year matches", "A"),
        ("Grade: A becauseof", None),
    ],
)
def test_verdict_tail_on_later_clauses_and_lines(text: str, expected: str | None) -> None:
    assert parse_judge_grade(text) == expected


class TestChatCompletionsJudge:
    """The Chat Completions judge path, which `configs/omniscience.yaml` selects (`use_chat_completions_for_judge`)."""

    @fixture
    def config(self) -> OmniscienceConfig:
        return OmniscienceConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            # Each value differs from the chat path's fallback (64, 0.0, 1.0), so a dropped setting fails the test.
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[], max_output_tokens=32768, temperature=0.2, top_p=0.9
            ),
            use_chat_completions_for_judge=True,
        )

    @staticmethod
    def _server(config: OmniscienceConfig, content: str | None) -> tuple[OmniscienceServer, MagicMock]:
        server_mock = MagicMock(spec=ServerClient)
        reply = AsyncMock()
        reply.ok = True
        reply.status = 200
        body = {
            "id": "chatcmpl-judge",
            "object": "chat.completion",
            "created": 0,
            "model": "judge_model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        }
        reply.read = AsyncMock(return_value=orjson.dumps(body))
        server_mock.post = AsyncMock(return_value=reply)
        return OmniscienceServer(config=config, server_client=server_mock), server_mock

    @staticmethod
    def _request(text: str) -> OmniscienceVerifyRequest:
        return OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
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
            ),
            question="Which Internal Revenue Code section allows expensing equipment?",
            expected_answer="179",
        )

    async def test_request_carries_the_configured_judge_settings(self, config: OmniscienceConfig) -> None:
        server, server_mock = self._server(config, "A")
        result = await server.verify(self._request("Section 179"))
        assert result.verdict == "correct"
        assert result.reward == approx(1.0)
        kwargs = server_mock.post.call_args.kwargs
        assert kwargs["url_path"] == "/v1/chat/completions"
        payload = kwargs["json"].model_dump()
        assert payload["max_tokens"] == 32768
        assert payload["temperature"] == approx(0.2)
        assert payload["top_p"] == approx(0.9)
        assert "Which Internal Revenue Code section" in payload["messages"][0]["content"]

    async def test_empty_judge_content_is_a_judge_failure(self, config: OmniscienceConfig) -> None:
        from nemo_gym.verification import JudgeError

        server, _ = self._server(config, None)
        with pytest.raises(JudgeError):
            await server.verify(self._request("Section 179"))

    async def test_unclosed_thinking_block_is_an_empty_answer(self, config: OmniscienceConfig) -> None:
        server, _ = self._server(config, "D")
        result = await server.verify(self._request("<thinking>Let me recall the section"))
        assert result.extracted_answer == ""
        assert result.verdict == "not_attempted"


def test_compute_metrics_index_and_hallucination_rate() -> None:
    config = OmniscienceConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    server = OmniscienceServer(config=config, server_client=MagicMock(spec=ServerClient))
    verdicts = [
        ["correct", "incorrect", "not_attempted", "partial"],
        ["incorrect", "incorrect", "correct", "not_attempted"],
    ]
    tasks = [
        [
            {
                "_ng_task_index": t,
                "_ng_rollout_index": r,
                "verdict": v,
                "reward": 1.0 if v == "correct" else 0.0,
                "extracted_answer": f"answer-{t}-{r}",
            }
            for r, v in enumerate(row)
        ]
        for t, row in enumerate(verdicts)
    ]
    metrics = server.compute_metrics(tasks)
    # NeMo Skills OmniMetrics: index = 100 * (correct - incorrect) / n; hallucination = 100 * incorrect / not correct.
    assert metrics["pass@1[avg-of-4]/judge_correct"] == approx(100 * 2 / 8)
    assert metrics["pass@1[avg-of-4]/judge_omni_index"] == approx(100 * (2 - 3) / 8)
    assert metrics["pass@1[avg-of-4]/judge_omni_hallucination"] == approx(100 * 3 / 6)
    key_metrics = server.get_key_metrics(metrics)
    for name in ("judge_omni_index", "judge_correct", "judge_omni_hallucination"):
        assert f"pass@1[avg-of-4]/{name}" in key_metrics
