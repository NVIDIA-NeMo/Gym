# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nemo_gym.server_utils import ServerClient
from resources_servers.simpleqa.app import (
    SimpleQAConfig,
    SimpleQAServer,
    SimpleQAVerifyRequest,
    extract_text_from_response,
    parse_judge_grade,
)


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

    def test_fallback_to_c(self) -> None:
        # Unparseable judgement → NOT_ATTEMPTED, matching Skills'
        # DEFAULT_GRADE_IF_UNPARSEABLE = "C".
        assert parse_judge_grade("no clear grade here") == "C"

    def test_whitespace_handling(self) -> None:
        assert parse_judge_grade("  B  ") == "B"

    def test_leading_letter_match(self) -> None:
        # Matches Skills' is_correct_judgement_label_matching's
        # judgement[0] fallback for "A: CORRECT" / "A " inputs.
        assert parse_judge_grade("A: CORRECT") == "A"
        assert parse_judge_grade("B - INCORRECT") == "B"
        assert parse_judge_grade("C, NOT_ATTEMPTED") == "C"


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

    def test_returns_content_with_inline_think_tags_verbatim(self) -> None:
        # If <think> blocks reach this function, the model server is
        # misconfigured (no --reasoning-parser). The function does NOT
        # strip them — it returns the message content verbatim and the
        # judge sees the same garbled text. Operator must fix server-side.
        response = self._make_response("<think>reasoning</think>Answer")
        assert extract_text_from_response(response) == "<think>reasoning</think>Answer"

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

    def test_returns_content_verbatim(self) -> None:
        # Reasoning-trace stripping is the model server's responsibility
        # (vLLM --reasoning-parser <name>); this function never inspects
        # <think> tags itself.
        response = self._make_response("Uzo Njoku was born in 1991.")
        assert extract_text_from_response(response) == "Uzo Njoku was born in 1991."


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
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


def _make_judge_response_dict(grade: str) -> dict:
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


def _make_chat_response_dict(grade: str) -> dict:
    return {
        "id": "chat",
        "object": "chat.completion",
        "created": 0,
        "model": "judge_model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": grade},
                "finish_reason": "stop",
            }
        ],
    }


class TestSimpleQAServer:
    @fixture
    def config(self) -> SimpleQAConfig:
        return SimpleQAConfig(
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

    async def test_verify_correct(self, config: SimpleQAConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = SimpleQAServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=_make_judge_response_dict("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        request = SimpleQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("Pancreas"),
            question="What organ in the human body produces insulin?",
            expected_answer="Pancreas",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.verdict == "correct"
        assert result.is_correct == approx(1.0)
        assert result.is_incorrect == approx(0.0)
        assert result.is_not_attempted == approx(0.0)

    async def test_verify_incorrect(self, config: SimpleQAConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = SimpleQAServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=_make_judge_response_dict("B"))
        server_mock.post = AsyncMock(return_value=response_mock)

        request = SimpleQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("Liver"),
            question="What organ in the human body produces insulin?",
            expected_answer="Pancreas",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "incorrect"
        assert result.is_incorrect == approx(1.0)

    async def test_verify_not_attempted(self, config: SimpleQAConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = SimpleQAServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=_make_judge_response_dict("C"))
        server_mock.post = AsyncMock(return_value=response_mock)

        request = SimpleQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("I'm not sure."),
            question="Some obscure question?",
            expected_answer="Some answer",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "not_attempted"
        assert result.is_not_attempted == approx(1.0)

    async def test_verify_passes_message_content_verbatim(self, config: SimpleQAConfig) -> None:
        """The model-server's --reasoning-parser strips <think> blocks
        before they reach the resource server, so verify() forwards the
        message content as-is to the judge — no inline tag handling."""
        server_mock = MagicMock(spec=ServerClient)
        server = SimpleQAServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=_make_judge_response_dict("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        request = SimpleQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("Pancreas"),
            question="What organ in the human body produces insulin?",
            expected_answer="Pancreas",
        )

        result = await server.verify(request)
        assert result.verdict == "correct"
        assert result.extracted_answer == "Pancreas"

    async def test_verify_chat_completions_path(self, config: SimpleQAConfig) -> None:
        config.use_chat_completions_for_judge = True
        server_mock = MagicMock(spec=ServerClient)
        server = SimpleQAServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=_make_chat_response_dict("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        request = SimpleQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("Paris"),
            question="What is the capital of France?",
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.verdict == "correct"

        # Confirm /v1/chat/completions was invoked, not /v1/responses.
        call_kwargs = server_mock.post.call_args
        assert call_kwargs.kwargs["url_path"] == "/v1/chat/completions"

    async def test_judge_prompt_contains_question_and_answer(self, config: SimpleQAConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = SimpleQAServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=_make_judge_response_dict("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        request = SimpleQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("Paris"),
            question="What is the capital of France?",
            expected_answer="Paris",
        )
        await server.verify(request)

        call_kwargs = server_mock.post.call_args
        json_payload = call_kwargs.kwargs["json"]
        judge_input = json_payload.input[0].content
        assert "What is the capital of France?" in judge_input
        assert "Paris" in judge_input

    async def test_judge_unparseable_falls_back_to_not_attempted(self, config: SimpleQAConfig) -> None:
        """Skills' DEFAULT_GRADE_IF_UNPARSEABLE = 'C' (NOT_ATTEMPTED)."""
        server_mock = MagicMock(spec=ServerClient)
        server = SimpleQAServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(
            return_value=_make_judge_response_dict("Garbled output with no recognizable grade")
        )
        server_mock.post = AsyncMock(return_value=response_mock)

        request = SimpleQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("Some answer"),
            question="Q?",
            expected_answer="Gold",
        )
        result = await server.verify(request)
        assert result.verdict == "not_attempted"


class TestSimpleQAScoreFn:
    def test_correct(self) -> None:
        assert SimpleQAServer._score_fn({"verdict": "correct"}) == {
            "correct": 1.0,
            "incorrect": 0.0,
            "not_attempted": 0.0,
        }

    def test_incorrect(self) -> None:
        assert SimpleQAServer._score_fn({"verdict": "incorrect"}) == {
            "correct": 0.0,
            "incorrect": 1.0,
            "not_attempted": 0.0,
        }

    def test_not_attempted(self) -> None:
        assert SimpleQAServer._score_fn({"verdict": "not_attempted"}) == {
            "correct": 0.0,
            "incorrect": 0.0,
            "not_attempted": 1.0,
        }


class TestSimpleQAComputeMetrics:
    """Validate F1 formula matches Skills' SimpleQAMetrics.

    Skills computes:
        precision = correct / (correct + incorrect)   (= accuracy_given_attempted)
        recall    = correct / total
        f1 = 2 * precision * recall / (precision + recall)
    """

    @fixture
    def config(self) -> SimpleQAConfig:
        return SimpleQAConfig(
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

    def _task_with_verdict(self, verdict: str, idx: int = 0) -> dict:
        return {
            "verdict": verdict,
            "extracted_answer": f"answer_{idx}",
            "is_correct": 1.0 if verdict == "correct" else 0.0,
            "is_incorrect": 1.0 if verdict == "incorrect" else 0.0,
            "is_not_attempted": 1.0 if verdict == "not_attempted" else 0.0,
        }

    def test_f1_all_correct(self, config: SimpleQAConfig) -> None:
        server = SimpleQAServer(config=config, server_client=MagicMock(spec=ServerClient))
        # 4 questions × 1 rollout, all correct.
        tasks = [[self._task_with_verdict("correct", i)] for i in range(4)]
        m = server.compute_metrics(tasks)
        # P = 4/(4+0) = 1, R = 4/4 = 1 → F1 = 1 (100% on the 0-100 scale)
        f1_keys = [k for k in m if k.endswith("/f1")]
        assert f1_keys, f"No /f1 keys in metrics: {sorted(m)}"
        for k in f1_keys:
            assert m[k] == approx(100.0), (k, m[k])

    def test_f1_all_incorrect(self, config: SimpleQAConfig) -> None:
        server = SimpleQAServer(config=config, server_client=MagicMock(spec=ServerClient))
        tasks = [[self._task_with_verdict("incorrect", i)] for i in range(4)]
        m = server.compute_metrics(tasks)
        # P = 0/(0+4) = 0, R = 0/4 = 0 → F1 = 0
        f1_keys = [k for k in m if k.endswith("/f1")]
        assert f1_keys
        for k in f1_keys:
            assert m[k] == approx(0.0), (k, m[k])

    def test_f1_all_not_attempted(self, config: SimpleQAConfig) -> None:
        server = SimpleQAServer(config=config, server_client=MagicMock(spec=ServerClient))
        tasks = [[self._task_with_verdict("not_attempted", i)] for i in range(4)]
        m = server.compute_metrics(tasks)
        # P = 0/(0+0) = 0 (no attempted), R = 0/4 = 0 → F1 = 0
        f1_keys = [k for k in m if k.endswith("/f1")]
        assert f1_keys
        for k in f1_keys:
            assert m[k] == approx(0.0), (k, m[k])

    def test_f1_mixed(self, config: SimpleQAConfig) -> None:
        server = SimpleQAServer(config=config, server_client=MagicMock(spec=ServerClient))
        # 2 correct, 1 incorrect, 1 not_attempted (4 questions total).
        verdicts = ["correct", "correct", "incorrect", "not_attempted"]
        tasks = [[self._task_with_verdict(v, i)] for i, v in enumerate(verdicts)]
        m = server.compute_metrics(tasks)
        # P = 2/3 ≈ 0.6667, R = 2/4 = 0.5
        # F1 = 2*P*R / (P+R) = 2 * (2/3) * 0.5 / (2/3 + 0.5)
        # = (2/3) / (7/6) = 4/7 ≈ 0.5714 → 57.14 on the 0-100 scale.
        f1_keys = [k for k in m if k.endswith("/f1")]
        agt_keys = [k for k in m if k.endswith("/accuracy_given_attempted")]
        assert f1_keys
        assert agt_keys
        expected_f1 = 100.0 * 4 / 7
        expected_agt = 100.0 * 2 / 3
        for k in f1_keys:
            assert m[k] == approx(expected_f1, rel=1e-3), (k, m[k])
        for k in agt_keys:
            assert m[k] == approx(expected_agt, rel=1e-3), (k, m[k])

    def test_f1_no_attempted(self, config: SimpleQAConfig) -> None:
        """All questions not attempted → P=0/0 (defined as 0), F1=0."""
        server = SimpleQAServer(config=config, server_client=MagicMock(spec=ServerClient))
        tasks = [[self._task_with_verdict("not_attempted", i)] for i in range(3)]
        m = server.compute_metrics(tasks)
        agt_keys = [k for k in m if k.endswith("/accuracy_given_attempted")]
        for k in agt_keys:
            assert m[k] == approx(0.0)


class TestSimpleQAGetKeyMetrics:
    @fixture
    def config(self) -> SimpleQAConfig:
        return SimpleQAConfig(
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

    def test_includes_token_means(self, config: SimpleQAConfig) -> None:
        server = SimpleQAServer(config=config, server_client=MagicMock(spec=ServerClient))
        agg = {
            "mean/input_tokens": 100.0,
            "mean/output_tokens": 200.0,
            "pass@1[avg-of-4]/correct": 0.5,
            "pass@4/correct": 0.7,
        }
        key = server.get_key_metrics(agg)
        assert key["mean/input_tokens"] == 100.0
        assert key["mean/output_tokens"] == 200.0
