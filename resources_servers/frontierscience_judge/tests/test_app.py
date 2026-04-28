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
from resources_servers.frontierscience_judge.app import (
    FrontierScienceJudgeConfig,
    FrontierScienceJudgeServer,
    FrontierScienceJudgeVerifyRequest,
    _strip_thinking_traces,
    extract_text_from_response,
    parse_judgement,
)


class TestStripThinkingTraces:
    def test_strips_think_tags(self) -> None:
        assert _strip_thinking_traces("<think>some reasoning</think>The answer is 42") == "The answer is 42"

    def test_strips_thinking_tags(self) -> None:
        assert _strip_thinking_traces("<thinking>deep thought</thinking>Result: yes") == "Result: yes"

    def test_strips_unpaired_closing_think(self) -> None:
        assert _strip_thinking_traces("reasoning here</think>The actual answer") == "The actual answer"

    def test_no_tags(self) -> None:
        assert _strip_thinking_traces("plain text") == "plain text"

    def test_multiline_think(self) -> None:
        assert _strip_thinking_traces("<think>\nline1\nline2\n</think>\nAnswer") == "Answer"


class TestParseJudgement:
    def test_yes(self) -> None:
        assert parse_judgement("Some reasoning\nJudgement: YES") == "YES"

    def test_no(self) -> None:
        assert parse_judgement("Some reasoning\nJudgement: NO") == "NO"

    def test_lowercase(self) -> None:
        assert parse_judgement("Judgement: yes") == "YES"

    def test_extra_whitespace(self) -> None:
        assert parse_judgement("Judgement:   YES  ") == "YES"

    def test_last_match_wins(self) -> None:
        # Mirror Skills' is_correct_judgement that anchors on the last line.
        assert parse_judgement("Judgement: NO\n... actually Judgement: YES") == "YES"

    def test_no_marker(self) -> None:
        assert parse_judgement("This response has no verdict marker") is None

    def test_empty(self) -> None:
        assert parse_judgement("") is None


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

    def test_strip_false_preserves_reasoning(self) -> None:
        response = self._make_response("reasoning here</think>answer")
        assert extract_text_from_response(response, strip_thinking=False) == "reasoning here</think>answer"

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


class TestFrontierScienceJudgeServer:
    @fixture
    def config(self) -> FrontierScienceJudgeConfig:
        return FrontierScienceJudgeConfig(
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

    def _make_judge_response(self, judge_text: str) -> dict:
        return NeMoGymResponse(
            id="judge_resp",
            created_at=0.0,
            model="judge_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="judge_msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=judge_text, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()

    async def test_verify_yes(self, config: FrontierScienceJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("Reasoning ...\nJudgement: YES"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("CO2")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What is the chemical formula of carbon dioxide?",
            expected_answer="CO2",
            subject="chemistry",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.verdict == "YES"
        assert result.extracted_answer == "CO2"
        assert result.expected_answer == "CO2"

    async def test_verify_no(self, config: FrontierScienceJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("Wrong: Judgement: NO"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("H2O")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What is the chemical formula of carbon dioxide?",
            expected_answer="CO2",
            subject="chemistry",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "NO"

    async def test_verify_unparseable_judge(self, config: FrontierScienceJudgeConfig) -> None:
        """Judge response with no Judgement: marker => verdict=None, reward=0."""
        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("hand-wavy reasoning, no marker"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("H2O")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Q",
            expected_answer="A",
            subject="biology",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict is None

    async def test_verify_with_thinking_traces(self, config: FrontierScienceJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("Judgement: YES"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("<think>Let me recall...</think>Pancreas")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What organ produces insulin?",
            expected_answer="Pancreas",
            subject="biology",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.verdict == "YES"
        assert result.extracted_answer == "Pancreas"

    async def test_verify_no_close_think_tag_treated_as_no_answer(self, config: FrontierScienceJudgeConfig) -> None:
        """Reasoning that never closed: <think>... with no </think> => generation empty."""
        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("Judgement: NO"))
        server_mock.post = AsyncMock(return_value=response_mock)

        # Model output has <think>... but no </think>: truncated mid-reasoning.
        model_response = self._make_model_response("<think>Let me think... I don't know the answer")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Q",
            expected_answer="A",
            subject="physics",
        )

        result = await server.verify(request)
        assert result.extracted_answer is None

    async def test_judge_prompt_contains_question_answer_generation(self, config: FrontierScienceJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("Judgement: YES"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("Paris")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What is the capital of France?",
            expected_answer="Paris",
            subject="other",
        )

        await server.verify(request)

        call_kwargs = server_mock.post.call_args
        json_payload = call_kwargs.kwargs["json"]
        judge_input = json_payload.input[0].content
        assert "What is the capital of France?" in judge_input
        assert "Paris" in judge_input
        # Skills' verbatim grading instructions:
        assert "olympiad" in judge_input
        assert "Judgement: YES" in judge_input or "Judgement:" in judge_input

    async def test_subject_propagated_to_response(self, config: FrontierScienceJudgeConfig) -> None:
        """The `subject` field must round-trip through verify so compute_subset_metrics sees it."""
        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("Judgement: YES"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("answer")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="q",
            expected_answer="a",
            subject="chemistry",
        )

        result = await server.verify(request)
        dump = result.model_dump()
        assert dump["subject"] == "chemistry"

    async def test_verify_chat_completions_path(self, config: FrontierScienceJudgeConfig) -> None:
        """Exercise the use_chat_completions_for_judge=True branch."""
        chat_config = config.model_copy(deep=True)
        chat_config.use_chat_completions_for_judge = True

        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=chat_config, server_client=server_mock)

        chat_response_dict = {
            "id": "chat-1",
            "created": 0,
            "model": "judge_model",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Reasoning step 1...\nJudgement: YES",
                    },
                }
            ],
        }
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=chat_response_dict)
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("Pancreas")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What organ produces insulin?",
            expected_answer="Pancreas",
            subject="biology",
        )

        result = await server.verify(request)
        # The chat-completions endpoint was hit, not /v1/responses.
        call_kwargs = server_mock.post.call_args
        assert call_kwargs.kwargs["url_path"] == "/v1/chat/completions"
        assert result.reward == approx(1.0)
        assert result.verdict == "YES"
        assert "Judgement: YES" in (result.judge_output or "")

    async def test_verify_chat_completions_empty_choices(self, config: FrontierScienceJudgeConfig) -> None:
        """Chat completions response with empty choices => verdict=None, reward=0."""
        chat_config = config.model_copy(deep=True)
        chat_config.use_chat_completions_for_judge = True

        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=chat_config, server_client=server_mock)

        chat_response_dict = {
            "id": "chat-empty",
            "created": 0,
            "model": "judge_model",
            "object": "chat.completion",
            "choices": [],
        }
        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=chat_response_dict)
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("anything")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="q",
            expected_answer="a",
            subject="physics",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict is None

    async def test_verify_response_fields(self, config: FrontierScienceJudgeConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = FrontierScienceJudgeServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.json = AsyncMock(return_value=self._make_judge_response("Judgement: YES"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("Test answer")
        request = FrontierScienceJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Test question?",
            expected_answer="Test answer",
            subject="biology",
        )

        result = await server.verify(request)
        dump = result.model_dump()
        for f in ("reward", "extracted_answer", "expected_answer", "verdict", "judge_output"):
            assert f in dump
        assert result.expected_answer == "Test answer"


class TestComputeMetrics:
    @fixture
    def server(self) -> FrontierScienceJudgeServer:
        config = FrontierScienceJudgeConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        return FrontierScienceJudgeServer(config=config, server_client=MagicMock(spec=ServerClient))

    @staticmethod
    def _rollout(reward: float, subject: str, ans: str = "x") -> dict:
        return {"reward": reward, "subject": subject, "extracted_answer": ans}

    def test_pass_at_k(self, server: FrontierScienceJudgeServer) -> None:
        tasks = [
            [self._rollout(1.0, "chemistry"), self._rollout(0.0, "chemistry")],
            [self._rollout(0.0, "biology"), self._rollout(0.0, "biology")],
        ]
        result = server.compute_metrics(tasks)
        assert "pass@1/accuracy" in result
        assert "pass@2/accuracy" in result
        # Task 0: 1/2 correct, Task 1: 0/2 correct => pass@1 = (50 + 0) / 2 = 25
        assert result["pass@1/accuracy"] == approx(25.0, abs=0.01)
        # pass@2 (any correct): 100 + 0 / 2 = 50
        assert result["pass@2/accuracy"] == approx(50.0, abs=0.01)

    def test_per_subject_breakdown(self, server: FrontierScienceJudgeServer) -> None:
        tasks = [
            [self._rollout(1.0, "chemistry"), self._rollout(1.0, "chemistry")],
            [self._rollout(0.0, "chemistry"), self._rollout(0.0, "chemistry")],
            [self._rollout(1.0, "physics"), self._rollout(0.0, "physics")],
        ]
        result = server.compute_metrics(tasks)
        assert "chemistry/pass@1/accuracy" in result
        assert "physics/pass@1/accuracy" in result
        # chemistry: task0=1.0, task1=0.0 -> 50% pass@1
        assert result["chemistry/pass@1/accuracy"] == approx(50.0, abs=0.01)
        # physics: task0=0.5 -> pass@1 = 50%
        assert result["physics/pass@1/accuracy"] == approx(50.0, abs=0.01)

    def test_empty_tasks(self, server: FrontierScienceJudgeServer) -> None:
        assert server.compute_metrics([]) == {}


class TestGetKeyMetrics:
    def test_selects_headlines(self) -> None:
        agent_metrics = {
            "mean/input_tokens": 100.0,
            "mean/output_tokens": 500.0,
            "mean/reward": 0.5,
            "pass@1/accuracy": 50.0,
            "pass@1[avg-of-4]/accuracy": 50.0,
            "pass@1[avg-of-4]/accuracy/std_dev_across_runs": 3.0,
            "pass@4/accuracy": 70.0,
            "majority@4/accuracy": 60.0,
        }
        result = FrontierScienceJudgeServer.get_key_metrics(None, agent_metrics)
        assert "mean/input_tokens" in result
        assert "mean/output_tokens" in result
        assert "mean/reward" not in result
        assert "pass@1[avg-of-4]/accuracy" in result
        assert "pass@4/accuracy" in result
        assert "majority@4/accuracy" in result
        assert "pass@1[avg-of-4]/accuracy/std_dev_across_runs" not in result


class TestScoreFn:
    def test_score_fn_correct(self) -> None:
        assert FrontierScienceJudgeServer._score_fn({"reward": 1.0}) == {"accuracy": 1.0}

    def test_score_fn_incorrect(self) -> None:
        assert FrontierScienceJudgeServer._score_fn({"reward": 0.0}) == {"accuracy": 0.0}

    def test_score_fn_missing_reward(self) -> None:
        assert FrontierScienceJudgeServer._score_fn({}) == {"accuracy": 0.0}
