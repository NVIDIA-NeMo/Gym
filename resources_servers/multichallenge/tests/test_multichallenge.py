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

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.multichallenge.app import (
    AggregationMode,
    MultiChallengeConfig,
    MultiChallengeServer,
    MultiChallengeVerifyRequest,
    RubricEvaluation,
    _build_context_from_messages,
    _extract_verdict,
)


def make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response_id",
        created_at=0.0,
        model="test-model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message_id",
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


def make_server() -> MultiChallengeServer:
    config = MultiChallengeConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="multichallenge",
        judge_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=8192, temperature=0.7, top_p=0.8
        ),
        judge_system_message=(
            "You are a precise evaluator. Assess responses objectively based on the given criteria. "
            "Analyze the response carefully against the evaluation question."
        ),
        aggregation_mode=AggregationMode.MEAN,
    )
    return MultiChallengeServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestMultiChallenge:
    """Tests for MultiChallenge environment utilities."""

    def test_extract_verdict_yes(self):
        """Test extracting YES verdict."""
        response = "After analysis, the model correctly addressed the user's allergy. [[YES]]"
        verdict = _extract_verdict(response, "[[YES]]", "[[NO]]")
        assert verdict == "YES"

    def test_extract_verdict_no(self):
        """Test extracting NO verdict."""
        response = "The model failed to remember the allergy. [[NO]]"
        verdict = _extract_verdict(response, "[[YES]]", "[[NO]]")
        assert verdict == "NO"

    def test_extract_verdict_fallback(self):
        """Test fallback when no label present."""
        response = "The model did well.\nYES"
        verdict = _extract_verdict(response, "[[YES]]", "[[NO]]")
        assert verdict == "YES"

    def test_extract_verdict_last_wins(self):
        """Test that last label wins when both present."""
        response = "Initially [[YES]] but actually [[NO]]"
        verdict = _extract_verdict(response, "[[YES]]", "[[NO]]")
        assert verdict == "NO"

    def test_build_context_excludes_thinking(self):
        """Test that thinking messages are excluded from context."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "thinking", "content": "Processing..."},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = _build_context_from_messages(messages, exclude_thinking=True)
        assert "Processing" not in context
        assert "[USER]: Hello" in context
        assert "[ASSISTANT]: Hi there!" in context

    def test_build_context_includes_thinking(self):
        """Test that thinking messages can be included."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "thinking", "content": "Processing..."},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = _build_context_from_messages(messages, exclude_thinking=False)
        assert "[THINKING]: Processing" in context


class TestAggregation:
    """Tests for score aggregation."""

    def create_evaluations(self, scores: list[float]) -> list[RubricEvaluation]:
        """Create mock evaluations with given scores."""
        return [
            RubricEvaluation(
                question=f"Q{i}",
                pass_criteria="YES",
                judge_prompt="...",
                judge_response="...",
                verdict="YES" if s >= 0.99 else "NO",
                score=s,
                weight=1.0,
            )
            for i, s in enumerate(scores)
        ]

    def test_aggregation_modes(self):
        """Test various aggregation modes."""
        config = MultiChallengeConfig(
            host="",
            port=0,
            entrypoint="",
            name="test",
            judge_model_server=ModelServerRef(type="responses_api_models", name="test"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        # Create a proper mock that passes pydantic validation
        mock_client = MagicMock(spec=ServerClient)
        server = MultiChallengeServer.model_construct(config=config, server_client=mock_client)
        evaluations = self.create_evaluations([1.0, 0.5, 0.0])

        # Test MEAN
        config.aggregation_mode = AggregationMode.MEAN
        assert server._aggregate_scores(evaluations) == pytest.approx(0.5)

        # Test MIN
        config.aggregation_mode = AggregationMode.MIN
        assert server._aggregate_scores(evaluations) == 0.0

        # Test MAX
        config.aggregation_mode = AggregationMode.MAX
        assert server._aggregate_scores(evaluations) == 1.0

        # Test ALL (only first passes)
        config.aggregation_mode = AggregationMode.ALL
        assert server._aggregate_scores(evaluations) == 0.0

        # Test ANY (first passes)
        config.aggregation_mode = AggregationMode.ANY
        assert server._aggregate_scores(evaluations) == 1.0


async def test_verify_scores_each_rubric_and_returns_mean_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_judge(*args, **kwargs) -> NeMoGymResponse:
        del args
        prompt = kwargs["json"].input[-1].content
        verdict = "[[YES]]" if "remember the allergy" in prompt else "[[NO]]"
        return make_response(verdict)

    judge = AsyncMock(side_effect=fake_call_judge)
    monkeypatch.setattr("resources_servers.multichallenge.app.call_judge", judge)
    server = make_server()
    request = MultiChallengeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Please suggest dinner."}]
        ),
        response=make_response("<think>private chain of thought</think>\nTry a nut-free pasta."),
        metadata={
            "messages": [{"role": "user", "content": "I have a peanut allergy."}],
            "rubric": [
                {"question": "Did the response remember the allergy?", "pass_criteria": "YES"},
                {"question": "Did the response include a dessert?", "pass_criteria": "YES"},
            ],
        },
    )

    result = await server.verify(request)

    assert result.reward == pytest.approx(0.5)
    assert result.generated_response == "Try a nut-free pasta."
    assert "private chain of thought" not in result.generated_response
    assert result.context == "[USER]: I have a peanut allergy."
    assert [evaluation.verdict for evaluation in result.rubric_evaluations] == ["YES", "NO"]
    assert result.num_passed == 1
    assert result.num_total == 2
    assert judge.await_count == 2
    for call in judge.await_args_list:
        assert call.kwargs["server_name"] == "policy_model"
        assert call.kwargs["url_path"] == "/v1/responses"
        assert call.kwargs["response_model"] is NeMoGymResponse
        judge_params = call.kwargs["json"]
        assert judge_params.max_output_tokens == 8192
        assert judge_params.temperature == pytest.approx(0.7)
        assert judge_params.top_p == pytest.approx(0.8)
        assert [message.role for message in judge_params.input] == ["system", "user"]


async def test_verify_empty_rubric_is_zero_without_judge_call(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = AsyncMock()
    monkeypatch.setattr("resources_servers.multichallenge.app.call_judge", judge)
    request = MultiChallengeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("A response without a rubric."),
    )

    result = await make_server().verify(request)

    assert result.reward == 0.0
    assert result.num_total == 0
    judge.assert_not_awaited()


async def test_verify_negative_pass_criteria_rewards_no_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = AsyncMock(return_value=make_response("[[NO]]"))
    monkeypatch.setattr("resources_servers.multichallenge.app.call_judge", judge)
    request = MultiChallengeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("I did not make the prohibited claim."),
        context="Earlier conversation",
        rubric=[{"question": "Did the response make the prohibited claim?", "pass_criteria": "NO"}],
    )

    result = await make_server().verify(request)

    assert result.reward == 1.0
    assert result.num_passed == 1
    assert result.rubric_evaluations[0].verdict == "NO"


async def test_verify_propagates_judge_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = AsyncMock(side_effect=JudgeError("RuntimeError: judge unavailable"))
    monkeypatch.setattr("resources_servers.multichallenge.app.call_judge", judge)
    request = MultiChallengeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("A response to judge."),
        context="Earlier conversation",
        rubric=[{"question": "Did it answer?", "pass_criteria": "YES"}],
    )

    with pytest.raises(JudgeError, match="RuntimeError: judge unavailable"):
        await make_server().verify(request)
