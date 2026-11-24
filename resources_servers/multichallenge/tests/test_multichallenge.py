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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    JUDGE_UNPARSEABLE,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.verification import JudgeError, judge_failsafe
from resources_servers.multichallenge.app import (
    AggregationMode,
    MultiChallengeConfig,
    MultiChallengeServer,
    MultiChallengeVerifyRequest,
    RubricEvaluation,
    _build_context_from_messages,
    _extract_verdict,
)


ROOT = Path(__file__).resolve().parents[1]


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
    config_path = ROOT / "configs/multichallenge.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["multichallenge"]["resources_servers"][
        "multichallenge"
    ]
    config = MultiChallengeConfig.model_validate(
        {"host": "127.0.0.1", "port": 8080, "name": "multichallenge", **raw_config}
    )
    return MultiChallengeServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_checked_in_examples_match_verify_request_contract() -> None:
    examples = [
        yaml.safe_load(line)
        for line in (ROOT / "data/example.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = [
        MultiChallengeVerifyRequest.model_validate({**example, "response": make_response("Acknowledged.")})
        for example in examples
    ]

    assert len(requests) == 5
    challenges = set()
    for request in requests:
        assert request.metadata is not None
        challenges.add(request.metadata["challenge"])
    assert challenges >= {
        "Context Tracking",
        "Memory Retention",
        "Preference Update",
    }
    for request in requests:
        assert request.context
        assert request.context.endswith(request.responses_create_params.input[-1].content)
        assert request.rubric is not None
        assert all(item["pass_criteria"] == "YES" for item in request.rubric)
        assert request.agent_ref is not None
        assert request.agent_ref["name"] == "multichallenge_simple_agent"


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

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("Reasoning.\n**No**", "NO"),
            ("Reasoning.\nVerdict: yes", "YES"),
            ("Reasoning.\n[NO].", "NO"),
        ],
    )
    def test_extract_verdict_bare_last_line(self, response, expected):
        assert _extract_verdict(response, "[[YES]]", "[[NO]]") == expected

    @pytest.mark.parametrize(
        "response",
        [
            "",
            "The response does not mention the allergy.",
            "I cannot determine whether the criterion is met.",
            "Analysis of the response follows, but it is unknown",
            "Reasoning.\nNot enough information to decide.",
            "Reasoning.\nNOTE: the response is unclear",
        ],
    )
    def test_extract_verdict_without_a_verdict_is_none(self, response):
        """A reply without a verdict is not a NO: substrings such as NOT, CANNOT or UNKNOWN are not verdicts."""
        assert _extract_verdict(response, "[[YES]]", "[[NO]]") is None

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


@pytest.mark.parametrize(
    "judge_text",
    ["I cannot determine whether the response makes the claim.", ""],
    ids=["prose-without-verdict", "empty-reply"],
)
async def test_verify_masks_a_judge_reply_without_a_verdict(monkeypatch: pytest.MonkeyPatch, judge_text: str) -> None:
    # Before, this reply defaulted to NO and passed the NO-criterion rubric item (reward 1.0, trained).
    judge = AsyncMock(return_value=make_response(judge_text))
    monkeypatch.setattr("resources_servers.multichallenge.app.call_judge", judge)
    server = make_server()
    request = MultiChallengeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("I did not make the prohibited claim."),
        context="Earlier conversation",
        rubric=[{"question": "Did the response make the prohibited claim?", "pass_criteria": "NO"}],
    )

    with pytest.raises(JudgeError) as raised:
        await server.verify(request)
    assert raised.value.failure_kind == JUDGE_UNPARSEABLE

    # Served through judge_failsafe, as the base server serves /verify: a masked row.
    result = json.loads((await judge_failsafe(server.verify)(body=request)).body)
    assert result["mask_sample"] is True
    assert result["failure_kind"] == JUDGE_UNPARSEABLE
    assert result["_ng_failure_class"] == "judge_failed"


async def test_verify_empty_answer_is_zero_without_judge_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # The model's own failure: trained at 0, not masked, and never judged (a NO-criterion item would pass it).
    judge = AsyncMock(return_value=make_response("[[NO]]"))
    monkeypatch.setattr("resources_servers.multichallenge.app.call_judge", judge)
    request = MultiChallengeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("<think>reasoning that never reached an answer</think>"),
        context="Earlier conversation",
        rubric=[{"question": "Did the response make the prohibited claim?", "pass_criteria": "NO"}],
    )

    result = await make_server().verify(request)

    assert result.reward == 0.0
    assert result.mask_sample is False
    assert result.generated_response == ""
    judge.assert_not_awaited()


async def test_verify_masks_a_judge_reply_cut_at_its_output_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    cut_reply = NeMoGymResponse.model_validate(
        make_response("If it never recommends peanuts, the answer is [[NO]]. But line 3 says").model_dump()
        | {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    )
    monkeypatch.setattr("resources_servers.multichallenge.app.call_judge", AsyncMock(return_value=cut_reply))
    request = MultiChallengeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("Peanut butter crackers are a great snack."),
        context="I am allergic to peanuts.",
        rubric=[{"question": "Did the response recommend peanuts?", "pass_criteria": "NO"}],
    )

    with pytest.raises(JudgeError, match="cut off") as raised:
        await make_server().verify(request)
    assert raised.value.failure_kind == JUDGE_UNPARSEABLE


async def test_verify_scores_a_prompt_the_judge_refused_for_its_length_as_a_failed_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # vllm_model answers a prompt longer than the judge's window with an empty reply cut before any output token.
    refused = NeMoGymResponse.model_validate(
        make_response("").model_dump()
        | {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )
    monkeypatch.setattr("resources_servers.multichallenge.app.call_judge", AsyncMock(return_value=refused))
    request = MultiChallengeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=make_response("Peanut butter crackers are a great snack. " * 50),
        context="I am allergic to peanuts.",
        rubric=[{"question": "Did the response recommend peanuts?", "pass_criteria": "NO"}],
    )

    result = await make_server().verify(request)

    # Before this rule, the empty reply read as NO and passed this NO-criterion item.
    assert (result.reward, result.mask_sample) == (0.0, False)
    [evaluation] = result.rubric_evaluations
    assert (evaluation.verdict, evaluation.score) == ("", 0.0)
