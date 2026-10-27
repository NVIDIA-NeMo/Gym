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
"""Tests for the DWRL pointwise utilities.

In DWRL pointwise every rollout is judged on its own. The reward of a rollout is the BT score of the judge call
(the probability that the judge answers with `score_token`), averaged over the judges. `score_1` is parsed from
the judge's JSON, but it does not feed into the reward.
"""

import pytest
from pytest import approx

from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from resources_servers.dwrl_genrm_compare_pointwise.utils import (
    EMPTY_OUTPUT_PLACEHOLDER,
    GenRMOutputParseError,
    _compute_length_weights,
    aggregate_scores,
    apply_length_bonuses,
    extract_conversation_history,
    extract_from_response_obj,
    extract_output_text,
    generate_comparison_pairs,
    get_prompt_key,
    get_prompt_key_from_input,
    parse_genrm_output,
)


def _response_obj(answer="", reasoning=""):
    """A Response API object with the given reasoning summary and final answer."""
    output = []
    if reasoning:
        output.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    if answer:
        output.append({"type": "message", "content": [{"type": "output_text", "text": answer}]})
    return {"output": output}


class TestGetPromptKeyFromInput:
    """The prompt key groups the rollouts of one prompt into a cohort."""

    def test_accepts_pydantic_messages(self) -> None:
        dict_messages = [{"role": "user", "content": "Hello", "type": "message"}]
        model_messages = [NeMoGymEasyInputMessage(role="user", content="Hello", type="message")]

        assert get_prompt_key_from_input(model_messages, "Be concise") == get_prompt_key_from_input(
            dict_messages, "Be concise"
        )

    def test_is_deterministic(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        assert get_prompt_key_from_input(messages, None) == get_prompt_key_from_input(list(messages), None)

    def test_different_input_gives_different_key(self) -> None:
        a = get_prompt_key_from_input([{"role": "user", "content": "Hello"}], None)
        b = get_prompt_key_from_input([{"role": "user", "content": "Goodbye"}], None)
        assert a != b

    def test_different_principle_gives_different_key(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        assert get_prompt_key_from_input(messages, "Be concise") != get_prompt_key_from_input(messages, "Be thorough")
        assert get_prompt_key_from_input(messages, "Be concise") != get_prompt_key_from_input(messages, None)

    def test_dict_key_order_does_not_matter(self) -> None:
        a = get_prompt_key_from_input([{"role": "user", "content": "Hello"}], None)
        b = get_prompt_key_from_input([{"content": "Hello", "role": "user"}], None)
        assert a == b

    def test_message_order_matters(self) -> None:
        first = {"role": "user", "content": "one"}
        second = {"role": "assistant", "content": "two"}
        assert get_prompt_key_from_input([first, second], None) != get_prompt_key_from_input([second, first], None)


class TestGetPromptKey:
    def test_prompt_id_is_used_when_present(self) -> None:
        assert get_prompt_key({"prompt_id": 17}) == "17"

    def test_prompt_id_is_combined_with_principle(self) -> None:
        assert get_prompt_key({"prompt_id": 17, "principle": "Be concise"}) == "17::Be concise"

    def test_falls_back_to_hashing_the_input(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        example = {"responses_create_params": {"input": messages}, "principle": "Be concise"}
        assert get_prompt_key(example) == get_prompt_key_from_input(messages, "Be concise")


class TestExtractConversationHistory:
    def test_returns_the_input_messages(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        assert extract_conversation_history({"responses_create_params": {"input": messages}}) == messages

    def test_missing_responses_create_params_raises(self) -> None:
        with pytest.raises(ValueError, match="responses_create_params"):
            extract_conversation_history({})

    def test_missing_input_raises(self) -> None:
        with pytest.raises(ValueError, match="input"):
            extract_conversation_history({"responses_create_params": {}})


class TestGenerateComparisonPairs:
    """Pair generation is shared with the pairwise server; DWRL pointwise does not use it to judge rollouts."""

    def test_circular(self) -> None:
        assert generate_comparison_pairs("circular", 3) == [(0, 1), (1, 2), (2, 0)]

    def test_all_pairs(self) -> None:
        assert generate_comparison_pairs("all_pairs", 3) == [(0, 1), (0, 2), (1, 2)]

    def test_unsupported_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown comparison strategy"):
            generate_comparison_pairs("unknown", 3)

    def test_less_than_2_responses_raises(self) -> None:
        with pytest.raises(ValueError, match="Need at least 2 responses"):
            generate_comparison_pairs("circular", 1)


class TestParseGenRMOutput:
    """The judge is asked to end with a JSON object containing `score_1`."""

    def test_fenced_json(self) -> None:
        output = 'My analysis...\n```json\n{"response_analysis": "Solid.", "score_1": 4}\n```\n'
        assert parse_genrm_output(output, -100.0) == approx(4.0)

    def test_unfenced_json(self) -> None:
        assert parse_genrm_output('The result is {"score_1": 5} overall.', -100.0) == approx(5.0)

    def test_returns_a_float(self) -> None:
        result = parse_genrm_output('{"score_1": 4}', -100.0)
        assert isinstance(result, float)

    def test_fractional_and_string_scores(self) -> None:
        assert parse_genrm_output('{"score_1": 3.5}', -100.0) == approx(3.5)
        assert parse_genrm_output('{"score_1": "2"}', -100.0) == approx(2.0)

    def test_last_unfenced_json_wins(self) -> None:
        output = 'First {"score_1": 1} then, after reconsidering, {"score_1": 4}'
        assert parse_genrm_output(output, -100.0) == approx(4.0)

    def test_fenced_json_is_preferred_over_unfenced(self) -> None:
        output = '{"score_1": 1}\n```json\n{"score_1": 5}\n```\n{"score_1": 2}'
        assert parse_genrm_output(output, -100.0) == approx(5.0)

    def test_pretty_printed_json(self) -> None:
        output = '```json\n{\n    "response_analysis": "Good.",\n    "score_1": 4\n}\n```'
        assert parse_genrm_output(output, -100.0) == approx(4.0)

    def test_braces_inside_strings_do_not_break_parsing(self) -> None:
        output = '{"response_analysis": "uses {braces} and \\"quotes\\"", "score_1": 3}'
        assert parse_genrm_output(output, -100.0) == approx(3.0)

    def test_nested_overall_object(self) -> None:
        output = '{"rubric_evaluations": [], "overall": {"score_1": 2}}'
        assert parse_genrm_output(output, -100.0) == approx(2.0)

    def test_no_json_returns_default(self) -> None:
        assert parse_genrm_output("I think the response is fine", -100.0) == approx(-100.0)

    def test_json_without_score_1_returns_default(self) -> None:
        assert parse_genrm_output('{"score_2": 4, "ranking": 2}', -100.0) == approx(-100.0)

    def test_non_numeric_score_returns_default(self) -> None:
        assert parse_genrm_output('{"score_1": "not a number"}', -100.0) == approx(-100.0)

    def test_non_object_json_returns_default(self) -> None:
        assert parse_genrm_output("[1, 2, 3]", -100.0) == approx(-100.0)

    def test_empty_output_returns_default(self) -> None:
        assert parse_genrm_output("", 3.0) == approx(3.0)

    def test_raise_on_fail_raises(self) -> None:
        with pytest.raises(GenRMOutputParseError):
            parse_genrm_output("No JSON here", 3.0, raise_on_fail=True)

    def test_raise_on_fail_does_not_raise_when_parsable(self) -> None:
        assert parse_genrm_output('{"score_1": 4}', 3.0, raise_on_fail=True) == approx(4.0)


class TestExtractFromResponseObj:
    def test_extract_output_text_from_message(self) -> None:
        assert extract_output_text(_response_obj(answer="Hello world")) == "Hello world"

    def test_extract_reasoning_and_output(self) -> None:
        response_obj = {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"text": "Let me think... ", "type": "summary_text"},
                        {"text": "The answer is clear.", "type": "summary_text"},
                    ],
                },
                {"type": "message", "content": [{"type": "output_text", "text": "Paris"}]},
            ]
        }
        reasoning, output = extract_from_response_obj(response_obj)
        assert reasoning == "Let me think... The answer is clear."
        assert output == "Paris"

    def test_multiple_output_text_parts_are_concatenated(self) -> None:
        response_obj = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Hello "},
                        {"type": "output_text", "text": "world"},
                    ],
                }
            ]
        }
        assert extract_output_text(response_obj) == "Hello world"

    def test_non_output_text_content_is_ignored(self) -> None:
        response_obj = {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}
        assert extract_from_response_obj(response_obj) == ("", "")

    def test_empty_output_returns_placeholder(self) -> None:
        assert extract_output_text({"output": []}) == EMPTY_OUTPUT_PLACEHOLDER

    def test_whitespace_only_output_returns_placeholder(self) -> None:
        assert extract_output_text(_response_obj(answer="  \n ")) == EMPTY_OUTPUT_PLACEHOLDER

    @pytest.mark.parametrize("bad", [None, "text", 3, {"output": None}, {"output": "text"}, {"output": [None, 3]}])
    def test_malformed_response_objects_are_handled(self, bad) -> None:
        assert extract_from_response_obj(bad) == ("", "")
        assert extract_output_text(bad) == EMPTY_OUTPUT_PLACEHOLDER


class TestComputeLengthWeights:
    def test_equal_lengths_give_zero_weights(self) -> None:
        assert _compute_length_weights([5, 5, 5]) == [0.0, 0.0, 0.0]

    def test_shorter_gets_higher_weight_and_weights_are_zero_centered(self) -> None:
        weights = _compute_length_weights([0, 5, 10])
        assert weights == approx([0.5, 0.0, -0.5])
        assert sum(weights) == approx(0.0)


class TestApplyLengthBonuses:
    """Length bonuses are implemented here, but aggregate_scores does not apply them (see TestAggregateScores)."""

    @staticmethod
    def _apply(scores, response_objs, **overrides):
        kwargs = dict(
            reasoning_bonus=0.0,
            answer_bonus=0.0,
            top_percentile=0.34,
            group_reasoning_length_penalty_coeff=0.0,
            group_answer_length_penalty_coeff=0.0,
        )
        kwargs.update(overrides)
        return apply_length_bonuses(list(scores), response_objs, **kwargs)

    def test_fewer_than_two_responses_is_a_no_op(self) -> None:
        scores, bonuses = self._apply([1.0], [_response_obj(answer="a")], reasoning_bonus=1.0)
        assert scores == [1.0]
        assert bonuses == [0.0]

    def test_reasoning_bonus_goes_to_the_shortest_reasoning_if_it_is_a_top_scorer(self) -> None:
        response_objs = [
            _response_obj(reasoning="r" * 1),
            _response_obj(reasoning="r" * 5),
            _response_obj(reasoning="r" * 10),
        ]
        scores, bonuses = self._apply([1.0, 0.5, 0.2], response_objs, reasoning_bonus=0.1)
        assert scores == approx([1.1, 0.5, 0.2])
        assert bonuses == approx([0.1, 0.0, 0.0])

    def test_no_reasoning_bonus_if_the_shortest_is_not_a_top_scorer(self) -> None:
        response_objs = [
            _response_obj(reasoning="r" * 10),
            _response_obj(reasoning="r" * 5),
            _response_obj(reasoning="r" * 1),
        ]
        scores, bonuses = self._apply([1.0, 0.5, 0.2], response_objs, reasoning_bonus=0.1)
        assert scores == approx([1.0, 0.5, 0.2])
        assert bonuses == approx([0.0, 0.0, 0.0])

    def test_empty_reasoning_is_not_eligible_for_the_bonus(self) -> None:
        response_objs = [_response_obj(), _response_obj(reasoning="r" * 5), _response_obj(reasoning="r" * 9)]
        scores, bonuses = self._apply([1.0, 0.9, 0.2], response_objs, reasoning_bonus=0.1, top_percentile=0.7)
        # Response 0 has no reasoning, so response 1 is the shortest eligible one, and it is a top scorer.
        assert bonuses == approx([0.0, 0.1, 0.0])

    def test_answer_bonus_goes_to_the_shortest_answer_if_it_is_a_top_scorer(self) -> None:
        response_objs = [_response_obj(answer="a"), _response_obj(answer="a" * 4), _response_obj(answer="a" * 9)]
        scores, bonuses = self._apply([1.0, 0.5, 0.2], response_objs, answer_bonus=0.2)
        assert scores == approx([1.2, 0.5, 0.2])
        assert bonuses == approx([0.2, 0.0, 0.0])

    def test_group_reasoning_penalty_is_zero_centered(self) -> None:
        response_objs = [
            _response_obj(reasoning="r" * 1),
            _response_obj(reasoning="r" * 6),
            _response_obj(reasoning="r" * 11),
        ]
        scores, bonuses = self._apply([1.0, 1.0, 1.0], response_objs, group_reasoning_length_penalty_coeff=2.0)
        assert bonuses == approx([1.0, 0.0, -1.0])
        assert scores == approx([2.0, 1.0, 0.0])
        assert sum(bonuses) == approx(0.0)

    def test_group_answer_penalty(self) -> None:
        response_objs = [_response_obj(answer="a" * 2), _response_obj(answer="a" * 12)]
        scores, bonuses = self._apply([1.0, 1.0], response_objs, group_answer_length_penalty_coeff=1.0)
        assert bonuses == approx([0.5, -0.5])
        assert scores == approx([1.5, 0.5])


class TestAggregateScores:
    @staticmethod
    def _aggregate(comparison_results, comparison_metadata, num_responses, **overrides):
        kwargs = dict(
            comparison_results=comparison_results,
            comparison_metadata=comparison_metadata,
            response_objs=[_response_obj(answer=f"answer {i}") for i in range(num_responses)],
            aggregator_method="simple_tiebreaker",
            default_score=3.0,
            reasoning_bonus=0.0,
            answer_bonus=0.0,
            top_percentile=0.2,
            group_reasoning_length_penalty_coeff=0.0,
            group_answer_length_penalty_coeff=0.0,
        )
        kwargs.update(overrides)
        return aggregate_scores(**kwargs)

    def test_reward_is_the_bt_score_of_each_response(self) -> None:
        rewards, _, base_scores, bonuses = self._aggregate(
            comparison_results=[(3.0, 0.9), (3.0, 0.1), (3.0, 0.5)],
            comparison_metadata=[(0, 0), (1, 0), (2, 0)],
            num_responses=3,
        )
        assert rewards == approx([0.9, 0.1, 0.5])
        assert base_scores == approx([0.9, 0.1, 0.5])
        assert bonuses == [0.0, 0.0, 0.0]

    def test_scores_are_assigned_by_response_index_not_by_position(self) -> None:
        rewards, _, _, _ = self._aggregate(
            comparison_results=[(3.0, 0.2), (3.0, 0.7), (3.0, 0.4)],
            comparison_metadata=[(2, 0), (0, 0), (1, 0)],
            num_responses=3,
        )
        assert rewards == approx([0.7, 0.4, 0.2])

    def test_score_1_does_not_affect_the_reward(self) -> None:
        low, _, _, _ = self._aggregate([(1.0, 0.6), (1.0, 0.3)], [(0, 0), (1, 0)], num_responses=2)
        high, _, _, _ = self._aggregate([(5.0, 0.6), (-100.0, 0.3)], [(0, 0), (1, 0)], num_responses=2)
        assert low == approx(high)

    def test_multiple_judges_are_averaged(self) -> None:
        rewards, _, _, _ = self._aggregate(
            comparison_results=[(3.0, 0.8), (3.0, 0.4), (3.0, 0.2), (3.0, 0.6)],
            comparison_metadata=[(0, 0), (1, 0), (0, 1), (1, 1)],
            num_responses=2,
        )
        assert rewards == approx([0.5, 0.5])

    def test_metrics_describe_the_individual_scores(self) -> None:
        _, metrics, _, _ = self._aggregate([(3.0, 0.9), (3.0, 0.1)], [(0, 0), (1, 0)], num_responses=2)
        assert metrics["mean_individual_score"] == approx(0.5)
        assert metrics["std_individual_score"] == approx(0.4)

    def test_response_without_any_judgment_raises(self) -> None:
        with pytest.raises(AssertionError, match="DIAG357_VIOLATION"):
            self._aggregate([(3.0, 0.9)], [(0, 0)], num_responses=2)

    def test_unsupported_aggregator_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported aggregator_method"):
            self._aggregate([(3.0, 0.9)], [(0, 0)], num_responses=1, aggregator_method="mean")

    def test_length_bonus_settings_are_currently_ignored(self) -> None:
        # The length bonus call in aggregate_scores is commented out, so these settings have no effect even though
        # apply_length_bonuses itself works. If it is switched back on, this test should be replaced.
        response_objs = [_response_obj(reasoning="r" * 1, answer="a" * 1), _response_obj(reasoning="r" * 9)]
        rewards, _, _, bonuses = self._aggregate(
            [(3.0, 0.9), (3.0, 0.1)],
            [(0, 0), (1, 0)],
            num_responses=2,
            response_objs=response_objs,
            reasoning_bonus=1.0,
            answer_bonus=1.0,
            group_reasoning_length_penalty_coeff=1.0,
            group_answer_length_penalty_coeff=1.0,
        )
        assert rewards == approx([0.9, 0.1])
        assert bonuses == [0.0, 0.0]
