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
"""Tests for the DWRL pairwise utilities.

In DWRL pairwise the third value of a comparison result is the probability that the GenRM gives to "response 1 is
better", and the rewards are the Bradley-Terry (Elo) ratings fitted to all comparisons of a prompt. These tests pin
down that aggregation, the prompt keys that group rollouts into cohorts, and the parsing of the GenRM output.
"""

import math
import random

import numpy as np
import pytest
from pydantic import BaseModel
from pytest import approx

from resources_servers.dwrl_genrm_compare_pairwise.utils import (
    EMPTY_OUTPUT_PLACEHOLDER,
    GenRMOutputParseError,
    aggregate_scores,
    apply_length_bonuses,
    apply_style_penalties,
    compute_style_density,
    count_style_elements,
    extract_conversation_history,
    extract_from_response_obj,
    extract_output_text,
    generate_comparison_pairs,
    get_elo_scores,
    get_prompt_key,
    get_prompt_key_from_input,
    parse_genrm_output,
)


def _response_obj(text: str) -> dict:
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _aggregate(results, metadata, texts, **overrides):
    """aggregate_scores with all of the optional settings off. `results` are win probabilities."""
    kwargs = dict(
        aggregator_method="simple_tiebreaker",
        default_score=3.0,
        reasoning_bonus=0.0,
        answer_bonus=0.0,
        top_percentile=0.2,
        group_reasoning_length_penalty_coeff=0.0,
        group_answer_length_penalty_coeff=0.0,
    )
    kwargs.update(overrides)
    return aggregate_scores(
        comparison_results=[(3.0, 3.0, p) if not isinstance(p, tuple) else p for p in results],
        comparison_metadata=metadata,
        response_objs=[_response_obj(t) for t in texts],
        **kwargs,
    )


CIRCLE_4 = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)]


class TestGetPromptKeyFromInput:
    MESSAGES = [{"role": "user", "content": "What is 2+2?"}]

    def test_is_stable(self):
        assert get_prompt_key_from_input(self.MESSAGES) == get_prompt_key_from_input(list(self.MESSAGES))

    def test_depends_on_the_input(self):
        other = [{"role": "user", "content": "What is 3+3?"}]

        assert get_prompt_key_from_input(self.MESSAGES) != get_prompt_key_from_input(other)

    def test_depends_on_the_principle(self):
        assert get_prompt_key_from_input(self.MESSAGES, "a") != get_prompt_key_from_input(self.MESSAGES, "b")
        assert get_prompt_key_from_input(self.MESSAGES, "a") != get_prompt_key_from_input(self.MESSAGES, None)

    def test_depends_on_the_reference_answer(self):
        # Rollouts that are compared against different reference answers cannot share a cohort.
        with_a = get_prompt_key_from_input(self.MESSAGES, None, "answer a")
        with_b = get_prompt_key_from_input(self.MESSAGES, None, "answer b")

        assert len({with_a, with_b, get_prompt_key_from_input(self.MESSAGES)}) == 3

    def test_accepts_pydantic_messages(self):
        class Message(BaseModel):
            role: str
            content: str

        pydantic_key = get_prompt_key_from_input([Message(role="user", content="What is 2+2?")])

        assert pydantic_key == get_prompt_key_from_input(self.MESSAGES)

    def test_ignores_dict_key_order(self):
        reordered = [{"content": "What is 2+2?", "role": "user"}]

        assert get_prompt_key_from_input(reordered) == get_prompt_key_from_input(self.MESSAGES)


class TestGetPromptKey:
    def test_prompt_id_is_used_when_present(self):
        assert get_prompt_key({"prompt_id": 7}) == "7"

    def test_prompt_id_is_combined_with_the_principle(self):
        assert get_prompt_key({"prompt_id": "p", "principle": "Be brief"}) == "p::Be brief"

    def test_without_a_prompt_id_the_input_is_hashed(self):
        example = {"responses_create_params": {"input": [{"role": "user", "content": "hi"}]}, "principle": "x"}

        assert get_prompt_key(example) == get_prompt_key_from_input([{"role": "user", "content": "hi"}], "x")


class TestExtractConversationHistory:
    def test_returns_the_input(self):
        history = [{"role": "user", "content": "hi"}]

        assert extract_conversation_history({"responses_create_params": {"input": history}}) == history

    def test_missing_responses_create_params(self):
        with pytest.raises(ValueError, match="responses_create_params"):
            extract_conversation_history({"other": 1})

    def test_missing_input(self):
        with pytest.raises(ValueError, match="input"):
            extract_conversation_history({"responses_create_params": {}})


class TestGenerateComparisonPairs:
    def test_circular(self):
        assert generate_comparison_pairs("circular", 4) == [(0, 1), (1, 2), (2, 3), (3, 0)]

    def test_circular_of_two_compares_both_orders(self):
        assert generate_comparison_pairs("circular", 2) == [(0, 1), (1, 0)]

    def test_all_pairs(self):
        assert generate_comparison_pairs("all_pairs", 4) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown comparison strategy"):
            generate_comparison_pairs("round_robin", 4)

    @pytest.mark.parametrize("num_responses", [0, 1])
    def test_needs_two_responses(self, num_responses):
        with pytest.raises(ValueError, match="at least 2"):
            generate_comparison_pairs("circular", num_responses)


class TestParseGenrmOutput:
    def test_fenced_json(self):
        output = 'Analysis.\n```json\n{"score_1": 4, "score_2": 2, "ranking": 2}\n```'

        assert parse_genrm_output(output, 3.0, 0.5) == (4.0, 2.0, 2.0)

    def test_bare_json_with_surrounding_prose(self):
        output = 'I think {"score_1": 5, "score_2": 1, "ranking": 1} is right.'

        assert parse_genrm_output(output, 3.0, 0.5) == (5.0, 1.0, 1.0)

    def test_the_last_json_object_wins(self):
        output = '{"score_1": 1, "score_2": 1, "ranking": 3} then {"score_1": 4, "score_2": 5, "ranking": 6}'

        assert parse_genrm_output(output, 3.0, 0.5) == (4.0, 5.0, 6.0)

    def test_pretty_printed_json_with_long_text_fields(self):
        output = (
            "```json\n{\n"
            '  "response_1_analysis": "Uses {braces} and \\"quotes\\".",\n'
            '  "response_2_analysis": "Fine.",\n'
            '  "score_1": 3,\n  "score_2": 4,\n  "ranking": 4\n}\n```'
        )

        assert parse_genrm_output(output, 0.0, 0.0) == (3.0, 4.0, 4.0)

    def test_nested_overall(self):
        output = '{"rubric_evaluations": [], "overall": {"score_1": 2, "score_2": 3, "ranking": 5}}'

        assert parse_genrm_output(output, 0.0, 0.0) == (2.0, 3.0, 5.0)

    def test_missing_keys_take_the_defaults(self):
        assert parse_genrm_output('{"score_1": 4}', 3.0, 0.5) == (4.0, 3.0, 0.5)

    def test_json_without_any_expected_key_is_not_a_result(self):
        assert parse_genrm_output('{"other": 1}', 3.0, 0.5) == (3.0, 3.0, 0.5)

    def test_non_numeric_score_is_not_a_result(self):
        assert parse_genrm_output('{"score_1": "high", "score_2": 2, "ranking": 1}', 3.0, 0.5) == (3.0, 3.0, 0.5)

    def test_failure_returns_the_defaults_by_default(self):
        assert parse_genrm_output("no json here", 3.0, 0.5) == (3.0, 3.0, 0.5)

    def test_failure_can_raise(self):
        with pytest.raises(GenRMOutputParseError, match="No parseable JSON"):
            parse_genrm_output("no json here", 3.0, 0.5, raise_on_fail=True)

    def test_a_truncated_json_object_is_a_failure(self):
        # What a judge that ran out of tokens leaves behind.
        with pytest.raises(GenRMOutputParseError):
            parse_genrm_output('```json\n{"score_1": 4, "score_2"', 3.0, 0.5, raise_on_fail=True)

    def test_error_message_previews_long_output(self):
        with pytest.raises(GenRMOutputParseError) as exc:
            parse_genrm_output("x" * 500, 3.0, 0.5, raise_on_fail=True)

        message = str(exc.value)
        assert "No parseable JSON" in message
        assert "x" * 300 not in message  # Only a preview of the output is kept
        assert message.endswith("...")


class TestResponseObjectExtraction:
    def test_reasoning_and_output_text_are_separated(self):
        response_obj = {
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking "}, {"text": "more"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "Answer"}]},
            ]
        }

        assert extract_from_response_obj(response_obj) == ("thinking more", "Answer")

    def test_several_messages_are_concatenated(self):
        response_obj = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "A"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "B"}]},
            ]
        }

        assert extract_output_text(response_obj) == "AB"

    @pytest.mark.parametrize("response_obj", [None, "text", {}, {"output": None}, {"output": [None, 1, "x"]}])
    def test_malformed_response_objects_give_nothing(self, response_obj):
        assert extract_from_response_obj(response_obj) == ("", "")

    def test_non_text_content_is_ignored(self):
        response_obj = {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}

        assert extract_from_response_obj(response_obj) == ("", "")

    @pytest.mark.parametrize("response_obj", [{}, _response_obj(""), _response_obj("  \n ")])
    def test_empty_output_text_becomes_a_placeholder(self, response_obj):
        assert extract_output_text(response_obj) == EMPTY_OUTPUT_PLACEHOLDER == "None"


class TestGetEloScores:
    """`get_elo_scores` fits Bradley-Terry ratings to the fractional win counts of the pairs."""

    @staticmethod
    def _tables(n, comparisons):
        a_win = np.zeros([n, n])
        b_win = np.zeros([n, n])
        for i, j, p in comparisons:
            a_win[i, j] += p
            b_win[i, j] += 1.0 - p
        return a_win, b_win

    def test_two_responses_differ_by_the_log_odds(self):
        a_win, b_win = self._tables(2, [(0, 1, 0.8)])

        ratings = get_elo_scores(2, a_win, b_win)

        assert ratings[0] - ratings[1] == approx(_logit(0.8), abs=1e-3)

    def test_ratings_sum_to_zero(self):
        a_win, b_win = self._tables(4, [(0, 1, 0.9), (1, 2, 0.7), (2, 3, 0.6), (3, 0, 0.2)])

        assert get_elo_scores(4, a_win, b_win).sum() == approx(0.0, abs=1e-6)

    def test_a_tied_pair_has_equal_ratings(self):
        a_win, b_win = self._tables(2, [(0, 1, 0.5)])

        ratings = get_elo_scores(2, a_win, b_win)

        assert ratings[0] == approx(ratings[1], abs=1e-6)

    def test_the_reference_answer_is_the_zero_point(self):
        # Response 1 beats the reference (the last player) with probability 0.75.
        a_win, b_win = self._tables(3, [(0, 1, 0.9), (1, 2, 0.4), (0, 2, 0.75)])

        ratings = get_elo_scores(3, a_win, b_win, ref_answer_last=True)

        assert ratings[2] == approx(0.0, abs=1e-9)

    def test_scale_and_initial_rating(self):
        a_win, b_win = self._tables(2, [(0, 1, 0.8)])

        plain = get_elo_scores(2, a_win, b_win)
        scaled = get_elo_scores(2, a_win, b_win, scale=400.0, init_rating=1000.0)

        assert scaled == approx(400.0 * plain + 1000.0)


class TestAggregateScores:
    def test_a_pair_is_rated_by_the_win_probability(self):
        rewards, _, _, _ = _aggregate([0.8], [(0, 1, 0)], ["a", "b"])

        assert rewards[0] - rewards[1] == approx(_logit(0.8), abs=1e-3)
        assert rewards[0] == approx(-rewards[1], abs=1e-6)

    def test_the_better_response_gets_the_higher_reward(self):
        # 0 beats 1 beats 2 beats 3, and 3 loses to 0 as well.
        rewards, _, _, _ = _aggregate([0.9, 0.8, 0.7, 0.1], CIRCLE_4, ["a", "b", "c", "d"])

        assert rewards[0] > rewards[1] > rewards[2] > rewards[3]

    def test_rewards_sum_to_zero_without_a_reference(self):
        rewards, _, _, _ = _aggregate([0.9, 0.3, 0.6, 0.45], CIRCLE_4, ["a", "b", "c", "d"])

        assert sum(rewards) == approx(0.0, abs=1e-6)

    def test_the_order_of_a_pair_does_not_matter(self):
        # "1 beats 0 with probability 0.2" says the same as "0 beats 1 with probability 0.8".
        forward, _, _, _ = _aggregate([0.8], [(0, 1, 0)], ["a", "b"])
        backward, _, _, _ = _aggregate([0.2], [(1, 0, 0)], ["a", "b"])

        assert forward == approx(backward, abs=1e-6)

    def test_a_reward_depends_on_comparisons_that_do_not_involve_the_response(self):
        # The reason why rewards cannot be returned one rollout at a time.
        base = [0.7, 0.6, 0.55, 0.4]
        only_pair_2_3_differs = [0.7, 0.6, 0.9, 0.4]

        before, _, _, _ = _aggregate(base, CIRCLE_4, ["a", "b", "c", "d"])
        after, _, _, _ = _aggregate(only_pair_2_3_differs, CIRCLE_4, ["a", "b", "c", "d"])

        # Response 0 is in neither of the comparisons that changed (2 against 3), and its reward changes.
        assert after[0] != approx(before[0], abs=1e-3)

    def test_several_judges_count_as_their_mean_win_probability(self):
        two_judges, _, _, _ = _aggregate([0.9, 0.5], [(0, 1, 0), (0, 1, 1)], ["a", "b"])
        mean_judge, _, _, _ = _aggregate([0.7], [(0, 1, 0)], ["a", "b"])

        assert two_judges == approx(mean_judge, abs=1e-6)

    def test_the_1_to_5_scores_do_not_influence_the_rewards(self):
        low, _, _, _ = _aggregate([(1.0, 5.0, 0.8)], [(0, 1, 0)], ["a", "b"])
        high, _, _, _ = _aggregate([(5.0, 1.0, 0.8)], [(0, 1, 0)], ["a", "b"])

        assert low == approx(high)

    def test_the_result_does_not_depend_on_the_order_of_the_comparisons(self):
        results = [0.9, 0.3, 0.6, 0.45]
        shuffled = list(zip(results, CIRCLE_4))
        random.Random(3).shuffle(shuffled)

        in_order, _, _, _ = _aggregate(results, CIRCLE_4, ["a", "b", "c", "d"])
        out_of_order, _, _, _ = _aggregate(
            [p for p, _ in shuffled], [m for _, m in shuffled], ["a", "b", "c", "d"]
        )

        assert out_of_order == approx(in_order, abs=1e-6)

    def test_metrics_summarize_the_1_to_5_scores(self):
        _, metrics, _, _ = _aggregate(
            [(2.0, 4.0, 0.6), (4.0, 4.0, 0.5)], [(0, 1, 0), (1, 2, 0)], ["a", "b", "c"]
        )

        assert metrics["mean_individual_score"] == approx(np.mean([2.0, 4.0, 4.0, 4.0]))
        assert metrics["std_individual_score"] == approx(np.std([2.0, 4.0, 4.0, 4.0]))
        assert metrics["tiebreak_usage_rate"] == 0.0

    def test_unsupported_aggregator_method(self):
        with pytest.raises(ValueError, match="Unsupported aggregator_method"):
            _aggregate([0.8], [(0, 1, 0)], ["a", "b"], aggregator_method="mean")

    def test_a_response_without_comparisons_is_an_error(self):
        # Response 2 takes part in no comparison, so it has no rating.
        with pytest.raises(AssertionError, match="DIAG357_VIOLATION"):
            _aggregate([0.8], [(0, 1, 0)], ["a", "b", "c"])

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_a_win_probability_outside_0_to_1_breaks_the_fit(self):
        # What the old default_ranking of 3.5 (a 1-6 ranking) did to a comparison that could not be judged.
        # The fit not converging on such an input is expected, so its ConvergenceWarning is ignored here.
        with pytest.raises(AssertionError, match="ELO_VIOLATION"):
            _aggregate([0.8, 3.5, 0.7, 0.4], CIRCLE_4, ["a", "b", "c", "d"])

    def test_a_neutral_comparison_is_harmless(self):
        rewards, _, _, _ = _aggregate([0.7, 0.5, 0.55, 0.4], CIRCLE_4, ["a", "b", "c", "d"])

        assert all(abs(r) < 10 for r in rewards)

    def test_length_bonuses_are_disabled(self):
        # The length settings are accepted but have no effect on the rewards.
        plain, _, _, _ = _aggregate([0.8], [(0, 1, 0)], ["short", "a much longer answer than the other one"])
        with_bonuses, _, _, _ = _aggregate(
            [0.8],
            [(0, 1, 0)],
            ["short", "a much longer answer than the other one"],
            reasoning_bonus=1.0,
            answer_bonus=1.0,
            group_reasoning_length_penalty_coeff=1.0,
            group_answer_length_penalty_coeff=1.0,
        )

        assert with_bonuses == plain


class TestAggregateScoresWithReferenceAnswer:
    # Two rollouts, and the reference answer as the third "response".
    METADATA = [(0, 1, 0), (1, 0, 0), (0, 2, 0), (1, 2, 0)]

    def test_the_reference_is_not_rewarded(self):
        rewards, _, base_scores, bonuses = _aggregate(
            [0.6, 0.4, 0.8, 0.3], self.METADATA, ["a", "b", "reference"], ref_answer_last=True
        )

        assert len(rewards) == len(base_scores) == len(bonuses) == 2

    def test_a_rollout_is_rated_relative_to_the_reference(self):
        # Only comparisons against the reference: the reward is the log odds of beating it.
        rewards, _, _, _ = _aggregate(
            [0.8, 0.3], [(0, 2, 0), (1, 2, 0)], ["a", "b", "reference"], ref_answer_last=True
        )

        assert rewards[0] == approx(_logit(0.8), abs=1e-3)
        assert rewards[1] == approx(_logit(0.3), abs=1e-3)

    def test_rewards_are_not_centered_when_there_is_a_reference(self):
        rewards, _, _, _ = _aggregate(
            [0.8, 0.9], [(0, 2, 0), (1, 2, 0)], ["a", "b", "reference"], ref_answer_last=True
        )

        assert all(r > 0 for r in rewards)  # Both beat the reference

    def test_the_comparisons_between_rollouts_still_matter(self):
        wins, _, _, _ = _aggregate(
            [0.9, 0.1, 0.5, 0.5], self.METADATA, ["a", "b", "reference"], ref_answer_last=True
        )

        assert wins[0] > wins[1]

    def test_without_the_flag_the_reference_is_rewarded_like_any_response(self):
        rewards, _, _, _ = _aggregate([0.6, 0.4, 0.8, 0.3], self.METADATA, ["a", "b", "reference"])

        assert len(rewards) == 3


class TestStyleElements:
    def test_counts_headers_lists_bold_and_emoji(self):
        text = "# Title\n## Sub\n- one\n* two\n1. three\nSome **bold** and __also bold__ text 😀 🎉\n"

        assert count_style_elements(text) == {
            "header_count": 2,
            "list_count": 3,
            "bold_count": 2,
            "emoji_count": 2,
        }

    def test_code_blocks_are_not_counted(self):
        text = "```\n# not a header\n- not a list\n**not bold**\n```\nplain"

        assert sum(count_style_elements(text).values()) == 0

    def test_plain_text_has_no_style_elements(self):
        assert sum(count_style_elements("Just a sentence.").values()) == 0

    def test_density_is_elements_per_character(self):
        text = "# Title\ntext"  # 1 header in 12 characters

        assert compute_style_density(text) == approx(1 / 12)

    def test_density_of_empty_text_is_zero(self):
        assert compute_style_density("") == 0.0
        assert compute_style_density("   ") == 0.0

    def test_density_ignores_code_blocks_in_the_length_too(self):
        with_code = "# Title\n```\n" + "x" * 1000 + "\n```\n"

        assert compute_style_density(with_code) == approx(compute_style_density("# Title"))


class TestApplyStylePenalties:
    PLAIN = "A plain answer without any formatting at all."
    FANCY = "# Answer\n- **one**\n- **two**\n- **three**\n"

    def test_the_more_formatted_response_is_penalized(self):
        scores, adjustments = apply_style_penalties(
            [1.0, 1.0], [_response_obj(self.PLAIN), _response_obj(self.FANCY)], group_style_penalty_coeff=0.5
        )

        assert scores[0] > 1.0 > scores[1]
        assert adjustments[0] > 0 > adjustments[1]

    def test_adjustments_are_centered(self):
        _, adjustments = apply_style_penalties(
            [0.0, 0.0, 0.0],
            [_response_obj(self.PLAIN), _response_obj(self.FANCY), _response_obj("- a\n- b")],
            group_style_penalty_coeff=1.0,
        )

        assert sum(adjustments) == approx(0.0, abs=1e-9)

    def test_the_adjustment_scales_with_the_coefficient(self):
        responses = [_response_obj(self.PLAIN), _response_obj(self.FANCY)]
        _, small = apply_style_penalties([0.0, 0.0], responses, group_style_penalty_coeff=0.1)
        _, large = apply_style_penalties([0.0, 0.0], responses, group_style_penalty_coeff=0.3)

        assert large == approx([3 * a for a in small])

    def test_equal_density_changes_nothing(self):
        scores, adjustments = apply_style_penalties(
            [1.0, 2.0], [_response_obj(self.PLAIN), _response_obj(self.PLAIN)], group_style_penalty_coeff=1.0
        )

        assert scores == [1.0, 2.0]
        assert adjustments == [0.0, 0.0]

    @pytest.mark.parametrize("coeff", [0.0, -1.0])
    def test_no_penalty_without_a_positive_coefficient(self, coeff):
        scores, adjustments = apply_style_penalties(
            [1.0, 1.0], [_response_obj(self.PLAIN), _response_obj(self.FANCY)], group_style_penalty_coeff=coeff
        )

        assert scores == [1.0, 1.0]
        assert adjustments == [0.0, 0.0]

    def test_a_single_response_is_not_adjusted(self):
        scores, adjustments = apply_style_penalties(
            [1.0], [_response_obj(self.FANCY)], group_style_penalty_coeff=1.0
        )

        assert scores == [1.0]
        assert adjustments == [0.0]


class TestAggregateScoresWithStylePenalty:
    TEXTS = ["A plain answer without formatting.", "# Answer\n- **one**\n- **two**\n", "Another plain one."]
    METADATA = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    RESULTS = [0.5, 0.5, 0.5]  # The judge sees no difference

    def test_the_penalty_is_off_by_default(self):
        default, _, _, _ = _aggregate(self.RESULTS, self.METADATA, self.TEXTS)
        explicit, _, _, _ = _aggregate(self.RESULTS, self.METADATA, self.TEXTS, group_style_penalty_coeff=0.0)

        assert default == explicit

    def test_a_formatted_response_is_rewarded_less(self):
        rewards, _, _, _ = _aggregate(self.RESULTS, self.METADATA, self.TEXTS, group_style_penalty_coeff=0.5)

        assert rewards[1] < rewards[0]
        assert rewards[1] < rewards[2]

    def test_base_scores_are_the_ratings_before_the_penalty(self):
        plain, _, _, _ = _aggregate(self.RESULTS, self.METADATA, self.TEXTS)
        rewards, _, base_scores, bonuses = _aggregate(
            self.RESULTS, self.METADATA, self.TEXTS, group_style_penalty_coeff=0.5
        )

        assert base_scores == approx(plain)
        assert rewards == approx([b + a for b, a in zip(base_scores, bonuses)])
        assert bonuses[1] < 0

    def test_the_reference_answer_is_left_out_of_the_penalty(self):
        results = [(3.0, 3.0, 0.7), (3.0, 3.0, 0.6)]
        rewards, _, base_scores, bonuses = _aggregate(
            results,
            [(0, 2, 0), (1, 2, 0)],
            ["A plain answer.", "# Answer\n- **one**\n- **two**\n", "The reference answer."],
            group_style_penalty_coeff=0.5,
            ref_answer_last=True,
        )

        assert len(rewards) == 2
        assert bonuses[0] > 0 > bonuses[1]
        assert sum(bonuses) == approx(0.0, abs=1e-9)
        assert rewards == approx([b + a for b, a in zip(base_scores, bonuses)])


class TestApplyLengthBonuses:
    """The length bonuses are not used by aggregate_scores at the moment, but the function is kept working."""

    LONG, SHORT = "a long answer " * 10, "short"

    def test_shortest_answer_among_the_top_scorers_gets_the_bonus(self):
        scores, bonuses = apply_length_bonuses(
            [1.0, 1.0],
            [_response_obj(self.LONG), _response_obj(self.SHORT)],
            reasoning_bonus=0.0,
            answer_bonus=0.5,
            top_percentile=1.0,
            group_reasoning_length_penalty_coeff=0.0,
            group_answer_length_penalty_coeff=0.0,
        )

        assert bonuses == [0.0, 0.5]
        assert scores == [1.0, 1.5]

    def test_group_relative_penalty_is_centered_and_favors_short_answers(self):
        _, bonuses = apply_length_bonuses(
            [0.0, 0.0],
            [_response_obj(self.LONG), _response_obj(self.SHORT)],
            reasoning_bonus=0.0,
            answer_bonus=0.0,
            top_percentile=0.2,
            group_reasoning_length_penalty_coeff=0.0,
            group_answer_length_penalty_coeff=1.0,
        )

        assert bonuses == approx([-0.5, 0.5])

    def test_a_single_response_is_not_adjusted(self):
        scores, bonuses = apply_length_bonuses(
            [1.0],
            [_response_obj(self.SHORT)],
            reasoning_bonus=1.0,
            answer_bonus=1.0,
            top_percentile=1.0,
            group_reasoning_length_penalty_coeff=1.0,
            group_answer_length_penalty_coeff=1.0,
        )

        assert scores == [1.0]
        assert bonuses == [0.0]
