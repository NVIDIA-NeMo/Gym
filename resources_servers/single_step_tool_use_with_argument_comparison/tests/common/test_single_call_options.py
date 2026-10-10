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
"""Tests for the comparator options that apply only to a single expected call matched against a single
actual call: list F1, strong list guardrails, argument filters, per-argument overrides and quote handling."""

import json
import logging

from pytest import approx, fixture

from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ActionComparator,
    FunctionCallAction,
    FunctionCallBatchAction,
    ParallelToolCallRewardMode,
    StepRewardCategory,
    ToolCallArgumentComparisonOverride,
    ToolCallArgumentFilter,
    ToolCallComparatorConfig,
)


def _call(name: str, arguments: dict) -> FunctionCallAction:
    return FunctionCallAction(type="function_call", name=name, arguments=json.dumps(arguments))


def _comparator(**config: object) -> ActionComparator:
    return ActionComparator(config=ToolCallComparatorConfig(**config))


class TestSingleCallOnly:
    """The single-call options must not change how a parallel batch is scored."""

    def test_without_single_call_options_keeps_only_base_and_parallel_settings(self) -> None:
        config = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.4,
            floating_point_comparison_threshold=1e-3,
            parallel_tool_call_rewarding=True,
            allow_subset=True,
            allow_superset=True,
            parallel_tool_call_reward_mode=ParallelToolCallRewardMode.F1,
            use_f1_for_list=True,
            use_strong_list_reward=True,
            keep_quotes=True,
            argument_filters={"search": ToolCallArgumentFilter(included_argument_names=[])},
            argument_comparison_overrides={"search": {"queries": ToolCallArgumentComparisonOverride()}},
            validate_against_declared_tool_schema=True,
        )
        expected = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.4,
            floating_point_comparison_threshold=1e-3,
            parallel_tool_call_rewarding=True,
            allow_subset=True,
            allow_superset=True,
            parallel_tool_call_reward_mode=ParallelToolCallRewardMode.F1,
        )
        assert config.without_single_call_options() == expected

    def test_single_call_uses_list_f1(self) -> None:
        comparator = _comparator(
            word_count_similarity_threshold=0.3, use_f1_for_list=True, use_list_f1_threshold=False
        )
        result = comparator.compare_action(_call("search", {"items": [1, 2, 3]}), _call("search", {"items": [1, 2]}))
        assert result.reward == approx(0.8)
        assert result.category == StepRewardCategory.ARGUMENT_LIST_F1_PARTIAL
        assert len(comparator.list_f1_match_details) == 1

    def test_batch_ignores_list_f1(self) -> None:
        comparator = _comparator(
            word_count_similarity_threshold=0.3, use_f1_for_list=True, use_list_f1_threshold=False
        )
        expected = FunctionCallBatchAction(
            type="function_call_batch",
            calls=[_call("search", {"items": [1, 2, 3]}), _call("lookup", {"id": 9})],
        )
        actual = FunctionCallBatchAction(
            type="function_call_batch",
            calls=[_call("search", {"items": [1, 2]}), _call("lookup", {"id": 9})],
        )
        result = comparator.compare_action(expected, actual)
        # Without list F1 the shorter list is a plain length mismatch, so one of two calls matches.
        assert result.reward == 0.0
        assert result.category == StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT
        assert comparator.list_f1_match_details == []

    def test_batch_ignores_argument_filters(self) -> None:
        comparator = _comparator(
            word_count_similarity_threshold=0.3,
            argument_filters={"search": ToolCallArgumentFilter(included_argument_names=[])},
        )
        expected = FunctionCallBatchAction(
            type="function_call_batch",
            calls=[_call("search", {"query": "alpha"}), _call("search", {"query": "beta"})],
        )
        actual = FunctionCallBatchAction(
            type="function_call_batch",
            calls=[_call("search", {"query": "alpha"}), _call("search", {"query": "gamma"})],
        )
        assert comparator.compare_action(expected, actual).reward == 0.0

        # The same filter makes a single call score on its tool name alone.
        single = comparator.compare_action(_call("search", {"query": "beta"}), _call("search", {"query": "gamma"}))
        assert single.reward == 1.0
        assert single.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_batch_ignores_keep_quotes(self) -> None:
        comparator = _comparator(word_count_similarity_threshold=0.3, keep_quotes=True)
        quoted, unquoted = '"a b c d e"', "a b c d e"
        expected = FunctionCallBatchAction(
            type="function_call_batch",
            calls=[_call("search", {"query": quoted}), _call("lookup", {"id": 1})],
        )
        actual = FunctionCallBatchAction(
            type="function_call_batch",
            calls=[_call("search", {"query": unquoted}), _call("lookup", {"id": 1})],
        )
        # Plain whitespace split: 3 shared words of 10, which meets the 0.3 threshold.
        assert comparator.compare_action(expected, actual).reward == 1.0

        # With keep_quotes the quoted phrase is one token, so the single call fails.
        single = comparator.compare_action(_call("search", {"query": quoted}), _call("search", {"query": unquoted}))
        assert single.reward == 0.0

    def test_goal_argument_is_compared_by_default(self) -> None:
        comparator = _comparator(word_count_similarity_threshold=0.3)
        result = comparator.compare_action(
            _call("browse", {"urls": ["https://example.com/a"], "goal": "find the founding year"}),
            _call("browse", {"urls": ["https://example.com/a"], "goal": "zzz"}),
        )
        assert result.reward == 0.0
        assert result.category == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT


class TestArgumentFilter:
    def test_apply_argument_filter(self) -> None:
        first_string_value = "first"
        list_value = ["element1", "element2", "element3"]
        single_item_dictionary_value = {"key4": "value4"}
        two_items_dictionary_value = {"first_key": "first_value", "second_key": "second_value"}
        four_items_dictionary_value = {"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4"}

        empty_filter = ToolCallArgumentFilter(included_argument_names=None)
        for value in (
            first_string_value,
            list_value,
            single_item_dictionary_value,
            two_items_dictionary_value,
            four_items_dictionary_value,
        ):
            assert ActionComparator._apply_argument_filter(empty_filter, value) is value

        included_arguments_filter = ToolCallArgumentFilter(
            included_argument_names=["second", "element3", "key1", "key4"]
        )
        assert (
            ActionComparator._apply_argument_filter(included_arguments_filter, first_string_value)
            is first_string_value
        )
        assert ActionComparator._apply_argument_filter(included_arguments_filter, list_value) is list_value
        assert (
            ActionComparator._apply_argument_filter(included_arguments_filter, single_item_dictionary_value)
            == single_item_dictionary_value
        )
        assert ActionComparator._apply_argument_filter(included_arguments_filter, two_items_dictionary_value) == {}
        assert ActionComparator._apply_argument_filter(included_arguments_filter, four_items_dictionary_value) == {
            "key1": "value1",
            "key4": "value4",
        }


class TestListF1PartialReward:
    """List F1 with `use_list_f1_threshold=False` returns the F1 score itself."""

    @fixture
    def comparator(self) -> ActionComparator:
        return _comparator(word_count_similarity_threshold=0.3, use_f1_for_list=True, use_list_f1_threshold=False)

    @fixture
    def comparator_with_threshold(self) -> ActionComparator:
        return _comparator(
            word_count_similarity_threshold=0.3,
            use_f1_for_list=True,
            use_list_f1_threshold=True,
            list_f1_threshold=0.5,
        )

    def test_exact_list_match_returns_1(self, comparator: ActionComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(["a", "b", "c"], ["a", "b", "c"])
        assert score == 1.0
        assert category is None
        assert len(comparator.list_f1_match_details) == 1
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 3
        assert detail.f1 == 1.0
        assert detail.matched_pairs == [(0, 0), (1, 1), (2, 2)]
        assert detail.score_matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def test_partial_list_match_returns_f1(self, comparator: ActionComparator) -> None:
        # tp=2, precision=2/2, recall=2/3, f1=0.8
        score, category = comparator.compare_tool_call_arguments([1, 2, 3], [1, 2])
        assert score == approx(0.8)
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_PARTIAL
        detail = comparator.list_f1_match_details[0]
        assert detail.expected_values == [1, 2, 3]
        assert detail.actual_values == [1, 2]
        assert detail.tp == 2
        assert detail.precision == 1.0
        assert detail.recall == approx(2 / 3)
        assert detail.f1 == approx(0.8)
        assert detail.matched_pairs == [(0, 0), (1, 1)]
        assert detail.score_matrix == [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]

    def test_no_list_match_returns_0(self, comparator: ActionComparator) -> None:
        score, category = comparator.compare_tool_call_arguments([1, 2, 3], [4, 5, 6])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 0
        assert detail.f1 == 0.0
        assert detail.matched_pairs == []

    def test_empty_lists_return_1(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments([], []) == (1.0, None)
        assert comparator.list_f1_match_details == []

    def test_one_empty_list_returns_0(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments([1, 2], []) == (
            0.0,
            StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD,
        )
        assert comparator.list_f1_match_details == []

    def test_threshold_mode_returns_binary(self, comparator_with_threshold: ActionComparator) -> None:
        # f1=0.8 meets the 0.5 threshold.
        assert comparator_with_threshold.compare_tool_call_arguments([1, 2, 3], [1, 2]) == (1.0, None)
        assert comparator_with_threshold.list_f1_match_details[0].f1 == approx(0.8)

    def test_threshold_mode_below_threshold_returns_0(self, comparator_with_threshold: ActionComparator) -> None:
        # tp=1, precision=1, recall=0.2, f1=0.333 < 0.5
        assert comparator_with_threshold.compare_tool_call_arguments([1, 2, 3, 4, 5], [1]) == (
            0.0,
            StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD,
        )

    def test_partial_f1_propagates_through_compare_tool_call(self, comparator: ActionComparator) -> None:
        result = comparator.compare_tool_call(
            _call("search", {"items": [1, 2, 3]}), _call("search", {"items": [1, 2]})
        )
        assert result.reward == approx(0.8)
        assert result.category == StepRewardCategory.ARGUMENT_LIST_F1_PARTIAL
        assert comparator.list_f1_match_details[0].matched_pairs == [(0, 0), (1, 1)]


class TestStrongListReward:
    @fixture
    def comparator(self) -> ActionComparator:
        return _comparator(
            word_count_similarity_threshold=0.5,
            use_f1_for_list=True,
            use_strong_list_reward=True,
            list_f1_threshold=0.8,
            list_f1_min_precision=1.0,
            list_f1_min_recall=1.0,
            list_f1_max_actual_to_expected_ratio=1.0,
        )

    def test_exact_list_match_returns_1(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments(["a", "b", "c"], ["a", "b", "c"]) == (1.0, None)
        assert comparator.list_f1_match_details[0].strong_match_failure_reasons == []

    def test_partial_list_match_returns_0_even_when_f1_is_high(self, comparator: ActionComparator) -> None:
        score, category = comparator.compare_tool_call_arguments([1, 2, 3], [1, 2])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert detail.f1 == approx(0.8)
        assert "recall_below_floor" in detail.strong_match_failure_reasons

    def test_duplicate_padding_returns_0(self, comparator: ActionComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(["a", "b"], ["a", "b", "a"])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert detail.actual_duplicate_count == 1
        assert "duplicate_actual_values" in detail.strong_match_failure_reasons

    def test_overbroad_list_returns_0(self, comparator: ActionComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(["a"], ["a", "b"])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert detail.actual_to_expected_ratio == 2.0
        assert "actual_list_overbroad" in detail.strong_match_failure_reasons

    def test_fuzzy_query_element_match_returns_0(self, comparator: ActionComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(["alpha beta gamma delta"], ["alpha beta"])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 0
        assert "f1_below_threshold" in detail.strong_match_failure_reasons

    def test_ambiguous_list_match_uses_optimal_matching(self) -> None:
        comparator = _comparator(
            word_count_similarity_threshold=0.4,
            use_f1_for_list=True,
            use_strong_list_reward=True,
            list_f1_threshold=0.8,
            list_f1_min_precision=1.0,
            list_f1_min_recall=1.0,
            list_f1_max_actual_to_expected_ratio=1.0,
        )
        score, category = comparator.compare_tool_call_arguments(
            ["alpha beta", "alpha beta gamma delta"],
            ["alpha beta gamma", "alpha beta"],
        )
        assert (score, category) == (1.0, None)
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 2
        assert detail.matched_pairs == [(0, 1), (1, 0)]


class TestToolSpecificStrongListReward:
    THREE_QUERIES = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]

    @fixture
    def comparator(self) -> ActionComparator:
        return _comparator(
            word_count_similarity_threshold=0.5,
            use_f1_for_list=True,
            use_strong_list_reward=True,
            list_f1_threshold=0.8,
            list_f1_min_precision=1.0,
            list_f1_min_recall=1.0,
            list_f1_max_actual_to_expected_ratio=1.0,
            keep_quotes=True,
            argument_comparison_overrides={
                "search": {
                    "queries": ToolCallArgumentComparisonOverride(
                        word_count_similarity_threshold=0.25,
                        word_count_min_precision=0.5,
                        word_count_min_recall=0.3,
                        word_count_max_actual_to_expected_ratio=1.5,
                        word_count_max_unmatched_actual_words=5,
                        list_f1_relaxed_min_expected_len=3,
                        list_f1_max_unmatched_expected=1,
                        list_f1_max_unmatched_actual=1,
                        keep_quotes=True,
                        coerce_actual_string_to_list=True,
                        coerce_actual_string_to_singleton_list=True,
                        validate_list_item_schema=True,
                        reject_empty_string_list_items=True,
                    )
                }
            },
        )

    def _search(self, comparator: ActionComparator, expected_queries: object, actual_queries: object):
        return comparator.compare_tool_call(
            _call("search", {"queries": expected_queries}), _call("search", {"queries": actual_queries})
        )

    def test_search_query_uses_relaxed_element_match(self, comparator: ActionComparator) -> None:
        result = self._search(
            comparator,
            ["Hawthorn Trophy bonus points Motorsport Ireland"],
            ["Hawthorn Trophy bonus points table Motorsport Ireland"],
        )
        assert (result.reward, result.category) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)

    def test_search_query_rejects_broad_element_padding(self, comparator: ActionComparator) -> None:
        result = self._search(
            comparator,
            ["Hawthorn Trophy bonus points Motorsport Ireland"],
            [
                "Hawthorn Trophy bonus points Motorsport Ireland official source "
                "wikipedia reddit news pdf overview details"
            ],
        )
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH)

    def test_search_query_json_string_list_is_coerced(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, ["alpha beta gamma"], json.dumps(["alpha beta gamma"]))
        assert (result.reward, result.category) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)

    def test_search_query_plain_string_is_coerced_for_singleton_gold(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, ["alpha beta gamma"], "alpha beta gamma")
        assert (result.reward, result.category) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)

    def test_search_query_plain_string_is_not_coerced_for_multi_query_gold(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, ["alpha beta gamma", "delta epsilon zeta"], "alpha beta gamma")
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT)

    def test_malformed_singleton_list_string_is_coerced(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, ["PayPal proxy Jamie Miller compensation"], "[PayPal proxy Jamie Miller]")
        assert (result.reward, result.category) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)

    def test_search_query_still_rejects_missing_list_items(self, comparator: ActionComparator) -> None:
        result = self._search(
            comparator,
            ["Hawthorn Trophy bonus points Motorsport Ireland", "Dunlop Hawthorn Trophy bonus points starters"],
            ["Hawthorn Trophy bonus points table Motorsport Ireland"],
        )
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH)
        assert "recall_below_floor" in comparator.list_f1_match_details[0].strong_match_failure_reasons

    def test_search_query_allows_one_missing_item_for_three_or_more_gold(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, self.THREE_QUERIES, self.THREE_QUERIES[:2])
        assert (result.reward, result.category) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 2
        assert detail.strong_match_failure_reasons == []

    def test_search_query_allows_one_extra_item_for_three_or_more_gold(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, self.THREE_QUERIES, self.THREE_QUERIES + ["unrelated reward check"])
        assert (result.reward, result.category) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 3
        assert detail.strong_match_failure_reasons == []

    def test_search_query_rejects_non_string_extra_even_when_one_extra_allowed(
        self, comparator: ActionComparator
    ) -> None:
        result = self._search(comparator, self.THREE_QUERIES, self.THREE_QUERIES + [123])
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_LIST_SCHEMA_MALFORMED)

    def test_search_query_rejects_empty_extra_even_when_one_extra_allowed(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, self.THREE_QUERIES, self.THREE_QUERIES + ["  "])
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_LIST_SCHEMA_MALFORMED)

    def test_search_query_allows_one_substitution_for_three_or_more_gold(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, self.THREE_QUERIES, self.THREE_QUERIES[:2] + ["unrelated reward check"])
        assert (result.reward, result.category) == (1.0, StepRewardCategory.EXPECTED_TOOL_CALL)
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 2
        assert detail.strong_match_failure_reasons == []

    def test_search_query_rejects_two_missing_items_for_three_or_more_gold(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, self.THREE_QUERIES, self.THREE_QUERIES[:1])
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH)
        assert "unmatched_expected_above_limit" in comparator.list_f1_match_details[0].strong_match_failure_reasons

    def test_search_query_rejects_two_extra_items_for_three_or_more_gold(self, comparator: ActionComparator) -> None:
        result = self._search(
            comparator, self.THREE_QUERIES, self.THREE_QUERIES + ["unrelated reward check", "another unrelated probe"]
        )
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH)
        assert "unmatched_actual_above_limit" in comparator.list_f1_match_details[0].strong_match_failure_reasons

    def test_search_query_rejects_duplicate_under_relaxed_list_rule(self, comparator: ActionComparator) -> None:
        result = self._search(comparator, self.THREE_QUERIES, self.THREE_QUERIES[:2] + [self.THREE_QUERIES[0]])
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH)
        assert "duplicate_actual_values" in comparator.list_f1_match_details[0].strong_match_failure_reasons

    def test_browse_url_does_not_use_search_query_override(self, comparator: ActionComparator) -> None:
        result = comparator.compare_tool_call(
            _call("browse", {"urls": ["https://example.com/alpha-beta-gamma"]}),
            _call("browse", {"urls": ["https://example.com/alpha-beta"]}),
        )
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH)
        assert comparator.list_f1_match_details[0].tp == 0

    def test_browse_url_does_not_use_search_query_list_relaxation(self, comparator: ActionComparator) -> None:
        urls = ["https://example.com/one", "https://example.com/two", "https://example.com/three"]
        result = comparator.compare_tool_call(_call("browse", {"urls": urls}), _call("browse", {"urls": urls[:2]}))
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH)
        assert "recall_below_floor" in comparator.list_f1_match_details[0].strong_match_failure_reasons


class TestStringInsteadOfList:
    """Model outputs where the `queries` list argument arrived as one JSON-looking string."""

    @fixture
    def comparator(self) -> ActionComparator:
        return _comparator(word_count_similarity_threshold=0.3, use_f1_for_list=True, use_list_f1_threshold=False)

    def test_quoted_phrases_in_a_string(self, comparator: ActionComparator) -> None:
        result = comparator.compare_tool_call(
            _call("search", {"queries": ['"Environmental Modeling Prediction" "Figure A" NOSIA']}),
            _call("search", {"queries": '["NOSIA-II" "Weather Ready Nation" Mission Service Areas]'}),
        )
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT)
        assert comparator.list_f1_match_details == []

    def test_mixed_quoted_and_bare_words_in_a_string(self, comparator: ActionComparator) -> None:
        result = comparator.compare_tool_call(
            _call("search", {"queries": ["non-original screenplay wga credit rules"]}),
            _call("search", {"queries": '["production executive" "non-original" screenplay credit percentage WGA]'}),
        )
        assert (result.reward, result.category) == (0.0, StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT)
        assert comparator.list_f1_match_details == []


class TestKeepQuotes:
    """With keep_quotes, a run inside matching quotes is one token, so a quoted phrase is distinct from
    the same words unquoted. Search queries use quotes to ask for an exact phrase."""

    @fixture
    def comparator(self) -> ActionComparator:
        return _comparator(word_count_similarity_threshold=0.3, keep_quotes=True)

    @fixture
    def comparator_default(self) -> ActionComparator:
        return _comparator(word_count_similarity_threshold=0.3)

    def test_default_keep_quotes_is_false(self) -> None:
        assert ToolCallComparatorConfig(word_count_similarity_threshold=0.3).keep_quotes is False

    def test_keep_quotes_flag_changes_tokenization(self, comparator: ActionComparator, comparator_default) -> None:
        # Off: ['"a', 'b', 'c', 'd', 'e"'] vs ['a', 'b', 'c', 'd', 'e'] share 3 of 10 words, which meets 0.3.
        # On: ['"a b c d e"'] is one token, so the exact-string rule applies and fails.
        assert comparator_default.compare_tool_call_arguments('"a b c d e"', "a b c d e") == (1.0, None)
        assert comparator.compare_tool_call_arguments('"a b c d e"', "a b c d e") == (
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )

    def test_baseline_quoted_vs_unquoted_without_flag(self, comparator_default: ActionComparator) -> None:
        # ['"hello', 'world"', 'zumba'] vs ['hello', 'world', 'zumba'] share 1 of 6 words.
        assert comparator_default.compare_tool_call_arguments('"hello world" zumba', "hello world zumba") == (
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )

    def test_quoted_phrase_matches_same_quoted_phrase(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments('"hello world" zumba', '"hello world" zumba') == (1.0, None)

    def test_quoted_phrase_order_independent(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments('"hello world" zumba', 'zumba "hello world"') == (1.0, None)

    def test_quoted_phrase_distinct_from_unquoted_same_words(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments('"hello world" zumba', "hello world zumba") == (
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )

    def test_inner_words_of_quoted_phrase_not_matched_separately(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments('"hello world"', "hello zumba world") == (
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )

    def test_multiple_quoted_phrases_order_independent(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments('"new york" "los angeles"', '"los angeles" "new york"') == (
            1.0,
            None,
        )

    def test_unbalanced_quote_strips_and_warns(self, comparator: ActionComparator, caplog) -> None:
        caplog.set_level(logging.WARNING)
        assert comparator.compare_tool_call_arguments('"hello world zumba', "hello world zumba") == (1.0, None)
        assert any("unbalanced" in record.message.lower() for record in caplog.records)

    def test_single_quoted_short_string_requires_exact_match(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments('"hi"', "hi") == (
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )

    def test_quoted_phrase_case_insensitive(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments('"Hello World" Zumba', '"hello world" zumba') == (1.0, None)

    def test_single_quote_chars_supported(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments("'hello world' zumba", "zumba 'hello world'") == (1.0, None)

    def test_mixed_quote_types_supported(self, comparator: ActionComparator) -> None:
        assert comparator.compare_tool_call_arguments("\"hi there\" 'bye now'", "'bye now' \"hi there\"") == (
            1.0,
            None,
        )

    def test_quoted_strings_inside_f1_list(self) -> None:
        comparator = _comparator(
            word_count_similarity_threshold=0.3,
            use_f1_for_list=True,
            use_list_f1_threshold=True,
            list_f1_threshold=0.5,
            keep_quotes=True,
        )
        assert comparator.compare_tool_call_arguments(
            ['"hello world" zumba', "other query here"],
            ["other query here", '"hello world" zumba'],
        ) == (1.0, None)
