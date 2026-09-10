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
import logging

from pytest import fixture

from nemo_gym.openai_utils import NeMoGymResponseFunctionToolCall
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedFunctionCall,
    StepRewardCategory,
    ToolCallArgumentComparisonOverride,
    ToolCallArgumentFilter,
    ToolCallComparator,
    ToolCallComparatorConfig,
)


class TestToolCallComparator:
    @fixture
    def tool_call_comparator(self) -> ToolCallComparator:
        comparator_config = ToolCallComparatorConfig(word_count_similarity_threshold=0.1)
        return ToolCallComparator(config=comparator_config)

    # TODO: test assertions below use True/False and 0.0/1.0 inconsistently due to return type
    # change from bool to float. Python's True == 1.0 and False == 0.0 so they still pass.
    def test_compare_tool_call(self, tool_call_comparator: ToolCallComparator) -> None:
        arguments_object = {
            "first": "one",
            "second": 2,
            "third": True,
            "fourth": [1, "element2"],
            "fifth": {
                "inner1": "value1",
                "inner2": False,
            },
        }
        arguments_string = json.dumps(arguments_object)
        expected_function_call = ExpectedFunctionCall(
            type="function_call",
            name="send",
            arguments=arguments_string,
        )

        different_tool_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="different_tool",
            name="receive",
            arguments=arguments_string,
        )
        assert tool_call_comparator.compare_tool_call(expected_function_call, different_tool_tool_call) == (
            0.0,
            StepRewardCategory.UNEXPECTED_TOOL,
        )

        invalid_arguments_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="invalid_arguments",
            name="send",
            arguments="first=one",
        )
        assert tool_call_comparator.compare_tool_call(expected_function_call, invalid_arguments_tool_call) == (
            0.0,
            StepRewardCategory.ARGUMENTS_DECODE_ERROR,
        )

        matching_arguments_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="matching_arguments",
            name="send",
            arguments=arguments_string,
        )
        assert tool_call_comparator.compare_tool_call(expected_function_call, matching_arguments_tool_call) == (
            1.0,
            StepRewardCategory.EXPECTED_TOOL_CALL,
        )

        different_argument_value_object = {
            "first": "one",
            "second": 2,
            "third": True,
            "fourth": [1, "element3"],
            "fifth": {
                "inner1": "value1",
                "inner2": False,
            },
        }
        different_argument_value_string = json.dumps(different_argument_value_object)
        different_argument_value_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="different_argument_value",
            name="send",
            arguments=different_argument_value_string,
        )
        assert tool_call_comparator.compare_tool_call(expected_function_call, different_argument_value_tool_call) == (
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )

        different_argument_key_object = {
            "first": "one",
            "second": 2,
            "third": True,
            "fourth": [1, "element2"],
            "fifth": {
                "inner": "value1",
                "inner2": False,
            },
        }
        different_argument_key_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="different_argument_key",
            name="send",
            arguments=json.dumps(different_argument_key_object),
        )
        assert tool_call_comparator.compare_tool_call(expected_function_call, different_argument_key_tool_call) == (
            0.0,
            StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT,
        )

        tool_call_comparator.config.argument_filters = {
            "send": ToolCallArgumentFilter(
                included_argument_names=["first", "second", "third", "fifth"],
            )
        }
        assert tool_call_comparator.compare_tool_call(expected_function_call, different_argument_value_tool_call) == (
            1.0,
            StepRewardCategory.EXPECTED_TOOL_CALL,
        )

        receive_expected_function_call = ExpectedFunctionCall(
            type="function_call",
            name="receive",
            arguments=arguments_string,
        )
        receive_different_argument_value_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="receive_different_argument_value",
            name="receive",
            arguments=different_argument_value_string,
        )
        assert tool_call_comparator.compare_tool_call(
            receive_expected_function_call, receive_different_argument_value_tool_call
        ) == (
            0.0,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )

    # TODO: test assertions below use True/False instead of 1.0/0.0 to match the old return type.
    # Python's True == 1.0 and False == 0.0 so they still pass, but should be updated for clarity.
    def test_compare_tool_call_arguments(self, tool_call_comparator: ToolCallComparator) -> None:
        assert tool_call_comparator.compare_tool_call_arguments(None, "None") == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT,
        )

        assert tool_call_comparator.compare_tool_call_arguments(
            {"x": 1},
            {
                "x": 1,
                "y": 2,
            },
        ) == (False, StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT)
        assert tool_call_comparator.compare_tool_call_arguments(
            {
                "x": 1,
                "y": 3,
            },
            {
                "x": 1,
                "y": 2,
            },
        ) == (False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT)
        assert tool_call_comparator.compare_tool_call_arguments(
            {
                "x": 1,
                "y": "two",
                "z": True,
            },
            {
                "y": "two",
                "x": 1,
                "z": True,
            },
        ) == (True, None)

        assert tool_call_comparator.compare_tool_call_arguments(
            [
                "first",
                2,
            ],
            [
                "first",
                2,
                "three",
            ],
        ) == (False, StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT)
        assert tool_call_comparator.compare_tool_call_arguments(
            [
                "first",
                2,
            ],
            [
                "one",
                2,
            ],
        ) == (False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT)
        assert tool_call_comparator.compare_tool_call_arguments(
            [
                "first",
                2,
                "three",
            ],
            [
                "first",
                2,
                "three",
            ],
        ) == (True, None)

        assert tool_call_comparator.compare_tool_call_arguments(3.1, 3.11) == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert tool_call_comparator.compare_tool_call_arguments(3.1, 3.1) == (True, None)

        assert tool_call_comparator.compare_tool_call_arguments("value1", "value2") == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert tool_call_comparator.compare_tool_call_arguments("value1", "value1") == (True, None)
        assert tool_call_comparator.compare_tool_call_arguments("the", "the cat") == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert tool_call_comparator.compare_tool_call_arguments("the dog", "the") == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert tool_call_comparator.compare_tool_call_arguments("the cat", "the dog") == (True, None)
        assert tool_call_comparator.compare_tool_call_arguments(
            "the cat ate some food", "the dog ran to the store"
        ) == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert tool_call_comparator.compare_tool_call_arguments("Birds are animals.", "The birds fly.") == (True, None)

        assert tool_call_comparator.compare_tool_call_arguments(26, 25) == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert tool_call_comparator.compare_tool_call_arguments(26, 26) == (True, None)

        assert tool_call_comparator.compare_tool_call_arguments(False, True) == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )
        assert tool_call_comparator.compare_tool_call_arguments(False, False) == (True, None)

    def test_apply_argument_filter(self) -> None:
        first_string_value = "first"
        second_string_value = "second"
        list_value = ["element1", "element2", "element3"]
        single_item_dictionary_value = {
            "key4": "value4",
        }
        two_items_dictionary_value = {
            "first_key": "first_value",
            "second_key": "second_value",
        }
        four_items_dictionary_value = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
            "key4": "value4",
        }

        empty_filter = ToolCallArgumentFilter(included_argument_names=None)
        assert ToolCallComparator._apply_argument_filter(empty_filter, first_string_value) is first_string_value
        assert ToolCallComparator._apply_argument_filter(empty_filter, second_string_value) is second_string_value
        assert ToolCallComparator._apply_argument_filter(empty_filter, list_value) is list_value
        assert (
            ToolCallComparator._apply_argument_filter(empty_filter, single_item_dictionary_value)
            is single_item_dictionary_value
        )
        assert (
            ToolCallComparator._apply_argument_filter(empty_filter, two_items_dictionary_value)
            is two_items_dictionary_value
        )
        assert (
            ToolCallComparator._apply_argument_filter(empty_filter, four_items_dictionary_value)
            is four_items_dictionary_value
        )

        included_arguments_filter = ToolCallArgumentFilter(
            included_argument_names=["second", "element3", "key1", "key4"]
        )
        assert (
            ToolCallComparator._apply_argument_filter(included_arguments_filter, first_string_value)
            is first_string_value
        )
        assert (
            ToolCallComparator._apply_argument_filter(included_arguments_filter, second_string_value)
            is second_string_value
        )
        assert ToolCallComparator._apply_argument_filter(included_arguments_filter, list_value) is list_value
        assert (
            ToolCallComparator._apply_argument_filter(included_arguments_filter, single_item_dictionary_value)
            == single_item_dictionary_value
        )
        assert ToolCallComparator._apply_argument_filter(included_arguments_filter, two_items_dictionary_value) == {}
        assert ToolCallComparator._apply_argument_filter(included_arguments_filter, four_items_dictionary_value) == {
            "key1": "value1",
            "key4": "value4",
        }


class TestToolCallComparatorF1PartialReward:
    """Tests for partial F1 reward (use_list_f1_threshold=False)."""

    @fixture
    def comparator(self) -> ToolCallComparator:
        config = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.3,
            use_f1_for_list=True,
            use_list_f1_threshold=False,
        )
        return ToolCallComparator(config=config)

    @fixture
    def comparator_with_threshold(self) -> ToolCallComparator:
        config = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.3,
            use_f1_for_list=True,
            use_list_f1_threshold=True,
            list_f1_threshold=0.5,
        )
        return ToolCallComparator(config=config)

    def test_exact_list_match_returns_1(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(["a", "b", "c"], ["a", "b", "c"])
        assert score == 1.0
        assert category is None
        assert len(comparator.list_f1_match_details) == 1
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 3
        assert detail.f1 == 1.0
        assert detail.matched_pairs == [(0, 0), (1, 1), (2, 2)]
        assert detail.score_matrix == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def test_partial_list_match_returns_f1(self, comparator: ToolCallComparator) -> None:
        # expected: [1, 2, 3], actual: [1, 2] -> tp=2, precision=2/2=1.0, recall=2/3=0.667, f1=0.8
        score, category = comparator.compare_tool_call_arguments([1, 2, 3], [1, 2])
        assert abs(score - 0.8) < 1e-6
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_PARTIAL
        assert len(comparator.list_f1_match_details) == 1
        detail = comparator.list_f1_match_details[0]
        assert detail.expected_values == [1, 2, 3]
        assert detail.actual_values == [1, 2]
        assert detail.tp == 2
        assert detail.precision == 1.0
        assert abs(detail.recall - 2 / 3) < 1e-6
        assert abs(detail.f1 - 0.8) < 1e-6
        assert detail.matched_pairs == [(0, 0), (1, 1)]
        assert detail.score_matrix == [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]

    def test_no_list_match_returns_0(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments([1, 2, 3], [4, 5, 6])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD
        assert len(comparator.list_f1_match_details) == 1
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 0
        assert detail.f1 == 0.0
        assert detail.matched_pairs == []
        assert detail.score_matrix == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

    def test_empty_lists_return_1(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments([], [])
        assert score == 1.0
        assert category is None
        assert len(comparator.list_f1_match_details) == 0

    def test_one_empty_list_returns_0(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments([1, 2], [])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD
        assert len(comparator.list_f1_match_details) == 0

    def test_threshold_mode_returns_binary(self, comparator_with_threshold: ToolCallComparator) -> None:
        # Same partial match as above (f1=0.8 >= threshold 0.5) -> binary 1.0
        score, category = comparator_with_threshold.compare_tool_call_arguments([1, 2, 3], [1, 2])
        assert score == 1.0
        assert category is None
        assert len(comparator_with_threshold.list_f1_match_details) == 1
        detail = comparator_with_threshold.list_f1_match_details[0]
        assert abs(detail.f1 - 0.8) < 1e-6

    def test_threshold_mode_below_threshold_returns_0(self, comparator_with_threshold: ToolCallComparator) -> None:
        # expected: [1, 2, 3, 4, 5], actual: [1] -> tp=1, precision=1/1=1.0, recall=1/5=0.2, f1=0.333 < 0.5
        score, category = comparator_with_threshold.compare_tool_call_arguments([1, 2, 3, 4, 5], [1])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD

    def test_partial_f1_propagates_through_compare_tool_call(self, comparator: ToolCallComparator) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"items": [1, 2, 3]}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="test",
            name="search",
            arguments=json.dumps({"items": [1, 2]}),
        )
        reward, category = comparator.compare_tool_call(expected, actual)
        assert abs(reward - 0.8) < 1e-6
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_PARTIAL
        assert len(comparator.list_f1_match_details) == 1
        detail = comparator.list_f1_match_details[0]
        assert detail.expected_values == [1, 2, 3]
        assert detail.actual_values == [1, 2]
        assert detail.matched_pairs == [(0, 0), (1, 1)]


class TestToolCallComparatorStrongListReward:
    @fixture
    def comparator(self) -> ToolCallComparator:
        config = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.5,
            use_f1_for_list=True,
            use_strong_list_reward=True,
            list_f1_threshold=0.8,
            list_f1_min_precision=1.0,
            list_f1_min_recall=1.0,
            list_f1_max_actual_to_expected_ratio=1.0,
        )
        return ToolCallComparator(config=config)

    def test_exact_list_match_returns_1(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(["a", "b", "c"], ["a", "b", "c"])
        assert score == 1.0
        assert category is None
        detail = comparator.list_f1_match_details[0]
        assert detail.strong_match_failure_reasons == []

    def test_partial_list_match_returns_0_even_when_f1_is_high(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        score, category = comparator.compare_tool_call_arguments([1, 2, 3], [1, 2])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert abs(detail.f1 - 0.8) < 1e-6
        assert "recall_below_floor" in detail.strong_match_failure_reasons

    def test_duplicate_padding_returns_0(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(["a", "b"], ["a", "b", "a"])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert detail.actual_duplicate_count == 1
        assert "duplicate_actual_values" in detail.strong_match_failure_reasons

    def test_overbroad_list_returns_0(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(["a"], ["a", "b"])
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert detail.actual_to_expected_ratio == 2.0
        assert "actual_list_overbroad" in detail.strong_match_failure_reasons

    def test_fuzzy_query_element_match_returns_0(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(
            ["alpha beta gamma delta"],
            ["alpha beta"],
        )
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 0
        assert "f1_below_threshold" in detail.strong_match_failure_reasons

    def test_ambiguous_list_match_uses_optimal_matching(self) -> None:
        comparator = ToolCallComparator(
            config=ToolCallComparatorConfig(
                word_count_similarity_threshold=0.4,
                use_f1_for_list=True,
                use_strong_list_reward=True,
                list_f1_threshold=0.8,
                list_f1_min_precision=1.0,
                list_f1_min_recall=1.0,
                list_f1_max_actual_to_expected_ratio=1.0,
            )
        )

        score, category = comparator.compare_tool_call_arguments(
            ["alpha beta", "alpha beta gamma delta"],
            ["alpha beta gamma", "alpha beta"],
        )

        assert score == 1.0
        assert category is None
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 2
        assert detail.matched_pairs == [(0, 1), (1, 0)]


class TestToolCallComparatorToolSpecificStrongListReward:
    @fixture
    def comparator(self) -> ToolCallComparator:
        config = ToolCallComparatorConfig(
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
        return ToolCallComparator(config=config)

    def test_search_query_uses_relaxed_element_match(self, comparator: ToolCallComparator) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"queries": ["Hawthorn Trophy bonus points Motorsport Ireland"]}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": ["Hawthorn Trophy bonus points table Motorsport Ireland"]}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 1.0
        assert category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_search_query_rejects_broad_element_padding(self, comparator: ToolCallComparator) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"queries": ["Hawthorn Trophy bonus points Motorsport Ireland"]}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "Hawthorn Trophy bonus points Motorsport Ireland official source "
                        "wikipedia reddit news pdf overview details"
                    ]
                }
            ),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH

    def test_search_query_json_string_list_is_coerced(self, comparator: ToolCallComparator) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"queries": ["alpha beta gamma"]}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": json.dumps(["alpha beta gamma"])}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 1.0
        assert category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_search_query_plain_string_is_coerced_for_singleton_gold(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"queries": ["alpha beta gamma"]}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": "alpha beta gamma"}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 1.0
        assert category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_search_query_plain_string_is_not_coerced_for_multi_query_gold(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"queries": ["alpha beta gamma", "delta epsilon zeta"]}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": "alpha beta gamma"}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT

    def test_malformed_singleton_list_string_is_coerced(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"queries": ["PayPal proxy Jamie Miller compensation"]}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": "[PayPal proxy Jamie Miller]"}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 1.0
        assert category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_search_query_still_rejects_missing_list_items(self, comparator: ToolCallComparator) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "Hawthorn Trophy bonus points Motorsport Ireland",
                        "Dunlop Hawthorn Trophy bonus points starters",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": ["Hawthorn Trophy bonus points table Motorsport Ireland"]}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert "recall_below_floor" in detail.strong_match_failure_reasons

    def test_search_query_allows_one_missing_item_for_three_or_more_gold(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": ["alpha beta gamma", "delta epsilon zeta"]}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 1.0
        assert category == StepRewardCategory.EXPECTED_TOOL_CALL
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 2
        assert detail.strong_match_failure_reasons == []

    def test_search_query_allows_one_extra_item_for_three_or_more_gold(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                        "unrelated reward check",
                    ]
                }
            ),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 1.0
        assert category == StepRewardCategory.EXPECTED_TOOL_CALL
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 3
        assert detail.strong_match_failure_reasons == []

    def test_search_query_rejects_non_string_extra_even_when_one_extra_allowed(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", 123]}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_SCHEMA_MALFORMED

    def test_search_query_rejects_empty_extra_even_when_one_extra_allowed(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "  "]}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_SCHEMA_MALFORMED

    def test_search_query_allows_one_substitution_for_three_or_more_gold(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "unrelated reward check",
                    ]
                }
            ),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 1.0
        assert category == StepRewardCategory.EXPECTED_TOOL_CALL
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 2
        assert detail.strong_match_failure_reasons == []

    def test_search_query_rejects_two_missing_items_for_three_or_more_gold(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": ["alpha beta gamma"]}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert "unmatched_expected_above_limit" in detail.strong_match_failure_reasons

    def test_search_query_rejects_two_extra_items_for_three_or_more_gold(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                        "unrelated reward check",
                        "another unrelated probe",
                    ]
                }
            ),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert "unmatched_actual_above_limit" in detail.strong_match_failure_reasons

    def test_search_query_rejects_duplicate_under_relaxed_list_rule(
        self,
        comparator: ToolCallComparator,
    ) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps(
                {
                    "queries": [
                        "alpha beta gamma",
                        "delta epsilon zeta",
                        "eta theta iota",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="search",
            arguments=json.dumps({"queries": ["alpha beta gamma", "delta epsilon zeta", "alpha beta gamma"]}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert "duplicate_actual_values" in detail.strong_match_failure_reasons

    def test_browse_url_does_not_use_search_query_override(self, comparator: ToolCallComparator) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="browse",
            arguments=json.dumps({"urls": ["https://example.com/alpha-beta-gamma"]}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="browse",
            arguments=json.dumps({"urls": ["https://example.com/alpha-beta"]}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert detail.tp == 0

    def test_browse_url_does_not_use_search_query_list_relaxation(self, comparator: ToolCallComparator) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="browse",
            arguments=json.dumps(
                {
                    "urls": [
                        "https://example.com/one",
                        "https://example.com/two",
                        "https://example.com/three",
                    ]
                }
            ),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="actual",
            name="browse",
            arguments=json.dumps({"urls": ["https://example.com/one", "https://example.com/two"]}),
        )
        score, category = comparator.compare_tool_call(expected, actual)
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
        detail = comparator.list_f1_match_details[0]
        assert "recall_below_floor" in detail.strong_match_failure_reasons


class TestToolCallComparatorRealWorldCases:
    """Tests from real model outputs where queries arg is a string instead of a list."""

    @fixture
    def comparator(self) -> ToolCallComparator:
        config = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.3,
            use_f1_for_list=True,
            use_list_f1_threshold=False,
        )
        return ToolCallComparator(config=config)

    def test_queries_string_instead_of_list_nosia(self, comparator: ToolCallComparator) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"queries": ['"Environmental Modeling Prediction" "Figure A" NOSIA']}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="chatcmpl-tool-b6eb12d375957a8c",
            name="search",
            arguments=json.dumps({"queries": '["NOSIA-II" "Weather Ready Nation" Mission Service Areas]'}),
        )
        reward, category = comparator.compare_tool_call(expected, actual)
        assert reward == 0.0
        assert category == StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT
        assert len(comparator.list_f1_match_details) == 0

    def test_queries_string_instead_of_list_wga(self, comparator: ToolCallComparator) -> None:
        expected = ExpectedFunctionCall(
            type="function_call",
            name="search",
            arguments=json.dumps({"queries": ["non-original screenplay wga credit rules"]}),
        )
        actual = NeMoGymResponseFunctionToolCall(
            call_id="chatcmpl-tool-ae08555b2ca7a66d",
            name="search",
            arguments=json.dumps(
                {"queries": '["production executive" "non-original" screenplay credit percentage WGA]'}
            ),
        )
        reward, category = comparator.compare_tool_call(expected, actual)
        assert reward == 0.0
        assert category == StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT
        assert len(comparator.list_f1_match_details) == 0


class TestToolCallComparatorKeepQuotes:
    """Tests for keep_quotes=True: quoted substrings are parsed as single tokens.

    With keep_quotes=True, a substring inside matching quote characters (single or double)
    is treated as one token, quotes preserved. This makes quoted phrases distinct from the
    same words unquoted, which matters for search-query-style arguments where quoting
    signals an exact-phrase intent.
    """

    @fixture
    def comparator(self) -> ToolCallComparator:
        config = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.3,
            keep_quotes=True,
        )
        return ToolCallComparator(config=config)

    @fixture
    def comparator_default(self) -> ToolCallComparator:
        config = ToolCallComparatorConfig(word_count_similarity_threshold=0.3)
        return ToolCallComparator(config=config)

    def test_default_keep_quotes_is_false(self) -> None:
        config = ToolCallComparatorConfig(word_count_similarity_threshold=0.3)
        assert config.keep_quotes is False

    def test_keep_quotes_flag_changes_tokenization(self) -> None:
        # Strong discriminator: same inputs, different score based on keep_quotes.
        #
        # Input: expected='"a b c d e"' vs actual='a b c d e'.
        #
        # With keep_quotes=False (plain whitespace split):
        #   expected tokens: ['"a', 'b', 'c', 'd', 'e"']  (5 tokens; first/last carry quotes)
        #   actual   tokens: ['a', 'b', 'c', 'd', 'e']    (5 tokens)
        #   intersection: {'b','c','d'} = 3; similarity = 3/10 = 0.3, not < 0.3 -> PASS.
        #
        # With keep_quotes=True (quoted-phrase-aware):
        #   expected tokens: ['"a b c d e"']  (1 token, short-string rule applies)
        #   exact equality with 'a b c d e' fails -> FAIL.
        expected = '"a b c d e"'
        actual = "a b c d e"

        config_off = ToolCallComparatorConfig(word_count_similarity_threshold=0.3)
        comp_off = ToolCallComparator(config=config_off)
        score_off, _ = comp_off.compare_tool_call_arguments(expected, actual)
        assert score_off == 1.0

        config_on = ToolCallComparatorConfig(word_count_similarity_threshold=0.3, keep_quotes=True)
        comp_on = ToolCallComparator(config=config_on)
        score_on, category_on = comp_on.compare_tool_call_arguments(expected, actual)
        assert score_on == 0.0
        assert category_on == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    def test_baseline_identical_strings_still_match_without_flag(self, comparator_default: ToolCallComparator) -> None:
        # Regression: with keep_quotes unset, today's behavior is unchanged.
        # Both sides tokenize identically via plain .split(), so they match.
        score, category = comparator_default.compare_tool_call_arguments('"hello world" zumba', '"hello world" zumba')
        assert score == 1.0
        assert category is None

    def test_baseline_quoted_vs_unquoted_without_flag(self, comparator_default: ToolCallComparator) -> None:
        # Without keep_quotes, quotes are just glued to adjacent words:
        #   expected tokens: ['"hello', 'world"', 'zumba']
        #   actual   tokens: ['hello', 'world', 'zumba']
        # Shared word total = 1 ('zumba'); similarity = 1 / (3+3) = 0.167 < 0.3 -> fail.
        # This locks in current behavior so we can detect drift once keep_quotes lands.
        score, category = comparator_default.compare_tool_call_arguments('"hello world" zumba', "hello world zumba")
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    def test_quoted_phrase_matches_same_quoted_phrase(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments('"hello world" zumba', '"hello world" zumba')
        assert score == 1.0
        assert category is None

    def test_quoted_phrase_order_independent(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments('"hello world" zumba', 'zumba "hello world"')
        assert score == 1.0
        assert category is None

    def test_quoted_phrase_distinct_from_unquoted_same_words(self, comparator: ToolCallComparator) -> None:
        # With keep_quotes, '"hello world"' is one token and does NOT match unquoted
        # 'hello' + 'world'. Only 'zumba' overlaps. Similarity = 0 / (2+3) = 0 -> fail.
        score, category = comparator.compare_tool_call_arguments('"hello world" zumba', "hello world zumba")
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    def test_inner_words_of_quoted_phrase_not_matched_separately(self, comparator: ToolCallComparator) -> None:
        # Expected: ['"hello world"'] (1 token). <2 tokens triggers exact-string rule.
        # Actual: 'hello zumba world' != '"hello world"' -> fail.
        score, category = comparator.compare_tool_call_arguments('"hello world"', "hello zumba world")
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    def test_multiple_quoted_phrases_order_independent(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments(
            '"new york" "los angeles"', '"los angeles" "new york"'
        )
        assert score == 1.0
        assert category is None

    def test_unbalanced_quote_strips_and_warns(self, comparator: ToolCallComparator, caplog) -> None:
        # Unbalanced quote on expected side -> strip all quote chars, fall back to
        # whitespace split, emit a warning. After strip both sides are identical.
        caplog.set_level(logging.WARNING)
        score, category = comparator.compare_tool_call_arguments('"hello world zumba', "hello world zumba")
        assert score == 1.0
        assert category is None
        assert any(
            "unbalanced" in rec.message.lower() or "unmatched" in rec.message.lower() for rec in caplog.records
        ), f"expected an unbalanced-quote warning, got: {[r.message for r in caplog.records]}"

    def test_single_quoted_short_string_requires_exact_match(self, comparator: ToolCallComparator) -> None:
        # Expected tokenizes to 1 token ['"hi"']; <2 triggers exact-equality rule.
        # '"hi"' != 'hi' -> fail.
        score, category = comparator.compare_tool_call_arguments('"hi"', "hi")
        assert score == 0.0
        assert category == StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    def test_quoted_phrase_case_insensitive(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments('"Hello World" Zumba', '"hello world" zumba')
        assert score == 1.0
        assert category is None

    def test_single_quote_chars_supported(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments("'hello world' zumba", "zumba 'hello world'")
        assert score == 1.0
        assert category is None

    def test_mixed_quote_types_supported(self, comparator: ToolCallComparator) -> None:
        score, category = comparator.compare_tool_call_arguments("\"hi there\" 'bye now'", "'bye now' \"hi there\"")
        assert score == 1.0
        assert category is None

    def test_quoted_strings_inside_f1_list(self) -> None:
        config = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.3,
            use_f1_for_list=True,
            use_list_f1_threshold=True,
            list_f1_threshold=0.5,
            keep_quotes=True,
        )
        comp = ToolCallComparator(config=config)
        score, category = comp.compare_tool_call_arguments(
            ['"hello world" zumba', "other query here"],
            ["other query here", '"hello world" zumba'],
        )
        assert score == 1.0
        assert category is None
