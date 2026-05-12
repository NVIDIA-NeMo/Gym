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
from collections import Counter
from enum import StrEnum
from json import JSONDecodeError
from typing import Annotated, Any, Literal, Optional, Protocol, TypeAlias, Union

from pydantic import BaseModel, Field

from nemo_gym.openai_utils import NeMoGymResponseFunctionToolCall


class ExpectedMessage(BaseModel):
    type: Literal["message"]
    content: str


class ExpectedFunctionCall(BaseModel):
    type: Literal["function_call"]
    name: str
    arguments: str


class ExpectedFunctionCallBatch(BaseModel):
    type: Literal["function_call_batch"]
    calls: list[ExpectedFunctionCall] = Field(min_length=1)


ExpectedAction: TypeAlias = Annotated[
    Union[ExpectedMessage, ExpectedFunctionCall, ExpectedFunctionCallBatch],
    Field(discriminator="type"),
]


ToolAction: TypeAlias = Union[ExpectedFunctionCall, ExpectedFunctionCallBatch, NeMoGymResponseFunctionToolCall]


class ToolCallLike(Protocol):
    type: str
    name: str
    arguments: str


class StepRewardCategory(StrEnum):
    NO_ACTION_FOUND = "No tool call or chat message was found in the response"
    NO_EXPECTED_TOOL_CALL = "No tool call was found when one was expected"
    EXPECTED_CHAT_MESSAGE_FOUND = "A chat message was found as expected"
    NO_EXPECTED_CHAT_MESSAGE = "A tool call was executed when a chat message was expected"
    UNEXPECTED_TOOL = "The tool in a tool call is not the expected tool"
    ARGUMENTS_DECODE_ERROR = "An error occurred when decoding the arguments string in a tool call as a JSON object"
    ARGUMENT_VALUE_TYPE_DIFFERENT = "The type of an argument value in a tool call is different than the expected type"
    ARGUMENT_OBJECT_KEYS_DIFFERENT = (
        "The keys in an object in an argument value in a tool call are different than the keys in the expected object"
    )
    ARGUMENT_LIST_LENGTH_DIFFERENT = (
        "A list in an argument value in a tool call has a different length than the expected list"
    )
    ARGUMENT_VALUE_DIFFERENT = "An argument value in a tool call is different than the expected value"
    EXPECTED_TOOL_CALL = "A tool call that matches the expected tool call was found"
    FUNCTION_CALL_BATCH_LENGTH_DIFFERENT = "The number of tool calls in a batch is different than expected"
    EXPECTED_TOOL_CALL_BATCH = "A tool-call batch that matches the expected tool calls was found"


class ToolCallComparatorConfig(BaseModel):
    word_count_similarity_threshold: float
    floating_point_comparison_threshold: float = 1e-6
    allow_subset: bool = False
    allow_superset: bool = False
    parallel_tool_call_reward_mode: Literal["binary_strict", "fractional"] = "binary_strict"


class ToolCallComparator(BaseModel):
    config: ToolCallComparatorConfig

    def compare_tool_call(
        self, expected_tool_call: ExpectedFunctionCall, actual_tool_call: ToolCallLike
    ) -> tuple[float, StepRewardCategory]:
        if expected_tool_call.name != actual_tool_call.name:
            return 0.0, StepRewardCategory.UNEXPECTED_TOOL

        # It is assumed that the expected arguments string is a string representation of a JSON object.
        expected_arguments = json.loads(expected_tool_call.arguments)

        try:
            actual_arguments = json.loads(actual_tool_call.arguments)
        except (JSONDecodeError, UnicodeDecodeError):
            return 0.0, StepRewardCategory.ARGUMENTS_DECODE_ERROR

        arguments_match, category = self.compare_tool_call_arguments(expected_arguments, actual_arguments)
        if arguments_match:
            return 1.0, StepRewardCategory.EXPECTED_TOOL_CALL
        else:
            return 0.0, category

    def compare_tool_action(
        self, expected_tool_action: ToolAction, actual_tool_action: ToolAction
    ) -> tuple[float, StepRewardCategory]:
        expected_calls = self.get_tool_calls(expected_tool_action)
        actual_calls = self.get_tool_calls(actual_tool_action)

        if len(expected_calls) == 1 and len(actual_calls) == 1:
            return self.compare_tool_call(expected_calls[0], actual_calls[0])

        if not self.is_call_count_allowed(len(expected_calls), len(actual_calls)):
            return 0.0, StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT

        matched_count, failure_category = self.count_optimistic_matches(expected_calls, actual_calls)
        required_count = self.get_required_match_count(len(expected_calls), len(actual_calls))
        if matched_count == required_count:
            return 1.0, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH

        if self.config.parallel_tool_call_reward_mode == "fractional" and required_count > 0:
            return matched_count / required_count, failure_category

        return 0.0, failure_category

    def get_tool_calls(self, tool_action: ToolAction) -> list[ToolCallLike]:
        if tool_action.type == "function_call_batch":
            return list(tool_action.calls)

        return [tool_action]

    def is_call_count_allowed(self, expected_count: int, actual_count: int) -> bool:
        if actual_count == expected_count:
            return True

        if actual_count < expected_count:
            return self.config.allow_subset

        return self.config.allow_superset

    def get_required_match_count(self, expected_count: int, actual_count: int) -> int:
        if self.config.allow_subset and self.config.allow_superset:
            return min(expected_count, actual_count)

        if self.config.allow_subset and actual_count < expected_count:
            return actual_count

        return expected_count

    def count_optimistic_matches(
        self, expected_calls: list[ToolCallLike], actual_calls: list[ToolCallLike]
    ) -> tuple[int, StepRewardCategory]:
        adjacency: list[list[int]] = []
        failure_category = StepRewardCategory.UNEXPECTED_TOOL

        for expected_call in expected_calls:
            matching_actual_indices: list[int] = []
            for actual_index, actual_call in enumerate(actual_calls):
                reward, category = self.compare_tool_call(expected_call, actual_call)
                if reward == 1.0:
                    matching_actual_indices.append(actual_index)
                elif (
                    failure_category == StepRewardCategory.UNEXPECTED_TOOL
                    or category != StepRewardCategory.UNEXPECTED_TOOL
                ):
                    failure_category = category

            adjacency.append(matching_actual_indices)

        actual_to_expected: dict[int, int] = {}
        for expected_index in sorted(range(len(expected_calls)), key=lambda index: len(adjacency[index])):
            self.try_match_expected_call(
                expected_index=expected_index,
                adjacency=adjacency,
                actual_to_expected=actual_to_expected,
                seen_actual_indices=set(),
            )

        return len(actual_to_expected), failure_category

    def try_match_expected_call(
        self,
        expected_index: int,
        adjacency: list[list[int]],
        actual_to_expected: dict[int, int],
        seen_actual_indices: set[int],
    ) -> bool:
        for actual_index in adjacency[expected_index]:
            if actual_index in seen_actual_indices:
                continue

            seen_actual_indices.add(actual_index)
            if actual_index not in actual_to_expected or self.try_match_expected_call(
                expected_index=actual_to_expected[actual_index],
                adjacency=adjacency,
                actual_to_expected=actual_to_expected,
                seen_actual_indices=seen_actual_indices,
            ):
                actual_to_expected[actual_index] = expected_index
                return True

        return False

    def compare_tool_call_arguments(
        self, expected_value: Any, actual_value: Any
    ) -> tuple[bool, Optional[StepRewardCategory]]:
        if not isinstance(actual_value, type(expected_value)):
            return False, StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT

        if isinstance(expected_value, dict):
            if set(expected_value.keys()) != set(actual_value.keys()):
                return False, StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT

            for expected_dict_key, expected_dict_value in expected_value.items():
                actual_dict_value = actual_value[expected_dict_key]
                dict_value_match, dict_value_category = self.compare_tool_call_arguments(
                    expected_dict_value, actual_dict_value
                )
                if not dict_value_match:
                    return dict_value_match, dict_value_category

            return True, None

        elif isinstance(expected_value, list):
            if len(expected_value) != len(actual_value):
                return False, StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT

            for expected_list_element, actual_list_element in zip(expected_value, actual_value):
                list_element_match, list_element_category = self.compare_tool_call_arguments(
                    expected_list_element, actual_list_element
                )
                if not list_element_match:
                    return list_element_match, list_element_category

            return True, None

        elif isinstance(expected_value, float):
            if abs(actual_value - expected_value) < self.config.floating_point_comparison_threshold:
                return True, None
            else:
                return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

        elif isinstance(expected_value, str):
            # For now, strings are compared by using whitespace to split them into lower-case
            # words, counting the words, and comparing the word counts using Jaccard similarity.
            expected_word_counts = Counter(expected_value.strip().lower().split())
            actual_word_counts = Counter(actual_value.strip().lower().split())
            expected_word_total = expected_word_counts.total()
            actual_word_total = actual_word_counts.total()

            if expected_word_total < 2 or actual_word_total < 2:
                if expected_value != actual_value:
                    return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

            else:
                intersection_word_counts = expected_word_counts & actual_word_counts

                word_count_similarity = intersection_word_counts.total() / (expected_word_total + actual_word_total)
                if word_count_similarity < self.config.word_count_similarity_threshold:
                    return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

            return True, None

        elif expected_value == actual_value:
            return True, None

        else:
            return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
