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
from typing import Annotated, Any, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field

from nemo_gym.openai_utils import NeMoGymResponseFunctionToolCall


class ExpectedMessage(BaseModel):
    type: Literal["message"]
    content: str


class ExpectedFunctionCall(BaseModel):
    type: Literal["function_call"]
    name: str
    arguments: str


ExpectedAction: TypeAlias = Annotated[Union[ExpectedMessage, ExpectedFunctionCall], Field(discriminator="type")]


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


class ToolCallComparatorConfig(BaseModel):
    word_count_similarity_threshold: float
    floating_point_comparison_threshold: float = 1e-6


class ToolCallComparator(BaseModel):
    config: ToolCallComparatorConfig

    def compare_tool_call(
        self, expected_tool_call: ExpectedFunctionCall, actual_tool_call: NeMoGymResponseFunctionToolCall
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
