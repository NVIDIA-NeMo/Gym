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

from pytest import fixture

from nemo_gym.openai_utils import NeMoGymResponseFunctionToolCall
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedFunctionCall,
    StepRewardCategory,
    ToolCallComparator,
    ToolCallComparatorConfig,
)


class TestToolCallComparator:
    @fixture
    def tool_call_comparator(self) -> ToolCallComparator:
        comparator_config = ToolCallComparatorConfig(word_count_similarity_threshold=0.1)
        return ToolCallComparator(config=comparator_config)

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
            False,
            StepRewardCategory.UNEXPECTED_TOOL,
        )

        invalid_arguments_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="invalid_arguments",
            name="send",
            arguments="first=one",
        )
        assert tool_call_comparator.compare_tool_call(expected_function_call, invalid_arguments_tool_call) == (
            False,
            StepRewardCategory.ARGUMENTS_DECODE_ERROR,
        )

        matching_arguments_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="matching_arguments",
            name="send",
            arguments=arguments_string,
        )
        assert tool_call_comparator.compare_tool_call(expected_function_call, matching_arguments_tool_call) == (
            True,
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
        different_argument_value_tool_call = NeMoGymResponseFunctionToolCall(
            call_id="different_argument_value",
            name="send",
            arguments=json.dumps(different_argument_value_object),
        )
        assert tool_call_comparator.compare_tool_call(expected_function_call, different_argument_value_tool_call) == (
            False,
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
            False,
            StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT,
        )

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
