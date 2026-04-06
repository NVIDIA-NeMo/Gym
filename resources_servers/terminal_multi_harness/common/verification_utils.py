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
from difflib import SequenceMatcher
from enum import StrEnum
from json import JSONDecodeError
from typing import Annotated, Any, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field


class MessageAction(BaseModel):
    type: Literal["message"]
    content: str


class FunctionCallAction(BaseModel):
    type: Literal["function_call"]
    name: str
    arguments: str


class FunctionCallBatchAction(BaseModel):
    type: Literal["function_call_batch"]
    calls: list[FunctionCallAction]
    ordered: bool = True


ExpectedAction: TypeAlias = Annotated[
    Union[MessageAction, FunctionCallAction, FunctionCallBatchAction],
    Field(discriminator="type"),
]


class StepRewardCategory(StrEnum):
    NO_ACTION_FOUND = "No tool call or chat message was found in the response"
    ACTION_TYPE_MISMATCH = "The actual action type does not match the expected action type"
    MESSAGE_CONTENT_DIFFERENT = "The assistant message content is different than expected"
    EXPECTED_CHAT_MESSAGE_FOUND = "A chat message that matches the expected message was found"
    UNEXPECTED_TOOL = "The tool in a tool call is not the expected tool"
    ARGUMENTS_DECODE_ERROR = "An error occurred when decoding the arguments string in a tool call as JSON"
    ARGUMENT_VALUE_TYPE_DIFFERENT = "The type of an argument value in a tool call is different than expected"
    ARGUMENT_OBJECT_KEYS_DIFFERENT = "The keys in an object argument value are different than expected"
    ARGUMENT_LIST_LENGTH_DIFFERENT = "A list in an argument value has a different length than expected"
    ARGUMENT_VALUE_DIFFERENT = "An argument value in a tool call is different than expected"
    FUNCTION_CALL_BATCH_LENGTH_DIFFERENT = "The number of tool calls in a batch is different than expected"
    FUNCTION_CALL_BATCH_CALL_DIFFERENT = "A tool call in the batch does not match the expected batch"
    EXPECTED_TOOL_CALL = "A tool call that matches the expected tool call was found"
    EXPECTED_TOOL_CALL_BATCH = "A tool-call batch that matches the expected batch was found"


class ToolCallComparatorConfig(BaseModel):
    string_similarity_threshold: float
    floating_point_comparison_threshold: float = 1e-6
    ignored_argument_keys_by_tool: dict[str, list[str]] = Field(default_factory=dict)


class ActionComparator(BaseModel):
    config: ToolCallComparatorConfig

    def compare_action(
        self, expected_action: ExpectedAction, actual_action: ExpectedAction, harness: str = "generic"
    ) -> tuple[bool, StepRewardCategory]:
        del harness

        if expected_action.type != actual_action.type:
            return False, StepRewardCategory.ACTION_TYPE_MISMATCH

        match expected_action.type:
            case "message":
                return self.compare_message(expected_action, actual_action)
            case "function_call":
                return self.compare_tool_call(expected_action, actual_action)
            case "function_call_batch":
                return self.compare_tool_call_batch(expected_action, actual_action)
            case _:
                raise NotImplementedError

    def compare_message(
        self, expected_message: MessageAction, actual_message: MessageAction
    ) -> tuple[bool, StepRewardCategory]:
        if self.compare_text(expected_message.content, actual_message.content):
            return True, StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND
        return False, StepRewardCategory.MESSAGE_CONTENT_DIFFERENT

    def compare_tool_call(
        self, expected_tool_call: FunctionCallAction, actual_tool_call: FunctionCallAction
    ) -> tuple[bool, StepRewardCategory]:
        if expected_tool_call.name != actual_tool_call.name:
            return False, StepRewardCategory.UNEXPECTED_TOOL

        try:
            expected_arguments = json.loads(expected_tool_call.arguments)
            actual_arguments = json.loads(actual_tool_call.arguments)
        except (JSONDecodeError, UnicodeDecodeError):
            return False, StepRewardCategory.ARGUMENTS_DECODE_ERROR

        expected_arguments = self.normalize_tool_call_arguments(expected_tool_call.name, expected_arguments)
        actual_arguments = self.normalize_tool_call_arguments(actual_tool_call.name, actual_arguments)

        arguments_match, category = self.compare_action_values(expected_arguments, actual_arguments)
        if arguments_match:
            return True, StepRewardCategory.EXPECTED_TOOL_CALL
        return False, category or StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    def compare_tool_call_batch(
        self, expected_batch: FunctionCallBatchAction, actual_batch: FunctionCallBatchAction
    ) -> tuple[bool, StepRewardCategory]:
        if len(expected_batch.calls) != len(actual_batch.calls):
            return False, StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT

        if expected_batch.ordered:
            for expected_call, actual_call in zip(expected_batch.calls, actual_batch.calls):
                call_matches, category = self.compare_tool_call(expected_call, actual_call)
                if not call_matches:
                    return False, category
            return True, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH

        unmatched_actual_calls = list(actual_batch.calls)
        for expected_call in expected_batch.calls:
            matched_index = None
            mismatch_category = StepRewardCategory.FUNCTION_CALL_BATCH_CALL_DIFFERENT

            for actual_index, actual_call in enumerate(unmatched_actual_calls):
                call_matches, category = self.compare_tool_call(expected_call, actual_call)
                if call_matches:
                    matched_index = actual_index
                    break
                mismatch_category = category

            if matched_index is None:
                return False, mismatch_category

            unmatched_actual_calls.pop(matched_index)

        return True, StepRewardCategory.EXPECTED_TOOL_CALL_BATCH

    def normalize_tool_call_arguments(self, tool_name: str, value: Any) -> Any:
        if isinstance(value, dict):
            ignored_keys = set(self.config.ignored_argument_keys_by_tool.get(tool_name, []))
            normalized_value = {}
            for key, item in value.items():
                if key in ignored_keys:
                    continue

                if tool_name == "batch" and key == "tool_calls" and isinstance(item, list):
                    normalized_value[key] = [self.normalize_batch_tool_call(tool_call) for tool_call in item]
                    continue

                normalized_value[key] = self.normalize_tool_call_arguments(tool_name, item)

            return normalized_value

        if isinstance(value, list):
            return [self.normalize_tool_call_arguments(tool_name, item) for item in value]

        return value

    def normalize_batch_tool_call(self, tool_call: Any) -> Any:
        if not isinstance(tool_call, dict):
            return tool_call

        normalized_tool_call = dict(tool_call)
        nested_tool_name = normalized_tool_call.get("tool")
        nested_parameters = normalized_tool_call.get("parameters")
        if isinstance(nested_tool_name, str):
            normalized_tool_call["parameters"] = self.normalize_tool_call_arguments(
                nested_tool_name,
                nested_parameters,
            )
        return normalized_tool_call

    def compare_action_values(self, expected_value: Any, actual_value: Any) -> tuple[bool, Optional[StepRewardCategory]]:
        if not isinstance(actual_value, type(expected_value)):
            return False, StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT

        if isinstance(expected_value, dict):
            if set(expected_value.keys()) != set(actual_value.keys()):
                return False, StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT

            for expected_dict_key, expected_dict_value in expected_value.items():
                actual_dict_value = actual_value[expected_dict_key]
                dict_value_match, dict_value_category = self.compare_action_values(
                    expected_dict_value,
                    actual_dict_value,
                )
                if not dict_value_match:
                    return False, dict_value_category
            return True, None

        if isinstance(expected_value, list):
            if len(expected_value) != len(actual_value):
                return False, StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT

            for expected_list_element, actual_list_element in zip(expected_value, actual_value):
                list_element_match, list_element_category = self.compare_action_values(
                    expected_list_element,
                    actual_list_element,
                )
                if not list_element_match:
                    return False, list_element_category
            return True, None

        if isinstance(expected_value, float):
            if abs(actual_value - expected_value) < self.config.floating_point_comparison_threshold:
                return True, None
            return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

        if isinstance(expected_value, str):
            if self.compare_text(expected_value, actual_value):
                return True, None
            return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

        if expected_value == actual_value:
            return True, None

        return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    def compare_text(self, expected_text: str, actual_text: str) -> bool:
        return SequenceMatcher(None, expected_text, actual_text).ratio() >= self.config.string_similarity_threshold
