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


class MessageAction(BaseModel):
    type: Literal["message"]
    content: str


class FunctionCallAction(BaseModel):
    type: Literal["function_call"]
    name: str
    arguments: str


class FunctionCallBatchAction(BaseModel):
    type: Literal["function_call_batch"]
    calls: list[FunctionCallAction] = Field(min_length=1)


# Actions are canonical on both sides of a comparison: dataset rows deserialize into them, and model
# responses are normalized into them by `response_utils.extract_action`. The alias keeps the name
# that dataset validation tooling imports.
ExpectedAction: TypeAlias = Annotated[
    Union[MessageAction, FunctionCallAction, FunctionCallBatchAction],
    Field(discriminator="type"),
]


def get_tool_calls(action: ExpectedAction) -> list[FunctionCallAction]:
    """Flatten an action into the tool calls it represents, so single and parallel calls share a path."""
    if isinstance(action, FunctionCallBatchAction):
        return list(action.calls)

    if isinstance(action, FunctionCallAction):
        return [action]

    return []


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


class ParallelToolCallRewardMode(StrEnum):
    """How an admissible parallel tool-call response converts its matched-call count into a reward."""

    BINARY_STRICT = "binary_strict"
    FRACTIONAL = "fractional"
    F1 = "f1"


class ActionComparisonResult(BaseModel):
    reward: float
    category: StepRewardCategory


class ToolCallComparatorConfig(BaseModel):
    word_count_similarity_threshold: float
    floating_point_comparison_threshold: float = 1e-6

    # Master switch: does the NUMBER of tool calls the model made affect its reward?
    #
    # False (default) turns parallel tool-call rewarding off entirely. The question asked of a
    # response is only "did it make the expected call(s)?", so surplus calls are never penalized.
    # This reproduces the behaviour that predates parallel tool-call support, which is why it is the
    # default: existing datasets score exactly as they did before. It is also the honest default,
    # because chat templates do not render differently for `parallel_tool_calls` (the Nemotron
    # template never references the flag), so a model is never told how many calls it may make.
    #
    # True turns it on: the call count becomes part of the verdict, and the three settings below
    # decide which counts are admissible and how partial matches are scored. Turn this on for
    # datasets that use `expected_action.type: function_call_batch`.
    parallel_tool_call_rewarding: bool = False

    # The three settings below are only consulted when `parallel_tool_call_rewarding` is True.

    # Cardinality gate: whether a response that makes fewer / more tool calls than expected is
    # admissible at all. Both default to False, so the call count must match exactly.
    allow_subset: bool = False
    allow_superset: bool = False

    # Scoring for responses that clear the gate. See `ActionComparator.compare_tool_calls`.
    parallel_tool_call_reward_mode: ParallelToolCallRewardMode = ParallelToolCallRewardMode.BINARY_STRICT


def find_maximum_matching(candidates: list[list[int]]) -> dict[int, int]:
    """Maximum bipartite matching between expected and actual tool calls (Kuhn's algorithm).

    `candidates[expected_index]` lists the actual-call indices that expected call could match, and the
    returned mapping goes from actual index to the expected index it was matched with. Greedy pairing
    is not enough here: argument matching is a fuzzy relation, so one actual call can satisfy several
    expected calls and an early arbitrary pairing can strand a later expected call that had no
    alternative. Augmenting paths undo those pairings and recover the true maximum.
    """
    matching: dict[int, int] = {}

    # Matching the most constrained expected calls first keeps the number of augmenting paths small.
    for expected_index in sorted(range(len(candidates)), key=lambda index: len(candidates[index])):
        _augment_matching(expected_index, candidates, matching, set())

    return matching


def _augment_matching(
    expected_index: int,
    candidates: list[list[int]],
    matching: dict[int, int],
    visited_actual_indices: set[int],
) -> bool:
    for actual_index in candidates[expected_index]:
        if actual_index in visited_actual_indices:
            continue

        visited_actual_indices.add(actual_index)
        is_unmatched = actual_index not in matching
        if is_unmatched or _augment_matching(matching[actual_index], candidates, matching, visited_actual_indices):
            matching[actual_index] = expected_index
            return True

    return False


class ActionComparator(BaseModel):
    config: ToolCallComparatorConfig

    def compare_action(self, expected_action: ExpectedAction, actual_action: ExpectedAction) -> ActionComparisonResult:
        match expected_action:
            case MessageAction():
                # Currently, any chat message is assigned a reward of one.
                if isinstance(actual_action, MessageAction):
                    return ActionComparisonResult(reward=1.0, category=StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND)

                return ActionComparisonResult(reward=0.0, category=StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE)

            case FunctionCallAction() | FunctionCallBatchAction():
                if isinstance(actual_action, MessageAction):
                    return ActionComparisonResult(reward=0.0, category=StepRewardCategory.NO_EXPECTED_TOOL_CALL)

                return self.compare_tool_calls(get_tool_calls(expected_action), get_tool_calls(actual_action))

            case _:
                raise NotImplementedError(f"Unsupported expected action: {expected_action!r}")

    def compare_tool_calls(
        self, expected_calls: list[FunctionCallAction], actual_calls: list[FunctionCallAction]
    ) -> ActionComparisonResult:
        """Score a set of tool calls against the expected set, ignoring the order they were emitted in.

        `parallel_tool_call_rewarding` is the master switch. While it is False (the default) the number
        of calls is simply not part of the verdict: the response is asked only whether it made the
        expected call(s), and surplus calls cost nothing. That reproduces the behaviour that predates
        parallel tool-call support.

        Turning it on makes the call count matter, in two independent stages. The cardinality gate
        (`allow_subset` / `allow_superset`) decides whether a response that under- or over-calls is
        admissible at all; anything it rejects scores zero. `parallel_tool_call_reward_mode` then decides
        how much credit an admissible response earns:

        - `binary_strict` — 1.0 only if every required call matched, else 0.0.
        - `fractional` — the matched fraction of the required calls. Surplus calls permitted by
          `allow_superset` are free, so this rewards recall but not precision.
        - `f1` — the harmonic mean of precision and recall, `2 * matched / (expected + actual)`. Missing
          and surplus calls are penalized symmetrically, so a response only reaches 1.0 by matching the
          expected calls exactly.
        """
        expected_count = len(expected_calls)
        actual_count = len(actual_calls)

        if expected_count == 1 and actual_count == 1:
            # Preserve the single-call categories that predate parallel tool-call support.
            return self.compare_tool_call(expected_calls[0], actual_calls[0])

        if not self.is_call_count_admissible(expected_count, actual_count):
            return ActionComparisonResult(reward=0.0, category=StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT)

        candidates, failure_categories = self.build_match_candidates(expected_calls, actual_calls)
        matching = find_maximum_matching(candidates)
        reward = self.score_matched_calls(len(matching), expected_count, actual_count)
        if reward == 1.0:
            # A single expected call keeps the category it had before parallel support existed.
            category = (
                StepRewardCategory.EXPECTED_TOOL_CALL
                if expected_count == 1
                else StepRewardCategory.EXPECTED_TOOL_CALL_BATCH
            )
            return ActionComparisonResult(reward=reward, category=category)

        category = self.resolve_failure_category(
            matched_expected_indices=set(matching.values()),
            failure_categories=failure_categories,
            expected_count=expected_count,
            actual_count=actual_count,
        )
        return ActionComparisonResult(reward=reward, category=category)

    def is_call_count_admissible(self, expected_count: int, actual_count: int) -> bool:
        if actual_count == expected_count:
            return True

        # With parallel tool-call rewarding off, the call count is not part of the verdict at all.
        if not self.config.parallel_tool_call_rewarding:
            return True

        if actual_count < expected_count:
            return self.config.allow_subset

        return self.config.allow_superset

    def required_match_count(self, expected_count: int, actual_count: int) -> int:
        """How many expected calls must match for full credit under `binary_strict` and `fractional`."""
        if self.config.allow_subset:
            return min(expected_count, actual_count)

        return expected_count

    def score_matched_calls(self, matched_count: int, expected_count: int, actual_count: int) -> float:
        if not self.config.parallel_tool_call_rewarding:
            # The call count is not part of the verdict, so neither the gate nor the reward mode
            # applies: full credit exactly when every expected call was matched.
            return 1.0 if matched_count == expected_count else 0.0

        if self.config.parallel_tool_call_reward_mode == ParallelToolCallRewardMode.F1:
            total_count = expected_count + actual_count
            return 2 * matched_count / total_count if total_count else 0.0

        required_count = self.required_match_count(expected_count, actual_count)
        if self.config.parallel_tool_call_reward_mode == ParallelToolCallRewardMode.FRACTIONAL:
            return matched_count / required_count if required_count else 0.0

        return 1.0 if matched_count == required_count else 0.0

    def build_match_candidates(
        self, expected_calls: list[FunctionCallAction], actual_calls: list[FunctionCallAction]
    ) -> tuple[list[list[int]], list[StepRewardCategory]]:
        """Pair every expected call with the actual calls it matches, keeping why the others missed."""
        candidates: list[list[int]] = []
        failure_categories: list[StepRewardCategory] = []

        for expected_call in expected_calls:
            matching_actual_indices: list[int] = []
            failure_category = StepRewardCategory.UNEXPECTED_TOOL

            for actual_index, actual_call in enumerate(actual_calls):
                result = self.compare_tool_call(expected_call, actual_call)
                if result.reward == 1.0:
                    matching_actual_indices.append(actual_index)

                elif failure_category == StepRewardCategory.UNEXPECTED_TOOL:
                    # Keep the first reason that got past the tool name; it explains the closest near miss.
                    failure_category = result.category

            candidates.append(matching_actual_indices)
            failure_categories.append(failure_category)

        return candidates, failure_categories

    def resolve_failure_category(
        self,
        matched_expected_indices: set[int],
        failure_categories: list[StepRewardCategory],
        expected_count: int,
        actual_count: int,
    ) -> StepRewardCategory:
        unmatched_expected_indices = [
            expected_index
            for expected_index in range(expected_count)
            if expected_index not in matched_expected_indices
        ]

        # Either every expected call was matched and the only defect left is surplus calls, or every
        # actual call was consumed and the only defect left is missing calls. Reporting an argument
        # mismatch in those cases would point at a comparison that is not why the reward was docked.
        if not unmatched_expected_indices or len(matched_expected_indices) == actual_count:
            return StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT

        return failure_categories[unmatched_expected_indices[0]]

    def compare_tool_call(
        self, expected_tool_call: FunctionCallAction, actual_tool_call: FunctionCallAction
    ) -> ActionComparisonResult:
        if expected_tool_call.name != actual_tool_call.name:
            return ActionComparisonResult(reward=0.0, category=StepRewardCategory.UNEXPECTED_TOOL)

        # It is assumed that the expected arguments string is a string representation of a JSON object.
        expected_arguments = json.loads(expected_tool_call.arguments)

        try:
            actual_arguments = json.loads(actual_tool_call.arguments)
        except (JSONDecodeError, UnicodeDecodeError):
            return ActionComparisonResult(reward=0.0, category=StepRewardCategory.ARGUMENTS_DECODE_ERROR)

        arguments_match, category = self.compare_tool_call_arguments(expected_arguments, actual_arguments)
        if arguments_match:
            return ActionComparisonResult(reward=1.0, category=StepRewardCategory.EXPECTED_TOOL_CALL)

        return ActionComparisonResult(reward=0.0, category=category)

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
