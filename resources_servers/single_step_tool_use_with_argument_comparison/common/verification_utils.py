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
from collections import Counter
from enum import StrEnum
from json import JSONDecodeError
from typing import Annotated, Any, Literal, Optional, TypeAlias, Union

import jsonschema.validators
from jsonschema.exceptions import SchemaError as JSONSchemaError
from jsonschema.exceptions import ValidationError as JSONValidationError
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
    ARGUMENT_LIST_F1_BELOW_THRESHOLD = "The F1 score for a list argument is below the threshold"
    ARGUMENT_LIST_F1_PARTIAL = "The F1 score for a list argument is a partial match"
    ARGUMENT_LIST_F1_WEAK_MATCH = "The list argument F1 match failed strong-match guardrails"
    ARGUMENT_LIST_SCHEMA_MALFORMED = "A list argument in a tool call does not conform to the expected schema"
    TOOL_SCHEMA_NOT_FOUND = "No declared tool schema was found for the tool call"
    TOOL_SCHEMA_VALIDATION_FAILED = "The tool call arguments do not conform to the declared tool schema"
    MULTIPLE_TOOL_CALLS_FOUND = "Multiple tool calls were found when exactly one was expected"
    ARGUMENT_VALUE_DIFFERENT = "An argument value in a tool call is different than the expected value"
    EXPECTED_TOOL_CALL = "A tool call that matches the expected tool call was found"


class ListF1MatchDetail(BaseModel):
    expected_values: list[Any]
    actual_values: list[Any]
    score_matrix: list[list[float]]
    matched_pairs: list[tuple[int, int]]
    tp: int
    precision: float
    recall: float
    f1: float
    actual_duplicate_count: int = 0
    actual_to_expected_ratio: Optional[float] = None
    strong_match_failure_reasons: list[str] = Field(default_factory=list)


class ToolCallArgumentFilter(BaseModel):
    included_argument_names: Optional[list[str]] = None


class ToolCallArgumentComparisonOverride(BaseModel):
    word_count_similarity_threshold: Optional[float] = None
    word_count_min_precision: Optional[float] = None
    word_count_min_recall: Optional[float] = None
    word_count_max_actual_to_expected_ratio: Optional[float] = None
    word_count_max_unmatched_actual_words: Optional[int] = None
    list_f1_relaxed_min_expected_len: Optional[int] = None
    list_f1_max_unmatched_expected: Optional[int] = None
    list_f1_max_unmatched_actual: Optional[int] = None
    keep_quotes: Optional[bool] = None
    coerce_actual_string_to_list: bool = False
    coerce_actual_string_to_singleton_list: bool = False
    validate_list_item_schema: bool = False
    reject_empty_string_list_items: bool = False


class ToolCallComparatorConfig(BaseModel):
    word_count_similarity_threshold: float
    floating_point_comparison_threshold: float = 1e-6
    use_f1_for_list: bool = False
    list_f1_threshold: float = 0.5
    use_list_f1_threshold: bool = True
    use_strong_list_reward: bool = False
    list_f1_min_precision: float = 1.0
    list_f1_min_recall: float = 1.0
    list_f1_max_actual_to_expected_ratio: float = 1.0
    list_f1_reject_duplicate_actual_values: bool = True
    argument_filters: Optional[dict[str, ToolCallArgumentFilter]] = None
    argument_comparison_overrides: Optional[dict[str, dict[str, ToolCallArgumentComparisonOverride]]] = None
    keep_quotes: bool = False


logger = logging.getLogger(__name__)


_QUOTE_CHARS = ('"', "'")


def tokenize_with_quoted_phrases(value: str) -> list[str]:
    """Tokenize on whitespace, but keep a run inside matching quotes (`"..."` or `'...'`)
    as one token with the quote characters preserved.

    If any quote is opened and never closed, all quote characters are stripped from the
    original string and the result is whitespace-split. A warning is logged so
    mis-quoted inputs are observable.
    """
    lowered = value.strip().lower()
    tokens: list[str] = []
    buf: list[str] = []
    active_quote: Optional[str] = None

    def flush_buf() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in lowered:
        if active_quote is not None:
            buf.append(ch)
            if ch == active_quote:
                flush_buf()
                active_quote = None
        elif ch in _QUOTE_CHARS:
            flush_buf()
            active_quote = ch
            buf.append(ch)
        elif ch.isspace():
            flush_buf()
        else:
            buf.append(ch)

    if active_quote is not None:
        logger.warning(
            "Unbalanced quote character %r in string %r; stripping all quote chars and falling back to whitespace split.",
            active_quote,
            value,
        )
        stripped = lowered
        for q in _QUOTE_CHARS:
            stripped = stripped.replace(q, "")
        return stripped.split()

    flush_buf()
    return tokens


def build_declared_tool_schema_map(declared_tools: list[Any] | None) -> dict[str, dict[str, Any]]:
    declared_tool_schemas: dict[str, dict[str, Any]] = {}
    for tool_definition in declared_tools or []:
        if hasattr(tool_definition, "model_dump"):
            tool_definition = tool_definition.model_dump(mode="python")

        if not isinstance(tool_definition, dict):
            continue

        function_definition = tool_definition.get("function")
        if isinstance(function_definition, dict):
            tool_name = function_definition.get("name")
            tool_schema = function_definition.get("parameters")
        else:
            tool_name = tool_definition.get("name")
            tool_schema = tool_definition.get("parameters")

        if isinstance(tool_name, str) and isinstance(tool_schema, dict):
            declared_tool_schemas[tool_name] = tool_schema

    return declared_tool_schemas


def validate_tool_call_against_declared_schema(
    tool_call: NeMoGymResponseFunctionToolCall,
    declared_tools: list[Any] | None,
) -> Optional[StepRewardCategory]:
    tool_schema = build_declared_tool_schema_map(declared_tools).get(tool_call.name)
    if tool_schema is None:
        return StepRewardCategory.TOOL_SCHEMA_NOT_FOUND

    try:
        actual_arguments = json.loads(tool_call.arguments)
    except (JSONDecodeError, UnicodeDecodeError):
        return StepRewardCategory.ARGUMENTS_DECODE_ERROR

    try:
        validator_class = jsonschema.validators.validator_for(tool_schema)
        validator_class.check_schema(tool_schema)
        validator = validator_class(tool_schema)
        validator.validate(actual_arguments)
    except (JSONSchemaError, JSONValidationError):
        return StepRewardCategory.TOOL_SCHEMA_VALIDATION_FAILED

    return None


class ToolCallComparator(BaseModel):
    config: ToolCallComparatorConfig
    list_f1_match_details: list[ListF1MatchDetail] = Field(default_factory=list)

    def compare_tool_call(
        self, expected_tool_call: ExpectedFunctionCall, actual_tool_call: NeMoGymResponseFunctionToolCall
    ) -> tuple[float, StepRewardCategory]:
        tool_name = expected_tool_call.name
        if tool_name != actual_tool_call.name:
            return 0.0, StepRewardCategory.UNEXPECTED_TOOL

        # It is assumed that the expected arguments string is a string representation of a JSON object.
        expected_arguments = json.loads(expected_tool_call.arguments)

        try:
            actual_arguments = json.loads(actual_tool_call.arguments)
        except (JSONDecodeError, UnicodeDecodeError):
            return 0.0, StepRewardCategory.ARGUMENTS_DECODE_ERROR

        argument_filter_map = self.config.argument_filters
        if argument_filter_map is not None:
            argument_filter = argument_filter_map.get(tool_name)
            if argument_filter is not None:
                expected_arguments = self._apply_argument_filter(argument_filter, expected_arguments)
                actual_arguments = self._apply_argument_filter(argument_filter, actual_arguments)

        score, category = self.compare_tool_call_arguments(
            expected_arguments,
            actual_arguments,
            tool_name=tool_name,
        )
        if score == 1.0:
            return 1.0, StepRewardCategory.EXPECTED_TOOL_CALL
        elif score > 0.0:
            return score, category
        else:
            return 0.0, category

    def compare_tool_call_arguments(
        self,
        expected_value: Any,
        actual_value: Any,
        *,
        tool_name: Optional[str] = None,
        argument_override: Optional[ToolCallArgumentComparisonOverride] = None,
    ) -> tuple[float, Optional[StepRewardCategory]]:
        if not isinstance(actual_value, type(expected_value)):
            return 0.0, StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT

        if isinstance(expected_value, dict):
            if set(expected_value.keys()) != set(actual_value.keys()):
                return 0.0, StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT

            for expected_dict_key, expected_dict_value in expected_value.items():
                if expected_dict_key == "goal":
                    continue
                actual_dict_value = actual_value[expected_dict_key]
                child_argument_override = argument_override or self._argument_comparison_override(
                    tool_name,
                    expected_dict_key,
                )
                actual_dict_value = self._maybe_coerce_actual_string_to_list(
                    expected_dict_value,
                    actual_dict_value,
                    child_argument_override,
                )
                dict_value_score, dict_value_category = self.compare_tool_call_arguments(
                    expected_dict_value,
                    actual_dict_value,
                    tool_name=tool_name,
                    argument_override=child_argument_override,
                )
                if dict_value_score < 1.0:
                    return dict_value_score, dict_value_category

            return 1.0, None

        elif isinstance(expected_value, list):
            list_schema_category = self._validate_actual_list_schema(
                expected_value,
                actual_value,
                argument_override,
            )
            if list_schema_category is not None:
                return 0.0, list_schema_category

            if self.config.use_f1_for_list:
                if not expected_value and not actual_value:
                    return 1.0, None
                if not expected_value or not actual_value:
                    return 0.0, StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD

                score_matrix: list[list[float]] = []
                for i, expected_element in enumerate(expected_value):
                    row: list[float] = []
                    for j, actual_element in enumerate(actual_value):
                        score, _ = self.compare_tool_call_arguments(
                            expected_element,
                            actual_element,
                            tool_name=tool_name,
                            argument_override=argument_override,
                        )
                        row.append(score)
                        logger.debug(
                            "List F1 match: expected[%d]=%r vs actual[%d]=%r -> score=%.4f",
                            i,
                            expected_element,
                            j,
                            actual_element,
                            score,
                        )
                    score_matrix.append(row)

                matched_pairs = self._maximum_list_match_pairs(score_matrix)
                tp = len(matched_pairs)

                precision = tp / len(actual_value)
                recall = tp / len(expected_value)
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                actual_duplicate_count = self._count_duplicate_list_values(actual_value)
                actual_to_expected_ratio = len(actual_value) / len(expected_value)
                strong_match_failure_reasons = self._strong_list_match_failure_reasons(
                    expected_count=len(expected_value),
                    actual_count=len(actual_value),
                    true_positive_count=tp,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    actual_to_expected_ratio=actual_to_expected_ratio,
                    actual_duplicate_count=actual_duplicate_count,
                    argument_override=argument_override,
                )

                detail = ListF1MatchDetail(
                    expected_values=expected_value,
                    actual_values=actual_value,
                    score_matrix=score_matrix,
                    matched_pairs=matched_pairs,
                    tp=tp,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    actual_duplicate_count=actual_duplicate_count,
                    actual_to_expected_ratio=actual_to_expected_ratio,
                    strong_match_failure_reasons=strong_match_failure_reasons,
                )
                self.list_f1_match_details.append(detail)
                logger.debug(
                    "List F1 summary: tp=%d, precision=%.4f, recall=%.4f, f1=%.4f, matched_pairs=%s",
                    tp,
                    precision,
                    recall,
                    f1,
                    matched_pairs,
                )

                if self.config.use_strong_list_reward:
                    if strong_match_failure_reasons:
                        return 0.0, StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
                    return 1.0, None
                elif self.config.use_list_f1_threshold:
                    if f1 >= self.config.list_f1_threshold:
                        return 1.0, None
                    return 0.0, StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD
                else:
                    if f1 == 1.0:
                        return 1.0, None
                    elif f1 > 0.0:
                        return f1, StepRewardCategory.ARGUMENT_LIST_F1_PARTIAL
                    else:
                        return 0.0, StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD

            else:
                if len(expected_value) != len(actual_value):
                    return 0.0, StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT

                for expected_list_element, actual_list_element in zip(expected_value, actual_value):
                    list_element_score, list_element_category = self.compare_tool_call_arguments(
                        expected_list_element, actual_list_element
                    )
                    if list_element_score < 1.0:
                        return list_element_score, list_element_category

                return 1.0, None

        elif isinstance(expected_value, float):
            if abs(actual_value - expected_value) < self.config.floating_point_comparison_threshold:
                return 1.0, None
            else:
                return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

        elif isinstance(expected_value, str):
            # For now, strings are compared by using whitespace to split them into lower-case
            # words, counting the words, and comparing the word counts using Jaccard similarity.
            # With keep_quotes=True, runs inside matching quotes are kept as one token so an
            # exact-phrase query like '"hello world"' is distinct from the unquoted words.
            keep_quotes = self.config.keep_quotes
            word_count_similarity_threshold = self.config.word_count_similarity_threshold
            if argument_override is not None:
                if argument_override.keep_quotes is not None:
                    keep_quotes = argument_override.keep_quotes
                if argument_override.word_count_similarity_threshold is not None:
                    word_count_similarity_threshold = argument_override.word_count_similarity_threshold

            if keep_quotes:
                expected_tokens = tokenize_with_quoted_phrases(expected_value)
                actual_tokens = tokenize_with_quoted_phrases(actual_value)
            else:
                expected_tokens = expected_value.strip().lower().split()
                actual_tokens = actual_value.strip().lower().split()
            expected_word_counts = Counter(expected_tokens)
            actual_word_counts = Counter(actual_tokens)
            expected_word_total = expected_word_counts.total()
            actual_word_total = actual_word_counts.total()

            if expected_word_total < 2 or actual_word_total < 2:
                if expected_value != actual_value:
                    return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

            else:
                intersection_word_counts = expected_word_counts & actual_word_counts
                intersection_word_total = intersection_word_counts.total()

                word_count_similarity = intersection_word_total / (expected_word_total + actual_word_total)
                if word_count_similarity < word_count_similarity_threshold:
                    return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
                if argument_override is not None:
                    word_count_precision = intersection_word_total / actual_word_total
                    word_count_recall = intersection_word_total / expected_word_total
                    actual_to_expected_ratio = actual_word_total / expected_word_total
                    unmatched_actual_words = actual_word_total - intersection_word_total

                    if (
                        argument_override.word_count_min_precision is not None
                        and word_count_precision < argument_override.word_count_min_precision
                    ):
                        return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
                    if (
                        argument_override.word_count_min_recall is not None
                        and word_count_recall < argument_override.word_count_min_recall
                    ):
                        return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
                    if (
                        argument_override.word_count_max_actual_to_expected_ratio is not None
                        and actual_to_expected_ratio > argument_override.word_count_max_actual_to_expected_ratio
                    ):
                        return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
                    if (
                        argument_override.word_count_max_unmatched_actual_words is not None
                        and unmatched_actual_words > argument_override.word_count_max_unmatched_actual_words
                    ):
                        return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

            return 1.0, None

        elif expected_value == actual_value:
            return 1.0, None

        else:
            return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    @classmethod
    def _apply_argument_filter(cls, argument_filter: ToolCallArgumentFilter, argument_value: Any) -> Any:
        included_argument_names = argument_filter.included_argument_names
        if included_argument_names is not None:
            if isinstance(argument_value, dict):
                return {key: value for key, value in argument_value.items() if key in included_argument_names}

        return argument_value

    def _argument_comparison_override(
        self,
        tool_name: Optional[str],
        argument_name: str,
    ) -> Optional[ToolCallArgumentComparisonOverride]:
        if tool_name is None or self.config.argument_comparison_overrides is None:
            return None
        tool_overrides = self.config.argument_comparison_overrides.get(tool_name)
        if tool_overrides is None:
            return None
        return tool_overrides.get(argument_name)

    @staticmethod
    def _maybe_coerce_actual_string_to_list(
        expected_value: Any,
        actual_value: Any,
        argument_override: Optional[ToolCallArgumentComparisonOverride],
    ) -> Any:
        if argument_override is None:
            return actual_value
        if not isinstance(expected_value, list) or not isinstance(actual_value, str):
            return actual_value

        if argument_override.coerce_actual_string_to_list:
            try:
                parsed_value = json.loads(actual_value)
            except (JSONDecodeError, TypeError):
                parsed_value = None

            if isinstance(parsed_value, list):
                return parsed_value

        if argument_override.coerce_actual_string_to_singleton_list and len(expected_value) == 1:
            singleton_value = actual_value.strip()
            if singleton_value.startswith("[") and singleton_value.endswith("]"):
                singleton_value = singleton_value[1:-1].strip()
            return [singleton_value]
        return actual_value

    @staticmethod
    def _validate_actual_list_schema(
        expected_value: list[Any],
        actual_value: list[Any],
        argument_override: Optional[ToolCallArgumentComparisonOverride],
    ) -> Optional[StepRewardCategory]:
        if argument_override is None or not argument_override.validate_list_item_schema:
            return None
        if not expected_value:
            return None

        expected_element_type = type(expected_value[0])
        if any(type(expected_element) is not expected_element_type for expected_element in expected_value):
            return None

        for actual_element in actual_value:
            if type(actual_element) is not expected_element_type:
                return StepRewardCategory.ARGUMENT_LIST_SCHEMA_MALFORMED
            if (
                argument_override.reject_empty_string_list_items
                and isinstance(actual_element, str)
                and not actual_element.strip()
            ):
                return StepRewardCategory.ARGUMENT_LIST_SCHEMA_MALFORMED
        return None

    @staticmethod
    def _maximum_list_match_pairs(score_matrix: list[list[float]]) -> list[tuple[int, int]]:
        if not score_matrix:
            return []
        actual_count = len(score_matrix[0]) if score_matrix[0] else 0
        matched_expected_by_actual: list[Optional[int]] = [None] * actual_count

        def try_match(expected_index: int, seen_actual: set[int]) -> bool:
            for actual_index, score in enumerate(score_matrix[expected_index]):
                if score != 1.0 or actual_index in seen_actual:
                    continue
                seen_actual.add(actual_index)
                previous_expected_index = matched_expected_by_actual[actual_index]
                if previous_expected_index is None or try_match(previous_expected_index, seen_actual):
                    matched_expected_by_actual[actual_index] = expected_index
                    return True
            return False

        for expected_index in range(len(score_matrix)):
            try_match(expected_index, set())

        return sorted(
            (expected_index, actual_index)
            for actual_index, expected_index in enumerate(matched_expected_by_actual)
            if expected_index is not None
        )

    def _strong_list_match_failure_reasons(
        self,
        *,
        expected_count: int,
        actual_count: int,
        true_positive_count: int,
        precision: float,
        recall: float,
        f1: float,
        actual_to_expected_ratio: float,
        actual_duplicate_count: int,
        argument_override: Optional[ToolCallArgumentComparisonOverride],
    ) -> list[str]:
        reasons: list[str] = []
        if self._uses_relaxed_list_guardrails(argument_override, expected_count):
            assert argument_override is not None
            unmatched_expected = expected_count - true_positive_count
            unmatched_actual = actual_count - true_positive_count
            if (
                argument_override.list_f1_max_unmatched_expected is not None
                and unmatched_expected > argument_override.list_f1_max_unmatched_expected
            ):
                reasons.append("unmatched_expected_above_limit")
            if (
                argument_override.list_f1_max_unmatched_actual is not None
                and unmatched_actual > argument_override.list_f1_max_unmatched_actual
            ):
                reasons.append("unmatched_actual_above_limit")
            if self.config.list_f1_reject_duplicate_actual_values and actual_duplicate_count:
                reasons.append("duplicate_actual_values")
            return reasons

        if f1 < self.config.list_f1_threshold:
            reasons.append("f1_below_threshold")
        if precision < self.config.list_f1_min_precision:
            reasons.append("precision_below_floor")
        if recall < self.config.list_f1_min_recall:
            reasons.append("recall_below_floor")
        if actual_to_expected_ratio > self.config.list_f1_max_actual_to_expected_ratio:
            reasons.append("actual_list_overbroad")
        if self.config.list_f1_reject_duplicate_actual_values and actual_duplicate_count:
            reasons.append("duplicate_actual_values")
        return reasons

    @staticmethod
    def _uses_relaxed_list_guardrails(
        argument_override: Optional[ToolCallArgumentComparisonOverride],
        expected_count: int,
    ) -> bool:
        if argument_override is None:
            return False
        min_expected_len = argument_override.list_f1_relaxed_min_expected_len
        if min_expected_len is None:
            return False
        return expected_count >= min_expected_len

    @classmethod
    def _count_duplicate_list_values(cls, values: list[Any]) -> int:
        seen: set[str] = set()
        duplicate_count = 0
        for value in values:
            identity = cls._list_value_identity(value)
            if identity in seen:
                duplicate_count += 1
            seen.add(identity)
        return duplicate_count

    @staticmethod
    def _list_value_identity(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return repr(value)
