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
from typing import Annotated, Any, Literal, TypeAlias, Union

from openapi_schema_validator import validate as validate_against_schema_openapi
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
    EMPTY_MESSAGE = "The assistant message is empty after trimming whitespace"
    EXPECTED_CHAT_MESSAGE_FOUND = "A chat message that matches the expected message was found"
    UNEXPECTED_TOOL = "The tool in a tool call is not the expected tool"
    ARGUMENTS_DECODE_ERROR = "An error occurred when decoding the arguments string in a tool call as JSON"
    ARGUMENTS_NOT_OBJECT = "The decoded tool-call arguments are not a JSON object"
    TOOL_SCHEMA_NOT_FOUND = "The declared tool schema for the tool call could not be found"
    TOOL_SCHEMA_VALIDATION_FAILED = "The actual tool-call arguments are not valid under the declared tool schema"
    UNEXPECTED_ARGUMENT_KEYS = "The actual tool-call arguments contain parameter keys absent from the expected answer"
    ARGUMENT_VALUE_MISMATCH = "The actual tool-call argument values do not match the expected answer"
    FUNCTION_CALL_BATCH_LENGTH_DIFFERENT = "The number of tool calls in a batch is different than expected"
    EXEC_COMMAND_MISSING_CMD = "The exec_command tool call does not contain a cmd argument"
    EXEC_COMMAND_CMD_SIMILARITY_BELOW_THRESHOLD = "The exec_command cmd similarity is below threshold"
    BASH_MISSING_COMMAND = "The bash tool call does not contain a command argument"
    BASH_COMMAND_SIMILARITY_BELOW_THRESHOLD = "The bash command similarity is below threshold"
    UPDATE_PLAN_EMPTY_PLAN = "The update_plan tool call does not contain a non-empty plan argument"
    EXECUTE_PYTHON_MISSING_CODE = "The execute_python tool call does not contain a code argument"
    EXECUTE_PYTHON_CODE_SIMILARITY_BELOW_THRESHOLD = (
        "The execute_python code similarity is below threshold"
    )
    RETURN_RESULT_MISSING_RESULT = "The return_result tool call does not contain a result argument"
    RETURN_RESULT_SIMILARITY_BELOW_THRESHOLD = (
        "The return_result result similarity is below threshold"
    )
    TODOWRITE_EMPTY_TODOS = "The todowrite tool call does not contain a non-empty todos list"
    EXPECTED_TOOL_CALL = "A tool call that matches the expected tool call was found"
    EXPECTED_TOOL_CALL_BATCH = "A tool-call batch that matches the expected batch was found"


class ActionComparisonResult(BaseModel):
    matches: bool
    category: StepRewardCategory
    similarity_score: float | None = None


class ToolCallComparatorConfig(BaseModel):
    string_similarity_threshold: float
    floating_point_comparison_threshold: float = 1e-6
    ignored_argument_keys_by_tool: dict[str, list[str]] = Field(default_factory=dict)


class ActionComparator(BaseModel):
    config: ToolCallComparatorConfig

    def compare_action(
        self,
        expected_action: ExpectedAction,
        actual_action: ExpectedAction,
        declared_tools: list[Any] | None = None,
        threshold_override: float | None = None,
        harness: str = "generic",
    ) -> ActionComparisonResult:
        declared_tool_schemas = self.build_declared_tool_schema_map(declared_tools)

        match expected_action.type:
            case "message":
                if actual_action.type != "message":
                    return ActionComparisonResult(
                        matches=False,
                        category=StepRewardCategory.ACTION_TYPE_MISMATCH,
                    )
                return self.compare_message(actual_action)
            case "function_call":
                if actual_action.type != "function_call":
                    return ActionComparisonResult(
                        matches=False,
                        category=StepRewardCategory.ACTION_TYPE_MISMATCH,
                    )
                return self.compare_tool_call(
                    expected_tool_call=expected_action,
                    actual_tool_call=actual_action,
                    declared_tool_schemas=declared_tool_schemas,
                    threshold_override=threshold_override,
                    harness=harness,
                )
            case "function_call_batch":
                if actual_action.type != "function_call_batch":
                    return ActionComparisonResult(
                        matches=False,
                        category=StepRewardCategory.ACTION_TYPE_MISMATCH,
                    )
                return self.compare_tool_call_batch(
                    expected_batch=expected_action,
                    actual_batch=actual_action,
                    declared_tool_schemas=declared_tool_schemas,
                    threshold_override=threshold_override,
                    harness=harness,
                )
            case _:
                raise NotImplementedError

    def compare_message(self, actual_message: MessageAction) -> ActionComparisonResult:
        if actual_message.content.strip():
            return ActionComparisonResult(
                matches=True,
                category=StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND,
            )
        return ActionComparisonResult(
            matches=False,
            category=StepRewardCategory.EMPTY_MESSAGE,
        )

    def compare_tool_call(
        self,
        expected_tool_call: FunctionCallAction,
        actual_tool_call: FunctionCallAction,
        declared_tool_schemas: dict[str, dict[str, Any]],
        threshold_override: float | None = None,
        harness: str = "generic",
    ) -> ActionComparisonResult:
        expected_arguments_result = self.decode_arguments(expected_tool_call.arguments)
        if expected_arguments_result.category is not None:
            return expected_arguments_result

        actual_arguments_result = self.decode_arguments(actual_tool_call.arguments)
        if actual_arguments_result.category is not None:
            return actual_arguments_result

        expected_arguments = expected_arguments_result.arguments
        actual_arguments = actual_arguments_result.arguments
        assert expected_arguments is not None
        assert actual_arguments is not None

        schema_validation_result = self.validate_against_declared_tool_schema(
            tool_name=actual_tool_call.name,
            actual_arguments=actual_arguments,
            declared_tool_schemas=declared_tool_schemas,
        )
        if schema_validation_result is not None:
            return schema_validation_result

        if expected_tool_call.name != actual_tool_call.name:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.UNEXPECTED_TOOL,
            )

        if not set(actual_arguments).issubset(expected_arguments):
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.UNEXPECTED_ARGUMENT_KEYS,
            )

        harness_specific_result = self.compare_harness_specific_tool_call(
            harness=harness,
            tool_name=expected_tool_call.name,
            expected_arguments=expected_arguments,
            actual_arguments=actual_arguments,
            threshold_override=threshold_override,
        )
        if harness_specific_result is not None:
            return harness_specific_result

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
        )

    def compare_tool_call_batch(
        self,
        expected_batch: FunctionCallBatchAction,
        actual_batch: FunctionCallBatchAction,
        declared_tool_schemas: dict[str, dict[str, Any]],
        threshold_override: float | None = None,
        harness: str = "generic",
    ) -> ActionComparisonResult:
        if len(expected_batch.calls) != len(actual_batch.calls):
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT,
            )

        expected_calls = sorted(
            expected_batch.calls,
            key=lambda call: self.batch_call_sort_key(call, harness=harness),
        )
        actual_calls = sorted(
            actual_batch.calls,
            key=lambda call: self.batch_call_sort_key(call, harness=harness),
        )
        for expected_call, actual_call in zip(expected_calls, actual_calls):
            comparison_result = self.compare_tool_call(
                expected_tool_call=expected_call,
                actual_tool_call=actual_call,
                declared_tool_schemas=declared_tool_schemas,
                threshold_override=threshold_override,
                harness=harness,
            )
            if not comparison_result.matches:
                return comparison_result

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL_BATCH,
        )

    def compare_exec_command(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
        threshold_override: float | None = None,
    ) -> ActionComparisonResult:
        actual_cmd = actual_arguments.get("cmd")
        if not isinstance(actual_cmd, str):
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.EXEC_COMMAND_MISSING_CMD,
            )

        expected_cmd = expected_arguments.get("cmd")
        if not isinstance(expected_cmd, str):
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.EXEC_COMMAND_MISSING_CMD,
            )

        normalized_expected_cmd = self.normalize_command_text(expected_cmd)
        normalized_actual_cmd = self.normalize_command_text(actual_cmd)
        similarity_score = SequenceMatcher(None, normalized_expected_cmd, normalized_actual_cmd).ratio()
        threshold = self.get_string_similarity_threshold(threshold_override)

        if similarity_score < threshold:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.EXEC_COMMAND_CMD_SIMILARITY_BELOW_THRESHOLD,
                similarity_score=similarity_score,
            )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
            similarity_score=similarity_score,
        )

    def compare_update_plan(self, actual_arguments: dict[str, Any]) -> ActionComparisonResult:
        if self.is_non_empty_value(actual_arguments.get("plan")):
            return ActionComparisonResult(
                matches=True,
                category=StepRewardCategory.EXPECTED_TOOL_CALL,
            )

        return ActionComparisonResult(
            matches=False,
            category=StepRewardCategory.UPDATE_PLAN_EMPTY_PLAN,
        )

    def compare_execute_python(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
        threshold_override: float | None = None,
    ) -> ActionComparisonResult:
        actual_code = actual_arguments.get("code")
        if not isinstance(actual_code, str) or not actual_code.strip():
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.EXECUTE_PYTHON_MISSING_CODE,
            )

        expected_code = expected_arguments.get("code")
        if not isinstance(expected_code, str) or not expected_code.strip():
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.EXECUTE_PYTHON_MISSING_CODE,
            )

        similarity_score = SequenceMatcher(None, expected_code, actual_code).ratio()
        threshold = self.get_string_similarity_threshold(threshold_override)
        if similarity_score < threshold:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.EXECUTE_PYTHON_CODE_SIMILARITY_BELOW_THRESHOLD,
                similarity_score=similarity_score,
            )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
            similarity_score=similarity_score,
        )

    def compare_return_result(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
        threshold_override: float | None = None,
    ) -> ActionComparisonResult:
        if "result" not in actual_arguments:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.RETURN_RESULT_MISSING_RESULT,
            )

        expected_result = expected_arguments.get("result")
        actual_result = actual_arguments.get("result")
        expected_serialized = json.dumps(expected_result, sort_keys=True, default=str)
        actual_serialized = json.dumps(actual_result, sort_keys=True, default=str)
        similarity_score = SequenceMatcher(None, expected_serialized, actual_serialized).ratio()
        threshold = self.get_string_similarity_threshold(threshold_override)
        if similarity_score < threshold:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.RETURN_RESULT_SIMILARITY_BELOW_THRESHOLD,
                similarity_score=similarity_score,
            )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
            similarity_score=similarity_score,
        )

    def compare_todowrite(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
    ) -> ActionComparisonResult:
        expected_todos = expected_arguments.get("todos")
        actual_todos = actual_arguments.get("todos")

        if expected_todos == []:
            if actual_todos == []:
                return ActionComparisonResult(
                    matches=True,
                    category=StepRewardCategory.EXPECTED_TOOL_CALL,
                )
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
            )

        if not isinstance(expected_todos, list) or not isinstance(actual_todos, list):
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.TODOWRITE_EMPTY_TODOS,
            )

        if len(expected_todos) != len(actual_todos):
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
            )

        for expected_todo, actual_todo in zip(expected_todos, actual_todos):
            if not isinstance(expected_todo, dict) or not isinstance(actual_todo, dict):
                return ActionComparisonResult(
                    matches=False,
                    category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
                )
            if expected_todo.get("status") != actual_todo.get("status"):
                return ActionComparisonResult(
                    matches=False,
                    category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
                )
            if expected_todo.get("priority") != actual_todo.get("priority"):
                return ActionComparisonResult(
                    matches=False,
                    category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
                )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
        )

    def compare_harness_specific_tool_call(
        self,
        harness: str,
        tool_name: str,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
        threshold_override: float | None = None,
    ) -> ActionComparisonResult | None:
        match harness:
            case "codex":
                match tool_name:
                    case "exec_command":
                        return self.compare_exec_command(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                            threshold_override=threshold_override,
                        )
                    case "update_plan":
                        return self.compare_update_plan(actual_arguments)
            case "agent006":
                match tool_name:
                    case "execute_python":
                        return self.compare_execute_python(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                            threshold_override=threshold_override,
                        )
                    case "return_result":
                        return self.compare_return_result(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                            threshold_override=threshold_override,
                        )
            case "opencode":
                match tool_name:
                    case "bash":
                        return self.compare_bash(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                            threshold_override=threshold_override,
                        )
                    case "todowrite":
                        return self.compare_todowrite(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                        )
                    case "read":
                        return self.compare_read(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                        )
                    case "write":
                        return self.compare_write(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                        )
                    case "edit":
                        return self.compare_edit(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                        )
                    case "glob":
                        return self.compare_selected_arguments(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                            keys=("pattern", "path"),
                        )
                    case "grep":
                        return self.compare_selected_arguments(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                            keys=("pattern", "path", "include"),
                        )
                    case "webfetch":
                        return self.compare_selected_arguments(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                            keys=("url", "format"),
                        )
                    case "skill":
                        return self.compare_selected_arguments(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                            keys=("name",),
                        )
                    case "task":
                        return self.compare_selected_arguments(
                            expected_arguments=expected_arguments,
                            actual_arguments=actual_arguments,
                            keys=("subagent_type",),
                        )

        return None

    def compare_read(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
    ) -> ActionComparisonResult:
        expected_file_path = expected_arguments.get("filePath")
        actual_file_path = actual_arguments.get("filePath")
        if expected_file_path != actual_file_path:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
            )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
        )

    def compare_write(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
    ) -> ActionComparisonResult:
        expected_file_path = expected_arguments.get("filePath")
        actual_file_path = actual_arguments.get("filePath")
        if expected_file_path != actual_file_path:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
            )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
        )

    def compare_edit(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
    ) -> ActionComparisonResult:
        expected_file_path = expected_arguments.get("filePath")
        actual_file_path = actual_arguments.get("filePath")
        if expected_file_path != actual_file_path:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
            )

        actual_old_string = actual_arguments.get("oldString")
        actual_new_string = actual_arguments.get("newString")
        if actual_old_string == actual_new_string:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
            )

        expected_replace_all = expected_arguments.get("replaceAll")
        actual_replace_all = actual_arguments.get("replaceAll")
        if expected_replace_all != actual_replace_all:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
            )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
        )

    def compare_bash(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
        threshold_override: float | None = None,
    ) -> ActionComparisonResult:
        actual_command = actual_arguments.get("command")
        if not isinstance(actual_command, str):
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.BASH_MISSING_COMMAND,
            )

        expected_command = expected_arguments.get("command")
        if not isinstance(expected_command, str):
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.BASH_MISSING_COMMAND,
            )

        normalized_expected_command = self.normalize_command_text(expected_command)
        normalized_actual_command = self.normalize_command_text(actual_command)
        similarity_score = SequenceMatcher(
            None,
            normalized_expected_command,
            normalized_actual_command,
        ).ratio()
        threshold = self.get_string_similarity_threshold(threshold_override)
        if similarity_score < threshold:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.BASH_COMMAND_SIMILARITY_BELOW_THRESHOLD,
                similarity_score=similarity_score,
            )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
            similarity_score=similarity_score,
        )

    def compare_exact_arguments(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
    ) -> ActionComparisonResult:
        if expected_arguments != actual_arguments:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
            )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
        )

    def compare_selected_arguments(
        self,
        expected_arguments: dict[str, Any],
        actual_arguments: dict[str, Any],
        keys: tuple[str, ...],
    ) -> ActionComparisonResult:
        for key in keys:
            if expected_arguments.get(key) != actual_arguments.get(key):
                return ActionComparisonResult(
                    matches=False,
                    category=StepRewardCategory.ARGUMENT_VALUE_MISMATCH,
                )

        return ActionComparisonResult(
            matches=True,
            category=StepRewardCategory.EXPECTED_TOOL_CALL,
        )

    def decode_arguments(self, arguments: str) -> "DecodedArgumentsResult":
        try:
            decoded_arguments = json.loads(arguments)
        except (JSONDecodeError, UnicodeDecodeError):
            return DecodedArgumentsResult(category=StepRewardCategory.ARGUMENTS_DECODE_ERROR)

        if not isinstance(decoded_arguments, dict):
            return DecodedArgumentsResult(category=StepRewardCategory.ARGUMENTS_NOT_OBJECT)

        return DecodedArgumentsResult(arguments=decoded_arguments)

    def validate_against_declared_tool_schema(
        self,
        tool_name: str,
        actual_arguments: dict[str, Any],
        declared_tool_schemas: dict[str, dict[str, Any]],
    ) -> ActionComparisonResult | None:
        tool_schema = declared_tool_schemas.get(tool_name)
        if tool_schema is None:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.TOOL_SCHEMA_NOT_FOUND,
            )

        try:
            validate_against_schema_openapi(actual_arguments, tool_schema)
        except Exception:
            return ActionComparisonResult(
                matches=False,
                category=StepRewardCategory.TOOL_SCHEMA_VALIDATION_FAILED,
            )

        return None

    def build_declared_tool_schema_map(self, declared_tools: list[Any] | None) -> dict[str, dict[str, Any]]:
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

    def get_string_similarity_threshold(self, threshold_override: float | None = None) -> float:
        if threshold_override is not None:
            return threshold_override
        return self.config.string_similarity_threshold

    def batch_call_sort_key(self, call: FunctionCallAction, harness: str = "generic") -> tuple[str, str]:
        normalized_arguments = self.normalize_arguments_for_batch_sort(call.arguments)

        match harness:
            case "codex":
                return (call.name, normalized_arguments)
            case "agent006":
                return (call.name, normalized_arguments)
            case "opencode":
                return (call.name, normalized_arguments)
            case _:
                return (call.name, normalized_arguments)

    def normalize_arguments_for_batch_sort(self, arguments: str) -> str:
        decoded_arguments = self.decode_arguments(arguments)
        if decoded_arguments.arguments is None:
            return arguments

        return json.dumps(
            decoded_arguments.arguments,
            sort_keys=True,
            separators=(",", ":"),
        )

    def normalize_command_text(self, command_text: str) -> str:
        return command_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    def is_non_empty_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict, tuple, set)):
            return bool(value)
        return True

class DecodedArgumentsResult(BaseModel):
    arguments: dict[str, Any] | None = None
    category: StepRewardCategory | None = None
