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

from pytest import approx, fixture

from resources_servers.terminal_multi_harness.common.verification_utils import (
    ActionComparator,
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
    StepRewardCategory,
    ToolCallComparatorConfig,
)


def build_declared_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "workdir": {"type": "string"},
                    },
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_plan",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {"type": "string"},
                                    "status": {"type": "string"},
                                },
                                "required": ["step", "status"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["plan"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_stdin",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "integer"},
                        "chars": {"type": "string"},
                    },
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
            },
        },
    ]


class TestActionComparator:
    @fixture
    def action_comparator(self) -> ActionComparator:
        comparator_config = ToolCallComparatorConfig(
            string_similarity_threshold=0.9,
        )
        return ActionComparator(config=comparator_config)

    @fixture
    def declared_tools(self) -> list[dict]:
        return build_declared_tools()

    def test_compare_message_ignores_expected_text(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=MessageAction(type="message", content="teacher final answer"),
            actual_action=MessageAction(type="message", content="policy says something else"),
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND

    def test_compare_message_requires_non_empty_text(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=MessageAction(type="message", content="teacher final answer"),
            actual_action=MessageAction(type="message", content="   "),
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.EMPTY_MESSAGE

    def test_compare_exec_command_uses_similarity_and_records_score(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="exec_command",
                arguments=json.dumps({"cmd": "pwd\n"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="exec_command",
                arguments=json.dumps({"cmd": "pwd"}),
            ),
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL
        assert comparison_result.similarity_score == approx(1.0)

    def test_compare_exec_command_rejects_extra_actual_keys(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="exec_command",
                arguments=json.dumps({"cmd": "pwd"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="exec_command",
                arguments=json.dumps({"cmd": "pwd", "workdir": "/repo"}),
            ),
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.UNEXPECTED_ARGUMENT_KEYS

    def test_compare_exec_command_schema_validation_happens_before_match(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="exec_command",
                arguments=json.dumps({"cmd": "pwd"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="exec_command",
                arguments=json.dumps({"cmd": 123}),
            ),
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.TOOL_SCHEMA_VALIDATION_FAILED

    def test_compare_tool_name_still_must_match_after_schema_validation(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="exec_command",
                arguments=json.dumps({"cmd": "pwd"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="write_stdin",
                arguments=json.dumps({"session_id": 7}),
            ),
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.UNEXPECTED_TOOL

    def test_compare_update_plan_only_checks_non_empty_plan(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="update_plan",
                arguments=json.dumps({"plan": [{"step": "a", "status": "pending"}]}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="update_plan",
                arguments=json.dumps({"plan": [{"step": "different", "status": "completed"}]}),
            ),
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_update_plan_rejects_empty_plan(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="update_plan",
                arguments=json.dumps({"plan": [{"step": "a", "status": "pending"}]}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="update_plan",
                arguments=json.dumps({"plan": []}),
            ),
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.UPDATE_PLAN_EMPTY_PLAN

    def test_compare_multiple_tool_calls_sorts_by_tool_name_and_compares_pairwise(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        expected_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(type="function_call", name="write_stdin", arguments='{"session_id": 7, "chars": ""}'),
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "pwd"}'),
            ],
        )
        actual_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "pwd"}'),
                FunctionCallAction(type="function_call", name="write_stdin", arguments='{"session_id": 7}'),
            ],
        )

        comparison_result = action_comparator.compare_action(
            expected_action=expected_batch,
            actual_action=actual_batch,
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL_BATCH

    def test_compare_multiple_tool_calls_fails_when_one_pair_fails(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        expected_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(type="function_call", name="write_stdin", arguments='{"session_id": 7}'),
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "pwd"}'),
            ],
        )
        actual_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "git status"}'),
                FunctionCallAction(type="function_call", name="write_stdin", arguments='{"session_id": 7}'),
            ],
        )

        comparison_result = action_comparator.compare_action(
            expected_action=expected_batch,
            actual_action=actual_batch,
            declared_tools=declared_tools,
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.EXEC_COMMAND_CMD_SIMILARITY_BELOW_THRESHOLD

    def test_compare_exec_command_uses_threshold_override(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="exec_command",
                arguments=json.dumps({"cmd": "pwd"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="exec_command",
                arguments=json.dumps({"cmd": "pwd\n"}),
            ),
            declared_tools=declared_tools,
            threshold_override=1.01,
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.EXEC_COMMAND_CMD_SIMILARITY_BELOW_THRESHOLD
