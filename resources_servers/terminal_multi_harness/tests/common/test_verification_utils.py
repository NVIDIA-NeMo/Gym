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

from resources_servers.terminal_multi_harness.common.verification_utils import (
    ActionComparator,
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
    StepRewardCategory,
    ToolCallComparatorConfig,
)


class TestActionComparator:
    @fixture
    def action_comparator(self) -> ActionComparator:
        comparator_config = ToolCallComparatorConfig(
            word_count_similarity_threshold=0.1,
            ignored_argument_keys_by_tool={
                "exec_command": ["yield_time_ms", "max_output_tokens"],
            },
        )
        return ActionComparator(config=comparator_config)

    def test_compare_message(self, action_comparator: ActionComparator) -> None:
        assert action_comparator.compare_message(
            MessageAction(type="message", content="Birds are animals."),
            MessageAction(type="message", content="The birds fly."),
        ) == (True, StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND)

        assert action_comparator.compare_message(
            MessageAction(type="message", content="hello"),
            MessageAction(type="message", content="goodbye"),
        ) == (False, StepRewardCategory.MESSAGE_CONTENT_DIFFERENT)

    def test_compare_tool_call(self, action_comparator: ActionComparator) -> None:
        expected_function_call = FunctionCallAction(
            type="function_call",
            name="exec_command",
            arguments=json.dumps(
                {
                    "cmd": "pwd",
                    "workdir": "/repo",
                }
            ),
        )

        ignored_field_tool_call = FunctionCallAction(
            type="function_call",
            name="exec_command",
            arguments=json.dumps(
                {
                    "cmd": "pwd",
                    "workdir": "/repo",
                    "yield_time_ms": 1000,
                }
            ),
        )
        assert action_comparator.compare_tool_call(expected_function_call, ignored_field_tool_call) == (
            True,
            StepRewardCategory.EXPECTED_TOOL_CALL,
        )

        different_tool_call = FunctionCallAction(
            type="function_call",
            name="write_stdin",
            arguments=json.dumps({"chars": "pwd\n"}),
        )
        assert action_comparator.compare_tool_call(expected_function_call, different_tool_call) == (
            False,
            StepRewardCategory.UNEXPECTED_TOOL,
        )

    def test_compare_tool_call_batch(self, action_comparator: ActionComparator) -> None:
        expected_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "pwd"}'),
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "git status"}'),
            ],
        )
        matching_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "pwd"}'),
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "git status"}'),
            ],
        )
        assert action_comparator.compare_tool_call_batch(expected_batch, matching_batch) == (
            True,
            StepRewardCategory.EXPECTED_TOOL_CALL_BATCH,
        )

        reordered_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "git status"}'),
                FunctionCallAction(type="function_call", name="exec_command", arguments='{"cmd": "pwd"}'),
            ],
        )
        assert action_comparator.compare_tool_call_batch(expected_batch, reordered_batch) == (
            False,
            StepRewardCategory.ARGUMENT_VALUE_DIFFERENT,
        )

        unordered_expected_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=False,
            calls=expected_batch.calls,
        )
        assert action_comparator.compare_tool_call_batch(unordered_expected_batch, reordered_batch) == (
            True,
            StepRewardCategory.EXPECTED_TOOL_CALL_BATCH,
        )

    def test_compare_batch_tool_arguments(self, action_comparator: ActionComparator) -> None:
        expected_batch_call = FunctionCallAction(
            type="function_call",
            name="batch",
            arguments=json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool": "exec_command",
                            "parameters": {
                                "cmd": "pwd",
                                "workdir": "/repo",
                            },
                        }
                    ]
                }
            ),
        )
        actual_batch_call = FunctionCallAction(
            type="function_call",
            name="batch",
            arguments=json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool": "exec_command",
                            "parameters": {
                                "cmd": "pwd",
                                "workdir": "/repo",
                                "yield_time_ms": 1000,
                            },
                        }
                    ]
                }
            ),
        )
        assert action_comparator.compare_tool_call(expected_batch_call, actual_batch_call) == (
            True,
            StepRewardCategory.EXPECTED_TOOL_CALL,
        )
