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
                "name": "execute_python",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "return_result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {},
                    },
                    "required": ["result"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                        "workdir": {"type": "string"},
                        "timeout": {"type": "number"},
                    },
                    "required": ["command", "description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["filePath"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["filePath", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "oldString": {"type": "string"},
                        "newString": {"type": "string"},
                        "replaceAll": {"type": "boolean"},
                    },
                    "required": ["filePath", "oldString", "newString"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "include": {"type": "string"},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "webfetch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "format": {"type": "string"},
                        "timeout": {"type": "number"},
                    },
                    "required": ["url", "format"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "skill",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "subagent_type": {"type": "string"},
                        "task_id": {"type": "string"},
                        "command": {"type": "string"},
                    },
                    "required": ["description", "prompt", "subagent_type"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "todowrite",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "status": {"type": "string"},
                                    "priority": {"type": "string"},
                                },
                                "required": ["content", "status", "priority"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["todos"],
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
            harness="codex",
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
            harness="codex",
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
            harness="codex",
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
            harness="codex",
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
            harness="codex",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_bash_uses_similarity_and_ignores_description(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="bash",
                arguments=json.dumps(
                    {
                        "command": "ls -la /repo\n",
                        "description": "Lists the repo files",
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="bash",
                arguments=json.dumps(
                    {
                        "command": "ls -la /repo",
                        "description": "Show repository listing",
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL
        assert comparison_result.similarity_score == approx(1.0)

    def test_compare_bash_ignores_workdir_mismatch(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="bash",
                arguments=json.dumps(
                    {
                        "command": "git status",
                        "description": "Shows git status",
                        "workdir": "/repo",
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="bash",
                arguments=json.dumps(
                    {
                        "command": "git status",
                        "description": "Status",
                        "workdir": "/tmp",
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL
        assert comparison_result.similarity_score == approx(1.0)

    def test_compare_bash_rejects_extra_metadata_keys_when_expected_omits_them(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="bash",
                arguments=json.dumps({"command": "git status"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="bash",
                arguments=json.dumps(
                    {
                        "command": "git status",
                        "description": "Show git status",
                        "workdir": "/repo",
                        "timeout": 120000,
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.UNEXPECTED_ARGUMENT_KEYS

    def test_compare_read_requires_matching_file_path(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="read",
                arguments=json.dumps({"filePath": "/app/a.txt", "limit": 10}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="read",
                arguments=json.dumps({"filePath": "/app/b.txt", "limit": 10}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.ARGUMENT_VALUE_MISMATCH

    def test_compare_read_ignores_offset_and_limit_mismatch(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="read",
                arguments=json.dumps({"filePath": "/app/a.txt", "offset": 1, "limit": 10}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="read",
                arguments=json.dumps({"filePath": "/app/a.txt", "offset": 200, "limit": 5}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_read_rejects_extra_offset_and_limit_keys_when_expected_omits_them(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="read",
                arguments=json.dumps({"filePath": "/app/a.txt"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="read",
                arguments=json.dumps({"filePath": "/app/a.txt", "offset": 10, "limit": 5}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.UNEXPECTED_ARGUMENT_KEYS

    def test_compare_write_requires_matching_file_path_only(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="write",
                arguments=json.dumps({"filePath": "/app/a.txt", "content": "expected text"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="write",
                arguments=json.dumps({"filePath": "/app/a.txt", "content": "different text"}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_write_allows_empty_content(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="write",
                arguments=json.dumps({"filePath": "/app/a.txt", "content": "expected text"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="write",
                arguments=json.dumps({"filePath": "/app/a.txt", "content": "   "}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_edit_requires_matching_file_path_and_allows_different_strings(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="edit",
                arguments=json.dumps(
                    {
                        "filePath": "/app/a.txt",
                        "oldString": "foo",
                        "newString": "bar",
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="edit",
                arguments=json.dumps(
                    {
                        "filePath": "/app/a.txt",
                        "oldString": "alpha",
                        "newString": "beta",
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_edit_rejects_noop_edit(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="edit",
                arguments=json.dumps(
                    {
                        "filePath": "/app/a.txt",
                        "oldString": "foo",
                        "newString": "bar",
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="edit",
                arguments=json.dumps(
                    {
                        "filePath": "/app/a.txt",
                        "oldString": "same",
                        "newString": "same",
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.ARGUMENT_VALUE_MISMATCH

    def test_compare_edit_requires_exact_replace_all_match(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="edit",
                arguments=json.dumps(
                    {
                        "filePath": "/app/a.txt",
                        "oldString": "foo",
                        "newString": "bar",
                        "replaceAll": True,
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="edit",
                arguments=json.dumps(
                    {
                        "filePath": "/app/a.txt",
                        "oldString": "alpha",
                        "newString": "beta",
                        "replaceAll": False,
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.ARGUMENT_VALUE_MISMATCH

    def test_compare_glob_requires_exact_pattern_and_optional_path(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="glob",
                arguments=json.dumps({"pattern": "**/*.py"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="glob",
                arguments=json.dumps({"pattern": "**/*.py"}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_grep_requires_exact_pattern_path_and_include(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="grep",
                arguments=json.dumps({"pattern": "TODO", "path": "/repo", "include": "*.py"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="grep",
                arguments=json.dumps({"pattern": "TODO", "path": "/repo", "include": "*.js"}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.ARGUMENT_VALUE_MISMATCH

    def test_compare_webfetch_ignores_timeout_and_matches_url_and_format(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="webfetch",
                arguments=json.dumps({"url": "https://example.com", "format": "markdown", "timeout": 10}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="webfetch",
                arguments=json.dumps({"url": "https://example.com", "format": "markdown", "timeout": 30}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_skill_requires_exact_name_only(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="skill",
                arguments=json.dumps({"name": "research-logbook-runs"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="skill",
                arguments=json.dumps({"name": "different-skill"}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.ARGUMENT_VALUE_MISMATCH

    def test_compare_task_requires_exact_subagent_type_only(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="task",
                arguments=json.dumps(
                    {
                        "description": "Inspect repo",
                        "prompt": "Search for API handlers",
                        "subagent_type": "explore",
                        "command": "/inspect",
                        "task_id": "task-1",
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="task",
                arguments=json.dumps(
                    {
                        "description": "Totally different",
                        "prompt": "Do another search",
                        "subagent_type": "explore",
                        "command": "/search-api",
                        "task_id": "task-2",
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
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
            harness="codex",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.UPDATE_PLAN_EMPTY_PLAN

    def test_compare_agent006_execute_python_uses_similarity_and_records_score(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="execute_python",
                arguments=json.dumps({"code": "print('hello')\n"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="execute_python",
                arguments=json.dumps({"code": "print('hello')\n"}),
            ),
            declared_tools=declared_tools,
            harness="agent006",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL
        assert comparison_result.similarity_score == approx(1.0)

    def test_compare_agent006_execute_python_rejects_blank_code(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="execute_python",
                arguments=json.dumps({"code": "print('hello')"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="execute_python",
                arguments=json.dumps({"code": "   "}),
            ),
            declared_tools=declared_tools,
            harness="agent006",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.ARGUMENT_VALUE_MISMATCH

    def test_compare_agent006_return_result_uses_canonical_json_similarity(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="return_result",
                arguments=json.dumps({"result": {"a": 1, "b": 2}}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="return_result",
                arguments=json.dumps({"result": {"b": 2, "a": 1}}),
            ),
            declared_tools=declared_tools,
            harness="agent006",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL
        assert comparison_result.similarity_score == approx(1.0)

    def test_compare_agent006_return_result_schema_validation_happens_before_missing_result_check(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="return_result",
                arguments=json.dumps({"result": {"a": 1}}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="return_result",
                arguments=json.dumps({}),
            ),
            declared_tools=declared_tools,
            harness="agent006",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.TOOL_SCHEMA_VALIDATION_FAILED

    def test_compare_todowrite_compares_status_and_priority_only(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="todowrite",
                arguments=json.dumps(
                    {
                        "todos": [
                            {
                                "content": "Inspect repo",
                                "status": "pending",
                                "priority": "high",
                            }
                        ]
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="todowrite",
                arguments=json.dumps(
                    {
                        "todos": [
                            {
                                "content": "Completely different content",
                                "status": "pending",
                                "priority": "high",
                            }
                        ]
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_todowrite_rejects_status_mismatch(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="todowrite",
                arguments=json.dumps(
                    {
                        "todos": [
                            {
                                "content": "Inspect repo",
                                "status": "pending",
                                "priority": "high",
                            }
                        ]
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="todowrite",
                arguments=json.dumps(
                    {
                        "todos": [
                            {
                                "content": "Different content",
                                "status": "completed",
                                "priority": "high",
                            }
                        ]
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.ARGUMENT_VALUE_MISMATCH

    def test_compare_todowrite_rejects_length_mismatch(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="todowrite",
                arguments=json.dumps(
                    {
                        "todos": [
                            {
                                "content": "Inspect repo",
                                "status": "pending",
                                "priority": "high",
                            }
                        ]
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="todowrite",
                arguments=json.dumps(
                    {
                        "todos": [
                            {
                                "content": "Inspect repo",
                                "status": "pending",
                                "priority": "high",
                            },
                            {
                                "content": "Run tests",
                                "status": "pending",
                                "priority": "low",
                            },
                        ]
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.ARGUMENT_VALUE_MISMATCH

    def test_compare_todowrite_accepts_matching_empty_clear(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="todowrite",
                arguments=json.dumps({"todos": []}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="todowrite",
                arguments=json.dumps({"todos": []}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_compare_multiple_tool_calls_sorts_by_tool_name_and_compares_pairwise(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        expected_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(
                    type="function_call", name="write_stdin", arguments='{"session_id": 7, "chars": ""}'
                ),
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
            harness="codex",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL_BATCH

    def test_compare_duplicate_name_batch_sorts_by_name_then_arguments(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        expected_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(
                    type="function_call",
                    name="read",
                    arguments=json.dumps({"filePath": "/app/a.txt", "limit": 10}),
                ),
                FunctionCallAction(
                    type="function_call",
                    name="read",
                    arguments=json.dumps({"filePath": "/app/b.txt", "limit": 10}),
                ),
            ],
        )
        actual_batch = FunctionCallBatchAction(
            type="function_call_batch",
            ordered=True,
            calls=[
                FunctionCallAction(
                    type="function_call",
                    name="read",
                    arguments=json.dumps({"filePath": "/app/b.txt", "limit": 10}),
                ),
                FunctionCallAction(
                    type="function_call",
                    name="read",
                    arguments=json.dumps({"filePath": "/app/a.txt", "limit": 10}),
                ),
            ],
        )

        comparison_result = action_comparator.compare_action(
            expected_action=expected_batch,
            actual_action=actual_batch,
            declared_tools=declared_tools,
            harness="opencode",
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
            harness="codex",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.EXEC_COMMAND_CMD_SIMILARITY_BELOW_THRESHOLD

    def test_codex_does_not_inherit_opencode_bash_matching_rules(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="bash",
                arguments=json.dumps(
                    {
                        "command": "pwd",
                        "description": "Show current directory",
                    }
                ),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="bash",
                arguments=json.dumps(
                    {
                        "command": "git status",
                        "description": "Show git status",
                    }
                ),
            ),
            declared_tools=declared_tools,
            harness="codex",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_opencode_does_not_inherit_codex_exec_command_matching_rules(
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
                arguments=json.dumps({"cmd": "git status"}),
            ),
            declared_tools=declared_tools,
            harness="opencode",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

    def test_codex_does_not_inherit_agent006_execute_python_matching_rules(
        self,
        action_comparator: ActionComparator,
        declared_tools: list[dict],
    ) -> None:
        comparison_result = action_comparator.compare_action(
            expected_action=FunctionCallAction(
                type="function_call",
                name="execute_python",
                arguments=json.dumps({"code": "print('teacher')"}),
            ),
            actual_action=FunctionCallAction(
                type="function_call",
                name="execute_python",
                arguments=json.dumps({"code": "print('policy')"}),
            ),
            declared_tools=declared_tools,
            harness="codex",
        )
        assert comparison_result.matches is True
        assert comparison_result.category == StepRewardCategory.EXPECTED_TOOL_CALL

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
            harness="codex",
        )
        assert comparison_result.matches is False
        assert comparison_result.category == StepRewardCategory.EXEC_COMMAND_CMD_SIMILARITY_BELOW_THRESHOLD
