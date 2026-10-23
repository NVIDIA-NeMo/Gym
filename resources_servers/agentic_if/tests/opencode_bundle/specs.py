# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RuntimeSpecs shared by the bundle tests: an unrestricted row, a private-shaped row, and a continuation."""

from collections.abc import Callable

from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


SELECTED = ["bash", "edit", "glob", "grep", "read", "task", "todowrite", "write"]


def unrestricted_spec() -> RuntimeSpec:
    """The shape of the public example rows at 00be1884: no tool selection, OpenCode's built-in tool inventory."""
    return RuntimeSpec(
        harness="opencode",
        task_backend="tb21",
        checks=[
            {
                "id": "final-token",
                "kind": "exact",
                "parameters": {"value": "DONE"},
                "instruction": {"text": "Reply with exactly DONE.", "at": "task_message"},
                "scope": {"places": ["final"]},
            }
        ],
    )


def private_like_spec() -> RuntimeSpec:
    """A row shape that uses every bundle feature: explicit selection, a description on every tool, interfaces with
    JSON-pointer parameter descriptions, read-first guards and a planted bash failure."""
    return RuntimeSpec(
        harness="opencode",
        task_backend="tb21",
        tools={
            "tool_names": {tool: tool for tool in SELECTED},
            "tool_descriptions": {tool: f"Row description of {tool}." for tool in SELECTED},
            "tool_parameter_descriptions": {
                "read": {"/properties/path": "Absolute path of the file."},
                "bash": {"/properties/command": "One shell command."},
            },
            "tool_interfaces": {
                "read": "cw/zed@3/read",
                "edit": "cw/pi@2/edit",
                "bash": "cw/openhands@2/bash",
                "task": "cw/zed@3/task",
            },
            "read_first": {
                "edit": {"unread": "Read {{path}} first.", "changed": "Read {{path}} again."},
                "write": {"unread": "Read {{path}} before overwriting."},
            },
        },
        events=[
            {
                "id": "push",
                "match": {"tool": "bash", "arg": {"field": "command", "regex": "^git push"}},
                "nth": 2,
                "action": "fail",
                "text": "Blocked: pushing is not allowed here.",
            }
        ],
    )


def continuation_spec() -> RuntimeSpec:
    """A continuation row: a restored prefix with one completed bash call and a witnessed rejected call."""
    prefix = {
        "info": {"id": "ses_source", "directory": "/app"},
        "messages": [
            {"info": {"id": "msg_u", "role": "user"}, "parts": [{"type": "text", "text": "Fix it"}]},
            {
                "info": {"id": "msg_a", "role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "call_ls",
                        "state": {"status": "completed", "input": {"command": "ls"}, "output": "a.py\n"},
                    },
                    {
                        "type": "tool",
                        "tool": "invalid",
                        "callID": "call_bad",
                        "state": {
                            "status": "completed",
                            "input": {"tool": "Missing", "error": "unavailable"},
                            "output": "unavailable",
                            "metadata": {
                                "nemo_gym_model_call": {
                                    "function": {"name": "Missing", "arguments": '{ "x": 1 }'},
                                    "available_names": ["bash", "edit", "read", "write"],
                                }
                            },
                        },
                    },
                ],
            },
        ],
    }
    return RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={"tool_names": {tool: tool for tool in ["bash", "edit", "read", "write"]}},
        continuation={
            "mode": "restore",
            "prefix_export": prefix,
            "workspace_diff": "",
            "source_task_sha256": "a" * 64,
            "source_tools_sha256": "b" * 64,
        },
    )


SPECS: dict[str, Callable[[], RuntimeSpec]] = {
    "unrestricted": unrestricted_spec,
    "private_like": private_like_spec,
    "continuation": continuation_spec,
}
