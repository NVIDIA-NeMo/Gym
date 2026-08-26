# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from responses_api_agents.hermes_agent.trajectory import (
    normalize_hermes_chat_messages,
    project_hermes_response_messages,
)


def test_normalize_dispatcher_call_exposes_actual_tool():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "tool_call",
                        "arguments": (
                            '{"name":"mcp__workplace__email_reply_email",'
                            '"arguments":{"email_id":"57","body":"Thanks"}}'
                        ),
                    },
                }
            ],
        }
    ]

    normalized = normalize_hermes_chat_messages(messages)

    assert normalized[0]["tool_calls"][0]["function"] == {
        "name": "mcp__workplace__email_reply_email",
        "arguments": '{"email_id": "57", "body": "Thanks"}',
    }
    assert messages[0]["tool_calls"][0]["function"]["name"] == "tool_call"


def test_response_projection_omits_internal_and_unexecuted_dispatch_calls():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "describe",
                    "function": {"name": "tool_describe", "arguments": '{"name":"mcp__workplace__reply"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "describe", "content": '{"name":"mcp__workplace__reply"}'},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "direct",
                    "function": {"name": "mcp__workplace__reply", "arguments": '{"body":"Thanks"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "direct",
            "content": "Tool 'mcp__workplace__reply' does not exist. Available tools: tool_call",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "executed",
                    "function": {
                        "name": "tool_call",
                        "arguments": '{"name":"mcp__workplace__reply","arguments":{"body":"Thanks"}}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "executed", "content": "Email replied successfully."},
        {"role": "assistant", "content": "done"},
    ]

    projected = project_hermes_response_messages(messages)

    assert [message.get("role") for message in projected] == ["assistant", "tool", "assistant"]
    assert projected[0]["tool_calls"][0]["function"] == {
        "name": "mcp__workplace__reply",
        "arguments": '{"body": "Thanks"}',
    }
    assert projected[1]["tool_call_id"] == "executed"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "tool_describe"
