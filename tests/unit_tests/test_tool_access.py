# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import TypeAdapter, ValidationError

from nemo_gym.tool_access import (
    ContextualToolCallRequest,
    ContextualToolCallResponse,
    DirectHTTPToolAccess,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    ToolAccess,
)


def test_tool_access_supports_named_direct_http_and_mcp_servers() -> None:
    accesses = TypeAdapter(list[ToolAccess]).validate_python(
        [
            {
                "kind": "direct_http",
                "name": "typed-tools",
                "required": True,
                "base_url": "http://resources:8000",
                "cookies": {"session": "value"},
            },
            {
                "kind": "mcp",
                "name": "task-tools",
                "required": True,
                "connection": {
                    "transport": "streamable_http",
                    "url": "http://resources:8000/mcp",
                    "headers": {"Authorization": "Bearer scoped"},
                },
            },
        ]
    )

    assert isinstance(accesses[0], DirectHTTPToolAccess)
    assert isinstance(accesses[1], MCPToolAccess)
    assert isinstance(accesses[1].connection, MCPStreamableHTTPConnection)


def test_mcp_http_connection_requires_an_absolute_url() -> None:
    with pytest.raises(ValidationError, match="valid URL"):
        MCPStreamableHTTPConnection(url="/mcp")


def test_tool_access_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MCPStreamableHTTPConnection(url="http://resources:8000/mcp", unknown=True)


def test_contextual_direct_http_calls_separate_arguments_output_and_hidden_context() -> None:
    request = ContextualToolCallRequest(
        arguments={"query": "example"},
        tool_call_context={"turn_id": "turn-1"},
        tool_call_id="call-1",
        round_id="round-1",
        assistant_response={"role": "assistant", "tool_calls": []},
    )
    response = ContextualToolCallResponse(
        output="model-visible",
        tool_call_context={"receipts": [{"call_id": "call-1"}]},
    )

    assert request.arguments == {"query": "example"}
    assert response.output == "model-visible"
    assert response.tool_call_context["receipts"] == [{"call_id": "call-1"}]


def test_direct_http_access_rejects_batch_and_context_modes_together() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        DirectHTTPToolAccess(
            name="tools",
            required=True,
            base_url="http://resources:8000",
            batch_path="/tool_calls",
            tool_call_context={"turn_id": "turn-1"},
        )
