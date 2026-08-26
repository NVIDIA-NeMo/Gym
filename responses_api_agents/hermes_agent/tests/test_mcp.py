# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.openai_utils import NeMoGymResponse
from responses_api_agents.hermes_agent.mcp import (
    hermes_mcp_tool_aliases,
    response_mcp_tool_call_provenance,
)


def _response(output: list[dict]) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "resp-1",
            "created_at": 1,
            "model": "model",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def test_builds_exact_alias_map_with_raw_identity():
    aliases = hermes_mcp_tool_aliases(
        {
            "mcp": {
                "server_name": "work-place",
                "tool_names": ["reply-email"],
            }
        }
    )

    assert aliases == {
        "mcp__work_place__reply_email": {
            "server_name": "work-place",
            "tool_name": "reply-email",
        }
    }


def test_alias_map_is_unavailable_without_advertised_tool_names():
    assert hermes_mcp_tool_aliases({}) is None
    assert hermes_mcp_tool_aliases({"mcp": {"server_name": "workplace"}}) is None


def test_sanitization_collision_is_omitted():
    aliases = hermes_mcp_tool_aliases(
        {
            "mcp": {
                "server_name": "workplace",
                "tool_names": ["reply-email", "reply_email"],
            }
        }
    )

    assert aliases == {}


def test_provenance_uses_call_id_and_ignores_non_mcp_calls():
    response = _response(
        [
            {
                "type": "function_call",
                "call_id": "mcp-call",
                "name": "mcp__workplace__reply",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "builtin-call",
                "name": "terminal",
                "arguments": "{}",
            },
        ]
    )

    provenance = response_mcp_tool_call_provenance(
        response,
        {
            "mcp__workplace__reply": {
                "server_name": "workplace",
                "tool_name": "reply",
            }
        },
    )

    assert provenance == {
        "mcp-call": {
            "server_name": "workplace",
            "tool_name": "reply",
        }
    }
