# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.mcp import (
    build_mcp_tool_aliases,
    parse_rollout_mcp_server,
    provenance_from_response_aliases,
    resources_server_base_url,
)
from nemo_gym.openai_utils import NeMoGymResponse


def test_resources_server_base_url_resolves_configured_instance() -> None:
    server_client = MagicMock()
    server_client.global_config_dict = {
        "workplace": {"resources_servers": {"workplace": {"host": "127.0.0.1", "port": 8123}}}
    }
    server_client._build_server_base_url.return_value = "http://127.0.0.1:8123"

    assert resources_server_base_url(server_client, "workplace") == "http://127.0.0.1:8123"


def test_parse_rollout_mcp_server_validates_and_resolves_metadata() -> None:
    server = parse_rollout_mcp_server(
        {
            "mcp": {
                "server_name": "workplace",
                "url_path": "tools/mcp",
                "transport": "streamable_http",
                "headers": {"Authorization": "Bearer token", "X-Retry": 3},
                "tool_names": ["reply", "", 7],
            }
        },
        resources_server_name="fallback",
        resources_server_base_url="http://resources/",
    )

    assert server is not None
    assert server.server_name == "workplace"
    assert server.url == "http://resources/tools/mcp"
    assert server.transport == "streamable-http"
    assert server.headers == {"Authorization": "Bearer token", "X-Retry": "3"}
    assert server.tool_names == ("reply",)


def test_parse_rollout_mcp_server_preserves_absent_metadata_and_tool_names() -> None:
    assert (
        parse_rollout_mcp_server(
            {},
            resources_server_name="workplace",
            resources_server_base_url="http://resources",
        )
        is None
    )
    server = parse_rollout_mcp_server(
        {"mcp": {}},
        resources_server_name="workplace",
        resources_server_base_url="http://resources",
    )
    assert server is not None
    assert server.server_name == "workplace"
    assert server.tool_names is None


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("invalid", "must be an object"),
        ({"server_name": []}, "server_name"),
        ({"url_path": []}, "url_path"),
        ({"transport": []}, "transport"),
        ({"headers": []}, "headers"),
        ({"headers": {"Authorization": {}}}, "scalar values"),
        ({"tool_names": "reply"}, "tool_names"),
    ],
)
def test_parse_rollout_mcp_server_rejects_malformed_metadata(metadata, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_rollout_mcp_server(
            {"mcp": metadata},
            resources_server_name="workplace",
            resources_server_base_url="http://resources",
        )


def test_alias_builder_omits_sanitization_collisions() -> None:
    server = parse_rollout_mcp_server(
        {
            "mcp": {
                "server_name": "workplace",
                "headers": {},
                "tool_names": ["email.reply", "email-reply", "email_reply"],
            }
        },
        resources_server_name="workplace",
        resources_server_base_url="http://resources",
    )
    assert server is not None

    aliases = build_mcp_tool_aliases(
        server,
        wire_name=lambda server_name, tool_name: (
            f"mcp__{server_name}__{tool_name.replace('.', '_').replace('-', '_')}"
        ),
    )

    assert aliases == {}


def test_response_provenance_joins_only_matching_function_calls() -> None:
    server = parse_rollout_mcp_server(
        {
            "mcp": {
                "server_name": "workplace",
                "headers": {},
                "tool_names": ["reply"],
            }
        },
        resources_server_name="workplace",
        resources_server_base_url="http://resources",
    )
    assert server is not None
    aliases = build_mcp_tool_aliases(
        server,
        wire_name=lambda server_name, tool_name: f"mcp__{server_name}__{tool_name}",
    )
    assert aliases is not None
    response = NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 0,
            "model": "model",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "function_call",
                    "call_id": "mcp-call",
                    "name": "mcp__workplace__reply",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "built-in-call",
                    "name": "terminal",
                    "arguments": "{}",
                },
            ],
        }
    )

    provenance = provenance_from_response_aliases(response, aliases)

    assert {call_id: identity.model_dump() for call_id, identity in provenance.items()} == {
        "mcp-call": {"server_name": "workplace", "tool_name": "reply"}
    }
