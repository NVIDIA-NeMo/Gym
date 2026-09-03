# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import MCPToolCallProvenance
from nemo_gym.mcp import (
    build_mcp_tool_aliases,
    build_mcp_verify_payload,
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
                "tool_names": ["reply"],
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
        ({"server_name": []}, "server_name"),
        ({"url_path": []}, "url_path"),
        ({"transport": []}, "transport"),
        ({"headers": []}, "headers"),
        ({"headers": {"Authorization": {}}}, "scalar values"),
        ({"tool_names": "reply"}, "tool_names"),
        ({"tool_names": ["reply", ""]}, "non-empty strings"),
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

    assert aliases is None


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

    assert provenance is not None
    assert {call_id: identity.model_dump() for call_id, identity in provenance.items()} == {
        "mcp-call": {"server_name": "workplace", "tool_name": "reply"}
    }


def test_parse_rollout_mcp_server_ignores_legacy_non_object_metadata() -> None:
    logger = MagicMock()

    assert (
        parse_rollout_mcp_server(
            {"mcp": None},
            resources_server_name="workplace",
            resources_server_base_url="http://resources",
            logger=logger,
        )
        is None
    )
    logger.warning.assert_called_once()


def test_parse_rollout_mcp_server_defaults_empty_path_and_transport() -> None:
    server = parse_rollout_mcp_server(
        {"mcp": {"url_path": "", "transport": ""}},
        resources_server_name="workplace",
        resources_server_base_url="http://resources",
    )

    assert server is not None
    assert server.url == "http://resources/mcp"
    assert server.transport == "http"


def test_response_provenance_returns_unknown_for_unmatched_mcp_call() -> None:
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
                    "call_id": "call",
                    "name": "mcp__external__reply",
                    "arguments": "{}",
                }
            ],
        }
    )

    aliases = {
        "mcp__workplace__reply": MCPToolCallProvenance(server_name="workplace", tool_name="reply"),
    }
    assert provenance_from_response_aliases(response, aliases) is None


def test_build_verify_payload_strips_caller_provenance_and_preserves_unknown() -> None:
    body = MagicMock()
    body.model_dump.return_value = {
        "mcp_tool_call_provenance": {
            "stale": {"server_name": "workplace", "tool_name": "forged"},
        }
    }
    response = NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 0,
            "model": "model",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [],
        }
    )

    payload = build_mcp_verify_payload(body, response, None)

    assert "mcp_tool_call_provenance" not in payload
    assert payload["response"]["id"] == "response"
    assert build_mcp_verify_payload(body, response, {})["mcp_tool_call_provenance"] == {}
