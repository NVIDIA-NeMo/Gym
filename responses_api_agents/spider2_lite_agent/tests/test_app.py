# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi import Response

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import SimpleAgentConfig
from responses_api_agents.spider2_lite_agent.app import Spider2LiteAgent


@pytest.mark.parametrize(
    "content",
    [
        '{"error":{"message":"litellm.ContextWindowExceededError: Max Input Tokens=131072, Got=743739"}}',
        '{"error":{"code":"context_length_exceeded"}}',
        "prompt is too long for the context window",
    ],
)
async def test_context_window_rejection_becomes_empty_response(content: str) -> None:
    agent = _agent_with_upstream_error(_upstream_error(content=content))
    request = MagicMock(path_params={}, cookies={"spider-session": "cookie"})
    response = Response()

    result = await agent.responses(
        request,
        response,
        NeMoGymResponseCreateParamsNonStreaming(input="oversized prompt"),
    )

    assert result.status == "failed"
    assert result.output_text == ""
    assert "spider-session=cookie" in response.headers["set-cookie"]


@pytest.mark.parametrize(
    "status, content",
    [
        (400, '{"error":{"message":"invalid model"}}'),
        (429, "context window"),
    ],
)
async def test_unrelated_upstream_errors_still_fail(status: int, content: str) -> None:
    error = _upstream_error(status=status, content=content)
    agent = _agent_with_upstream_error(error)

    with pytest.raises(ClientResponseError) as exc_info:
        await agent.responses(
            MagicMock(path_params={}, cookies={}),
            Response(),
            NeMoGymResponseCreateParamsNonStreaming(input="ordinary prompt"),
        )

    assert exc_info.value is error


def _agent_with_upstream_error(error: ClientResponseError) -> Spider2LiteAgent:
    """Build a Spider2LiteAgent whose next model-server call raises `error`."""
    config = SimpleAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="spider2_lite_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="spider2_lite"),
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    client.post = AsyncMock(side_effect=error)
    return Spider2LiteAgent(config=config, server_client=client)


def _upstream_error(*, status: int = 400, content: str) -> ClientResponseError:
    """Build the ClientResponseError that nemo_gym.server_utils.raise_for_status produces for a 4xx."""
    error = ClientResponseError(None, (), status=status, message="upstream failure")
    error.response_content = content.encode()
    return error
