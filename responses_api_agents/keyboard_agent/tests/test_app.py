# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from nemo_gym.multi_agent import AgentActRequest
from nemo_gym.server_utils import ServerClient
from responses_api_agents.keyboard_agent.app import KeyboardAgent, KeyboardAgentConfig


def _agent() -> KeyboardAgent:
    config = KeyboardAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="kuhn_player0",
        agent_id="player0",
        display_name="Player 0",
    )
    return KeyboardAgent(config=config, server_client=MagicMock(spec=ServerClient))


def test_act_route_is_registered() -> None:
    assert "/act" in {route.path for route in _agent().setup_webserver().routes}


@pytest.mark.asyncio
async def test_act_reads_action_from_own_terminal() -> None:
    agent = _agent()
    request = AgentActRequest(agent_id="player0", observation="Your card is K.")

    with patch("builtins.input", return_value="[bet]") as mocked_input:
        response = await agent.act(request)

    assert response.action == "[bet]"
    assert "Player 0 (player0)" in mocked_input.call_args.args[0]
    assert "Your card is K." in mocked_input.call_args.args[0]


@pytest.mark.asyncio
async def test_rejects_another_agents_request() -> None:
    agent = _agent()
    request = AgentActRequest(agent_id="player1", observation="private")

    with pytest.raises(HTTPException, match="controls player0"):
        await agent.act(request)
