# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
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


def test_responses_route_is_registered() -> None:
    paths = {route.path for route in _agent().setup_webserver().routes}
    assert "/v1/responses" in paths
    assert "/act" not in paths


@pytest.mark.asyncio
async def test_responses_reads_action_from_own_terminal() -> None:
    agent = _agent()
    request = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "user", "content": "Earlier observation."},
            {"role": "assistant", "content": "[check]"},
            {"role": "user", "content": "Your card is K."},
        ]
    )

    with patch("builtins.input", return_value="[bet]") as mocked_input:
        response = await agent.responses(request)

    assert response.output_text == "[bet]"
    assert "Player 0 (player0)" in mocked_input.call_args.args[0]
    assert "Your card is K." in mocked_input.call_args.args[0]
    assert "Earlier observation." not in mocked_input.call_args.args[0]


@pytest.mark.asyncio
async def test_responses_accepts_string_input() -> None:
    agent = _agent()
    request = NeMoGymResponseCreateParamsNonStreaming(input="Your card is Q.")

    with patch("builtins.input", return_value="[check]") as mocked_input:
        response = await agent.responses(request)

    assert response.output_text == "[check]"
    assert "Your card is Q." in mocked_input.call_args.args[0]
