# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.multi_agent_agent.app import (
    KeyboardController,
    KeyboardControllerConfig,
    MultiAgentAgent,
    MultiAgentAgentConfig,
    MultiAgentRunRequest,
)


class _FakeHttpResponse:
    def __init__(self, payload: dict, cookies: dict | None = None) -> None:
        self._payload = payload
        self.cookies = cookies or {}
        self.status = 200
        self.ok = True

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode()

    @property
    def content(self):
        payload = self._payload

        class _Body:
            async def read(self):
                return json.dumps(payload).encode()

        return _Body()

    def raise_for_status(self):
        return None


class _ScriptedController:
    def __init__(self, action: str) -> None:
        self.action = action
        self.observations: list[str] = []

    async def act(self, agent_id: str, observation: str, history: list) -> str:
        self.observations.append(observation)
        return self.action


def _agent(max_turns: int = 8) -> MultiAgentAgent:
    config = MultiAgentAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="multi_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="environment"),
        controllers={
            "player0": KeyboardControllerConfig(display_name="Player 0"),
            "player1": KeyboardControllerConfig(display_name="Player 1"),
        },
        focal_agent="player0",
        max_turns=max_turns,
    )
    return MultiAgentAgent(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.asyncio
async def test_responses_endpoint_requires_full_episode_run() -> None:
    agent = _agent()
    with pytest.raises(HTTPException, match="Use /run"):
        await agent.responses({"input": "play"})


def test_builds_one_keyboard_controller_per_configured_agent() -> None:
    controllers = _agent()._controllers()
    assert set(controllers) == {"player0", "player1"}
    assert all(isinstance(controller, KeyboardController) for controller in controllers.values())


@pytest.mark.asyncio
async def test_keyboard_controller_reads_input_without_blocking_event_loop() -> None:
    controller = KeyboardController("Alice")
    with patch("builtins.input", return_value="[check]") as mocked_input:
        assert await controller.act("player0", "Private observation", []) == "[check]"
    assert "Alice (player0)" in mocked_input.call_args.args[0]
    assert "Private observation" in mocked_input.call_args.args[0]


@pytest.mark.asyncio
async def test_orchestrator_routes_private_observations_and_preserves_cookies() -> None:
    agent = _agent()
    player0 = _ScriptedController("[check]")
    player1 = _ScriptedController("[check]")
    agent._controllers = MagicMock(return_value={"player0": player0, "player1": player1})
    calls = []
    responses = [
        _FakeHttpResponse(
            {"active_agent": "player0", "observation": "P0 private", "info": {"seed": 0}},
            {"session": "reset"},
        ),
        _FakeHttpResponse(
            {
                "active_agent": "player1",
                "observation": "P1 private",
                "rewards": {},
                "terminated": False,
                "truncated": False,
                "info": {},
            },
            {"session": "step-1"},
        ),
        _FakeHttpResponse(
            {
                "active_agent": None,
                "observation": None,
                "rewards": {"player0": 1.0, "player1": -1.0},
                "terminated": True,
                "truncated": False,
                "info": {"history": "check-check"},
            },
            {"session": "step-2"},
        ),
    ]

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        calls.append((url_path, cookies))
        return responses.pop(0)

    agent.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock()
    request.cookies = {"incoming": "cookie"}
    body = MultiAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})

    result = await agent.run(request, body)

    assert player0.observations == ["P0 private"]
    assert player1.observations == ["P1 private"]
    assert result.agent_rewards == {"player0": 1.0, "player1": -1.0}
    assert result.reward == 1.0
    assert result.terminated is True
    assert [call[0] for call in calls] == ["/reset", "/step", "/step"]
    assert calls[0][1] == {"incoming": "cookie"}
    assert calls[1][1] == {"session": "reset"}
    assert calls[2][1] == {"session": "step-1"}


@pytest.mark.asyncio
async def test_turn_limit_truncates_and_closes_environment() -> None:
    agent = _agent(max_turns=1)
    controller = _ScriptedController("[check]")
    agent._controllers = MagicMock(return_value={"player0": controller, "player1": _ScriptedController("[check]")})
    paths = []
    responses = {
        "/reset": _FakeHttpResponse({"active_agent": "player0", "observation": "P0", "info": {}}),
        "/step": _FakeHttpResponse(
            {
                "active_agent": "player1",
                "observation": "P1",
                "rewards": {},
                "terminated": False,
                "truncated": False,
                "info": {},
            }
        ),
        "/close": _FakeHttpResponse({"closed": True}),
    }

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        paths.append(url_path)
        return responses[url_path]

    agent.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock()
    request.cookies = {}
    body = MultiAgentRunRequest(responses_create_params={"input": "play"})

    result = await agent.run(request, body)

    assert result.truncated is True
    assert result.terminated is False
    assert paths == ["/reset", "/step", "/close"]
