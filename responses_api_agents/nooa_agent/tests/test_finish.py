# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.app import NOOAAgent, NOOASessionState
from responses_api_agents.nooa_agent.tests.test_app import request, result, seed
from responses_api_agents.nooa_agent.tests.test_config import agent_config, invocation_config
from responses_api_agents.nooa_agent.tests.test_sandbox_runner import runner


async def test_finish_freezes_execution_preserves_services_and_close_is_idempotent(monkeypatch) -> None:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    instance = NOOAAgent(config=agent_config(nooa=invocation_config(execution_mode="sandboxed")), server_client=client)
    r, _ = runner()
    r.run = AsyncMock(return_value=result())
    r.close = AsyncMock()

    async def seed_state(self, body):
        return NOOASessionState(request=body, runner=r)

    monkeypatch.setattr(NOOAAgent, "_seed_agent_session_state", seed_state)
    req = request()
    await instance.seed_agent_session(req, seed())
    close_body = AgentCloseSessionRequest(agent_session_id="session", episode_id=seed().episode_id)
    with pytest.raises(HTTPException, match="not finished"):
        await instance.finish_agent_session(req, close_body)
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    await instance.responses(req, body)
    first = await instance.finish_agent_session(req, close_body)
    second = await instance.finish_agent_session(req, close_body)
    assert first == second
    assert first.partial_response is not None
    assert first.agent_trajectory is not None
    r.close.assert_not_awaited()
    with pytest.raises(HTTPException, match="closing"):
        await instance.responses(req, body)
    with pytest.raises(HTTPException, match="cookie"):
        await instance.finish_agent_session(request(session_id="other"), close_body)
    await instance.close_agent_session(req, close_body)
    await instance.close_agent_session(req, close_body)
    r.close.assert_awaited_once()
