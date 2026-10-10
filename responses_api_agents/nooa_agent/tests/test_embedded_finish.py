# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentCloseSessionResponse
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.tests.test_app import agent, close, open_session, seed


async def test_embedded_finish_freezes_activation_and_replays_existing_receipt() -> None:
    instance = agent()
    request = await open_session(instance)
    close_body = AgentCloseSessionRequest(agent_session_id="session", episode_id=seed().episode_id)
    with pytest.raises(HTTPException, match="not finished"):
        await instance.finish_agent_session(request, close_body)
    body = NeMoGymResponseCreateParamsNonStreaming(input="question")
    await instance.responses(request, body)
    receipt = await instance.finish_agent_session(request, close_body)
    assert receipt.agent_observations.source == "nooa"
    assert set(receipt.model_dump()) == set(AgentCloseSessionResponse.model_fields)
    assert await instance.finish_agent_session(request, close_body) == receipt
    with pytest.raises(HTTPException, match="closing"):
        await instance.responses(request, body)
    assert await close(instance) == receipt
    assert await instance.finish_agent_session(request, close_body) == receipt
    instance.runner.run.assert_awaited_once()
