# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.tests.test_app import agent, close, open_session


async def test_embedded_close_replays_evidence_and_rejects_another_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, runner_run = agent(monkeypatch)
    request = await open_session(instance)
    body = NeMoGymResponseCreateParamsNonStreaming(input="question")
    await instance.responses(request, body)
    receipt = await close(instance)
    assert receipt.agent_observations.source == "nooa"
    assert set(receipt.model_dump()) == set(AgentCloseSessionResponse.model_fields)
    assert await close(instance) == receipt
    with pytest.raises(HTTPException, match="closing|closed"):
        await instance.responses(request, body)
    runner_run.assert_awaited_once()


async def test_embedded_close_before_activation_releases_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, runner_run = agent(monkeypatch)
    request = await open_session(instance)
    receipt = await close(instance)
    assert receipt.agent_observations is None
    assert await close(instance) == receipt
    with pytest.raises(HTTPException, match="closing|closed"):
        await instance.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="late activation"))
    runner_run.assert_not_awaited()
