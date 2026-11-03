# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from fastapi import HTTPException

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.tests.test_app import agent, close, open_session


async def test_embedded_close_replays_evidence_and_rejects_another_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, runner_run = agent(monkeypatch)
    request = await open_session(instance)
    body = NeMoGymResponseCreateParamsNonStreaming(input="question")
    await instance.responses(request, body)
    receipt = await close(instance)
    assert receipt.agent_observations.source == "nooa"
    # Continuation-only fields stay absent from legacy receipts.
    assert set(receipt.model_dump()) == {"agent_session_id", "agent_observations", "resources_cookies"}
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


async def test_embedded_activation_join_is_bounded_and_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, runner_run = agent(monkeypatch)
    instance.config.session_close_timeout_seconds = 0.01
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def run(_):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            raise

    runner_run.side_effect = run
    request = await open_session(instance)
    state = instance._require_agent_session("session")
    activation = asyncio.create_task(
        instance.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="question"))
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        with pytest.raises(TimeoutError, match="activation did not stop"):
            await asyncio.wait_for(close(instance), timeout=1)
        assert cancelled.is_set()
        assert not state.execution.done()
        assert instance._session_records["session"].state is state
    finally:
        release.set()
        with pytest.raises(HTTPException, match="Agent session is closing") as caught:
            await activation
        assert caught.value.status_code == 409
    receipt = await close(instance)
    assert await close(instance) == receipt
