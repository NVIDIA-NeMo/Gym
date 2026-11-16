# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA NeMo Gym (Apache-2.0), original path: tests/unit_tests/models/test_rollout_budget.py
# Source: https://gitlab-master.nvidia.com/tongc/gym/-/tree/d8c548915431e59169890d34cd8cea56ef2a905e
import asyncio
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.episode_context import rollout_context
from nemo_gym.core.responses_api_types import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.models.model_call_capture import _fail_uncommitted_external_call
from nemo_gym.models.rollout_budget import BUDGET_STOP_REASON, RolloutGenerationBudget, RolloutGenerationBudgetConfig
from nemo_gym.models.token_capture.lineage import InMemoryLineageStore
from nemo_gym.models.token_capture.sink import CaptureContext


def _reserve(root):
    budget = RolloutGenerationBudget(
        RolloutGenerationBudgetConfig(max_calls=10, state_dir=root, agent_servers=["agent"])
    )
    return budget.reserve("shared")


def test_processes_and_restarts_share_hard_cap(tmp_path):
    with ProcessPoolExecutor(max_workers=4) as pool:
        assert sorted(value for value in pool.map(_reserve, [tmp_path] * 40) if value) == list(range(1, 11))
    assert not _reserve(tmp_path)
    budget = RolloutGenerationBudget(
        RolloutGenerationBudgetConfig(max_calls=10, state_dir=tmp_path, agent_servers=["agent"])
    )
    assert budget.reserve("another-rollout")


class Model(SimpleResponsesAPIModel):
    async def responses(self, body):
        # A Responses -> Chat adapter invokes the concrete method, not dispatch.
        return await self.chat_completions(body=body)

    async def chat_completions(self, body):
        return {"output": []}


def _model(tmp_path):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client.post = AsyncMock(
        return_value=SimpleNamespace(ok=True, json=AsyncMock(return_value={"rollout_id": "r", "stopped": True}))
    )
    model = Model(
        config=BaseResponsesAPIModelConfig(
            host="",
            port=0,
            openai_api_key="test",
            entrypoint="",
            name="model",
            rollout_generation_budget={"max_calls": 10, "state_dir": tmp_path, "agent_servers": ["agent"]},
        ),
        server_client=client,
    )
    model.setup_webserver()
    return model, client


@pytest.mark.asyncio
async def test_dispatch_counts_generation_once_and_stops_before_eleventh(tmp_path, monkeypatch):
    model, client = _model(tmp_path)
    generated = AsyncMock(return_value={"output": []})
    monkeypatch.setattr(Model, "chat_completions", generated)
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
    response_params = NeMoGymResponseCreateParamsNonStreaming(input="same input")
    chat_params = NeMoGymChatCompletionCreateParamsNonStreaming(model="policy", messages=[])
    with rollout_context("r"):
        for index in range(10):
            if index % 2:
                await model._invoke_chat_completions(request, chat_params)
            else:
                await model._invoke_responses(request, response_params)
        with pytest.raises(HTTPException) as error:
            await model._invoke_responses(request, response_params)
    assert generated.await_count == 10
    assert error.value.status_code == 409
    client.post.assert_awaited_once_with(
        server_name="agent", url_path="/rollouts/r/stop", json={"reason": BUDGET_STOP_REASON}
    )


@pytest.mark.asyncio
async def test_refused_generation_has_no_capture_intent_or_backend_call(tmp_path, monkeypatch):
    import nemo_gym.models.base_responses_api_model as base
    import nemo_gym.models.rollout_budget as module

    model, _ = _model(tmp_path)
    ledger = InMemoryLineageStore()
    context = CaptureContext(
        capture_key="r",
        model_call_id="blocked-eleventh",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        parent_resolution_mode="source_history",
    )
    monkeypatch.setattr(base, "current_capture_context", lambda: context)
    monkeypatch.setattr(module, "current_capture_context", lambda: context)
    resolve = AsyncMock()
    register = AsyncMock()
    capture = AsyncMock()
    monkeypatch.setattr(base, "resolve_parent", resolve)
    monkeypatch.setattr(base, "register_call_intent", register)
    monkeypatch.setattr(base, "capture_tokens", capture)
    for _ in range(10):
        assert model._generation_budget.reserve("r")
    with rollout_context("r"), pytest.raises(HTTPException):
        await model._invoke_responses(
            SimpleNamespace(is_disconnected=AsyncMock(return_value=True)),
            NeMoGymResponseCreateParamsNonStreaming(input="test"),
        )
    assert context.refused
    assert not context.intent_registered
    register.assert_not_awaited()
    capture.assert_not_awaited()
    await _fail_uncommitted_external_call(context)
    manifest = await ledger.manifest("r")
    assert manifest["pending_call_ids"] == []
    assert manifest["attempted_call_ids"] == []
    assert manifest["failures"] == []


@pytest.mark.asyncio
async def test_missing_correlation_and_stop_failure_fail_closed(tmp_path, monkeypatch):
    model, client = _model(tmp_path)
    generated = AsyncMock(return_value={"output": []})
    monkeypatch.setattr(Model, "chat_completions", generated)
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
    with pytest.raises(HTTPException, match="correlation"):
        await model._invoke_responses(request, NeMoGymResponseCreateParamsNonStreaming(input="test"))
    generated.assert_not_awaited()
    for _ in range(10):
        model._generation_budget.reserve("r")
    client.post.side_effect = RuntimeError("transport failed")
    with rollout_context("r"), pytest.raises(RuntimeError, match="transport failed"):
        await model._generation_budget.admit(request, client)


@pytest.mark.asyncio
async def test_concurrent_admissions_never_exceed_limit(tmp_path):
    model, client = _model(tmp_path)
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
    with rollout_context("r"):
        results = await asyncio.gather(
            *(model._generation_budget.admit(request, client) for _ in range(25)), return_exceptions=True
        )
    assert sorted(result for result in results if isinstance(result, int)) == list(range(1, 11))
    assert sum(isinstance(result, HTTPException) and result.status_code == 409 for result in results) == 15


@pytest.mark.asyncio
async def test_multi_completion_cannot_bypass_budget(tmp_path):
    model, _ = _model(tmp_path)
    with rollout_context("r"), pytest.raises(HTTPException, match="one completion"):
        await model._invoke_chat_completions(
            SimpleNamespace(), NeMoGymChatCompletionCreateParamsNonStreaming(model="policy", messages=[], n=2)
        )
    assert not list(tmp_path.glob("*.generation-count"))


@pytest.mark.asyncio
async def test_stop_waits_for_cli_disconnect(tmp_path, monkeypatch):
    import nemo_gym.models.rollout_budget as module

    model, client = _model(tmp_path)
    for _ in range(10):
        model._generation_budget.reserve("r")
    request = SimpleNamespace(is_disconnected=AsyncMock(side_effect=[False, False, True]))
    sleep = AsyncMock()
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    with rollout_context("r"), pytest.raises(HTTPException):
        await model._generation_budget.admit(request, client)
    assert sleep.await_count == 2
    assert request.is_disconnected.await_count == 3
