# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise swallowed model exceptions through the pinned native LLM node."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError

from responses_api_agents.spatialclaw_agent import app


@pytest.mark.skipif(not os.environ.get("SPATIALCLAW_ROOT"), reason="requires pinned SPATIALCLAW_ROOT")
@pytest.mark.parametrize("status,masked", [(400, False), (503, True)])
async def test_native_llm_step_preserves_transport_failure(monkeypatch, tmp_path, status, masked):
    root = app._validate_spatialclaw_checkout(Path(os.environ["SPATIALCLAW_ROOT"]), app.SPATIALCLAW_COMMIT)
    monkeypatch.syspath_prepend(str(root))
    from spatial_agent.nodes.llm_step_node import llm_step_node

    journal = tmp_path / "gym-model-responses.jsonl"
    transport = app._AiohttpChatCompletions("http://model/v1", str(journal))
    request = AsyncMock(return_value=SimpleNamespace(cookies={}))
    monkeypatch.setattr(app, "http_request", request)
    monkeypatch.setattr(
        app,
        "raise_for_status",
        AsyncMock(
            side_effect=ClientResponseError(
                SimpleNamespace(real_url="http://model/v1/chat/completions"),
                (),
                status=status,
                message="model unavailable",
            )
        ),
    )

    async def generate(messages, **kwargs):
        return await transport.create(model="test", messages=messages)

    state = {"messages": [], "session_id": "native", "step_count": 0, "failure_count": 0}
    config = {
        "configurable": {
            "llm_client": SimpleNamespace(generate=generate),
            "agent_config": SimpleNamespace(main_params={}),
        }
    }
    for _ in range(2):
        result = await llm_step_node(state, config)
        assert result["current_llm_response"] is None
    request.assert_awaited_once()
    failure = app._model_transport_failure(journal)
    assert failure["http_status"] == status
    assert failure["mask_sample"] is masked
