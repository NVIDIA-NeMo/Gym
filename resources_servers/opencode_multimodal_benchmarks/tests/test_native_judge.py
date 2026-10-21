# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.judge import JudgeError
from nemo_gym.server_utils import ServerClient
from resources_servers.opencode_multimodal_benchmarks.agentx import AgentXConfig, AgentXServer
from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksConfig, VisualBenchmarksServer
from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request


@pytest.mark.asyncio
@pytest.mark.parametrize("agentx", [False, True])
async def test_native_judge_uses_captured_model_route_without_local_key(monkeypatch, agentx):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = MagicMock(spec=ServerClient)
    http = MagicMock()
    http.status = 200
    http.read = AsyncMock(
        return_value=json.dumps(
            {
                "id": "judge-call-123",
                "object": "chat.completion",
                "created": 1,
                "model": "judge",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"Score": 0.65}' if agentx else "A",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )
    client.post = AsyncMock(return_value=http)
    cfg = AgentXConfig if agentx else VisualBenchmarksConfig
    cls = AgentXServer if agentx else VisualBenchmarksServer
    server = cls(
        config=cfg(
            host="localhost",
            port=1,
            name="visual",
            entrypoint="app.py",
            judge_server="visual_judge",
            judge_attempts=1,
            judge_model="test-judge",
        ),
        server_client=client,
    )
    body = request("AgentX_GoalAcc" if agentx else "VStarBench", "the fuzzy feline")
    if agentx:
        body.verifier_metadata.update(GT_json="reference", query_type="subjective", query_type_raw="Subjective")
    result = await server.verify(body)
    assert result.reward == (0.65 if agentx else 1)
    assert client.post.call_args.kwargs["server_name"] == "visual_judge"
    assert client.post.call_args.kwargs["url_path"] == "/v1/chat/completions"
    assert "headers" not in client.post.call_args.kwargs
    client.post.side_effect = RuntimeError("judge transport unavailable")
    with pytest.raises(JudgeError, match="judge transport unavailable"):
        await server.verify(body)


@pytest.mark.asyncio
@pytest.mark.parametrize("agentx", [False, True])
@pytest.mark.parametrize("missing", ["model", "route"])
async def test_missing_judge_configuration_fails_before_network(monkeypatch, agentx, missing):
    from resources_servers.opencode_multimodal_benchmarks import agentx as agentx_module
    from resources_servers.opencode_multimodal_benchmarks import app as app_module

    transport = AsyncMock(side_effect=AssertionError("Must not make a network call"))
    monkeypatch.setattr(app_module, "request", transport)
    monkeypatch.setattr(agentx_module, "request", transport)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    config_type = AgentXConfig if agentx else VisualBenchmarksConfig
    server_type = AgentXServer if agentx else VisualBenchmarksServer
    config = config_type(host="localhost", port=1, name="visual", entrypoint="app.py")
    assert config.judge_endpoint is None and config.judge_model is None
    if missing == "route":
        config.judge_model = "test-judge"
    server = server_type(config=config, server_client=MagicMock(spec=ServerClient))
    with pytest.raises(JudgeError, match="Configure judge_"):
        if agentx:
            await server._judge_answer({}, "answer")
        else:
            await server._judge("prompt")
    transport.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("agentx", [False, True])
@pytest.mark.parametrize("recover", [False, True])
async def test_native_judge_deadline_cancels_attempt_and_preserves_retry_policy(monkeypatch, agentx, recover):
    from resources_servers.opencode_multimodal_benchmarks import agentx as agentx_module
    from resources_servers.opencode_multimodal_benchmarks import app as app_module

    client = MagicMock(spec=ServerClient)
    config_type = AgentXConfig if agentx else VisualBenchmarksConfig
    server_type = AgentXServer if agentx else VisualBenchmarksServer
    server = server_type(
        config=config_type(
            host="localhost",
            port=1,
            name="visual",
            entrypoint="app.py",
            judge_server="visual_judge",
            judge_model="test-judge",
            judge_timeout=0.01,
            judge_concurrency=1,
            judge_attempts=2 if recover else 1,
        ),
        server_client=client,
    )
    attempts = 0
    cancelled = []

    async def post(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(attempts)
                raise
        response = MagicMock(status=200)
        response.read = AsyncMock(
            return_value=json.dumps(
                {
                    "id": "judge",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "judge",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"Score": 1}' if agentx else "A",
                            },
                        }
                    ],
                }
            ).encode()
        )
        return response

    client.post = AsyncMock(side_effect=post)
    module = agentx_module if agentx else app_module
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    body = request("AgentX_GoalAcc" if agentx else "VStarBench", "the fuzzy feline")
    if agentx:
        body.verifier_metadata.update(GT_json="reference", query_type="objective", query_type_raw="Objective")
    if recover:
        result = await asyncio.wait_for(server.verify(body), timeout=1)
        assert result.reward == 1
    else:
        with pytest.raises(JudgeError, match="TimeoutError"):
            await asyncio.wait_for(server.verify(body), timeout=1)
    assert attempts == (2 if recover else 1)
    assert cancelled == [1]
    assert not server._judge_sem.locked()
