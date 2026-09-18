# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.miniswe_sandboxed_agent import app as module


@pytest.mark.parametrize("capture", [False, True])
async def test_proxy_preserves_runner_result_and_stable_session(monkeypatch, capture):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": capture, "token_id_capture": {"enabled": capture}}
    client.post = AsyncMock()
    params = {"input": []}
    result = dict(
        responses_create_params=params,
        reward=0.5,
        evaluation_completed=True,
        session_id="task",
        harness_version="test",
        response=NeMoGymResponse(
            id="response",
            created_at=0,
            model="model",
            object="response",
            output=[],
            tools=[],
            tool_choice="auto",
            parallel_tool_calls=False,
        ).model_dump(),
    )
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", AsyncMock(return_value=result))
    proxy = module.MiniSWESandboxedAgent(
        config=module.MiniSWESandboxedConfig(
            host="localhost",
            port=1,
            entrypoint="app.py",
            name="agent",
            token_id_capture=capture,
            resources_server={"type": "resources_servers", "name": "resources"},
        ),
        server_client=client,
    )
    body = module.MiniSWERunRequest(responses_create_params=params, _ng_rollout_id="rollout-1", task_name="generic")
    response = await proxy.run(SimpleNamespace(cookies={"session": "cookie"}, session={SESSION_ID_KEY: "owner"}), body)
    assert response.reward == 0.5 and response.harness_version == "test"
    call = client.post.await_args.kwargs
    assert call["url_path"] == "/run" and call["server_name"] == "resources"
    assert call["cookies"] == {"session": "cookie"}
    assert call["json"]["client_session_id"] == "owner"
    assert call["json"]["rollout_id"] == "rollout-1"
    assert call["json"]["capture_model_calls"] == capture
    assert call["json"]["capture_token_ids"] == capture
    assert "client_session_id" not in body.model_dump()
    client.post.assert_awaited_once()
