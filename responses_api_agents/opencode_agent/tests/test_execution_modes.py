# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig, OpenCodeAgentRunRequest
from responses_api_agents.opencode_agent.tests.test_native_sessions import seed


def make_agent(mode: str) -> OpenCodeAgent:
    return OpenCodeAgent(
        config=OpenCodeAgentConfig(
            host="localhost",
            port=8001,
            name="opencode",
            entrypoint="app.py",
            execution_mode=mode,
            model_server={"type": "responses_api_models", "name": "policy"},
            opencode_version="1.17.11",
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize("mode", ["local", "sandbox", "legacy_sandbox"])
def test_constructing_agent_never_installs_or_creates_legacy_bridge(mode: str) -> None:
    with patch("responses_api_agents.opencode_agent.app.ensure_opencode") as install:
        agent = make_agent(mode)
    install.assert_not_called()
    assert agent._legacy_agent is None


async def test_local_runtime_install_is_deferred_until_execution() -> None:
    with patch("responses_api_agents.opencode_agent.app.ensure_opencode", side_effect=RuntimeError("install failed")):
        agent = make_agent("local")
        assert not agent._local_runtime_ready
        with pytest.raises(RuntimeError, match="install failed"):
            await agent._run_opencode("task", None)
        assert not agent._local_runtime_ready


async def test_unseeded_native_mode_cannot_fall_back_to_host_or_bridge() -> None:
    agent = make_agent("sandbox")
    request = Request({"type": "http", "session": {}})
    with patch.object(agent, "_create_episode", AsyncMock()) as local:
        with pytest.raises(HTTPException) as response_error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        with pytest.raises(HTTPException) as run_error:
            await agent.run(request, OpenCodeAgentRunRequest(responses_create_params={"input": "task"}))
    assert response_error.value.status_code == run_error.value.status_code == 409
    local.assert_not_awaited()
    assert agent._legacy_agent is None
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize("mode", ["local", "legacy_sandbox"])
@pytest.mark.parametrize("marker", [None, "closed-session"])
async def test_native_markers_never_enter_other_modes(mode: str, marker: str | None) -> None:
    agent = make_agent(mode)
    request = Request({"type": "http", "session": {"nemo_gym_opencode_native_session": marker}})
    with patch.object(agent, "_create_episode", AsyncMock()) as local:
        with pytest.raises(HTTPException) as response_error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        with pytest.raises(HTTPException) as run_error:
            await agent.run(request, OpenCodeAgentRunRequest(responses_create_params={"input": "task"}))
    assert response_error.value.status_code == run_error.value.status_code == 409
    local.assert_not_awaited()
    assert agent._legacy_agent is None


@pytest.mark.parametrize("mode", ["local", "legacy_sandbox"])
async def test_native_seed_requires_explicit_sandbox_mode(mode: str) -> None:
    agent = make_agent(mode)
    with pytest.raises(HTTPException, match="execution_mode=sandbox"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    assert not agent._native_sessions


async def test_explicit_legacy_dispatch_retains_client_request_and_configuration() -> None:
    agent = make_agent("legacy_sandbox")
    agent.config.opencode_max_context_window = 32768
    agent.config.sandbox_timeout = 1200
    request = Request({"type": "http", "session": {}})
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    legacy = agent._legacy()
    assert legacy is agent._legacy()
    assert legacy.server_client is agent.server_client
    assert legacy.config.opencode_max_context_window == 32768
    assert legacy.config.sandbox_timeout == 1200
    with patch.object(type(legacy), "responses", AsyncMock(return_value="legacy response")) as responses:
        assert await agent.responses(request, body) == "legacy response"
    responses.assert_awaited_once_with(request, body)
    run_body = OpenCodeAgentRunRequest(responses_create_params={"input": "task"})
    with patch.object(type(legacy), "run", AsyncMock(return_value="legacy result")) as run:
        assert await agent.run(request, run_body) == "legacy result"
    run.assert_awaited_once_with(request, run_body)


def test_legacy_mode_retains_its_pinned_default_without_mutating_local_config() -> None:
    agent = make_agent("legacy_sandbox")
    agent.config.opencode_version = None
    legacy = agent._legacy()
    assert legacy.config.opencode_version == "1.17.11"
    assert agent.config.opencode_version is None
